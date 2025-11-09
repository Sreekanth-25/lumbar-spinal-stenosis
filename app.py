import uvicorn
import io
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Import your existing ML/processing logic
# These files (segmentation_inference.py, feature_extraction.py, utils_report.py)
# must be in the same directory.
from segmentation_inference import load_unet, segment_image, make_overlay
from feature_extraction import extract_features
from utils_report import severity_from_ratio, generate_summary, generate_pdf

# --- Configuration & Model Loading ---

# Create a directory for saving result images
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the segmentation model on startup
# Make sure "best_unet_resnet50.pth" is in the same folder
try:
    model = load_unet("best_unet_resnet50.pth")
    print("AI model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file 'best_unet_resnet50.pth' not found.")
    print("Please download the model file and place it in the same directory.")
    model = None
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    model = None


# Define the FastAPI app
app = FastAPI(title="LumbarPro Analyzer API")

# --- Pydantic Models for Data Validation ---

class ReportMetrics(BaseModel):
    area_px: int
    perimeter_px: float
    compactness: float
    opening_ratio: float

class ReportData(BaseModel):
    severity: str
    confidence: float
    metrics: ReportMetrics
    summary_html: str

# --- API Endpoints ---

@app.post("/analyze", response_model=ReportData)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded MRI image.
    Receives an image, runs segmentation, extracts features,
    and returns a JSON object with all results.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Server configuration error.")

    # Read image file
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # --- Run the full analysis pipeline ---
    # 1. Segment the image
    mask, _ = segment_image(model, pil_img)

    # 2. Extract features
    feats = extract_features(mask)

    # 3. Get severity and summary
    sev, conf = severity_from_ratio(feats["opening_ratio"])
    summary_html = generate_summary(sev, conf, feats)

    # 4. Create and save overlay image
    overlay_img = make_overlay(pil_img, mask)
    
    # Generate a unique filename for the result
    filename = f"{uuid.uuid4()}.png"
    overlay_path = os.path.join(RESULTS_DIR, filename)
    overlay_img.save(overlay_path)
    
    # URL path to access the saved image
    overlay_url = f"/{overlay_path}"

    # 5. Return all data as JSON
    return {
        "severity": sev,
        "confidence": conf,
        "metrics": feats,
        "summary_html": summary_html,
        "overlay_url": overlay_url
    }

@app.post("/download_report")
async def download_pdf_report(data: ReportData):
    """
    Generates a PDF report from the JSON analysis data.
    Receives JSON data, creates a PDF in memory,
    and returns it as a downloadable file.
    """
    try:
        # Generate the PDF bytes using your existing utility
        pdf_bytes = generate_pdf(
            filename="analysis_report.pdf",
            sev=data.severity,
            conf=data.confidence,
            feats=data.metrics.dict(), # Convert Pydantic model to dict
            summary_html=data.summary_html
        )
        
        # Create a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{data.severity.lower()}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")


# --- Static File Serving ---

# Mount the 'results' directory to serve generated images
app.mount(f"/{RESULTS_DIR}", StaticFiles(directory=RESULTS_DIR), name="results")

# Serve the main index.html file at the root
@app.get("/")
async def get_index():
    """Serves the main front-end HTML file."""
    return FileResponse("index.html")

# --- Run the Server ---
if __name__ == "__main__":
    if not model:
        print("\n--- WARNING: SERVER IS RUNNING WITHOUT A MODEL ---")
        print("--- Upload analysis will fail. Please add 'best_unet_resnet50.pth' ---\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
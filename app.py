import uvicorn
import io
import os
import uuid
import requests
import tqdm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

# --- NEW: Import CORS ---
from fastapi.middleware.cors import CORSMiddleware

# Import your existing ML/processing logic
from segmentation_inference import load_unet, segment_image, make_overlay
from feature_extraction import extract_features
from utils_report import severity_from_ratio, generate_summary, generate_pdf

# --- Configuration & Model Loading ---
MODEL_FILE = "best_unet_resnet50.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE" # <<<--- MAKE SURE YOU REPLACED THIS

if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found.")
    print(f"Downloading from {MODEL_URL}...")
    
    if "YOUR_FILE_ID_HERE" in MODEL_URL:
        print("\n" + "="*50)
        print("ERROR: Please update MODEL_URL in app.py")
        print("="*50 + "\n")
        model = None
    else:
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(MODEL_FILE, 'wb') as f, tqdm.tqdm(
                    desc=MODEL_FILE, total=total_size, unit='iB',
                    unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
            print("Model download complete.")
        except Exception as e:
            print(f"ERROR: Failed to download model: {e}")
            model = None
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
else:
    print(f"Model file '{MODEL_FILE}' already exists. Skipping download.")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

model_instance = None
if model is None and os.path.exists(MODEL_FILE):
    try:
        model_instance = load_unet(MODEL_FILE)
        print("AI model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from disk. {e}")
        model_instance = None
elif model is None:
     print("ERROR: Model not downloaded, cannot load.")

# Define the FastAPI app
app = FastAPI(title="LumbarPro Analyzer API")

# --- NEW: Add CORS Middleware ---
# This allows all origins (e.g., your GitHub Pages site)
# to make requests to your API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# --- End of CORS section ---


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

@app.post("/analyze", response_model=ReportData)
async def analyze_image(file: UploadFile = File(...)):
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model is not loaded. Server configuration error.")
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    mask, _ = segment_image(model_instance, pil_img)
    feats = extract_features(mask)
    sev, conf = severity_from_ratio(feats["opening_ratio"])
    summary_html = generate_summary(sev, conf, feats)
    overlay_img = make_overlay(pil_img, mask)
    
    filename = f"{uuid.uuid4()}.png"
    overlay_path = os.path.join(RESULTS_DIR, filename)
    overlay_img.save(overlay_path)
    overlay_url = f"/{overlay_path}"

    return {
        "severity": sev,
        "confidence": conf,
        "metrics": feats,
        "summary_html": summary_html,
        "overlay_url": overlay_url
    }

@app.post("/download_report")
async def download_pdf_report(data: ReportData):
    try:
        pdf_bytes = generate_pdf(
            filename="analysis_report.pdf",
            sev=data.severity,
            conf=data.confidence,
            feats=data.metrics.dict(),
            summary_html=data.summary_html
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{data.severity.lower()}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")

app.mount(f"/{RESULTS_DIR}", StaticFiles(directory=RESULTS_DIR), name="results")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    if not model_instance:
        print("\n--- WARNING: SERVER IS RUNNING WITHOUT A MODEL ---")
    
    # Use gunicorn for production, but uvicorn for local dev
    # You would run: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run(app, host="0.0.0.0", port=8000)

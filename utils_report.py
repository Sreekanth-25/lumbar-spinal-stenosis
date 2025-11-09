from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import re

# ------------------ Severity Logic ------------------
def severity_from_ratio(r):
    """Rule-based grading for opening_ratio typical 0.05–0.7 range."""
    r = max(0.0, min(float(r), 1.0))
    if r >= 0.55:
        sev, conf = "Normal", 0.85 + (r - 0.55) / 0.45 * 0.15
    elif r >= 0.4:
        sev, conf = "Mild", 0.7
    elif r >= 0.25:
        sev, conf = "Moderate", 0.6
    else:
        sev, conf = "Severe", 0.5
    return sev, round(min(conf, 1.0), 2)


# ------------------ Professional Narrative ------------------
def generate_summary(severity, confidence, feats):
    """Return formatted HTML summary without algorithm hints."""
    ratio = feats["opening_ratio"]
    area = feats["area_px"]
    perimeter = feats["perimeter_px"]
    compactness = feats["compactness"]

    parts = []
    parts.append("<ul style='font-size:15px; color:#dee2e6; line-height:1.6;'>")

    # Generic quantitative mention
    parts.append(
        f"<li>Calculated morphometric parameters include "
        f"an opening ratio of <b>{ratio:.3f}</b>, area of <b>{area}</b> px², "
        f"perimeter of <b>{perimeter}</b> px, and compactness index <b>{compactness}</b>.</li>"
    )

    if severity == "Normal":
        parts.append(
            "<li>The lumbar canal exhibits normal patency with preserved cerebrospinal fluid space. "
            "No significant narrowing or compression of the thecal sac is identified.</li>"
        )
        rec = "Routine clinical observation is adequate; no immediate intervention is required."
    elif severity == "Mild":
        parts.append(
            "<li>Mild reduction in canal calibre is observed, compatible with early degenerative changes. "
            "No definite neural compromise is apparent.</li>"
        )
        rec = "Recommend conservative management such as physiotherapy and posture training, with follow‑up if symptoms persist."
    elif severity == "Moderate":
        parts.append(
            "<li>Moderate concentric narrowing of the spinal canal is present with partial effacement of cerebrospinal fluid surrounding the thecal sac.</li>"
        )
        rec = "Clinical correlation advised; consider physical therapy and re‑evaluation within 6–12 months."
    else:  # Severe
        parts.append(
            "<li>Severe central canal narrowing is evident with marked reduction of thecal sac calibre and possible impingement on neural structures.</li>"
        )
        rec = "Specialist spinal consultation is recommended; surgical decompression may be considered depending on clinical symptoms."

    parts.append(f"<li><b>Impression:</b> {severity.upper()} lumbar canal stenosis.</li>")
    parts.append(f"<li><b>Recommendation:</b> {rec}</li>")
    parts.append("</ul>")
    return "\n".join(parts)


# ------------------ PDF REPORT ------------------
def generate_pdf(filename, sev, conf, feats, summary_html):
    """Generate a clean, readable, properly formatted PDF report."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # -------- TITLE --------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Lumbar Stenosis Diagnostic Report")

    # -------- SEVERITY --------
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, height - 1.3*inch, f"Predicted Severity: {sev}  (Confidence {conf:.2f})")

    # -------- METRICS --------
    c.drawString(1*inch, height - 1.7*inch, "Quantitative Metrics:")
    metrics = [
        f"• Area (px): {feats['area_px']}",
        f"• Perimeter (px): {feats['perimeter_px']}",
        f"• Compactness: {feats['compactness']}",
        f"• Opening Ratio: {feats['opening_ratio']}",
    ]

    y = height - 1.9*inch
    c.setFont("Helvetica", 11)
    for line in metrics:
        c.drawString(1.2*inch, y, line)
        y -= 0.2*inch

    # -------- SUMMARY TEXT --------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y - 0.2*inch, "Diagnostic Summary:")
    y -= 0.45*inch

    # Remove HTML tags and convert list bullets
    clean = re.sub(r"<ul[^>]*>", "", summary_html)

    clean = (
        clean
        .replace("</ul>", "")
        .replace("<li>", "• ").replace("</li>", "")
        .replace("<b>", "").replace("</b>", "")
        .replace("<br>", "\n")
        .strip()
    )

    c.setFont("Helvetica", 10)

    # Auto-wrap text to page width
    wrapped_lines = simpleSplit(clean, "Helvetica", 10, width - 2*inch)

    for line in wrapped_lines:
        if y < 1*inch:
            c.showPage()
            y = height - 1*inch
            c.setFont("Helvetica", 10)
        c.drawString(1*inch, y, line)
        y -= 0.18*inch

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
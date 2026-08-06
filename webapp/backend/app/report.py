"""Builds a standalone, printable HTML forensic report for one analyzed
image - prediction, Grad-CAM evidence, methodology, and the reliability
caveats established in the Milestone 5 evaluation (real numbers, not
generic disclaimers)."""

import uuid
from datetime import datetime, timezone

FACE_ALIGNMENT_LABELS = {
    "retinaface": "RetinaFace (automatic face detection & alignment)",
    "center_crop_fallback": "Center-crop fallback (RetinaFace unavailable on this server)",
}


def build_report_html(
    *,
    input_image_b64: str,
    filename: str,
    model_name: str,
    face_alignment_used: str,
    label: str,
    real_pct: float,
    fake_pct: float,
    heatmap_b64: str,
    overlay_b64: str,
) -> str:
    report_id = str(uuid.uuid4())[:8].upper()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    alignment_label = FACE_ALIGNMENT_LABELS.get(face_alignment_used, face_alignment_used)
    verdict_class = "real" if label == "Real" else "fake"
    confidence = real_pct if label == "Real" else fake_pct

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Face Authenticity Report — {report_id}</title>
<style>
  @media print {{
    body {{ margin: 0; }}
    .no-print {{ display: none; }}
  }}
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 800px;
    margin: 40px auto;
    padding: 0 24px;
    color: #1a1a1a;
    line-height: 1.55;
  }}
  header {{
    border-bottom: 3px solid #1a1a1a;
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px 0; }}
  .meta {{ font-family: 'Courier New', monospace; font-size: 0.85rem; color: #444; }}
  .meta span {{ display: inline-block; margin-right: 24px; }}
  section {{ margin-bottom: 28px; page-break-inside: avoid; }}
  h2 {{
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 6px;
    margin-bottom: 14px;
  }}
  .verdict-box {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 18px;
    border: 2px solid;
    border-radius: 6px;
  }}
  .verdict-box.real {{ border-color: #2e7d32; background: #f1f8f2; }}
  .verdict-box.fake {{ border-color: #c62828; background: #fdf1f1; }}
  .verdict-label {{ font-size: 1.6rem; font-weight: bold; }}
  .verdict-box.real .verdict-label {{ color: #2e7d32; }}
  .verdict-box.fake .verdict-label {{ color: #c62828; }}
  .pct-row {{ font-family: 'Courier New', monospace; font-size: 0.95rem; margin-top: 4px; }}
  .images-grid {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .images-grid figure {{ margin: 0; text-align: center; }}
  .images-grid img {{
    width: 200px;
    height: 200px;
    object-fit: cover;
    border: 1px solid #ccc;
    border-radius: 4px;
  }}
  figcaption {{ font-size: 0.8rem; color: #555; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  table td {{ padding: 6px 0; border-bottom: 1px solid #eee; }}
  table td:first-child {{ color: #555; width: 220px; }}
  .caveats {{
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 14px 18px;
    font-size: 0.9rem;
  }}
  .caveats ul {{ margin: 8px 0 0 0; padding-left: 20px; }}
  .caveats li {{ margin-bottom: 6px; }}
  .disclaimer {{
    font-size: 0.8rem;
    color: #666;
    border-top: 1px solid #ccc;
    padding-top: 14px;
    margin-top: 32px;
  }}
  .print-btn {{
    background: #1a1a1a;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    margin-bottom: 20px;
  }}
</style>
</head>
<body>

<button class="print-btn no-print" onclick="window.print()">Print / Save as PDF</button>

<header>
  <h1>Face Authenticity Analysis Report</h1>
  <div class="meta">
    <span>Report ID: {report_id}</span>
    <span>Generated: {generated_at}</span>
  </div>
</header>

<section>
  <h2>1. Verdict</h2>
  <div class="verdict-box {verdict_class}">
    <div>
      <div class="verdict-label">{label}</div>
      <div class="pct-row">Confidence: {confidence:.2f}%</div>
      <div class="pct-row">Real: {real_pct:.2f}% &nbsp;|&nbsp; Fake: {fake_pct:.2f}%</div>
    </div>
  </div>
</section>

<section>
  <h2>2. Case Details</h2>
  <table>
    <tr><td>Input filename</td><td>{filename}</td></tr>
    <tr><td>Model checkpoint used</td><td><code>{model_name}</code></td></tr>
    <tr><td>Face alignment method</td><td>{alignment_label}</td></tr>
  </table>
</section>

<section>
  <h2>3. Input &amp; Explainability Evidence</h2>
  <div class="images-grid">
    <figure>
      <img src="data:image/png;base64,{input_image_b64}" alt="Input image" />
      <figcaption>Input (as analyzed)</figcaption>
    </figure>
    <figure>
      <img src="data:image/png;base64,{heatmap_b64}" alt="Grad-CAM heatmap" />
      <figcaption>Grad-CAM heatmap</figcaption>
    </figure>
    <figure>
      <img src="data:image/png;base64,{overlay_b64}" alt="Grad-CAM overlay" />
      <figcaption>Grad-CAM overlay ({label})</figcaption>
    </figure>
  </div>
  <p style="font-size:0.85rem; color:#555; margin-top:12px;">
    The heatmap highlights which regions of the image most influenced the
    model's decision — brighter regions contributed more. A heatmap
    concentrated on facial structure (eyes, nose, mouth) is consistent
    with the model attending to genuine facial evidence; a heatmap
    concentrated on background or edges may indicate a less reliable
    prediction.
  </p>
</section>

<section>
  <h2>4. Reliability &amp; Known Limitations</h2>
  <div class="caveats">
    This model was evaluated in detail as part of a separate technical
    milestone report. Key findings relevant to interpreting this result:
    <ul>
      <li><strong>Strong on in-distribution data:</strong> 99.63% accuracy on the model's own held-out test set.</li>
      <li><strong>Weak on modern smartphone photos:</strong> accuracy on genuine recent smartphone photographs dropped to 8.6% in cross-domain testing, and 62.0% in a separate local check — both substantially below the in-distribution figure. If this input is a modern smartphone photo (HDR, heavy sharpening, high saturation), treat a "Fake" verdict with caution.</li>
      <li><strong>Dominant error type is false positives:</strong> genuine photos being misclassified as fake, not fakes evading detection.</li>
      <li><strong>Non-face images produce meaningless results:</strong> if the input does not contain a clearly detectable face, this verdict should be disregarded entirely.</li>
    </ul>
  </div>
</section>

<div class="disclaimer">
  This is an automated, educational analysis tool. It does not constitute
  a certified, legally admissible forensic conclusion. No chain-of-custody
  or examiner certification is implied. Results should be interpreted
  alongside the limitations stated above, not as a standalone determination
  of authenticity.
</div>

</body>
</html>
"""

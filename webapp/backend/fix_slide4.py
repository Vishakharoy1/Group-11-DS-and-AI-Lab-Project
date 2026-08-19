"""Corrects Slide 4 of Milestone-6-presentation.pdf (0-indexed page 3):

'Low-Latency Inference: ~15.5ms CPU forward pass execution suitable for real-time
Web API requests.' implies the deployed web app is fast, but 15.5ms was only ever
measured on local CPU dev hardware (M5/M6 reports) - never on Render. Per live-demo/
reviewer feedback, the actual deployed Render instance runs markedly slower. Corrects
the caption to stop implying Render-suitability from a local-only number.
"""
import shutil
from pathlib import Path

import pymupdf

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PDF_PATH = PROJECT_DIR / "doc" / "Milestone-6" / "Milestone-6-presentation.pdf"
BACKUP_PATH = PDF_PATH.with_name("Milestone-6-presentation.ORIGINAL_BACKUP.pdf")

if not BACKUP_PATH.exists():
    shutil.copy2(PDF_PATH, BACKUP_PATH)
    print(f"Backed up original to {BACKUP_PATH}")
else:
    print(f"Backup already exists at {BACKUP_PATH}, not overwriting.")

doc = pymupdf.open(PDF_PATH)
page = doc[3]  # Slide 4

WHITE = (1.0, 1.0, 1.0)

# Redact just the regular-weight text after the bold "Low-Latency Inference:" label
old_area = pymupdf.Rect(186.0, 270.0, 460.0, 305.5)
page.add_redact_annot(old_area, fill=WHITE)
page.apply_redactions()

page.insert_text(
    (186.54800415039062, 283.5),
    " ~15.5ms CPU forward pass on local dev",
    fontsize=11.31,
    fontname="helv",
    color=(0, 0, 0),
)
page.insert_text(
    (55.50199890136719, 300.4),
    "hardware only - Render latency runs markedly higher.",
    fontsize=11.31,
    fontname="helv",
    color=(0, 0, 0),
)

doc.saveIncr()
doc.close()
print(f"Slide 4 corrected in place: {PDF_PATH}")

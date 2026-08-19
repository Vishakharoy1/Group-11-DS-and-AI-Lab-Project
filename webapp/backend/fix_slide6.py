"""Corrects Slide 6 of Milestone-6-presentation.pdf (0-indexed page 5):

The 'Deployment Speed' card claims '15.5 ms ... on standard Render cloud instance.'
That number was actually measured on local CPU dev hardware (M5/M6 reports, Intel
i7-7700 desktop) and never benchmarked on Render itself. Per live-demo/reviewer
feedback, the deployed Render instance is in fact markedly slower than this figure
suggests - the caption is corrected to stop misattributing a local number to Render.

The 15.5 ms number itself is left in place (it's a real, correctly-cited local
measurement) - only the misleading "on standard Render cloud instance" attribution
is fixed.
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
page = doc[5]  # Slide 6

CARD_BG = (241 / 255, 253 / 255, 251 / 255)

# Redact the old 2-line caption ("Average CPU inference time per image on standard
# Render cloud instance.") - full width of the card, a little padding on all sides.
old_caption = pymupdf.Rect(654.0, 185.5, 892.0, 214.0)
page.add_redact_annot(old_caption, fill=CARD_BG)
page.apply_redactions()

x = 654.7459716796875
lines = [
    "Measured on local CPU dev hardware",
    "(M5/M6 report), NOT on Render - live",
    "Render inference ran markedly slower.",
]
baselines = [198.6, 210.3, 222.0]
for y, text in zip(baselines, lines):
    page.insert_text((x, y), text, fontsize=9.5, fontname="helv", color=(0, 0, 0))

doc.saveIncr()
doc.close()
print(f"Slide 6 corrected in place: {PDF_PATH}")

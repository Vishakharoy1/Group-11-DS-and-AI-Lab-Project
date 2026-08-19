"""Corrects Slide 11 of Milestone-6-presentation.pdf (0-indexed page 10):

1. Fabricated 91.20% 'In-Domain Acc' for mobilenetv3_cross_domain.pth -> corrected to the
   actual audited numbers from webapp/output/cross_domain_results.csv (59.89% / 70.0%).
2. Reorders the model matrix so the 3 actively-used models come first, in this order:
   1) mobilenetv3_noaug.pth (unchanged - already row 1)
   2) mobilenetv3_best.pth   (moved up from row 3)
   3) mobilenetv3_cross_domain.pth (moved down from row 2, with corrected accuracy)
   Rows 4-5 (manipulations, tuned) are relabeled "Removed" per current project status.

Backs up the original PDF first (skips backup if one already exists from a prior run).
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
page = doc[10]  # Slide 11

# Column left-x positions (shared across all rows) and each row slot's y-range + background.
COLS_X = [42.49100112915039, 249.61900329589844, 460.0060119628906, 558.4249877929688, 690.6610107421875]
COL_RIGHT_EDGE = 922.4219970703125  # last column's right boundary, for width sanity only

ROW2_SLOT = (175.29849243164062, 191.22976684570312, (0.9411764740943909, 0.9921568632125854, 0.9803921580314636))
ROW3_SLOT = (206.79147338867188, 222.72274780273438, (1.0, 1.0, 1.0))
ROW4_SLOT = (238.28347778320312, 254.21475219726562, (0.9725490212440491, 0.9803921580314636, 0.9882352948188782))
ROW5_SLOT = (269.7764892578125, 285.707763671875, (1.0, 1.0, 1.0))

FONT = "helv"
FONT_SIZE = 10.49
TEXT_COLOR = (0, 0, 0)


def replace_row(y_top, y_bottom, bg, cell_texts):
    """Redact the full-width row band, then write the 5 new cell texts."""
    row_rect = pymupdf.Rect(COLS_X[0] - 2, y_top - 2, COL_RIGHT_EDGE, y_bottom + 2)
    page.add_redact_annot(row_rect, fill=bg)
    page.apply_redactions()
    baseline_y = y_bottom - 3.5  # matches original text baseline offset within its row band
    for x, text in zip(COLS_X, cell_texts):
        page.insert_text((x, baseline_y), text, fontsize=FONT_SIZE, fontname=FONT, color=TEXT_COLOR)


# Row 1 (mobilenetv3_noaug.pth) is already correct and already in position 1 - left untouched.

# Row 2 slot now holds mobilenetv3_best.pth (moved up from the old row 3)
replace_row(*ROW2_SLOT, [
    "mobilenetv3_best.pth",
    "3-Stage Fine-Tuning + CelebA-HD",
    "99.63%",
    "No (RAM Limit)",
    "Research Comparison Baseline",
])

# Row 3 slot now holds mobilenetv3_cross_domain.pth (moved down from the old row 2),
# with the accuracy corrected from the fabricated 91.20% to the audited CSV figures.
replace_row(*ROW3_SLOT, [
    "mobilenetv3_cross_domain.pth",
    "Nano Banana, CIFAKE, Places365",
    "59.89-70.0%*",
    "Yes (Secondary)",
    "Cross-Domain Generalization Probe",
])

# Row 4: mobilenetv3_manipulations.pth - removed from the active model set.
replace_row(*ROW4_SLOT, [
    "mobilenetv3_manipulations.pth",
    "Trained with 11 corruption passes",
    "Removed",
    "Removed",
    "Removed from active model set",
])

# Row 5: mobilenetv3_tuned.pth - removed from the active model set.
replace_row(*ROW5_SLOT, [
    "mobilenetv3_tuned.pth",
    "Automated Hyperparameter Sweep",
    "Removed",
    "Removed",
    "Removed from active model set",
])

# --- Footnote citing the audited source for the corrected accuracy figure (2 lines, fits slide width) ---
footnote_line1 = (
    "* cross_domain In-Domain Acc corrected from unsupported 91.20% to audited CSV: "
    "face_main 59.89% (N=2,401) / nano_banana 70.0% (N=4,000)."
)
footnote_line2 = (
    "Rows reordered to used-models-first (noaug, best, cross_domain); "
    "manipulations and tuned checkpoints marked Removed."
)
page.insert_text((37.5, 295.5), footnote_line1, fontsize=6.6, fontname="helv", color=(0.35, 0.35, 0.35))
page.insert_text((37.5, 302.5), footnote_line2, fontsize=6.6, fontname="helv", color=(0.35, 0.35, 0.35))

doc.saveIncr()
doc.close()
print(f"Slide 11 rebuilt in place: {PDF_PATH}")

"""Inserts a new Slide 2 ("Team & Ownership") into Milestone-6-presentation.pdf,
between the title slide and the existing "Context & Objectives" slide, then
renumbers every subsequent slide's "Slide N" footer by +1.

Content is drawn directly from doc/Final-Contribution-Summary.md's Role
Matrix and Per-Member Summary sections - not invented.
"""
import pymupdf

PDF_PATH = "doc/Milestone-6/Milestone-6-presentation.pdf"

NAVY = (11 / 255, 31 / 255, 58 / 255)
TEAL = (13 / 255, 148 / 255, 136 / 255)
SLATE = (100 / 255, 116 / 255, 139 / 255)
BODY = (51 / 255, 65 / 255, 85 / 255)
WHITE = (1, 1, 1)
MINT = (240 / 255, 253 / 255, 250 / 255)
BORDER = (0.85, 0.87, 0.9)

doc = pymupdf.open(PDF_PATH)

# ---- 1. Renumber every existing slide from N to N+1 (do this FIRST, before
# insertion shifts indices) ----
for i in range(1, doc.page_count):  # skip slide 1 (title page has no "Slide N")
    page = doc[i]
    old_label = f"Slide {i + 1}"
    new_label = f"Slide {i + 2}"
    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                if span["text"] == old_label:
                    x0, y0, x1, y1 = span["bbox"]
                    page.add_redact_annot(pymupdf.Rect(x0 - 2, y0 - 2, x1 + 15, y1 + 2), fill=WHITE)
                    page.apply_redactions()
                    page.insert_text((x0, y1 - 3.3), new_label, fontsize=9.014, fontname="helv", color=SLATE)

# ---- 2. Insert the new page at index 1 (becomes Slide 2) ----
new_page = doc.new_page(pno=1, width=960, height=540)

f_bold = pymupdf.Font("hebo")
f_reg = pymupdf.Font("helv")

# Header
new_page.insert_text((37.5, 40.5), "SECTION 0: TEAM & OWNERSHIP", fontsize=9.014, fontname="hebo", color=TEAL)
new_page.insert_text((37.5, 71.0), "Meet the Team", fontsize=22.507, fontname="hebo", color=NAVY)
badge_text = "5 MEMBERS - M1 TO M6"
bw = f_bold.text_length(badge_text, fontsize=8.2)
badge_rect = pymupdf.Rect(922.48 - bw - 24, 32.0, 922.48, 32.0 + 17)
new_page.draw_rect(badge_rect, color=TEAL, fill=TEAL, radius=0.5, width=0)
new_page.insert_text((badge_rect.x0 + 12, badge_rect.y0 + 12), badge_text, fontsize=8.2, fontname="hebo", color=WHITE)

new_page.insert_text(
    (37.5, 93.0),
    "Each member's contribution forms a consistent thread from problem definition (M1) through final deployment (M6) - drawn from the Final Contribution Summary.",
    fontsize=9.5, fontname="helv", color=SLATE,
)

# ---- 3. Table: Member | Core Thread (M1 -> M6) | M6 Deliverable ----
col1_x, col2_x, col3_x, right_edge = 37.5, 150.0, 670.0, 922.48
table_top = 112.0
header_h = 26.0
row_h = 62.0

# Header row
new_page.draw_rect(pymupdf.Rect(col1_x, table_top, right_edge, table_top + header_h), color=NAVY, fill=NAVY, width=0)
new_page.insert_text((col1_x + 8, table_top + 17.5), "Member", fontsize=10.5, fontname="hebo", color=WHITE)
new_page.insert_text((col2_x + 8, table_top + 17.5), "Core Thread (M1 -> M6)", fontsize=10.5, fontname="hebo", color=WHITE)
new_page.insert_text((col3_x + 8, table_top + 17.5), "M6 Deliverable", fontsize=10.5, fontname="hebo", color=WHITE)

rows = [
    ("Vishakha", "Pipeline &\nPresentation Lead",
     ["Research & comparative analysis (M1) -> EDA (M2) -> dataset",
      "validation (M3) -> training/inference pipeline (M4) -> web app +",
      "robustness testing (M5) -> presentation & deployment (M6)."],
     ["Final Presentation,", "Deployment Stability,", "Contribution Summary"]),
    ("Rohit", "Training\nStability Lead",
     ["Problem definition (M1) -> dataset docs (M2) -> architecture",
      "selection (M3) -> hyperparameter optimization (M4) -> held-out",
      "evaluation (M5) -> full ROC-AUC + report (M6)."],
     ["Final Technical Report,", "ROC/PR/Confusion", "Matrix Plots"]),
    ("Aman", "Preprocessing &\nTransfer Learning Lead",
     ["Baseline evaluation strategy (M1) -> data splitting (M2) ->",
      "pipeline optimization (M3) -> robustness test generation (M4) ->",
      "3-stage transfer learning docs (M5) -> report (M6)."],
     ["Non-Technical Report"]),
    ("Raunak", "Dataset & Bias\nAnalysis Lead",
     ["Literature review (M1) -> augmentation design (M2) -> spatial",
      "baseline testing (M3) -> cross-domain evaluation (M4) -> shortcut-",
      "learning root cause (M5) -> Real-Latest + bias analysis (M6)."],
     ["8.6% Domain-Shift Root", "Cause, Bias Limitations"]),
    ("Somendu", "Explainability &\nOptimisation Lead",
     ["Data research (M1) -> dataset statistics (M2) -> hyperparameter",
      "search (M3) -> Grad-CAM integration (M4) -> failure-case",
      "verification (M5) -> Grad-CAM latency fix + guide (M6)."],
     ["User Guide, Grad-CAM", "Latency Optimization"]),
]

y = table_top + header_h
for idx, (name, role, thread_lines, deliv_lines) in enumerate(rows):
    bg = MINT if idx % 2 == 0 else WHITE
    row_rect = pymupdf.Rect(col1_x, y, right_edge, y + row_h)
    new_page.draw_rect(row_rect, color=bg, fill=bg, width=0)
    new_page.draw_line((col1_x, y + row_h), (right_edge, y + row_h), color=BORDER, width=0.75)

    name_y = y + 22
    new_page.insert_text((col1_x + 8, name_y), name, fontsize=11.5, fontname="hebo", color=NAVY)
    role_lines = role.split("\n")
    ry = name_y + 13
    for rl in role_lines:
        new_page.insert_text((col1_x + 8, ry), rl, fontsize=8.3, fontname="helv", color=SLATE)
        ry += 10.5

    ty = y + 20
    for tl in thread_lines:
        new_page.insert_text((col2_x + 8, ty), tl, fontsize=9.0, fontname="helv", color=BODY)
        ty += 13.5

    dy = y + 22
    for dl in deliv_lines:
        new_page.insert_text((col3_x + 8, dy), dl, fontsize=9.0, fontname="hebo", color=TEAL)
        dy += 13.0

    y += row_h

# Outer table border
new_page.draw_rect(pymupdf.Rect(col1_x, table_top, right_edge, y), color=BORDER, fill=None, width=1)
new_page.draw_line((col2_x, table_top), (col2_x, y), color=BORDER, width=0.75)
new_page.draw_line((col3_x, table_top), (col3_x, y), color=BORDER, width=0.75)

# ---- Footer ----
new_page.draw_line((37.5, 508.0), (922.48, 508.0), color=BORDER, width=0.75)
new_page.insert_text((37.5, 521.0), "Group 11 - Face Forensics", fontsize=9.014, fontname="helv", color=SLATE)
new_page.insert_text((893.0, 521.0), "Slide 2", fontsize=9.014, fontname="helv", color=SLATE)

doc.saveIncr()
doc.close()
print(f"Inserted new Slide 2. Total pages now: {pymupdf.open(PDF_PATH).page_count}")

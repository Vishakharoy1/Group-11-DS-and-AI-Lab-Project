"""Replaces the old Render URL (group-11-ds-and-ai-lab-project.onrender.com)
with the new one (face-forensics.onrender.com) on the 3 presentation slides
that mention it (title slide, deployment slide, live-demo slide).
"""
import pymupdf

PDF_PATH = "doc/Milestone-6/Milestone-6-presentation.pdf"
OLD = "group-11-ds-and-ai-lab-project.onrender.com"
NEW = "face-forensics.onrender.com"

doc = pymupdf.open(PDF_PATH)

for i in range(doc.page_count):
    page = doc[i]
    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                if OLD not in span["text"]:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                fontsize = span["size"]
                color_int = span["color"]
                color = (
                    ((color_int >> 16) & 255) / 255,
                    ((color_int >> 8) & 255) / 255,
                    (color_int & 255) / 255,
                )

                # sample background just left of the text to pick the redaction fill
                pix = page.get_pixmap(dpi=72)
                sx, sy = max(0, int(x0 - 5)), int((y0 + y1) / 2)
                px = pix.pixel(min(sx, pix.width - 1), min(sy, pix.height - 1))
                bg = (px[0] / 255, px[1] / 255, px[2] / 255)

                new_text = span["text"].replace(OLD, NEW)

                page.add_redact_annot(pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=bg)
                page.apply_redactions()

                baseline_y = y1 - (fontsize * 3.3 / 9.014)
                page.insert_text((x0, baseline_y), new_text, fontsize=fontsize, fontname="helv", color=color)
                print(f"page {i+1}: replaced -> {new_text!r}")

doc.saveIncr()
doc.close()
print("done")

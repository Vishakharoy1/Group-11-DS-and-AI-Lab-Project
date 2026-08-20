"""One-off: curates 10 real + 10 fake sample images for the Main Model
page and 10 real + 9 fake for the Cross-Domain page (only 9 fake
cross-domain sources exist in the repo), downscales/re-encodes them
(some sources are 2-3MB PNGs) and drops them into
webapp/backend/app/static/samples/ for the frontend's "Try a Sample"
gallery. Source images come from Test Sample/ (already in the repo,
used for manual QA testing).
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_SAMPLE = ROOT / "Test Sample"
OUT_DIR = Path(__file__).resolve().parent / "app" / "static" / "samples"

MAX_DIM = 900
JPEG_QUALITY = 85

SELECTIONS = {
    "main-real": [
        TEST_SAMPLE / "Test_real_vs_Fake" / "real" / f"{i}.jpg" for i in range(1, 10)
    ] + [TEST_SAMPLE / "Test_real_vs_Fake" / "real" / "real1.jpg"],
    "main-fake": [
        TEST_SAMPLE / "Test_real_vs_Fake" / "fake" / f"{i}.jpg" for i in range(1, 10)
    ] + [TEST_SAMPLE / "Test_real_vs_Fake" / "fake" / "fake.png"],
    "cross-real": [
        TEST_SAMPLE / "Cross Domain" / "Real" / f"{i}.jpg" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ] + [TEST_SAMPLE / "Cross Domain" / "Real" / "WhatsApp Image 2026-08-02 at 6.57.27 PM.jpeg"],
    "cross-fake": [
        TEST_SAMPLE / "Cross Domain" / "Fake" / name
        for name in [
            "ChatGPT Image Aug 2, 2026, 06_55_57 PM.png",
            "ChatGPT Image Aug 2, 2026, 08_25_38 PM.png",
            "ChatGPT Image Aug 2, 2026, 08_27_43 PM.png",
            "ChatGPT Image Aug 2, 2026, 08_29_51 PM.png",
            "ChatGPT Image Aug 2, 2026, 08_39_14 PM.png",
            "ChatGPT Image Aug 2, 2026, 08_43_39 PM.png",
            "cross domain 2.png",
            "cross domain.png",
            "crossdomain1.png",
        ]
    ],
}

for category, paths in SELECTIONS.items():
    out_sub = OUT_DIR / category
    out_sub.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths, start=1):
        if not src.is_file():
            print(f"MISSING: {src}")
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > MAX_DIM:
            scale = MAX_DIM / longest
            img = img.resize((round(w * scale), round(h * scale)))
        out_path = out_sub / f"{idx}.jpg"
        img.save(out_path, format="JPEG", quality=JPEG_QUALITY)
        print(f"{category}/{idx}.jpg <- {src.name} ({out_path.stat().st_size // 1024} KB)")

print("done")

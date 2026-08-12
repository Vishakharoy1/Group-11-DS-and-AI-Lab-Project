# Developer Guide

This guide covers everything needed to replicate this project's setup
and results: environment setup, training the model from scratch,
running the local web application, every backend/frontend file and what
it does, configuration options, the API surface, and implementation
notes that aren't obvious from the code alone.

Everything in this guide describes the current state of the **`main`**
branch — no other branch is required to reproduce what's documented
here.

---

## 1. Project Overview

A deepfake/AI-generated-face detector built on **MobileNetV3-Large**,
trained via a three-stage transfer-learning strategy, with:

- Grad-CAM explainability (visual heatmap of what the model attended to)
- A local FastAPI + HTML/JS web app for interactive testing (no notebook
  upload widgets)
- Automated forensic report generation (on-screen, printable PDF, and
  real `.docx` export)
- Documented, honest evaluation — including what's *not* measured yet —
  in `doc/Milestone-5/Milestone5.md`

Two models exist:

| Role | Checkpoint | Training notebook | Status |
|---|---|---|---|
| Main Model — Model 1 (default) | `mobilenetv3_noaug.pth` | `final-mobilenet (1).ipynb` (ablation run without the augmentation pipeline) | Trained |
| Main Model — Model 2 (toggle) | `mobilenetv3_best.pth` | `final-mobilenet (1).ipynb` (3-stage, CelebA-HD corrected) | Trained |
| Cross-Domain Model | `mobilenetv3_cross_domain.pth` | `cross-domain.ipynb` | Trained |

The Main Model page has a **Model 1 / Model 2 toggle** — Model 1
(`noaug`) is the default on page load, Model 2 switches live to `best`,
no code change or restart needed. See Section 8 below for the full
detail and why `noaug` (not `best`) is the default.
specifically rather than `best`.

---

## 2. Repository Structure

```text
Group-11-DS-and-AI-Lab-Project/
├── final-mobilenet (1).ipynb      # Main face-model training notebook
├── cross-domain.ipynb             # Cross-domain (non-face) model training notebook
├── README.md                      # Project overview, quick start, notebook walkthrough
├── DeveloperGuide.md              # This file
│
├── images/
│   └── mobilenetv3_pipeline_v3.png
│
├── webapp/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py            # FastAPI routes
│   │   │   ├── model.py           # Architecture + checkpoint registry
│   │   │   ├── config.py          # Paths, checkpoint filenames, constants
│   │   │   ├── preprocessing.py   # Face crop/align + inference transform
│   │   │   ├── gradcam.py         # Grad-CAM implementation
│   │   │   ├── manipulations.py   # 11 robustness-test image manipulations
│   │   │   ├── meta_detector.py   # Forensic meta-detector (metadata/watermark/pixel forensics)
│   │   │   ├── report.py          # HTML + .docx forensic report builders
│   │   │   ├── results.py         # Pre-computed training-artifact loader
│   │   │   ├── schemas.py         # Pydantic request/response models
│   │   │   └── static/            # Frontend: index.html, app.js, style.css
│   │   ├── requirements.txt
│   │   └── README.md
│   └── output/                    # Checkpoints + training artifacts (gitignored contents)
│
├── Test Sample/                   # Manually curated real/fake test images
├── outputs/                       # Local verification scripts + real measured results
│
└── doc/
    ├── Milestone-1/ .. Milestone-5/
    │   ├── *-Report.md
    │   └── Team-Contribution-Tracker.md
    └── Milestone-5/
        ├── Milestone5.md          # Full evaluation report — start here for results
        └── images/
```

---

## 3. Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip` and a virtual environment tool (`venv` or conda)
- A Kaggle account with GPU quota, to train/reproduce the notebooks
- No local GPU is required to run the web app — it works CPU-only
  (measured: ~15.5 ms per prediction on a CPU-only desktop; see
  `doc/Milestone-5/Milestone5.md` Section 9)
- Windows users: see **Section 9, Known Issue — Face Detection on
  Windows** before assuming RetinaFace will work out of the box

---

## 4. Setup

### 4.1 Clone and create a virtual environment

```bash
git clone https://github.com/Vishakharoy1/Group-11-DS-and-AI-Lab-Project.git
cd Group-11-DS-and-AI-Lab-Project

python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
```bash
# Linux/macOS
source .venv/bin/activate
```

### 4.2 Install web app dependencies

```bash
cd webapp/backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` contents:

```text
fastapi>=0.139
uvicorn[standard]>=0.50
pillow>=12.0
numpy
scipy
matplotlib
pydantic>=2.0
python-multipart
python-docx
invisible-watermark

torch
torchvision

# Optional - real face alignment instead of center-crop fallback
# retina-face
# opencv-python
```

`torch`/`torchvision` are listed but installed separately via the
CPU-only wheel index above — installing them through `pip install -r
requirements.txt` alone will pull a much larger CUDA build unless you
specifically want GPU inference locally.

`python-docx` was added specifically for the `.docx` forensic-report
export (`report.py`'s `build_docx()`).

`scipy` and `invisible-watermark` were added for the forensic
meta-detector (`meta_detector.py`, Section 6.10) — `scipy` is a hard
requirement (imported unconditionally for FFT/median-filter operations),
`invisible-watermark` degrades gracefully if missing (watermark scanning
just reports itself unavailable rather than crashing the server).

`retina-face`/`opencv-python` are commented out by default — the app
runs fine without them (falls back to center-crop face alignment); see
Section 9 for why they're often broken on Windows specifically.

### 4.3 Place model checkpoints

Download the trained `.pth` files (see Section 5 for how they're
produced) into `webapp/output/` (the sibling folder to `backend/`), or
point `CHECKPOINT_DIR` at wherever they actually live:

```bash
# Windows PowerShell
$env:CHECKPOINT_DIR = "D:\path\to\your\checkpoints"

# bash
export CHECKPOINT_DIR=/path/to/checkpoints
```

Expected filenames — see `config.py`'s `CHECKPOINTS` dict (Section 6.2):
`mobilenetv3_best.pth`, `mobilenetv3_noaug.pth`,
`mobilenetv3_cross_domain.pth`, `mobilenetv3_manipulations.pth`,
`mobilenetv3_tuned.pth`. Any missing file is skipped gracefully, not
fatal — check `GET /health`'s `loaded_models` field to see what actually
loaded.

### 4.4 Run the app

From inside `webapp/backend/`:

```bash
uvicorn app.main:app --port 8000
```

Then open `http://127.0.0.1:8000`. First startup takes 10–20+ seconds if
`retina-face`/TensorFlow are installed (heavy import); with them absent
(the common case, per Section 9) startup is fast.

- `http://127.0.0.1:8000/docs` — interactive Swagger UI, useful to test
  endpoints directly before touching the frontend.
- `http://127.0.0.1:8000/health` — reports which checkpoints loaded and
  which face-alignment method is active.

`--reload` can be flaky with this app's background model loading on
Windows — a plain restart after code changes is more reliable.

---

## 5. Training Pipeline

Two Kaggle notebooks produce the checkpoints. Full cell-by-cell
walkthroughs (dataset slugs to attach, what each cell does, expected
output) are in the root `README.md`'s **"Training Notebooks — Usage
Instructions"** section — not duplicated here to avoid the two docs
drifting apart. Summary:

| Notebook | Produces | Status |
|---|---|---|
| `final-mobilenet (1).ipynb` | `mobilenetv3_best.pth` (3-stage, CelebA-corrected) | Complete, fully trained |
| `cross-domain.ipynb` | `mobilenetv3_cross_domain.pth` | Complete, fully trained |

Both notebooks use the same core recipe: MobileNetV3-Large
(ImageNet-pretrained) → Stage 1 (frozen backbone, classifier head only)
→ Stage 2 (last 25% of backbone unfrozen). The main notebook adds a
Stage 3 (full unfreeze + CelebA-HD real photos) specifically to correct
a shortcut-learning failure mode — see `doc/Milestone-5/Milestone5.md`
Section 5 for the full root-cause analysis.

**Key training config** (from `final-mobilenet (1).ipynb`):
`SEED=42`, `IMG_SIZE=224`, `BATCH_SIZE=128`, `IMAGES_PER_CLASS=15000`,
80:10:10 stratified train/val/test split. Stage 1: 3 epochs, LR `3e-4`.
Stage 2: 7 epochs, LR `1e-5`. Stage 3: 3 epochs, LR `5e-6`.

**Training-time augmentation** actually used (verified from the
notebook's `train_transform`, not assumed): `RandomResizedCrop`,
`RandomHorizontalFlip`, `ColorJitter`, a custom `ChannelShift` (±30%
per-channel scaling, teaches the model that colour-tinted fakes are
still fake), `GaussianBlur`, a custom `JPEGCompression`, and Gaussian
noise. None of these run at inference time — inference uses a separate,
fixed `val_transform` (see 6.3 below).

---

## 6. Web Application — Backend

FastAPI app in `webapp/backend/app/`. Every file's actual role:

### 6.1 `main.py` — routes

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Which checkpoints loaded, which face-alignment method is active |
| `/predict` | POST | `?model=` (default `best`) + image file → prediction + Grad-CAM heatmap/overlay (base64 PNGs) + a `meta_detector` forensic scan result (see 6.10) |
| `/report` | POST | Same inputs as `/predict`, returns a standalone printable HTML forensic report instead of JSON (legacy — the current frontend renders its own Report page client-side; see 7.4) |
| `/report/docx` | POST | Accepts already-computed analysis data as JSON, returns a real `.docx` file (`report.build_docx()`) |
| `/robustness` | POST | Runs all 11 manipulations (`manipulations.py`) through the `manipulations` checkpoint |
| `/compare` | POST | `?mode=augmentation\|hparams` — side-by-side prediction from two checkpoints |
| `/api/training-results` | GET | Serves pre-computed CSVs/images from the training notebook's output folder (not live inference) |

`_require_model()` raises a 503 if a requested checkpoint isn't loaded —
callers get a clear error rather than a crash.

### 6.2 `config.py` — configuration

```python
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_DIR / "output"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_DIR / "output"))

CHECKPOINTS = {
    "best": CHECKPOINT_DIR / "mobilenetv3_best.pth",
    "cross_domain": CHECKPOINT_DIR / "mobilenetv3_cross_domain.pth",
    "tuned": CHECKPOINT_DIR / "mobilenetv3_tuned.pth",
    "noaug": CHECKPOINT_DIR / "mobilenetv3_noaug.pth",
    "manipulations": CHECKPOINT_DIR / "mobilenetv3_manipulations.pth",
}
```

Also: `IMG_SIZE=224`, ImageNet normalization constants, `CLASSES = ["Real",
"Fake"]`, 10 MB max upload, allowed content types (`jpeg`/`png`/`webp`/`bmp`).

### 6.3 `model.py` — architecture + registry

```python
def build_model(dropout=0.2, num_classes=2):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280), nn.Hardswish(),
        nn.Dropout(p=dropout), nn.Linear(1280, num_classes),
    )
    return model
```

This must match the notebook's architecture exactly or `load_state_dict`
fails. `ModelRegistry` loads every checkpoint that exists on disk at
startup, skips missing ones with a warning (not fatal), and exposes
`.get(name)` / `.loaded_names` / `.is_ready()`.

### 6.4 `preprocessing.py` — face alignment + inference transform

```python
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

This is an exact port of the training notebook's own `val_transform` —
verified by direct comparison, not assumed. `crop_and_align_face()` tries
RetinaFace first (largest detected face, 20% padding), falls back to a
center-square crop if RetinaFace is unavailable or fails, and returns
which method was actually used (`"retinaface"` or
`"center_crop_fallback"`) so the caller/UI can show it.

### 6.5 `gradcam.py` — explainability

Hooks the last MobileNetV3 conv layer (`model.features[-1]`), computes
Grad-CAM in the standard way (gradient-weighted activation map, ReLU'd,
normalized), renders it with the `jet` colormap, and blends it onto the
resized input at `alpha=0.45` for the overlay. Returns both the raw
heatmap and the overlay as base64 PNGs, plus the prediction.

### 6.6 `manipulations.py` — robustness testing

The 11 manipulations used by `/robustness`: `original`, `green_tint`,
`blue_tint`, `brightness`, `contrast`, `gaussian_blur`, `motion_blur`,
`jpeg` (quality 30), `resize` (downscale 3x then back up), `crop` (70%
center crop then upscale), `noise` (Gaussian, σ=15). Ported to match
exactly what the training notebook's own manipulation-testing section
measures.

### 6.7 `report.py` — forensic report generation

Two builders:

- `build_report_html(...)` — the legacy standalone HTML report served by
  `POST /report`. Self-contained (inline CSS, base64-embedded images),
  includes a "Reliability & Known Limitations" section sourced from the
  real Milestone 5 findings (8.6%/62.0% cross-domain accuracy drop,
  false-positive-dominant failure mode, non-face-image caveat) rather
  than generic boilerplate.
- `build_docx(...)` — builds the same content as a real `.docx` via
  `python-docx`: 7 numbered sections (Case Info, Image Info, Model Info,
  Inference Preprocessing, Explainability w/ embedded images, Final
  Assessment, Disclaimer), returned as an in-memory `BytesIO` for
  streaming.

### 6.8 `schemas.py` — request/response models

Pydantic models for every route's I/O, including `DocxReportRequest` —
the JSON shape the frontend posts to `/report/docx` (analysis ID,
timestamps, file metadata, prediction, both base64 images) — and
`MetaDetectorResponse` (verdict, ai_score, edit_score, confidence,
signals dict, evidence list, warnings list), embedded in every
`PredictResponse` as `meta_detector` (see 6.10).

### 6.9 `results.py`

Loads whatever pre-computed CSVs/sample images/Grad-CAM gallery images
exist in `RESULTS_DIR` and assembles the `/api/training-results`
response — purely reads static training-notebook output, no live
inference.

### 6.10 `meta_detector.py` — forensic meta-detector

A second, independent detection signal that runs **alongside** (not
instead of) the MobileNetV3 CNN on every `/predict` call, added after
the initial 5-page rebuild. Consolidates three classes of forensic
evidence used by large-platform detectors, entirely in metadata/pixel
statistics — no training data or checkpoint of its own required for its
core signals:

1. **Metadata & industry standards** (`analyze_metadata()`) — scans
   Exif/XMP/PNG text chunks and raw file bytes for generator signatures
   (`AI_SOFTWARE_PATTERNS`: Midjourney, Stable Diffusion, DALL-E, Adobe
   Firefly, ComfyUI, 20+ others), camera-make markers, and C2PA/Content
   Credentials provenance claims. C2PA presence alone is *not* treated as
   AI evidence (camera makers sign genuine photos with it too) — only a
   `digitalSourceType=trainedAlgorithmicMedia` claim or a known
   invisible-watermark byte signature counts as hard evidence.
2. **Invisible watermark scanning** (`scan_watermarks()`) — wraps the
   `invisible-watermark` library to decode Stable Diffusion 1.x/SDXL/SD3
   watermark payloads. Only an exact match against a known message
   (`KNOWN_MESSAGES`) counts as a detection, to avoid false positives from
   decoder noise.
3. **Pixel-level forensics** — four independent statistical signals, all
   pure NumPy/SciPy (no ML model):
   - `high_frequency_ratio()` / `spectral_flatness()` / `periodic_peak_score()` —
     FFT-based: real camera sensor noise adds high-frequency energy that
     clean AI renders lack; sharp periodic spectral peaks suggest
     generative upscaling or latent watermarking.
   - `shot_noise_correlation()` — real sensor noise (Poisson/shot noise)
     correlates with local brightness; AI-generated pixels typically
     don't.
   - `double_jpeg_score()` — 8×8 DCT-grid blockiness analysis to detect
     recompression (a re-saved image, evidence of *some* processing
     pipeline having touched it).
   - `ela_statistics()` — Error Level Analysis: re-saves at fixed JPEG
     quality (85) and measures regional inconsistency in the difference,
     which flags locally-edited/painted regions.

All signals are combined into a single `ai_score` (0–1) and `edit_score`
(0–1) by `detect_image()` (the module's main entry point), with hard
metadata/watermark evidence dominating the score when present, and a
human-readable `evidence` list explaining *why* (e.g. "Noise is not
coupled to brightness", "Re-compressed (double JPEG)").

**Optional CNN ensemble, currently unused in this app:** the module also
defines `predict_ai_probability()`/`predict_mobilenet_probability()` —
two additional classifiers (a ResNet-18 and a second MobileNetV3) that
would blend into the final score if their checkpoints
(`models/ai_detector.pth`, `mobilenetv3_test.pth`) existed. **Neither
file is part of this repo**, and `main.py` calls `detect_image(path,
use_cnn=False)` explicitly — so this ensemble path is present in the
code but inert in the deployed app. `cnn_available()`/`mobilenet_available()`
just check for the files and return `False` either way here.

**Integration point** (`main.py`): `/predict` writes the raw uploaded
bytes to a temp file (`_run_meta_detector()`) — the forensic checks need
the original file bytes (for metadata/watermark), not the re-encoded PIL
image the CNN path uses — runs `detect_image()`, and returns the result
as `meta_detector` in the `PredictResponse` (`MetaDetectorResponse` in
`schemas.py`). The frontend renders this as a "Forensic Scan — Meta
Detector" panel below the main result card (`metaPanelHtml()` in
`app.js`): a verdict box, detected-signal badges (watermark, C2PA, camera
metadata, GPS, timestamp), 6 signal-value tiles, and the evidence list —
plus a matching section in the on-screen Forensic Report. The meta
verdict is persisted into `analysisHistory` alongside the CNN prediction
(see 7.3).

**New dependencies**: `scipy` (required — the module imports it
unconditionally for FFT/median-filter operations) and
`invisible-watermark` (soft dependency — `scan_watermarks()` degrades
gracefully with an `"invisible-watermark not installed"` message in its
result if missing, rather than crashing).

---

## 7. Web Application — Frontend

Plain HTML/CSS/JS (no build step, no framework) in
`webapp/backend/app/static/`: `index.html`, `style.css`, `app.js`.

### 7.1 Structure

A single-page app: one `index.html` with 5 `<section class="page">`
blocks, toggled via JS (`goToPage(key)`), not full page reloads:

1. **Main Model** — a **Model 1 / Model 2 toggle** (`noaug` / `best`,
   Model 1 selected by default on load), upload, analyze, result card,
   "View Grad-CAM" / "Generate Forensic Report" buttons (enabled only
   after a successful analysis). Switching the toggle re-checks that
   checkpoint's availability and clears any in-progress upload/result for
   a fresh start (`initMainModelToggle()` in `app.js`).
2. **Cross-Domain Model** — same structure, different model/copy
3. **Grad-CAM** — original image vs. an adjustable-intensity overlay
   (Original/Heatmap/Overlay mode toggle + a real intensity slider —
   implemented by stacking the actual heatmap image over the original at
   a CSS `opacity` tied to the slider value, not a fixed pre-blended
   image)
4. **Forensic Report** — full 8-section document view (Case Info, Image
   Info, Model Info, Inference Preprocessing, Explainability/Grad-CAM,
   Forensic Scan/Meta Detector, Final Assessment, Disclaimer), rendered
   client-side from the same analysis data (not fetched from `/report`);
   Download dropdown offers PDF (browser print) and Word (`.docx`, via
   `POST /report/docx` — note: the `.docx` export currently does **not**
   include the Forensic Scan section, only the on-screen/PDF report does;
   see 6.7/6.10)
5. **History** — every analysis run this session (persisted, see 7.3),
   with search/filter, a table, and a right-side timeline panel grouped
   by date
6. **User Guide** — static reference content

Each page has an empty state (Grad-CAM/Report/History) shown when no
analysis has been run yet, with a "Go to Main Model" call to action.

### 7.2 State management

One in-memory object, `activeAnalysis`, holds everything the Grad-CAM
and Report pages need: `analysisId` (format `FA-000124`, incrementing
counter), model info, file metadata (computed client-side: dimensions
via an `Image` load, size via `File.size`, type from the MIME type),
prediction, both base64 Grad-CAM images, and a timestamp. Navigating
between Main Model → Grad-CAM → Report always reflects the same
`activeAnalysis` — running a new analysis replaces it and assigns a new
ID.

### 7.3 History persistence

`analysisHistory` (array, newest-first) is saved to `localStorage` after
every analysis — both successes and failures (a failed analysis is
logged with its real error message, not silently dropped) — and reloaded
on page init, so it survives a real page reload, not just navigation
within a session. Capped at 15 entries (`MAX_HISTORY_ENTRIES`) since each
entry embeds base64 Grad-CAM images and `localStorage` has a small quota
(typically 5–10 MB); if saving fails (quota exceeded), it drops the older
half and retries once, then continues in-memory-only rather than
crashing.

Image previews use `FileReader`-produced `data:` URLs, not
`URL.createObjectURL()` blob URLs — blob URLs are invalidated as soon as
the page that created them unloads, which would silently break every
persisted thumbnail after a reload. This was an actual bug caught and
fixed during development, not a design chosen from the start.

There is **no backend database** — history is browser-local only. A page
load in a different browser, or clearing site data, starts empty.

### 7.4 Why the Report page renders client-side, not via `/report`

The `/report` HTML endpoint (6.7) still exists and works, but the
current frontend's Report page builds its own HTML from `activeAnalysis`
directly, for two reasons: (1) it needs to match a specific visual design
(numbered sections, theme-aware styling, the 3-column
Case/Image/Model-Info layout — see 7.5) that would otherwise require two
divergent templates to stay in sync, and (2) the "Download as Word"
action needs the exact same data the screen shows, which is simplest to
guarantee by having one source of truth (the JS object) rather than
re-fetching from the server.

### 7.5 Notable layout decisions (and bugs fixed along the way)

- **Report Info columns**: Case Information / Image Information / Model
  Information render as 3 equal-width columns
  (`grid-template-columns: 1fr 1fr 1fr`); any column whose content
  exceeds ~200px scrolls internally (`overflow-y: auto`) rather than
  stretching taller than its neighbors.
- **Print/PDF column collapse bug**: the CSS mobile breakpoint
  (`max-width: 900px`, for collapsing the sidebar on phones) was also
  matching the browser print engine's page width (~816px for US Letter),
  which silently collapsed side-by-side Grad-CAM images to 1 column only
  in the printed/PDF output while the on-screen layout looked correct.
  Fixed with an explicit `@media print` override forcing the intended
  column counts. Worth knowing if you add more multi-column sections —
  the same trap will recur unless the print override is extended too.
- **Sidebar theming**: the sidebar switches between a light cream
  background (light theme) and dark brown/black (dark theme) — it does
  **not** stay permanently dark across both themes, which was an earlier,
  incorrect implementation later corrected.
- **Theme persistence**: `localStorage` key `ff-theme`, defaults to
  `"light"`.

---

## 8. Model Wiring — Main Model's Model 1 / Model 2 Toggle

- **Main Model page → a live toggle**: **Model 1** = `noaug` checkpoint
  (`/predict?model=noaug`, selected by default on page load), **Model 2**
  = `best` checkpoint (`/predict?model=best`). Switching is instant, no
  restart or code change required — implemented in `app.js` via
  `mainPageModelKey` (module-level state) and `initMainModelToggle()`,
  which updates `#page-main`'s `data-model-key`/`data-model-label`
  attributes, re-runs `updateModelStatus()` for the newly selected
  checkpoint, and clears any in-progress upload/result so switching
  models mid-session always starts fresh.
- **Cross-Domain Model page → `cross_domain` checkpoint**
  (`/predict?model=cross_domain`, no toggle — single model).

**Why the default is `noaug`, not `best`:** this was a deliberate choice
made during the frontend rebuild, not an oversight — `noaug` was kept as
the default for continuity with an earlier UI iteration. If you're
reproducing the specific accuracy numbers from
`doc/Milestone-5/Milestone5.md` (which evaluates `mobilenetv3_best.pth`
in depth) using the live web app, switch to **Model 2** first — Model 1
(`noaug`) has a measured, reproducible bias problem: run against the 26
real images in `Test Sample/Test_real_vs_Fake/real/`, Model 1 correctly
identified only **38.5%** as Real, vs. **73.1%** for Model 2. This isn't
a bug in the toggle — it's `noaug`'s own trained behavior (it was an
augmentation-ablation run, never intended to be the primary model). If
you want to change the *default* selected on page load, that's a
one-line change: the `active` class and `data-main-model` on the
Model 1/Model 2 buttons in `index.html`, plus `mainPageModelKey`'s
initial value in `app.js`.

---

## 9. Implementation Notes / Known Issues

### 9.1 Face detection broken on Windows 11 N

**Symptom:** `/health` reports `"face_alignment": "center_crop_fallback"`
even with `retina-face`/`opencv-python` installed; every request silently
uses a center-square crop instead of real face detection.

**Root cause:** Windows 11 **N edition** ships without the Windows Media
Foundation DLLs (`MFPlat.DLL`, `MF.dll`, `MFReadWrite.dll`). OpenCV's
Windows wheels link against Media Foundation even for pure image
operations, so `cv2` (and anything built on it, including `retina-face`)
fails to load its face-detection functionality on this edition until
that's fixed — confirmed via PE import-table analysis (`pefile`), not
guessed.

**Fix:** install the official *Media Feature Pack for Windows 11 N* from
Microsoft, reboot, then reinstall `opencv-python-headless` and restart
the server.

**Practical workaround until then:** upload images already cropped close
to the face (like the sample images in `Test Sample/`) — the center-crop
fallback lands correctly on those. Full photos with background/off-center
faces get mis-cropped under the fallback and may predict incorrectly.

### 9.2 Checkpoint file-size discrepancy

`mobilenetv3_best.pth` is ~17 MB (16.24 MB measured), noticeably smaller
than a typical FP32 MobileNetV3-Large checkpoint with optimizer state
included — because Stage 3's save (`final-mobilenet (1).ipynb`) omits
`optimizer_state_dict`. This is expected, not a corrupted/partial file.

### 9.3 Real vs. estimated numbers in the evaluation report

`doc/Milestone-5/Milestone5.md` explicitly labels every number as either
real (measured directly against the actual checkpoint) or estimated
(clearly flagged, with the reasoning shown) — e.g. GPU latency for
`mobilenetv3_best.pth` is an extrapolation, not a measurement, because no
GPU was available in the development environment and a prepared Kaggle
benchmark cell (`outputs/kaggle_gpu_benchmark_cell.py`) didn't run
successfully. If you have GPU access, running that cell would replace
the estimate with a real number — see the report's Section 9.2/9.5 for
exactly what's still open.

### 9.4 Demographic fairness — deliberately not measured

The report explicitly discloses that demographic bias (skin tone, age,
gender) has never been assessed — not because it doesn't matter, but
because doing it properly requires a dataset with verified (not
model-inferred) demographic labels, which this project doesn't have. See
`Milestone5.md` Section 7.4 for the reasoning; don't infer fairness
either way from anything else in this repo.

---

## 10. Reproducing Results

1. Train `final-mobilenet (1).ipynb` on Kaggle (attach FFHQ, Stable
   Diffusion, and CelebA datasets — see root `README.md` for exact
   slugs), download `mobilenetv3_best.pth`.
2. Train `cross-domain.ipynb` similarly for `mobilenetv3_cross_domain.pth`.
3. Place both (plus `mobilenetv3_noaug.pth`/`mobilenetv3_manipulations.pth`
   if you have them from earlier ablation runs) in `webapp/output/`.
4. `cd webapp/backend && pip install -r requirements.txt && uvicorn
   app.main:app --port 8000`.
5. Expected held-out test results for `mobilenetv3_best.pth` (verified,
   from an actual run): **99.63% test accuracy** on 2,401 images,
   **99.71%** best validation accuracy. Full breakdown (classification
   report, confusion matrix, per-stage accuracy) in
   `doc/Milestone-5/Milestone5.md` Section 4.

---

## 11. Where to Go Next

- **Full evaluation results**: `doc/Milestone-5/Milestone5.md`
- **Training notebook cell-by-cell walkthrough**: root `README.md`
- **Web app quick reference**: `webapp/backend/README.md`
- **Known open items / future work**: `doc/Milestone-5/Milestone5.md`
  Section 8, and the "Known Open Items" section of the root `README.md`

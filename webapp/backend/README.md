# Face Forensics — Local Web App

FastAPI backend + a static HTML/JS frontend for testing trained
MobileNetV3-Large face-authenticity checkpoints interactively in a
browser, instead of via notebook upload widgets — plus automated
forensic report generation (on-screen, printable PDF, and real `.docx`
export).

For a deeper walkthrough of every file and implementation decision, see
`../../DeveloperGuide.md` at the repo root — this file is the quick
reference.

## Setup

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

`python-docx` is a required dependency (not optional) — the app imports
it at startup for the `.docx` forensic-report export, so the server
won't start without it.

## Checkpoints

By default the app looks for checkpoints (and the pre-computed result
CSVs/images used by `/api/training-results`) in `webapp/output/` — the
sibling folder to `backend/`. Expected filenames (`app/config.py`):

| File | Role | Required? |
|---|---|---|
| `mobilenetv3_noaug.pth` | **Main Model** page | Required |
| `mobilenetv3_cross_domain.pth` | **Cross-Domain Model** page | Required |
| `mobilenetv3_best.pth` | Internal `"best"` role — used as the default for a few endpoints (`/report`, `/compare`) even though neither page in the UI currently calls it directly | Recommended |
| `mobilenetv3_manipulations.pth` | Used by `/robustness` (11-manipulation stress test, not exposed as its own page in the current UI) | Optional |
| `mobilenetv3_tuned.pth` | Swept-hyperparameters comparison (`/compare?mode=hparams`) | Optional |

Any missing file is skipped, not fatal — check `GET /health`'s
`loaded_models` field to see what's actually loaded; a page whose
checkpoint isn't loaded shows a clear "model not available" message
instead of erroring.

If your checkpoints live somewhere else:

```bash
# Windows PowerShell
$env:CHECKPOINT_DIR = "D:\path\to\your\checkpoints"
uvicorn app.main:app --port 8000

# bash
CHECKPOINT_DIR=/path/to/checkpoints uvicorn app.main:app --port 8000
```

## Run

From inside `backend/`:

```bash
uvicorn app.main:app --port 8000
```

Then open **http://localhost:8000**. (`--reload` can be flaky with this
app's background model loading on Windows — a plain restart after code
changes is more reliable.)

- `http://localhost:8000/docs` — interactive Swagger UI, useful to test
  endpoints directly before touching the frontend.
- `http://localhost:8000/health` — reports which checkpoints loaded and
  which face-alignment method is active.

## What's in the app

A 5-page single-page app (`app/static/index.html` + `app.js`, no build
step):

1. **Main Model** — upload an image, analyze, get a Real/AI-Generated
   verdict with confidence, powered by the `noaug` checkpoint.
2. **Cross-Domain Model** — same flow, powered by the `cross_domain`
   checkpoint (trained on general/non-face images across multiple
   domains).
3. **Grad-CAM** — original image vs. an adjustable-intensity heatmap
   overlay (Original/Heatmap/Overlay toggle + a real intensity slider),
   for whichever image was most recently analyzed on either page above.
4. **Forensic Report** — a full document-style report (case info, image
   info, model info, inference preprocessing steps, Grad-CAM evidence,
   final assessment, disclaimer) for the most recent analysis, with a
   Download dropdown: PDF (browser print) or a real `.docx` file (server
   generated via `POST /report/docx`).
5. **History** — every analysis run this browser session, persisted to
   `localStorage` (survives page reloads; no backend database), with
   search/filter and a "View Details" action that reloads that entry into
   the Grad-CAM/Report pages. Capped at 15 entries.

Plus a static **User Guide** page.

**Note on model naming:** the "Main Model" page intentionally serves the
`noaug` checkpoint, not `mobilenetv3_best.pth` (the checkpoint
`doc/Milestone-5/Milestone5.md` evaluates in depth). See
`DeveloperGuide.md` Section 8 for why, and how to change it if you want
"Main Model" to serve `best` instead.

## API endpoints

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Loaded checkpoints + active face-alignment method |
| `/predict` | POST | `?model=noaug\|cross_domain\|best\|...` + image → prediction + Grad-CAM |
| `/report` | POST | Same inputs as `/predict`, returns a standalone HTML report (legacy — the frontend's own Report page is the one actually used) |
| `/report/docx` | POST | Takes already-computed analysis data as JSON, returns a downloadable `.docx` |
| `/robustness` | POST | Runs the 11-manipulation stress test via the `manipulations` checkpoint |
| `/compare` | POST | `?mode=augmentation\|hparams` — side-by-side prediction from two checkpoints |
| `/api/training-results` | GET | Pre-computed CSVs/images from the training notebooks (not live inference) |

## Known issue: face alignment

The app tries to auto-crop the uploaded image down to just the detected
face before running the model (matching what the training notebook did
with RetinaFace) and otherwise falls back to a plain center-square crop.
**On a typical Windows 11 N development machine, real face detection is
broken** and every request uses the center-crop fallback — check
`/health`'s `face_alignment` field (`"center_crop_fallback"` means no
real detection is active).

Root cause: Windows 11 **N edition** ships without the Windows Media
Foundation DLLs (`MFPlat.DLL`, `MF.dll`, `MFReadWrite.dll`). OpenCV's
Windows wheels link against Media Foundation even for pure image use, so
`cv2` (and anything built on it, including `retina-face`) fails to load
face-detection functionality on this edition until that's installed.

**Fix**: install the official *Media Feature Pack for Windows 11 N* from
Microsoft, reboot, then reinstall `opencv-python-headless` and restart
the server.

**Practical workaround until then:** upload images already cropped close
to just the face (like `Test Sample/`'s own images) — center-crop lands
correctly on those. Full photos with background/off-center faces will
get mis-cropped under the fallback and may predict incorrectly.

## Notes

- Inference runs on CPU by default (or GPU automatically if a
  CUDA-enabled torch + GPU are available).
- `/predict` and `/report` take an optional `?model=` query param —
  defaults to `best`.

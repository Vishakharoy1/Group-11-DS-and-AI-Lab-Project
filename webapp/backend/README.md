# Deepfake Face Detector — Local Web App

FastAPI backend + a static HTML/JS frontend for testing trained
MobileNetV3-Large deepfake-detection checkpoints interactively in a
browser, instead of via notebook upload widgets.

## Setup

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Checkpoints

By default the app looks for checkpoints (and the pre-computed result
CSVs/images from `/api/training-results`) in `Deepfake/output/` — the
sibling folder to `backend/`. Expected filenames:

| File | Role | Required? |
|---|---|---|
| `mobilenetv3_best1.pth` | Main model — Prediction/Grad-CAM, Cross-Domain Testing, and the baseline side of Model Comparison | **Required** |
| `mobilenetv3_noaug.pth` | No-Augmentation model — its own independent upload section, plus the "With vs. Without Augmentation" comparison panel | Optional |
| `mobilenetv3_manipulations.pth` | Manipulation-robustness model — used exclusively by the Manipulation Robustness Testing section instead of `best1` | Optional (falls back to a 503 error on that endpoint if missing) |
| `mobilenetv3_tuned.pth` | Swept-hyperparameters model — "Baseline vs. Tuned Hyperparameters" comparison panel | Optional |

Any missing file is skipped, not fatal — check `GET /health` for
`loaded_models` to see what's actually active; the corresponding UI
section/panel shows a clear "not available" message instead of erroring.

If your checkpoints live somewhere else, point at that folder instead of
copying files (this also moves where `/api/training-results` looks for
CSVs/sample images, via `RESULTS_DIR`):

```bash
# Windows PowerShell
$env:CHECKPOINT_DIR = "D:\path\to\your\checkpoints"
uvicorn app.main:app --reload --port 8000

# bash
CHECKPOINT_DIR=/path/to/checkpoints uvicorn app.main:app --reload --port 8000
```

## Run

From inside `backend/`:

```bash
uvicorn app.main:app --port 8000
```

Then open **http://localhost:8000** in a browser. (Note: `--reload` has
been flaky with this app's background model loading on Windows — a plain
restart after code changes is more reliable.)

- `http://localhost:8000/docs` — interactive Swagger UI, useful to test
  `/health`, `/predict`, `/robustness`, `/compare` directly before touching
  the frontend.
- `http://localhost:8000/health` — reports which checkpoints loaded and
  which face-alignment method is active.

## What's in the app

1. **Real vs. AI Prediction & Explainability** — upload a face image, get a
   Real/Fake prediction with a Grad-CAM heatmap, from the main (`best1`)
   model.
2. **No-Augmentation Model** — independent upload, runs the `noaug`
   checkpoint on its own (highest measured test accuracy of the available
   checkpoints).
3. **Cross-Domain Testing** — independent upload, still the `best1` model,
   intended for images outside the core face domain.
4. **Manipulation Robustness Testing** — independent upload, runs all 11
   manipulations (tint, brightness, contrast, blur, jpeg, resize, crop,
   noise) through the `manipulations` checkpoint specifically. Shows only a
   plain Real/Fake verdict + confidence score (majority vote across the 11
   manipulations) — no per-manipulation table in the UI.
5. **Model Comparison** — independent upload, side-by-side
   with-vs-without-augmentation and baseline-vs-tuned-hparams panels
   (each hidden/disabled if its checkpoint isn't loaded).
6. **Training & Evaluation Results** — static dashboard of the
   pre-computed CSVs/sample images/Grad-CAM gallery from the training
   notebook's `output/` folder (not live inference).

## Known issue: face alignment

The app tries to auto-crop the uploaded image down to just the detected
face before running the model (matching what the training notebook did
with RetinaFace) and otherwise falls back to a plain center-square crop.
**On this development machine, real face detection is currently broken**
and every request uses the center-crop fallback — check `/health`'s
`face_alignment` field (`"center_crop_fallback"` means no real detection
is active).

Root cause: this machine is **Windows 11 Pro N**, which ships without the
Windows Media Foundation DLLs (`MFPlat.DLL`, `MF.dll`, `MFReadWrite.dll`).
OpenCV's Windows wheels link against Media Foundation even for pure image
use, so `cv2` (and anything built on it, including `retina-face`) fails to
load face-detection functionality on this machine until that's installed.

**Fix** (not yet applied): install the official *Media Feature Pack for
Windows 11 N* from Microsoft
(`https://www.microsoft.com/en-us/software-download/mediafeaturepack`),
reboot, then reinstall `opencv-python-headless` and restart the server.

**Practical workaround until then:** upload images already cropped close
to just the face (like the dataset's own sample images) — center-crop
lands correctly on those and predictions are accurate. Full photos with
background/off-center faces will get mis-cropped and may predict
incorrectly under the fallback.

## Notes

- Inference runs on CPU by default (or GPU automatically if a CUDA-enabled
  torch + GPU are available). A single prediction is fast either way; the
  `/robustness` endpoint runs 11 forward passes so it's the slowest call.
- `/predict` takes an optional `?model=` query param (`best` / `noaug` /
  `tuned`) — defaults to `best`.

---

## Team Declaration

We certify that all team members have actively contributed to the preparation of this document. Each member has reviewed the contents, understands the work presented, and agrees with the submitted report.

**Project:** Deep Learning-Based Human Face Authenticity Detection  
**Team:** Group 11 — Vishakha · Rohit · Aman · Raunak · Somendu  
**Course:** DS & AI Lab Project

| Team Member | Role | Signature |
| --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Vishakha |
| Rohit | Training Stability Lead | Rohit |
| Aman | Preprocessing & Transfer Learning Lead | Aman |
| Raunak | Dataset & Bias Analysis Lead | Raunak |
| Somendu | Explainability & Optimisation Lead | Somendu |

# Deepfake Face Detector — Local Web App

FastAPI backend + a static HTML/JS frontend for testing the trained
MobileNetV3-Large deepfake-detection model interactively in a browser,
instead of via notebook upload widgets.

## Setup

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Checkpoints

Drop your trained `.pth` files into `backend/checkpoints/`:

- `mobilenetv3_best.pth` — **required**. The main trained model.
- `mobilenetv3_tuned.pth` — optional. Enables the "Baseline vs. Tuned
  Hyperparameters" comparison panel.
- `mobilenetv3_noaug.pth` — optional. Enables the "With vs. Without
  Augmentation" comparison panel.

Any missing file is skipped, not fatal — the corresponding UI panel just
shows "not available" instead of erroring.

If your checkpoints live somewhere else, point at that folder instead of
copying files:

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
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser.

- `http://localhost:8000/docs` — interactive Swagger UI, useful to test
  `/health`, `/predict`, `/robustness`, `/compare` directly before touching
  the frontend.
- `http://localhost:8000/health` — reports which checkpoints loaded and
  whether real RetinaFace face-alignment is available or it's falling back
  to center-crop.

## Notes

- Face alignment uses RetinaFace if installed (`pip install retina-face
  opencv-python`), otherwise falls back to center-crop automatically —
  check `/health` to see which is active.
- Inference runs on CPU by default (or GPU automatically if a CUDA-enabled
  torch + GPU are available). A single image is fast either way; the
  `/robustness` endpoint runs 11 forward passes so it's the slowest call.

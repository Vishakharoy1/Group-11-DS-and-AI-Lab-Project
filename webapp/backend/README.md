# Deepfake Face Detector — Web App

FastAPI backend + static HTML/JS frontend for interactive face
authenticity testing with trained MobileNetV3-Large checkpoints.

**Production (Render):** https://group-11-ds-and-ai-lab-project.onrender.com  
Deployed from **`main`** — see **`READMEdeployment.md`**.

## Setup (local — `main` branch)

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Checkpoints

By default the app looks for checkpoints (and pre-computed result
CSVs/images from `/api/training-results`) in `webapp/output/`.

| File | Role | Production (Render) | Local |
|---|---|---|---|
| `mobilenetv3_noaug.pth` | Main Model default | ✅ Loaded | Optional |
| `mobilenetv3_cross_domain.pth` | Cross-Domain Model | ✅ Loaded | Optional |
| `mobilenetv3_best.pth` | Main Model toggle (Model 2) | ❌ RAM limit | Optional |
| `mobilenetv3_manipulations.pth` | Manipulation Robustness page | ❌ RAM limit | Optional |
| `mobilenetv3_tuned.pth` | Hyperparameter comparison | ❌ RAM limit | Optional |

Any missing file is skipped gracefully — check `GET /health` for
`loaded_models`. Affected UI sections show "model not available".

Override checkpoint location:

```bash
# Windows PowerShell
$env:CHECKPOINT_DIR = "D:\path\to\checkpoints"
uvicorn app.main:app --port 8000

# bash
CHECKPOINT_DIR=/path/to/checkpoints uvicorn app.main:app --port 8000
```

## Run locally

From inside `webapp/backend/`:

```bash
uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**

- `http://localhost:8000/docs` — Swagger UI
- `http://localhost:8000/health` — loaded models + face alignment method

## What's in the app

1. **Main Model** — Real/Fake prediction + Grad-CAM (`noaug` default;
   Model 2 toggle to `best` when that checkpoint is loaded locally).
2. **Cross-Domain Model** — uses `cross_domain` checkpoint.
3. **Manipulation Robustness** — 11 corruptions via `manipulations`
   checkpoint (local only unless deployed).
4. **Model Comparison** — side-by-side augmentation / hyperparameter
   panels (local only unless all checkpoints loaded).
5. **Training & Evaluation Results** — static dashboard from `output/`
   CSVs and PNGs.
6. **Forensic Report** — HTML + `.docx` export.

## Production notes

- Render loads **`noaug` + `cross_domain`** only (~310 MB RAM).
- Default `/predict` model on deploy branch is **`noaug`** (not `best`).
- RetinaFace is not installed on Render — center-crop fallback; upload
  face-focused images for best accuracy.

## Known issue: face alignment

The app tries RetinaFace face detection before inference, falling back
to center-crop. On **Windows 11 N**, OpenCV/RetinaFace often fail —
check `/health` for `"center_crop_fallback"`. Upload pre-cropped face
images as a workaround. See `DeveloperGuide.md` Section 9.

## Notes

- CPU inference by default; `/robustness` runs 11 forward passes (slowest).
- `/predict?model=` accepts `noaug`, `best`, `cross_domain`, `tuned`
  (whichever are loaded).

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

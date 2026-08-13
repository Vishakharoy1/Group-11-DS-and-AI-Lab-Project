# Face Forensics — Deepfake Detection Web Application

A production-deployed web application for AI-generated face detection, powered by
a fine-tuned **MobileNetV3-Large** CNN. Upload any face image and get a
Real/AI-Generated verdict with confidence scores, Grad-CAM explainability
heatmaps, and a downloadable forensic report — all running entirely on CPU in
the browser.

**Live URL**: [https://group-11-ds-and-ai-lab-project.onrender.com](https://group-11-ds-and-ai-lab-project.onrender.com)

> **Note**: Render free tier spins down after 15 minutes of inactivity. The first
> request after idle takes ~30–60 seconds (container restart + model loading).
> Subsequent requests are fast (~1–3 seconds).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Model & Checkpoints](#model--checkpoints)
- [API Endpoints](#api-endpoints)
- [Deployment on Render](#deployment-on-render)
  - [Infrastructure](#infrastructure)
  - [Docker Configuration](#docker-configuration)
  - [Environment Variables](#environment-variables)
  - [Git LFS for Model Weights](#git-lfs-for-model-weights)
  - [Build Pipeline](#build-pipeline)
- [Render Free Tier — Limitations & Mitigations](#render-free-tier--limitations--mitigations)
- [Changes from Main Branch](#changes-from-main-branch)
- [Local Development](#local-development)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Render (Docker)                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ Static Files  │    │     FastAPI Backend           │   │
│  │ index.html    │    │                              │   │
│  │ app.js        │◄──►│  /predict   → MobileNetV3   │   │
│  │ style.css     │    │  /report    → Grad-CAM + PDF │   │
│  │               │    │  /health    → Status check   │   │
│  └──────────────┘    │  /robustness → 11 manipulations│  │
│                       │  /compare   → A/B models     │   │
│                       └──────────┬───────────────────┘   │
│                                  │                       │
│                       ┌──────────▼───────────────────┐   │
│                       │  Model Registry               │   │
│                       │  ┌─────────────────────────┐  │   │
│                       │  │ mobilenetv3_noaug.pth    │  │   │
│                       │  │ mobilenetv3_cross_domain  │  │   │
│                       │  └─────────────────────────┘  │   │
│                       │  + Forensic Meta-Detector     │   │
│                       │  (metadata/spectral/noise/ELA)│   │
│                       └──────────────────────────────┘   │
│                                                         │
│  Port: 10000 (Render default)                           │
│  Runtime: python:3.11-slim + CPU PyTorch                │
└─────────────────────────────────────────────────────────┘
```

The application is a **single-container** deployment: FastAPI serves both the
REST API and the static frontend from one process. No database, no external
services, no CUDA — everything runs on CPU in a single Render web service
container.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Vanilla HTML/JS/CSS | 5-page SPA, no build step |
| **Backend** | FastAPI + Uvicorn | Async REST API |
| **ML Model** | MobileNetV3-Large (PyTorch) | Binary classification (Real vs Fake) |
| **Explainability** | Grad-CAM | Visual attention heatmaps |
| **Forensics** | Custom meta-detector | Metadata, spectral, sensor-noise, ELA analysis |
| **Reports** | python-docx + HTML | Downloadable .docx forensic reports |
| **Container** | Docker (python:3.11-slim) | Reproducible deployment |
| **Hosting** | Render (Free Tier) | Cloud deployment with auto-deploy from GitHub |
| **Model Storage** | Git LFS | Large file tracking for .pth checkpoints |

---

## Repository Structure

This is an **orphan branch** (`render-deploy`) containing only the deployment
essentials — no notebooks, milestone documents, or training code from `main`.

```
render-deploy/
├── render.yaml                        # Render Blueprint (IaC service definition)
├── .gitattributes                     # Git LFS tracking rules for .pth files
├── .gitignore
│
└── webapp/
    ├── backend/
    │   ├── Dockerfile                 # Docker image definition
    │   ├── requirements.txt           # Python deps (torch installed separately)
    │   └── app/
    │       ├── __init__.py
    │       ├── config.py              # Paths, constants, checkpoint registry
    │       ├── main.py                # FastAPI routes + static mount
    │       ├── model.py               # MobileNetV3 architecture + ModelRegistry
    │       ├── gradcam.py             # Grad-CAM implementation
    │       ├── preprocessing.py       # Face crop/align + inference transforms
    │       ├── meta_detector.py       # Forensic meta-detector (1030 lines)
    │       ├── manipulations.py       # 11 image manipulation functions
    │       ├── report.py              # HTML + .docx report generation
    │       ├── results.py             # Pre-computed training result loader
    │       ├── schemas.py             # Pydantic response models
    │       └── static/
    │           ├── index.html         # SPA entry point
    │           ├── app.js             # Frontend logic (~53KB)
    │           ├── style.css          # Styling (~23KB)
    │           ├── sample_fake.png    # Demo sample (AI-generated face)
    │           └── sample_real.jpg    # Demo sample (real photograph)
    │
    └── output/
        ├── mobilenetv3_noaug.pth      # Primary model checkpoint (45 MB, LFS)
        ├── mobilenetv3_cross_domain.pth # Cross-domain model checkpoint (45 MB, LFS)
        ├── *.csv                      # Pre-computed evaluation metrics
        └── *.png                      # Grad-CAM gallery + sample grids
```

---

## Model & Checkpoints

### Architecture

**MobileNetV3-Large** with a custom classifier head, matching the training
notebook exactly:

```python
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Linear(960, 1280),    # in_features from backbone
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(1280, 2),      # Binary: [Real, Fake]
)
```

### Deployed Checkpoints

| Checkpoint | Size | Role | Deployed? |
|---|---|---|---|
| `mobilenetv3_noaug.pth` | 45 MB | **Primary model** — no-augmentation baseline, default for all endpoints | ✅ Yes |
| `mobilenetv3_cross_domain.pth` | 45 MB | Cross-domain model (trained on general/non-face images across multiple domains) | ✅ Yes |
| `mobilenetv3_best.pth` | 16 MB | Best model — 3-stage fine-tuned with CelebA-HD | ❌ Not deployed (RAM limit) |
| `mobilenetv3_manipulations.pth` | 45 MB | Robustness-trained (11 corruptions) | ❌ Not deployed (RAM limit) |
| `mobilenetv3_tuned.pth` | — | Swept hyperparameters variant | ❌ Not deployed |

The `ModelRegistry` loads whatever `.pth` files exist at startup and skips
missing ones gracefully. Pages whose checkpoint isn't loaded show a "model not
available" message.

### Inference Pipeline

```
Upload Image
    │
    ▼
┌─────────────────────────┐
│ Face Crop & Alignment    │  RetinaFace (if available) or center-crop fallback
│ → 224×224 RGB            │
└───────────┬─────────────┘
            │
    ┌───────▼───────┐    ┌──────────────────┐
    │ MobileNetV3   │    │ Meta-Detector     │
    │ + Grad-CAM    │    │ (metadata/spectral│
    │               │    │  /noise/ELA)      │
    └───────┬───────┘    └────────┬─────────┘
            │                     │
            ▼                     ▼
    ┌───────────────────────────────┐
    │ JSON Response                 │
    │  prediction: Real/Fake + %    │
    │  gradcam_heatmap: base64 PNG  │
    │  gradcam_overlay: base64 PNG  │
    │  meta_detector: forensic data │
    └───────────────────────────────┘
```

---

## API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Loaded checkpoints + active face-alignment method |
| `/predict` | POST | `?model=noaug\|best\|cross_domain` + image → prediction + Grad-CAM |
| `/report` | POST | Same as `/predict`, returns standalone HTML report |
| `/report/docx` | POST | Takes pre-computed analysis JSON, returns downloadable `.docx` |
| `/robustness` | POST | 11-manipulation stress test (requires `manipulations` checkpoint) |
| `/compare` | POST | `?mode=augmentation\|hparams` — side-by-side prediction from two models |
| `/api/training-results` | GET | Pre-computed CSVs/images from training notebooks |
| `/docs` | GET | Auto-generated Swagger UI for interactive API testing |

Default model for `/predict` and `/report` is `noaug` (changed from `best` in
the deployment branch to match the deployed checkpoint strategy).

---

## Deployment on Render

### Infrastructure

| Component | Configuration |
|---|---|
| **Platform** | [Render](https://render.com) |
| **Service Type** | Web Service |
| **Runtime** | Docker |
| **Plan** | Free |
| **Region** | Oregon (US West) — configurable |
| **Branch** | `render-deploy` (orphan branch, deployment-only) |
| **Auto-Deploy** | Yes — triggers on push to `render-deploy` |

### Docker Configuration

The Dockerfile (`webapp/backend/Dockerfile`) builds a production image from
`python:3.11-slim` (Debian Bookworm):

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# System deps for Pillow, scipy, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (saves ~1.5 GB vs full CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy checkpoint(s) + result assets → /app/output/
COPY output/ /app/output/

# Copy backend app → /app/app/
COPY backend/app/ /app/app/

EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
```

**Key decisions:**
- **CPU-only PyTorch**: Installed via `--index-url https://download.pytorch.org/whl/cpu`
  to avoid pulling CUDA libraries (~1.5 GB saved). Render free tier has no GPU.
- **`--no-cache-dir`**: Prevents pip from caching wheels inside the image layer,
  keeping the final image smaller.
- **Port 10000**: Render's default port for web services. Configurable via `PORT`
  env var, but 10000 is the convention.
- **Docker context is `./webapp`**: So `COPY output/` and `COPY backend/` resolve
  relative to `webapp/`, not the repo root.

### Environment Variables

Set in `render.yaml` and/or the Render dashboard:

| Variable | Value | Purpose |
|---|---|---|
| `CHECKPOINT_DIR` | `/app/output` | Where `config.py` looks for `.pth` files |
| `RESULTS_DIR` | `/app/output` | Where pre-computed CSVs/PNGs live |

### Git LFS for Model Weights

The `.pth` checkpoint files are tracked via [Git LFS](https://git-lfs.github.com/):

```gitattributes
webapp/output/*.pth filter=lfs diff=lfs merge=lfs -text
```

**Why LFS?**
- `mobilenetv3_noaug.pth` is 45 MB, `mobilenetv3_cross_domain.pth` is 45 MB — too large
  for regular Git (GitHub's hard limit is 100 MB per file).
- Render natively supports Git LFS during builds — it automatically pulls LFS
  objects when cloning the repo.
- Total LFS usage: **~90 MB** of GitHub's 1 GB free LFS quota.

### Build Pipeline

What happens when you push to `render-deploy`:

```
1. Render detects push to render-deploy
   │
2. Clones repo + pulls Git LFS objects (checkpoints)
   │
3. Builds Docker image:
   │  ├─ Install system libs (libglib2, libjpeg, zlib)
   │  ├─ Install CPU PyTorch (~250 MB wheel)
   │  ├─ Install requirements.txt (FastAPI, Pillow, scipy, etc.)
   │  ├─ COPY output/ (checkpoints + assets → /app/output/)
   │  └─ COPY backend/app/ (Python code → /app/app/)
   │
4. Starts container:
   │  └─ uvicorn app.main:app --host 0.0.0.0 --port 10000
   │
5. FastAPI startup event:
   │  ├─ ModelRegistry scans /app/output/ for .pth files
   │  ├─ Loads mobilenetv3_noaug.pth → model.eval()
   │  ├─ Loads mobilenetv3_cross_domain.pth → model.eval()
   │  └─ Mounts static files + results assets
   │
6. Service is live ✅
```

First build takes **~5–10 minutes** (PyTorch wheel download). Subsequent builds
use Render's Docker layer cache and are faster.

---

## Render Free Tier — Limitations & Mitigations

### Resource Constraints

| Resource | Free Tier Limit | Our Usage | Status |
|---|---|---|---|
| **RAM** | 512 MB | ~310 MB (PyTorch + 2 models loaded) | ✅ Within limit |
| **CPU** | 0.1 vCPU (shared) | Single-threaded inference, ~1–3s per prediction | ✅ Acceptable |
| **Disk** | Ephemeral (no persistent storage) | ~61 MB checkpoints baked into image | ✅ No disk needed |
| **Bandwidth** | 100 GB/month | Minimal (JSON responses, base64 images) | ✅ Well within |
| **Build time** | 15 min timeout | ~5–10 min first build | ✅ Within limit |
| **Uptime** | Spins down after 15 min idle | Cold start ~30–60s | ⚠️ See below |

### Cold Start Problem

Render free tier **spins down the container** after 15 minutes of no inbound
traffic. The next request triggers a full container restart:

```
Cold start timeline:
  0s    — Render receives request, begins container start
  ~10s  — Docker container boots, Python process starts
  ~20s  — PyTorch loads, ModelRegistry begins loading checkpoints
  ~40s  — mobilenetv3_noaug.pth loaded into memory (45 MB)
  ~45s  — mobilenetv3_cross_domain.pth loaded (45 MB)
  ~50s  — Uvicorn ready, request served
```

**Mitigations:**
- Hit `/health` before a demo to "warm up" the container
- The free tier does not support Render's "Health Check Path" to keep services
  alive — that's a paid feature
- Consider an external uptime monitor (e.g., UptimeRobot free tier) to ping
  `/health` every 14 minutes to prevent spin-down

### What We Deliberately Excluded

To stay within the 512 MB RAM limit:

| Excluded | Size Impact | Mitigation |
|---|---|---|
| `mobilenetv3_best.pth` | +17 MB RAM (third model would exceed limit) | Model 2 toggle uses `cross_domain` instead; `best` not shipped |
| `mobilenetv3_manipulations.pth` | +45 MB RAM | Robustness page shows "model not available" |
| `mobilenetv3_tuned.pth` | — | Comparison mode shows "checkpoint not loaded" |
| `retina-face` + TensorFlow | +500 MB disk/RAM | Falls back to center-crop (works for cropped face photos) |
| `invisible-watermark` | +50 MB | Meta-detector's watermark scan returns "not installed" |

All exclusions use **graceful degradation** — the app never crashes, it just
reports reduced capability via the API and UI.

---

## Changes from Main Branch

This `render-deploy` branch is an **orphan branch** (no shared Git history with
`main`) containing only deployment-essential files. The code changes from the
original `main` branch:

| File | Change | Reason |
|---|---|---|
| `app/main.py` | Default model: `"best"` → `"noaug"` in `/predict`, `/report`, `_require_model()` | `noaug` is the guaranteed-deployed checkpoint |
| `app/model.py` | `is_ready()`: checks `len(self.models) > 0` instead of `"best" in self.models` | App should work if only `noaug` is loaded |
| `requirements.txt` | Removed `invisible-watermark`, `torch`, `torchvision` | `torch` installed via CPU index in Dockerfile; watermark lib is optional |
| `Dockerfile` | **New** | Docker build for Render |
| `render.yaml` | **New** | Render Blueprint (IaC) |
| `.gitattributes` | **New** | Git LFS tracking for `.pth` files |

---

## Local Development

### Run with Docker (matches production)

```bash
# From the repo root (render-deploy branch)
docker build -t face-forensics -f webapp/backend/Dockerfile webapp/
docker run -p 10000:10000 face-forensics

# Open http://localhost:10000
```

### Run without Docker

```bash
cd webapp/backend

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set checkpoint paths (if not in default location)
export CHECKPOINT_DIR=$(pwd)/../output
export RESULTS_DIR=$(pwd)/../output

# Run
uvicorn app.main:app --port 8000
# Open http://localhost:8000
```

### Verify Health

```bash
curl http://localhost:10000/health
# {"status":"ok","loaded_models":["noaug","cross_domain"],"face_alignment":"center_crop_fallback"}
```

### Test Prediction

```bash
curl -X POST "http://localhost:10000/predict?model=noaug" \
  -F "file=@webapp/backend/app/static/sample_fake.png"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| First request takes 30–60s | Render cold start (free tier spin-down) | Hit `/health` beforehand to warm up |
| `/predict` returns 503 "Model not loaded" | Checkpoint `.pth` missing from `/app/output/` | Verify Git LFS files are pulled; check Render build logs |
| "center_crop_fallback" in `/health` | `retina-face` not installed (expected on Render) | Upload pre-cropped face images for best accuracy |
| Build fails with `libgl1-mesa-glx` error | Old Debian package name | Fixed in latest commit — use `libgl1` instead |
| Build exceeds 15 min timeout | PyTorch wheel download is slow | Retry — Render caches Docker layers after first successful build |
| RAM exceeded / OOM kill | Too many models loaded | Only deploy `noaug` + `cross_domain` (current config) |

---

## References

- **MobileNetV3 Paper**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) (Howard et al., 2019)
- **Grad-CAM Paper**: [Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017)
- **Render Docker Docs**: [https://docs.render.com/docker](https://docs.render.com/docker)
- **Git LFS**: [https://git-lfs.github.com](https://git-lfs.github.com)
- **Training Notebook**: See `main` branch — `final-mobilenet.ipynb` and `cross-domain.ipynb`

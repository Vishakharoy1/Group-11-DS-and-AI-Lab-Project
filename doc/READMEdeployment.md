# Face Forensics — Deployment Guide

Simple guide for running and deploying the **Face Forensics** web app.

**Live app (no install):** [https://face-forensics.onrender.com](https://face-forensics.onrender.com)

---

## How deployment works

Everything lives on the **`main`** branch. Render builds a **minimal Docker
image** from the same repo — notebooks and docs are ignored by Docker;
only the web app and **two production checkpoints** are copied into the
container.

```
GitHub (main branch)
        │
        ▼
   render.yaml → Render Docker build
   webapp/backend/Dockerfile
   webapp/.dockerignore  (excludes extra .pth files)
        │
        ▼
   Live web app (CPU, free tier, port 10000)
```

---

## What is live in production

| Item | Details |
|---|---|
| **Platform** | [Render](https://render.com) — free-tier web service |
| **URL** | https://face-forensics.onrender.com |
| **Source branch** | `main` |
| **Models in container** | `mobilenetv3_noaug.pth` + `mobilenetv3_cross_domain.pth` |
| **Why only 2 models?** | Render free tier = **512 MB RAM**. Extra checkpoints cause OOM. |
| **Face detection** | Center-crop fallback on Render — upload face-focused images |
| **Cold start** | Free tier sleeps after ~15 min idle; first visit may take **30–60 s** |

### Pages available in production

- Main Model (`noaug`; Model 2 toggle → `cross_domain`)
- Cross-Domain Model
- Grad-CAM, Forensic Report, History

### Local-only pages (need extra checkpoints)

- Manipulation Robustness — `mobilenetv3_manipulations.pth`
- Model Comparison — `mobilenetv3_best.pth`, `mobilenetv3_tuned.pth`

---

## Use the live app

1. Open [https://face-forensics.onrender.com](https://face-forensics.onrender.com)
2. Wait on first load if the service was idle (cold start)
3. Upload a face image → **Analyze Image**

---

## Run locally (full features)

```bash
git clone https://github.com/Vishakharoy1/Group-11-DS-and-AI-Lab-Project.git
cd Group-11-DS-and-AI-Lab-Project
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

cd webapp/backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All checkpoints optional in webapp/output/ — see ../doc/DeveloperGuide.md
uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**

---

## Run with Docker (matches production)

From repo root:

```bash
docker build -t face-forensics -f webapp/backend/Dockerfile webapp/
docker run -p 10000:10000 face-forensics
```

Open **http://localhost:10000**

`webapp/.dockerignore` excludes `best`, `manipulations`, and `tuned`
checkpoints so the image matches Render (~310 MB RAM with 2 models).

---

## Deployment files on `main`

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint — service definition, branch `main` |
| `webapp/backend/Dockerfile` | Production Docker image (Python 3.11, CPU PyTorch, port 10000) |
| `webapp/backend/requirements-docker.txt` | Python deps for Docker (no torch — installed in Dockerfile) |
| `webapp/.dockerignore` | Excludes non-production checkpoints from Docker context |
| `.gitattributes` | Git LFS tracking for `webapp/output/*.pth` |

---

## Update production

1. Commit changes to **`main`**
2. Push:

```bash
git push origin main
```

3. Render auto-rebuilds if the service is linked to `main` (see Render dashboard → Settings → Branch)

**After migrating from `render-deploy`:** set Render service branch to **`main`** in the dashboard.

---

## Verify

```bash
curl https://face-forensics.onrender.com/health
```

Expected: `"status":"ok"`, `"loaded_models"` includes `noaug` and `cross_domain`.

API docs: https://face-forensics.onrender.com/docs

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Slow first load | Render free-tier cold start — wait or hit `/health` first |
| 503 Model not loaded | Checkpoints missing in image — verify `.dockerignore` and LFS |
| Wrong predictions on full photos | Upload cropped face images (no RetinaFace on Render) |
| OOM on Render | Do not remove `.dockerignore` exclusions — keep 2 models only |

---

## More detail

- **Developer setup:** `../doc/DeveloperGuide.md` §11
- **User instructions:** `../doc/User guide.md`
- **Milestone 6 report:** `doc/Milestone-6/Milestone6-Report.md`

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

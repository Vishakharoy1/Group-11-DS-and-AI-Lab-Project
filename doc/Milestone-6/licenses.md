# Licensing and Dataset References — Group 11

**Project:** Deep Learning-Based Human Face Authenticity Detection  
**Course:** DS & AI Lab Project · Milestone 6

---

## Project License

This project is released under the **MIT License**.

Copyright (c) 2026 Group 11 — DS & AI Lab Project  
(Vishakha, Rohit, Aman, Raunak, Somendu)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the MIT License conditions.

---

## Dataset Licenses

| Dataset | Source | License | Usage in Project |
|---|---|---|---|
| **FFHQ (Flickr-Faces-HQ)** | NVIDIA / Kaggle mirrors | [FFHQ License](https://github.com/NVlabs/ffhq-dataset/blob/master/LICENSE.txt) | Real training faces (~15,000 used) |
| **Stable Diffusion Face Dataset** | Kaggle: `mohannadaymansalah/stable-diffusion-dataaaaaaaaa` | Per dataset page on Kaggle | 9,001 synthetic training faces |
| **CelebA / CelebA-HD** | MMLab / academic release | Non-commercial research | 8,000 HD real faces (Stage 3 fine-tuning in `../../notebooks/final-mobilenet (1).ipynb`) |
| **Real vs AI Generated Faces** | Kaggle: `philosopher0808/real-vs-ai-generated-faces-dataset` | Per dataset page | FFHQ real images for training notebook |
| **Cross-domain corpora** | Nano Banana, CIFAKE, Places365, etc. | Per-source terms | `../../notebooks/cross-domain.ipynb` training |

**Milestone 2 reference:** FFHQ authentic images are governed by **Creative Commons BY-NC-SA 4.0** for non-commercial research use. See `doc/Milestone-2/Milestone-2-Report.md` Section 2 for the original dataset documentation.

**Citation (EMSCAD-style dataset ethics):**

> Vidros, S., Kolias, C., Kambourakis, G., & Maglaras, L. (2017). Automatic Detection of Online Recruitment Frauds. *Future Internet*, 9(1), 6.  
> *(Reference model for documenting public dataset use — our face dataset sources are FFHQ + Stable Diffusion, not EMSCAD.)*

---

## Pre-Trained Model License — MobileNetV3-Large

| Attribute | Value |
|---|---|
| Model | `mobilenetv3_large_100` |
| Organization | PyTorch / torchvision |
| License | BSD-style |
| Paper | Howard et al. (2019), "Searching for MobileNetV3" |

**Citation:**

> Howard, A., Sandler, M., Chen, B., et al. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF ICCV*.

---

## Third-Party Library Licenses

| Library | License | Usage |
|---|---|---|
| PyTorch | BSD-style | Deep learning framework |
| torchvision | BSD | MobileNetV3 backbone + transforms |
| FastAPI | MIT | REST API framework |
| Uvicorn | BSD | ASGI server |
| Pydantic | MIT | Request/response validation |
| Pillow | HPND | Image I/O |
| python-docx | MIT | Word report export |
| invisible-watermark | MIT | Watermark detection (meta-detector) |
| matplotlib | PSF | Grad-CAM colormap rendering |
| scikit-learn | BSD-3-Clause | Metrics, data splitting |
| LangChain | MIT | Not used in final deployed app |

---

## AI Service & Hosting Terms

| Service | Usage | Terms |
|---|---|---|
| **Render** | **Production web app** at [face-forensics.onrender.com](https://face-forensics.onrender.com) — FastAPI + Docker deploy from `main` | [render.com/terms](https://render.com/terms) |
| Google Colab / Kaggle GPU | Model training | Platform-specific ToS |
| Hugging Face Hub / Spaces | Optional Gradio demo at [somendu007/deepfake-detection](https://huggingface.co/spaces/somendu007/deepfake-detection) (not the primary live deployment) | [huggingface.co/terms-of-service](https://huggingface.co/terms-of-service) |
| OpenAI / Gemini | Not used in deployed app | N/A for core deployment |

---

## Fine-Tuned Checkpoints

| Checkpoint | Training Data | Deployed (Render) | License |
|---|---|---|---|
| `mobilenetv3_noaug.pth` | FFHQ + Stable Diffusion (no-augmentation ablation) | ✅ Main Model (default) | MIT (project) + dataset terms above |
| `mobilenetv3_cross_domain.pth` | Multi-domain synthetic corpora | ✅ Cross-Domain Model | MIT (project) + per-dataset terms |
| `mobilenetv3_best.pth` | Same + CelebA-HD Stage 3 | ❌ Local dev only (RAM limit on Render) | MIT (project) + dataset terms above |
| `mobilenetv3_manipulations.pth` | Robustness training (11 corruptions) | ❌ Local dev only | MIT (project) + dataset terms above |

Based on **MobileNetV3-Large** pre-trained weights (torchvision, BSD-style).

---

*For the full technical context, see `doc/Milestone-6/Milestone6-Report.md` Section E and `doc/Milestone-2/Milestone-2-Report.md`.*

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

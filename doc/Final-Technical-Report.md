# Final Technical Report — Deep Learning-Based Human Face Authenticity Detection

**Group 11 · DS & AI Lab Project**
**Deliverable 2 — Full M1 → M6 Technical Write-Up**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Literature Review & Existing Solutions](#3-literature-review--existing-solutions)
4. [System Design & Architecture](#4-system-design--architecture)
5. [Dataset Engineering](#5-dataset-engineering)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Model Architecture — Selection & Justification](#7-model-architecture--selection--justification)
8. [Training Strategy & Hyperparameter Optimisation](#8-training-strategy--hyperparameter-optimisation)
9. [Results & Quantitative Performance](#9-results--quantitative-performance)
10. [Challenges Faced & How They Were Resolved](#10-challenges-faced--how-they-were-resolved)
11. [Explainability Framework](#11-explainability-framework)
12. [Web Application & Deployment](#12-web-application--deployment)
13. [Robustness & Stress Testing](#13-robustness--stress-testing)
14. [Deployment Readiness Assessment](#14-deployment-readiness-assessment)
15. [Tools & Technologies Used](#15-tools--technologies-used)
16. [Lessons Learned & Reflections](#16-lessons-learned--reflections)
17. [Future Work](#17-future-work)
18. [Milestone-by-Milestone Summary](#18-milestone-by-milestone-summary)
19. [References](#19-references)
20. [Team Declaration](#20-team-declaration)

---

## 1. Executive Summary

This report documents the complete development lifecycle of a deep learning-based human face authenticity detection system — from the initial proposal in Milestone 1 through the final deployment and evaluation in Milestone 6. The project delivers:

- A **MobileNetV3-Large** classifier (4.2M parameters) trained via a **three-stage transfer-learning strategy** (frozen backbone → partial unfreeze → full unfreeze with CelebA-HD enrichment) to distinguish genuine human face photographs from AI-generated facial images.
- **99.63% accuracy** on a 2,401-image held-out test set (Precision 0.9993, Recall 0.9947 for the Real class; Precision 0.9912, Recall 0.9989 for the Fake class).
- **80% out-of-distribution accuracy** on unseen ChatGPT- and Gemini-generated images — generators the model never encountered during training.
- A **Grad-CAM explainability** module that produces visual heatmaps highlighting the facial regions driving each prediction.
- A fully functional **FastAPI web application** with interactive prediction, Grad-CAM visualisation, manipulation robustness testing, model comparison, forensic report generation (HTML and .docx), and cross-domain evaluation capabilities.
- An **honest evaluation** documenting both the model's strengths and its critical limitations — specifically the shortcut-learning failure mode on modern smartphone photographs, the preprocessing mismatch between training and deployment, and the domain-shift problem — rather than presenting only headline accuracy figures.

The original Milestone 1 proposal envisioned an Explainable Dual-Stream Vision Transformer (ViT) with RGB–Frequency (FFT/DCT) cross-attention fusion. During Milestone 3's competitive architecture bake-off, **MobileNetV3-Large was selected instead**, based on demonstrated real-world generalisation and a 10× smaller footprint. The frequency-domain analysis originally proposed was partially explored through the secondary Dual-Stream model (ConvNeXt-V2 + ResNet-18 FFT stream) but was not adopted as the primary production architecture. This pivot from proposal to implementation is a key part of the project's story and is documented throughout this report.

---

## 2. Problem Statement & Motivation

### 2.1 The Threat Landscape

The rapid advancement of Artificial Intelligence — particularly Generative Adversarial Networks (GANs) and diffusion models — has enabled the creation of highly realistic AI-generated and manipulated human faces in both images and videos. While these technologies have numerous legitimate applications in entertainment, gaming, and art, they are increasingly being exploited for:

- **Deepfakes and misinformation** — fabricating video/audio of public figures
- **Identity theft and impersonation** — creating fake identity documents or social media profiles
- **Digital fraud** — bypassing facial recognition-based authentication systems
- **Manipulated evidence** — fabricating visual evidence for legal, political, or personal harm

Modern diffusion-based models (Stable Diffusion, DALL·E, Midjourney) and commercial AI image generators (ChatGPT, Gemini) produce synthetic facial images with such high fidelity that they are virtually indistinguishable from genuine photographs to the human eye. Early GAN-generated faces exhibited obvious artefacts — blurry ears, mismatched eye reflections, distorted teeth — but state-of-the-art generators produce realistic skin texture, consistent facial features, natural lighting, accurate facial geometry, and realistic reflections and shadows.

### 2.2 Limitations of Existing Approaches

Existing deepfake detection methods suffer from several critical shortcomings:

| Limitation | Description |
|---|---|
| **Spatial feature dependency** | Most detectors rely primarily on RGB spatial features, making them ineffective against generators that produce spatially flawless images |
| **Dataset overfitting** | Models trained on a single dataset learn dataset-specific artefacts rather than generalisable forensic cues, leading to poor cross-dataset performance |
| **Compression sensitivity** | Detection accuracy degrades significantly on compressed images (JPEG, social media re-encoding) because subtle forensic artefacts are destroyed |
| **Black-box predictions** | Many systems provide prediction labels without explaining which facial regions or features influenced the decision, limiting user trust |
| **Generational gap** | Detectors trained on GAN-generated faces may fail on diffusion-generated faces (and vice versa), as the two generator families leave different artefact signatures |

### 2.3 Project Objectives

1. Develop an accurate binary classifier that distinguishes authentic human faces from AI-generated facial images.
2. Achieve strong in-domain performance (>99% accuracy on held-out test sets).
3. Demonstrate out-of-distribution generalisation to images from unseen AI generators (ChatGPT, Gemini).
4. Integrate Grad-CAM explainability to produce transparent, interpretable predictions that highlight the facial regions driving each decision.
5. Build a deployable web application for interactive testing, including forensic report generation.
6. Honestly evaluate and document the model's limitations — including shortcut learning, domain shift, and preprocessing inconsistencies — rather than presenting only headline accuracy figures.
7. Provide actionable recommendations for addressing identified limitations in future work.

---

## 3. Literature Review & Existing Solutions

### 3.1 Detection Paradigms

Five major paradigms for deepfake detection were surveyed during Milestone 1:

| Paradigm | How It Works | Strengths | Limitations |
|---|---|---|---|
| **Spatial CNN-Based** | Binary classification from RGB pixel features (skin texture, lighting, edges) | Simple, fast inference, high benchmark accuracy | Learns dataset-specific artefacts; poor cross-dataset generalisation |
| **Vision Transformer (ViT)-Based** | Divides images into patches; learns long-range dependencies via self-attention | Better global feature representation; improved cross-domain generalisation | High computational cost; requires larger datasets |
| **Frequency/Wavelet-Based** | Analyses FFT/DCT spectral information for generator-induced frequency artefacts | Effective against highly realistic deepfakes; captures artefacts invisible in RGB | Sensitive to JPEG compression; performance varies across generators |
| **Noise Residual-Based** | Analyses camera sensor fingerprints (SPN, demosaicing artefacts) | Effective for camera-captured vs. synthetic images | Degrades after heavy compression or editing |
| **Multi-Scale** | Processes images at multiple resolutions to capture fine-grained and large-scale manipulations | Improved robustness | Higher computational complexity |

### 3.2 Benchmark Baselines (DeepfakeBench)

To establish a fair comparison baseline, the project adopted **DeepfakeBench** — a unified evaluation framework for deepfake detection. Key baseline results:

| Detector | Backbone | Within-Domain AUC | Cross-Domain AUC |
|---|---|---|---|
| Xception | Xception | 0.9450 | 0.7718 |
| EfficientNet-B4 | EfficientNet-B4 | 0.9389 | 0.7718 |
| UCF | Xception | 0.9527 | 0.7801 |
| F3Net (frequency-based) | Xception | 0.9449 | 0.7645 |
| SPSL (frequency-based) | Xception | 0.9408 | 0.7875 |

**Critical observation**: all baselines achieve >0.94 AUC within-domain but drop to 0.76–0.79 cross-domain — confirming that **cross-dataset generalisation is the primary unsolved challenge**, not in-distribution accuracy.

### 3.3 Benchmark Datasets

| Dataset | Characteristics | Significance |
|---|---|---|
| FaceForensics++ (FF++) | 1,000 real videos, 4,000 manipulated (4 methods) | Standard benchmark for deepfake detection |
| Celeb-DF v2 | 590 real videos, 5,639 manipulated | Higher-quality face swaps |
| DFDC | 119,197 video clips | Largest public benchmark; varied compression and lighting |
| FFHQ (Flickr-Faces-HQ) | 70,000 high-resolution real face photographs | Widely used real-face source; high diversity in age, ethnicity, lighting |

### 3.4 Original Proposed Approach vs. Final Implementation

| Aspect | M1 Proposal | Final Implementation |
|---|---|---|
| **Architecture** | Explainable Dual-Stream ViT with RGB–FFT/DCT cross-attention fusion | MobileNetV3-Large (single spatial stream) |
| **Feature domains** | RGB + frequency-domain | RGB spatial only (primary); frequency explored as secondary model |
| **Explainability** | Attention Rollout + frequency saliency | Grad-CAM on MobileNetV3 conv layers |
| **Parameters** | Not specified (ViT-base: ~86M) | 4.2M (20× smaller than ViT-base) |
| **Reason for change** | — | MobileNetV3-Large was the only model to demonstrate out-of-distribution generalisation to consumer AI generators; 10× smaller and faster than alternatives |

---

## 4. System Design & Architecture

### 4.1 High-Level System Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    User Interface (Web App)                   │
│  HTML/CSS/JS Frontend  ←→  FastAPI Backend (Python)          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Image Preprocessing Pipeline                     │
│  Upload → Validate → RetinaFace Detection → Crop/Align      │
│  → Resize (224×224) → ImageNet Normalisation                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                 ┌─────────────┼──────────────┐
                 │             │              │
                 ▼             ▼              ▼
          ┌──────────┐  ┌───────────┐  ┌──────────────┐
          │ Primary   │  │ Cross-    │  │ Manipulation │
          │ Model     │  │ Domain    │  │ Robustness   │
          │ (best/    │  │ Model     │  │ Model        │
          │  noaug)   │  │           │  │              │
          └─────┬─────┘  └─────┬─────┘  └──────┬───────┘
                │              │               │
                ▼              ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                   Output Layer                                │
│  Prediction (Real/Fake + confidence) + Grad-CAM heatmap      │
│  + Forensic report (HTML/DOCX) + Robustness breakdown        │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Model Checkpoint Registry

The system supports multiple checkpoints managed by a `ModelRegistry`.
**Production (Render)** loads the 3 checkpoints the frontend actually
uses; **local development (`main`)** can load all available files from
`webapp/output/`.

| Checkpoint | Role | Held-out Test Accuracy | Training Source | Production (Render) |
|---|---|---|---|---|
| `mobilenetv3_noaug.pth` | Main Model default (Model 1) — no-augmentation baseline | **95.98%** | `../notebooks/final-mobilenet (1).ipynb` | ✅ Deployed |
| `mobilenetv3_best.pth` | 3-stage CelebA-HD corrected model (Model 2 toggle) | **99.63%** | `../notebooks/final-mobilenet (1).ipynb` | ✅ Deployed |
| `mobilenetv3_cross_domain.pth` | Cross-domain (non-face, multi-domain) evaluation | — | `../notebooks/cross-domain.ipynb` | ✅ Deployed |
| `mobilenetv3_manipulations.pth` | Manipulation robustness testing | — | Specialised training run | ❌ Not deployed (unused — `/robustness` isn't called from the UI) |
| `mobilenetv3_tuned.pth` | Hyperparameter sweep comparison | — | Hyperparameter experiment | ❌ Not deployed (unused — `/compare?mode=hparams` isn't called from the UI) |

**Model 1 vs. Model 2 trade-off**: Model 1 (`noaug`, deployed default) favours recall — it catches 100% of fake images in the held-out set at the cost of more false alarms on real photos. Model 2 (`best`) is the more heavily-tuned research checkpoint, higher overall accuracy but trained on a smaller, more curated slice of data. Both are available in production via the Model 1/Model 2 toggle on the Main Model page.

Production URL: **https://face-forensics.onrender.com**  
Deployed from Git branch **`main`** via Docker on Render free tier
(512 MB RAM; 2-model image via `webapp/.dockerignore`). See `../doc/READMEdeployment.md`.

### 4.3 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Primary prediction with Grad-CAM overlay |
| `/robustness` | POST | 11-manipulation robustness stress test |
| `/compare` | POST | Side-by-side model comparison (augmentation, hyperparameters) |
| `/report` | POST | HTML forensic report generation |
| `/report/docx` | POST | Downloadable Word document report |
| `/health` | GET | System status and loaded model check |
| `/api/training-results` | GET | Pre-computed training artefacts (CSVs, images) |

---

## 5. Dataset Engineering

### 5.1 Primary Datasets

The dataset evolved across milestones as new challenges were discovered:

#### 5.1.1 FFHQ (Flickr-Faces-HQ) — Real Images

- **Source**: NVIDIA's FFHQ dataset via Kaggle (`philosopher0808`)
- **Images used**: 15,000 (subsampled from 70,000 available)
- **Class**: Real (Class 0)
- **Properties**: High-quality portraits with significant diversity in age, ethnicity, lighting, facial expression, pose, accessories, and background. Recognised as a standard benchmark for facial analysis tasks.
- **Licence**: Creative Commons BY-NC-SA 4.0

#### 5.1.2 Stable Diffusion — AI-Generated Images

- **Source**: Stable Diffusion Face Dataset via Kaggle
- **Images used**: 9,001
- **Class**: Fake (Class 1)
- **Properties**: Highly realistic diffusion-model-generated faces; orders of magnitude more realistic than early GAN outputs.

#### 5.1.3 Nano Banana 2.0 — Cross-Domain Augmentation (M4)

- **Source**: AI Image with Nano Banana 2.0 vs Real Image Dataset via Kaggle
- **Contributed**: 31,999 training + 4,000 validation + 4,000 testing images
- **Purpose**: Improve cross-domain generalisation by exposing the model to a different generator family during training.

#### 5.1.4 CelebA-HD — Modern Real Photo Correction (M5)

- **Source**: CelebA dataset (`jessicali9530/celeba-dataset`) via Kaggle
- **Images used**: 8,000 (capped to preserve class balance)
- **Class**: Real
- **Purpose**: Specifically added in Stage 3 fine-tuning to correct the shortcut-learning failure mode — the model was misclassifying modern, high-resolution smartphone photographs as AI-generated because FFHQ did not adequately represent HDR processing, computational photography enhancement, and contemporary camera post-processing styles.

### 5.2 Dataset Construction & Splitting

The primary dataset was constructed by combining FFHQ and Stable Diffusion images, randomly shuffled with a fixed seed (42) for reproducibility:

| Split | Percentage | Real | Fake | Total |
|---|---|---|---|---|
| Training | 80% | 12,000 | 7,200 | 19,200 |
| Validation | 10% | 1,500 | 900 | 2,400 |
| Test | 10% | 1,500 | 901 | 2,401 |

**Stratified splitting**: `sklearn.model_selection.train_test_split` with `stratify=labels` preserved the Real:Fake ratio (~62.5:37.5) identically across all three sets.

**Zero data leakage**: The splitting was performed once, before any model training. The `test_loader` was only touched once — for the final evaluation — and was never used for any training or validation decisions.

### 5.3 Expanded Dataset (with Nano Banana 2.0)

When cross-domain data was incorporated in M4:

| Split | Number of Images |
|---|---|
| Training | 51,199 |
| Validation | 6,400 |
| Testing | 6,401 |
| **Total** | **64,000** |

### 5.4 Exploratory Data Analysis (EDA)

Key findings from the M2 EDA:

1. **Class balance**: The dataset is balanced (~50/50 Real/Fake), eliminating the need for SMOTE or class-weighted loss functions.
2. **Aspect ratios**: The overwhelming majority of images have an aspect ratio of exactly 1.0 (perfect squares), meaning resize to 224×224 introduces no geometric distortion.
3. **Pixel intensity**: Images utilise the full 8-bit dynamic range (0–255), confirming that 0.0–1.0 normalisation preserves meaningful information.
4. **Visual inspection**: AI-generated images exhibit extraordinary realism — skin texture, lighting physics, and facial geometry are mathematically precise. Simple spatial edge detectors or traditional ML features (HOG, Haar cascades) are insufficient; deep feature extraction is required.

---

## 6. Preprocessing Pipeline

### 6.1 Pipeline Overview

```
Raw Image
    │
    ▼
Face Detection (RetinaFace)
    │
    ▼
Face Alignment (5-point landmark-based)
    │
    ▼
Face Cropping (20% padding around bounding box)
    │
    ▼
Resize (224 × 224 pixels, LANCZOS interpolation)
    │
    ▼
ImageNet Normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    │
    ▼
Data Augmentation (Training Only)
    │
    ▼
Processed Image → Model Input
```

### 6.2 Face Detection & Alignment

**RetinaFace** was selected as the face detector because it provides:

- Robust single-stage detection under challenging conditions (pose variation, occlusion, expression)
- Five facial landmarks (two eyes, nose tip, two mouth corners) for alignment
- Bounding box localisation with the largest-face selection heuristic

**Fallback strategy**: If RetinaFace fails to detect a face (or is not installed), a center-crop strategy is employed as a fallback — the image is center-cropped to a square and resized to 224×224. This ensures the pipeline never discards potentially useful samples.

**Channel order**: A critical implementation detail discovered during M5 — the RetinaFace integration must receive RGB input (matching PIL's native channel order). An earlier version incorrectly converted to BGR via OpenCV, causing every detection to silently fail (returning an empty dict rather than raising an error), so the system fell through to center-crop for every request with no error logged.

### 6.3 Normalisation

All images are normalised using ImageNet statistics:

| Parameter | Value |
|---|---|
| Mean | (0.485, 0.456, 0.406) |
| Standard Deviation | (0.229, 0.224, 0.225) |

This ensures compatibility with the MobileNetV3-Large backbone's pre-trained weight distribution, improving convergence speed and numerical stability.

### 6.4 Data Augmentation (Training Only)

The following on-the-fly augmentations were applied during training:

| Augmentation | Parameters | Purpose |
|---|---|---|
| **Random Resized Crop** | Scale 0.9–1.0, size 224 | Simulates variations in framing, improves scale invariance |
| **Random Horizontal Flip** | p=0.5 | Doubles effective dataset; disrupts memorised noise patterns |
| **Colour Jitter** | Brightness, contrast, saturation, hue variation | Simulates different lighting/camera conditions; prevents colour-based shortcuts |
| **JPEG Compression** | Quality 30–100 | Simulates social media re-encoding and lossy compression |
| **Gaussian Blur** | Variable radius | Simulates camera defocus and motion blur |
| **Gaussian Noise** | σ=15 | Simulates sensor noise and low-light conditions |
| **Channel Shift** | ±30% per-channel, p=0.5 | Teaches robustness to colour tints |

**Critical finding (M5)**: The existing augmentation pipeline, while comprehensive, did not adequately address the actual failure mode. `ChannelShift`/`ColorJitter` vary hue/brightness/saturation but do not simulate **HDR tone-mapping or sharpening artefacts** — the specific properties identified as driving false positives on modern smartphone photographs. This gap between the augmentation strategy and the actual failure mode is a key technical lesson from the project.

**Validation and test sets**: Strictly no augmentation — only rescale + normalisation. Augmenting evaluation data would introduce artificial noise into metrics, invalidating accuracy and AUC scores.

---

## 7. Model Architecture — Selection & Justification

### 7.1 Architecture Bake-Off (Milestone 3)

Three architectures were independently developed and evaluated in parallel:

| Aspect | Model 1: MobileNetV3-Large | Model 2: Dual-Stream Fusion | Model 3: EfficientNet-B2 |
|---|---|---|---|
| **Priority rank** | **1st — SELECTED** | 2nd — Secondary | 3rd — Tertiary |
| **Backbone(s)** | MobileNetV3-Large | ConvNeXt-V2 + ResNet-18 | EfficientNet-B2 |
| **Input domain** | Spatial RGB | Spatial RGB + FFT spectral | Spatial RGB (video frames) |
| **Total parameters** | **~4.2M** | ~40.7M | ~7.8M |
| **Primary data** | FFHQ + Stable Diffusion | FFHQ + 12 generator families | FaceForensics++ / Celeb-DF |
| **In-domain accuracy** | **99.96%** | 97.42% | 58.33% (bug-affected) |
| **OOD generalisation** | **80% on ChatGPT/Gemini** | 88.7% avg. on academic GAN/diffusion benchmarks | Not evaluated |
| **Design goal** | Speed / edge deployment | Cross-generator generalisation | Interpretability (Grad-CAM) |
| **Framework** | PyTorch | PyTorch | TensorFlow / Keras |

### 7.2 Why MobileNetV3-Large Was Selected

The deciding factor was **not** in-domain accuracy alone (all three scored well there). MobileNetV3-Large was selected because:

1. **Demonstrated real-world generalisation**: 80% accuracy on unseen ChatGPT/Gemini-generated images — the only model tested against modern consumer AI generators. This is direct evidence that it learned transferable local-artefact signatures rather than overfitting to Stable Diffusion's specific generative fingerprint.

2. **Order-of-magnitude parameter efficiency**: ~4.2M parameters — roughly 10× smaller than the Dual-Stream network (40.7M) and about half the size of EfficientNet-B2 (7.8M).

3. **Real-time inference**: ~8.2 ms per image on a Tesla T4 GPU (~112 images/second); ~15.5 ms on CPU.

4. **Pipeline simplicity**: A single RGB spatial stream needs only resize + normalise — no FFT preprocessing, no video/face-detection dependency at training time.

5. **Architecturally well-matched**: Inverted-residual blocks and squeeze-and-excitation attention capture localised texture patterns (skin smoothing, blending boundaries, edge sharpness) — exactly the artefact types that are largely generator-agnostic.

### 7.3 MobileNetV3-Large Architecture Detail

```
Input Image (224 × 224 × 3)
         │
         ▼
MobileNetV3-Large Backbone (ImageNet pretrained)
  ├── Depthwise Separable Convolutions
  ├── Squeeze-and-Excitation (SE) Attention Modules
  ├── Hardswish / ReLU Activations
  └── 17 Feature Blocks (~2.97M params)
         │
         ▼
Global Average Pooling → [B, 960]
         │
         ▼
Modified Classification Head:
  Linear(960 → 1280) → Hardswish → Dropout(0.2) → Linear(1280 → 2)
         │
         ▼
Softmax → P(Real), P(Fake)
```

| Component | Total Params | Trainable (Stage 1) | Trainable (Stage 2) | Trainable (Stage 3) |
|---|---:|---:|---:|---:|
| Backbone | ~2.97M | 0 (frozen) | ~1.21M (blocks 12–16) | ~2.97M (all) |
| Classification head | ~1.23M | ~1.23M | ~1.23M | ~1.23M |
| **Total** | **~4.20M** | **~1.23M** | **~2.44M** | **~4.20M** |

### 7.4 Secondary Models (Retained for Specific Use Cases)

**Dual-Stream Spatial-Frequency Fusion (Model 2)**: ConvNeXt-V2 (RGB spatial stream) + ResNet-18 (FFT spectral stream) fused via cross-attention. The frequency stream applies a 15%-radius high-pass filter mask to the 2D-FFT magnitude spectrum, suppressing low-frequency face semantics and forcing focus on fine-grained generator grid artefacts. Invoked as a spectral cross-check when primary model confidence is low.

**EfficientNet-B2 (Model 3)**: Compound-scaled spatial classifier with Grad-CAM explainability, designed for video-frame ingestion from FaceForensics++/Celeb-DF. Its tuned run was affected by a documented Keras optimizer-reload bug (the `load_model()` call silently restored the Stage-1 learning rate instead of the intended Stage-2 rate, causing validation accuracy to collapse). Retained for video-frame pipeline and pixel-localised Grad-CAM output.

---

## 8. Training Strategy & Hyperparameter Optimisation

### 8.1 Three-Stage Transfer Learning

The training strategy evolved across milestones. Originally two-stage (M3–M4), it was extended to three stages in M5 to address the shortcut-learning failure mode:

#### Stage 1: Feature Extraction (Frozen Backbone)

- **Objective**: Adapt only the new classification head to the Real/Fake binary task, preserving ImageNet features.
- **Frozen layers**: Entire MobileNetV3-Large backbone
- **Trainable layers**: Classification head only (~1.23M parameters)
- **Epochs**: 3
- **Learning rate**: 3 × 10⁻⁴
- **Result**: Best validation accuracy 98.75%

#### Stage 2: Partial Fine-Tuning

- **Objective**: Adapt higher-level backbone features to deepfake-specific patterns without disturbing low-level filters.
- **Unfrozen layers**: Last 25% of backbone (blocks 12–16 of 17)
- **Epochs**: 7
- **Learning rate**: 1 × 10⁻⁵ (30× lower to avoid catastrophic forgetting)
- **Result**: Best validation accuracy 99.71%

#### Stage 3: HD Fine-Tuning (Added in M5)

- **Objective**: Correct the shortcut-learning failure mode — the model was misclassifying modern, high-resolution real photographs as AI-generated.
- **Unfrozen layers**: All backbone layers (full model)
- **Additional data**: 8,000 CelebA-HD real photographs added to training set
- **Epochs**: 3
- **Learning rate**: 5 × 10⁻⁶
- **Result**: Validation accuracy held at 99.71% (no regression); the intended effect was corrected behaviour on modern real photographs specifically.

### 8.2 Optimiser & Scheduler Configuration

| Parameter | Configuration |
|---|---|
| **Optimiser** | AdamW (decoupled weight decay) |
| **Weight Decay** | 1 × 10⁻⁴ |
| **LR Scheduler** | CosineAnnealingLR |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 128 |
| **Mixed Precision** | Automatic Mixed Precision (AMP) enabled |
| **Gradient Clipping** | Enabled for training stability |

### 8.3 Hyperparameter Ablation Study

A systematic 24-experiment ablation study was conducted in M4, varying one hyperparameter at a time:

#### Learning Rate

| Learning Rate | Test Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| 1×10⁻⁴ | 92.71% | 0.9281 | 0.9801 |
| **5×10⁻⁴** | **95.43%** | **0.9552** | **0.9921** |
| 1×10⁻³ | 95.41% | 0.9550 | 0.9925 |
| 5×10⁻³ | 90.10% | 0.8941 | 0.9791 |

#### Optimiser

| Optimiser | Test Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Adam | 86.51% | 0.8531 | 0.9557 |
| **AdamW** | **95.41%** | **0.9550** | **0.9925** |

AdamW significantly outperformed standard Adam — the decoupled weight decay mechanism provided better regularisation and improved generalisation.

#### Dropout

| Dropout Rate | Test Accuracy | F1-Score |
|---|---|---|
| **0.0** | **95.73%** | **0.9575** |
| 0.2 | 95.41% | 0.9550 |
| 0.3 | 95.16% | 0.9518 |
| 0.5 | 94.56% | 0.9447 |

No additional dropout produced the best results — MobileNetV3-Large's built-in architectural regularisation (SE blocks, H-swish activations) was sufficient.

#### Weight Decay

| Weight Decay | Test Accuracy | F1-Score |
|---|---|---|
| 0.00 | 95.28% | 0.9530 |
| 0.01 | 95.41% | 0.9550 |
| 0.05 | 95.54% | 0.9558 |
| **0.10** | **95.57%** | **0.9564** |

#### Label Smoothing

| Label Smoothing | Test Accuracy | F1-Score |
|---|---|---|
| **0.00** | **95.41%** | **0.9550** |
| 0.05 | 93.10% | 0.9293 |
| 0.10 | 94.30% | 0.9426 |

Label smoothing did not improve performance — the balanced dataset already provided sufficient supervision.

#### LR Scheduler

| Scheduler | Test Accuracy | F1-Score |
|---|---|---|
| **CosineAnnealingLR** | **95.41%** | **0.9550** |
| ReduceLROnPlateau | 93.23% | 0.9302 |

### 8.4 Final Optimal Configuration

| Hyperparameter | Selected Value |
|---|---|
| Optimiser | AdamW |
| Learning Rate (Stage 1/2/3) | 3×10⁻⁴ / 1×10⁻⁵ / 5×10⁻⁶ |
| Batch Size | 128 |
| Weight Decay | 0.05–0.10 |
| Dropout Rate | 0.0–0.2 |
| LR Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.0 |
| Mixed Precision | Enabled (AMP) |

### 8.5 Training Environment

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Mixed Precision | AMP enabled via `torch.cuda.amp` |
| Input Resolution | 224 × 224 pixels |
| Batch Size | 128 |

---

## 9. Results & Quantitative Performance

### 9.1 Final Held-Out Test Set Results

Classification report on the **2,401-image held-out test set**, evaluated with the final Stage 3 weights:

```
              precision    recall  f1-score   support

        Real     0.9993    0.9947    0.9970      1500
        Fake     0.9912    0.9989    0.9950       901

    accuracy                         0.9963      2401
   macro avg     0.9953    0.9968    0.9960      2401
weighted avg     0.9963    0.9963    0.9963      2401
```

| Metric | Value |
|---|---|
| **Test Accuracy** | 99.63% |
| **Test Loss** | 0.0109 |
| **Macro Precision** | 99.53% |
| **Macro Recall** | 99.68% |
| **Macro F1** | 99.60% |

**Confusion matrix**: Only 9 of 2,401 test images were misclassified — 8 Real images predicted Fake, 1 Fake image predicted Real.

### 9.2 Training Dynamics

| Stage | Epoch-by-Epoch Validation Accuracy |
|---|---|
| Stage 1 (frozen) | 96.96% → 97.92% → 98.75% |
| Stage 2 (partial) | 99.29% → 99.67% → 99.62% → 99.71% → 99.71% → 99.71% → 99.67% |
| Stage 3 (full + CelebA) | 99.62% → 99.71% → 99.71% |

### 9.3 Out-of-Distribution Generalisation (M3)

| Evaluation Set | Relationship to Training Data | Accuracy |
|---|---|---|
| FFHQ vs. Stable Diffusion test set | In-distribution | 99.96% |
| ChatGPT-generated images | **Out-of-distribution** (unseen generator) | 80% |
| Gemini-generated images | **Out-of-distribution** (unseen generator) | 80% |

### 9.4 Cross-Domain Evaluation (M5 — Critical Finding)

| Dataset | Images | Predicted Real | Predicted Fake | Accuracy |
|---|---|---|---|---|
| **Real-Old (FFHQ-like)** | 73 | 73 | 0 | **100.0%** |
| **Real-Latest (smartphone photos)** | 70 | 6 | 64 | **8.6%** |

The model classified every FFHQ-like image correctly but misclassified the vast majority (91.4%) of recent real smartphone photographs as AI-generated. This is the central finding of Milestone 5 and the defining challenge of this project.

### 9.5 Inference Performance

| Device | Mean Latency | Throughput |
|---|---|---|
| CPU (Intel i7-7700) | 15.53 ms | 64.4 images/sec |
| GPU (Tesla T4, estimated) | ~7–8 ms | ~125–140 images/sec |
| `/predict` request, CPU (M6: forward pass only) | ~2,200 ms locally · 33.3 s on Render | — |
| `/gradcam` request, CPU (on-demand, M6) | ~2,200 ms locally · 22.2 s on Render | — |

**Critical finding**: The model itself is not the deployment bottleneck. The ~140× gap between the raw forward pass (15.5 ms) and a single `/predict` or `/gradcam` request (~2.2 s locally) comes from face detection/preprocessing plus (for `/gradcam`) a full backward pass and heatmap rendering, not the classifier.

**M6 update**: `/predict` originally ran Grad-CAM on every call regardless of whether the user ever viewed it, roughly doubling every analysis. It was split into a forward-pass-only `/predict` and a separate, on-demand `/gradcam` (triggered only by the **Run Grad-CAM** button or report generation) — see `doc/Milestone-6/Milestone6-Report.md` §B.7a for the full write-up and the measured live-Render numbers above.

### 9.6 Comparison Against M4 Baseline

| Metric | M4 (2-stage) | M5/M6 (3-stage + CelebA-HD) |
|---|---|---|
| Validation Accuracy | 99.30% | 99.71% |
| Test Accuracy | 99.06% | 99.63% |
| Real Precision | 0.9900 | 0.9993 |
| Fake Recall | 0.9879 | 0.9989 |
| Real-Latest Accuracy | Not tested | 8.6% (identified problem) |

---

## 10. Challenges Faced & How They Were Resolved

### 10.1 Challenge 1: Shortcut Learning on Real Images

**The Problem**: During M4's faculty review, the model was observed to misclassify genuine human photographs as AI-generated. Investigation in M5 revealed the root cause:

- The **Real** class was primarily represented by FFHQ (a dataset of older, lower-resolution, naturally-lit photographs)
- The **Fake** class consisted of Stable Diffusion-generated images (which happen to be high-resolution, sharp, with vibrant colours)
- Instead of learning genuine forgery artefacts, the model learned to associate **FFHQ-style colour distribution** with "Real" and **modern HDR processing, strong sharpening, vibrant colour** with "Fake"

**Evidence**: Median P(Real) on the Real-Latest dataset was **0.00060** — the model was nearly 100% confident that genuine smartphone photos were fake.

| Learned Shortcut Feature | Effect on Prediction |
|---|---|
| FFHQ-style colour distribution | Classified as Real |
| Modern HDR processing | Classified as Fake |
| Strong sharpening / computational photography | Increased false positives |
| High colour saturation | Increased false positives |

**How It Was Addressed**:

1. **Root cause identified** (M5 Section 5.4): The failure was traced to specific, nameable image properties (HDR, sharpening, saturation) — not left as an unexplained accuracy gap.
2. **Stage 3 fine-tuning added**: 8,000 CelebA-HD real photographs were incorporated into a third training stage, specifically to teach the already-converged model that high-resolution, heavily post-processed real photographs are still genuinely real.
3. **Result**: Stage 3 maintained 99.71% validation accuracy (no regression) while partially correcting the false-positive pattern on modern photographs.

**Honest assessment**: Stage 3's CelebA-HD addition was a partial fix, not a complete solution. The Real-Latest accuracy (8.6%) remained low even after this correction, indicating that a more fundamental approach — expanding the Real class with diverse modern photography sources, adding HDR/sharpening-specific augmentations, or incorporating frequency-domain features as a shortcut-learning countermeasure — is required.

### 10.2 Challenge 2: Preprocessing Mismatch Between Training and Deployment

**The Problem**: M4's faculty review identified a risk that the face-crop/alignment and channel-order handling at inference time were not verified to be identical to what the model was trained on. If the deployed web app preprocesses images differently from the training notebook, the model receives out-of-distribution inputs at every inference — silently degrading accuracy.

**How It Was Addressed**:

1. **Direct code comparison** (M5): The deployed `preprocessing.py`'s `val_transform` (resize → tensor → ImageNet normalise) was verified to be an exact port of the training notebook's own `val_transform`.
2. **Channel-order bug found and fixed**: The RetinaFace integration was receiving BGR input (via an OpenCV conversion) instead of RGB (PIL's native order). This caused every face detection to silently fail, falling through to center-crop with no error logged. Fixed by removing the OpenCV conversion and passing RGB directly.
3. **Padding consistency**: The face crop padding was standardised at 20% across both training and inference.

### 10.3 Challenge 3: Domain Shift

**The Problem**: Models trained on FFHQ + Stable Diffusion data generalise well within that distribution but fail when the input distribution shifts — even when the task is identical (classifying real human faces).

**Evidence (two independent confirmations)**:
- Raunak's Real-Latest probe: 8.6% accuracy (70 images)
- Local Test Sample check: 62.0% accuracy (50 images)

**How It Was Addressed**:

1. **Multi-source dataset expansion** (M4): Nano Banana 2.0 dataset added to expose the model to a different generator family.
2. **CelebA-HD addition** (M5): Modern real photographs added specifically for Stage 3 fine-tuning.
3. **Cross-domain model** (`../notebooks/cross-domain.ipynb`): A separate checkpoint trained on non-face, multi-domain data (Nano Banana, CIFAKE, CrossDomain, Places365, Artefact) for broader generalisation.
4. **Decision-threshold recalibration identified** (M5): The deployed app uses a plain argmax (50% threshold), but raising the threshold to require higher confidence before predicting "Fake" is a no-retraining lever to improve real-world precision.

### 10.4 Challenge 4: EfficientNet-B2 Optimizer-Reload Bug

**The Problem**: During Model 3 (EfficientNet-B2) fine-tuning, Keras's `load_model()` silently restored the Stage-1 learning rate (1×10⁻³) instead of the intended Stage-2 fine-tuning rate (1×10⁻⁵). This caused:
- Fine-tuning at an excessively high learning rate
- Rapid overfitting and disrupted pretrained weights
- Validation accuracy collapse
- Early Stopping fired at epoch 8
- Final test accuracy of only 58.33%

**How It Was Addressed**: The bug was documented in M3 (Section 2.4.4) as a Keras-specific tradeoff. In future TensorFlow/Keras workflows, the optimizer state must be explicitly recompiled after checkpoint loading rather than relying on saved optimizer state.

### 10.5 Challenge 5: Computational Resource Constraints

**The Problem**: Most experiments were conducted on Kaggle Notebooks with limited GPU runtime and memory. Long training sessions occasionally required restarting notebooks and reloading checkpoints.

**How It Was Addressed**:
- **Mixed-precision training (AMP)** reduced GPU memory usage by ~50%, enabling batch size 128 on a T4 GPU.
- **MobileNetV3-Large's efficiency**: 4.2M parameters meant training completed faster than alternatives (40.7M Dual-Stream, 7.8M EfficientNet-B2).
- **Staged checkpoint saving**: Best model saved after each stage, so training could resume from the last best checkpoint after interruptions.

### 10.6 Challenge 6: Non-Face Image Handling

**The Problem**: The model has no "I don't know, this isn't a face" option — softmax always forces a confident Real/Fake decision. Testing on 10 non-face images (objects, animals, textures, landscapes) showed:
- 4/10 predicted Real, 6/10 predicted Fake
- Average confidence 86.67%
- Results were close to a coin flip with no discernible pattern

**How It Was Addressed**: Documented as a hard operational boundary — any prediction on a non-face image is meaningless and should be treated as such. The RetinaFace face-detection step at the front of the pipeline serves as a partial gate, but its center-crop fallback means non-face images can still reach the classifier.

### 10.7 Challenge 7: Unaudited Demographic & Ethical Bias

**The Problem**: The current evaluation does not establish whether the detector performs equally across demographic groups. Dataset splits (FFHQ, Stable Diffusion, CelebA-HD, and the Real-Latest probe set) were never audited for skin tone, age, gender, or ethnicity balance — either at the source-dataset level or after train/val/test partitioning. Given that the project's headline failure mode (§10.1, §10.3) is a shortcut-learning problem where the model keys on *processing* artefacts (HDR tone-mapping, sharpening, saturation) rather than genuine authenticity cues, it is plausible — though currently unmeasured — that these same shortcuts could produce uneven error rates across demographic groups if camera/processing conventions correlate with any of them.

**Why This Was Not Addressed**: No dataset used in this project (FFHQ, Stable Diffusion outputs, CelebA-HD) ships with verified demographic labels; only FFHQ's source documentation notes general diversity in age, ethnicity, and lighting without per-image annotations. A rigorous audit requires either verified demographic labels or a separate, carefully-validated labelling pass, which was out of scope given the M5/M6 timeline and the higher-priority shortcut-learning root-cause work.

**Scope for a Future Audit** (not performed in this project):

- Stratify the held-out test set and the Real-Latest probe by skin-tone group, age group, gender group, and any other reliably available demographic category.
- For each group, report Accuracy, Precision, Recall, F1-score, false-positive rate, and false-negative rate separately — the same metrics already computed in aggregate in §9.1 and §9.4.
- Treat this as a measurement exercise, not a presumption of bias: the goal is to determine whether meaningful performance differences exist, not to assume they do.
- Source demographic labels from a dataset that provides them natively (rather than inferring them), to avoid introducing a second, unverified layer of bias into the audit itself.

**Honest assessment**: This is a genuine limitation of the current evaluation, not a solved problem. Every accuracy figure reported in this report (§9) — including the well-documented 8.6% Real-Latest failure — is an aggregate number that could conceal unequal performance across demographic groups. Until the audit above is performed, no claim of demographic fairness (or unfairness) can honestly be made either way.

---

## 11. Explainability Framework

### 11.1 Grad-CAM Implementation

**Gradient-weighted Class Activation Mapping (Grad-CAM)** was implemented to provide visual interpretability of the model's predictions. The implementation:

1. Registers a forward hook on the last convolutional layer of MobileNetV3-Large (`model.features[-1]`)
2. Captures activations during the forward pass
3. Computes gradients of the predicted class score with respect to the activations
4. Produces a weighted combination of activation maps (global average pooling of gradients as weights)
5. Applies ReLU to retain only positive contributions
6. Upsamples the heatmap to 224×224 using bilinear interpolation
7. Overlays the heatmap on the original image using the JET colourmap with configurable alpha blending (default 0.45)

### 11.2 Verification Results

**Correct predictions**: Grad-CAM heatmaps concentrate on the central face — eyes, nose bridge, and mouth area — rather than the background or image border. No shortcut cues (corner watermarks, uniform background patches) are visibly dominant.

**Failure cases (false positives — real photos predicted as fake)**: Heatmaps concentrate on skin-texture/sharpness regions rather than forgery-typical blending seams, consistent with the model reacting to modern camera post-processing (HDR, sharpening, saturation) rather than genuine forgery evidence.

**Failure cases (false negatives — fake images predicted as real)**: Markedly rarer than false positives. The model's practical weak point is false alarms on real images, not missed fakes.

### 11.3 Forensic Report Generation

The system generates comprehensive forensic reports in two formats:

1. **HTML report** (via `/report` endpoint): Standalone, printable document containing input image, prediction result, confidence scores, Grad-CAM overlay, and model information.
2. **Word document** (via `/report/docx` endpoint): Downloadable `.docx` file with embedded images and structured analysis, built using the `python-docx` library.

---

## 12. Web Application & Deployment

### 12.1 Technology Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI (Python) |
| **Frontend** | HTML5, CSS3, vanilla JavaScript |
| **ML Framework** | PyTorch |
| **Face Detection** | RetinaFace (with center-crop fallback) |
| **Explainability** | Grad-CAM (custom implementation) |
| **Report Generation** | python-docx (Word), Jinja2-style HTML templates |
| **Server** | Uvicorn (ASGI) |
| **Metadata Analysis** | Custom forensic meta-detector (EXIF, watermark, spectral forensics) |

### 12.2 Application Features

The web application provides a 5-page interface:

1. **Main Model**: Primary prediction with Model 1/Model 2 toggle, Grad-CAM overlay, and confidence scores.
2. **Cross-Domain Model**: Prediction using the cross-domain checkpoint for non-face or multi-domain images.
3. **Grad-CAM Explainability**: Dedicated visualisation page with detailed heatmap analysis.
4. **Forensic Report**: Full forensic analysis with HTML rendering and `.docx` export.
5. **History**: Session-based prediction history.

### 12.3 Backend Architecture

```
webapp/backend/
├── app/
│   ├── main.py            # FastAPI routes (7 endpoints)
│   ├── model.py           # MobileNetV3-Large architecture + checkpoint registry
│   ├── config.py          # Paths, checkpoint filenames, constants
│   ├── preprocessing.py   # Face crop/align + inference transform
│   ├── gradcam.py         # Grad-CAM implementation
│   ├── manipulations.py   # 11 robustness-test image manipulations
│   ├── meta_detector.py   # Forensic meta-detector (metadata/watermark/pixel forensics)
│   ├── report.py          # HTML + .docx forensic report builders
│   ├── results.py         # Pre-computed training-artefact loader
│   ├── schemas.py         # Pydantic request/response models
│   └── static/            # Frontend: index.html, app.js, style.css
├── requirements.txt
└── README.md
```

### 12.4 Key Implementation Details

- **Model Registry**: Loads all available checkpoints at startup; missing checkpoints are skipped gracefully (the API degrades — individual endpoints report unavailability rather than crashing the whole application).
- **Preprocessing pipeline consistency**: The deployed `val_transform` is an exact port of the training notebook's transform, verified by direct code comparison.
- **Image validation**: Uploads are validated for content type (JPEG, PNG, WebP, BMP), file size (max 10 MB), and decodability before processing.
- **Forensic meta-detector**: Analyses raw uploaded bytes for EXIF metadata, invisible watermarks, spectral anomalies, and sensor-noise forensics — information that can complement the CNN-based prediction.

### 12.5 Production Deployment (Render)

| Component | Configuration |
|---|---|
| **Platform** | Render — free-tier web service |
| **Live URL** | https://face-forensics.onrender.com |
| **Source branch** | **`main`** |
| **Build** | Docker — `webapp/backend/Dockerfile`, context `./webapp` |
| **Infrastructure** | `render.yaml` (Render Blueprint) |
| **Models in image** | `mobilenetv3_noaug.pth` + `mobilenetv3_cross_domain.pth` |
| **Slimming** | `webapp/.dockerignore` excludes other checkpoints from Docker |
| **Default API model** | `noaug` |
| **Port** | 10000 |
| **Cold start** | ~30–60 s on free tier after idle spin-down |

The production Docker image deliberately excludes `mobilenetv3_best.pth`,
`mobilenetv3_manipulations.pth`, and RetinaFace/TensorFlow to stay
within Render's **512 MB RAM** limit (~310 MB with two models loaded).

Full deployment instructions: **`../doc/READMEdeployment.md`**.

---

## 13. Robustness & Stress Testing

### 13.1 The 11 Manipulation Tests

The manipulation robustness testing applies 11 different image corruptions and evaluates model stability:

| # | Manipulation | How It Works |
|---|---|---|
| 1 | Original | No modification (baseline) |
| 2 | Green tint | Scale green channel by 1.3× |
| 3 | Blue tint | Scale blue channel by 1.3× |
| 4 | Brightness | Increase brightness by 40% |
| 5 | Contrast | Increase contrast by 40% |
| 6 | Gaussian blur | Radius 2 Gaussian filter |
| 7 | Motion blur | 9×9 horizontal motion kernel |
| 8 | JPEG compression | Quality 30 re-encoding |
| 9 | Resize | Downscale to 1/3 then upscale back |
| 10 | Crop | 70% center crop then resize to original |
| 11 | Noise | Gaussian noise σ=15 |

### 13.2 Robustness Results

**True Real sample**: The model correctly held "Real" across all 11 manipulations — confidence dipped under brightness/motion_blur/noise but never crossed 50%.

**True Fake sample**: The model incorrectly predicted "Real" across all 11 manipulations, including the unmanipulated original — a concrete demonstration that robustness and correctness are not the same thing. The model can be stably wrong.

### 13.3 Out-of-Domain (Non-Face) Behaviour

Testing on 10 non-face images produced near-random predictions (4 Real, 6 Fake, average confidence 86.67%), confirming that the model's predictions on non-face images are meaningless.

---

## 14. Deployment Readiness Assessment

### 14.1 Model Size

| Metric | Value |
|---|---|
| Checkpoint size on disk | 16.24 MB |
| Total parameters | 4,204,594 |
| Optimizer state | Not saved (Stage 3 checkpoint omits it) |

### 14.2 Latency Profile

| Measurement | Value |
|---|---|
| Raw forward pass (CPU) | 15.53 ms average |
| Raw forward pass (GPU, T4, estimated) | ~7–8 ms |
| Full `/predict` with Grad-CAM (CPU) | ~2.2 s |
| CPU throughput | 64.4 images/sec |
| GPU throughput (estimated) | ~125–140 images/sec |

### 14.3 Memory Footprint

| Measurement | Value |
|---|---|
| Python + PyTorch baseline | 272.5 MB RSS |
| After loading model | 311.1 MB (+38.7 MB for model) |
| Steady-state inference (20 passes) | 332.6 MB |

### 14.4 Compression Options

| Technique | Expected Impact | Recommended? |
|---|---|---|
| **FP16** | Halves checkpoint to ~8 MB; negligible accuracy loss on GPU | Yes — simplest, lowest-risk |
| **INT8 (post-training)** | ~4 MB checkpoint; accuracy trade-off needs empirical measurement | Conditional — measure first |
| **Pruning** | Risky at 4.2M params (already compact) | Not recommended |
| **Knowledge distillation** | Only if mobile/edge target needed | Not for current server deployment |

**Critical insight**: Quantising the model alone would not meaningfully improve end-to-end latency — the bottleneck is Grad-CAM rendering, not the classifier. Making Grad-CAM optional or caching heatmaps would have far greater impact.

---

## 15. Tools & Technologies Used

### 15.1 Development & Training

| Tool | Version / Details | Purpose |
|---|---|---|
| Python | 3.10+ | Core development language |
| PyTorch | Latest stable | Primary deep learning framework |
| Torchvision | Latest stable | Pre-trained models, transforms, data utilities |
| TensorFlow/Keras | Used for Model 3 (EfficientNet-B2) | Secondary framework |
| Kaggle Notebooks | T4 GPU runtime | Training environment with GPU acceleration |
| scikit-learn | Latest stable | Data splitting, metrics computation |
| NumPy / Pandas | Latest stable | Data manipulation and analysis |
| Matplotlib / Seaborn | Latest stable | Visualisation and EDA |
| timm | Latest stable | Pre-trained model zoo (ConvNeXt-V2 for Dual-Stream) |

### 15.2 Preprocessing & Face Detection

| Tool | Purpose |
|---|---|
| RetinaFace | Face detection and landmark-based alignment |
| OpenCV (cv2) | Image manipulation (motion blur, manipulations) |
| Pillow (PIL) | Image loading, resizing, format conversion |

### 15.3 Web Application

| Tool | Purpose |
|---|---|
| FastAPI | Backend API framework |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |
| python-multipart | File upload handling |
| python-docx | Word document generation |
| HTML5 / CSS3 / JavaScript | Frontend interface |

### 15.4 Explainability & Forensics

| Tool | Purpose |
|---|---|
| Grad-CAM (custom) | Visual heatmap generation |
| invisible-watermark | Watermark detection in forensic analysis |
| scipy | Signal processing for spectral forensics |

---

## 16. Lessons Learned & Reflections

### 16.1 High In-Domain Accuracy Does Not Guarantee Real-World Performance

The single most important lesson from this project: **99%+ accuracy on a held-out test set can coexist with 8.6% accuracy on real-world data from the same task domain**. The M4 report's headline 99.06% metrics were never wrong — they accurately reflected in-distribution performance. But in-distribution evaluation alone cannot reveal shortcut learning, domain shift, or preprocessing mismatches. Cross-domain evaluation is not optional.

### 16.2 Augmentation Must Target the Actual Failure Mode

The training pipeline included comprehensive augmentation (rotation, flip, colour jitter, JPEG compression, blur, noise, channel shift). Despite this, the model still learned dataset-specific shortcuts. The augmentations varied hue/brightness/saturation but did not simulate **HDR tone-mapping or sharpening artefacts** — the specific properties driving false positives. Augmentation strategy must be informed by error analysis, not applied generically.

### 16.3 Preprocessing Consistency Is a First-Class Concern

A channel-order mismatch in the deployed face-detection pipeline (BGR instead of RGB) caused every face detection to silently fail, with no error logged. The system fell through to a center-crop fallback without any visible indication of the problem. Preprocessing verification — not just model accuracy — must be part of every deployment checklist.

### 16.4 Lightweight Models Can Outperform Larger Ones

MobileNetV3-Large (4.2M parameters) outperformed both the 40.7M-parameter Dual-Stream network and the 7.8M-parameter EfficientNet-B2 on the metric that mattered most: out-of-distribution generalisation to consumer AI generators. Larger models are not automatically better; architecture-task fit and training strategy matter more.

### 16.5 Explainability Is the Latency Bottleneck

The raw classifier runs at 15.5 ms (CPU), but adding Grad-CAM inflates the full request to ~2.2 s — a 140× increase. Any future optimisation effort should target the explainability layer, not the model itself.

### 16.6 Silent Failures Are More Dangerous Than Loud Ones

Several issues in this project (RetinaFace channel-order mismatch, Keras optimizer-reload bug, dataset-specific shortcut learning) were silent — they produced plausible-looking results that concealed the underlying problem. Designing for observability (logging, validation checks, cross-domain probes) is essential.

---

## 17. Future Work

### 17.1 Short-Term (No Retraining Required)

1. **Decision-threshold recalibration**: Raise the deployed Fake prediction threshold above 50% to improve real-world precision at the cost of some recall.
2. **Make Grad-CAM optional**: A prediction-only endpoint without Grad-CAM would bring interactive use into the tens-of-ms range on CPU.
3. **Demographic audit**: Stratify cross-domain evaluation by skin tone, age, and gender to check for uneven performance — currently unassessed.
4. **GPU benchmark**: Run the prepared Kaggle benchmark cell to get real GPU latency and VRAM numbers.

### 17.2 Long-Term (Requires Retraining or New Data)

1. **Targeted HDR/sharpening augmentation**: Add augmentations that specifically simulate HDR tone-mapping and sharpening artefacts, rather than relying on generic colour jitter.
2. **Expand the Real class**: Incorporate photographs from diverse modern cameras, lighting conditions, and capture eras beyond FFHQ and CelebA.
3. **Hard-negative mining**: Fold the misclassified Real-Latest images back into future fine-tuning stages.
4. **Frequency-domain analysis as shortcut-learning countermeasure**: Revisit FFT/DCT features — not as a full architecture change, but specifically to help the model distinguish HDR/sharpening artefacts (which have identifiable frequency signatures) from genuine synthesis artefacts.
5. **Complete and integrate the cross-domain model**: Finish `../notebooks/cross-domain.ipynb` training and integrate it as a proper secondary verification path.
6. **Video support**: Extend the system to detect deepfake videos by integrating temporal information across consecutive frames.
7. **INT8 quantisation**: Measure accuracy trade-off empirically and deploy if acceptable for edge use cases.

---

## 18. Milestone-by-Milestone Summary

### Milestone 1 — Problem Definition & Literature Review

- Defined the problem statement: detecting AI-generated facial images in an era of increasingly realistic synthetic media.
- Surveyed five detection paradigms (spatial CNN, ViT, frequency-domain, noise residual, multi-scale).
- Reviewed benchmark datasets (FF++, Celeb-DF, DFDC, WildDeepfake) and baseline models (XceptionNet, EfficientNet, ViT, Face X-ray).
- Proposed an Explainable Dual-Stream ViT with RGB–FFT/DCT cross-attention fusion (later pivoted in M3).
- Established evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Inference Time.

### Milestone 2 — Data Preparation, EDA & Preprocessing

- Selected the "Real vs AI Generated Faces Dataset" (FFHQ + StyleGAN/StyleGAN2, 120,000+ images).
- Conducted comprehensive EDA: class balance (50/50), aspect ratio analysis (mostly 1.0), pixel intensity distribution (full 8-bit range), visual inspection confirming the difficulty of the task.
- Built the preprocessing pipeline: rescale to [0,1], resize to 224×224, batch size 32.
- Designed on-the-fly augmentation: rotation (20°), shifts (15%), shear (15%), zoom (20%), horizontal flip, brightness (0.8–1.2).
- Established strict Train/Validation/Test isolation with zero data leakage.

### Milestone 3 — Model Architecture, Selection & Baseline Performance

- Developed three candidate architectures in parallel: MobileNetV3-Large, Dual-Stream Spatial-Frequency Fusion, EfficientNet-B2.
- **Selected MobileNetV3-Large** based on: 80% out-of-distribution accuracy on ChatGPT/Gemini images, 4.2M parameters, ~8.2 ms inference on T4 GPU, 99.96% in-domain test accuracy.
- Documented the Dual-Stream Fusion model (ConvNeXt-V2 + ResNet-18 FFT stream with 15% HPF mask, cross-attention fusion) as a secondary reference.
- Identified the EfficientNet-B2 optimizer-reload bug that caused its poor 58.33% test accuracy.
- Designed the unified pipeline and model selection flowchart.

### Milestone 4 — Training, Hyperparameters & Full-Scale Evaluation

- Expanded dataset with Nano Banana 2.0 (64,000 total images across train/val/test).
- Added RetinaFace face detection and alignment to the preprocessing pipeline.
- Conducted 24-experiment hyperparameter ablation study (LR, batch size, weight decay, dropout, optimizer, scheduler, label smoothing).
- Trained the two-stage MobileNetV3-Large: 99.30% validation accuracy, 99.06% test accuracy.
- Documented training artefacts: checkpoint, notebooks, training logs, performance visualisations.

### Milestone 5 — Model Evaluation, Error Analysis & Deployment

- **Identified the shortcut-learning failure mode**: 8.6% accuracy on Real-Latest smartphone photos.
- **Root-caused the failure**: Domain shift between FFHQ training distribution and modern smartphone photography (HDR, sharpening, saturation).
- **Added Stage 3 fine-tuning**: Full backbone unfreeze with CelebA-HD enrichment at 5×10⁻⁶ LR.
- **Fixed the preprocessing mismatch**: BGR→RGB channel-order bug in RetinaFace integration.
- **Verified Grad-CAM explainability**: Confirmed attention on facial regions for correct predictions, and on post-processing artefacts for false positives.
- **Conducted robustness stress testing**: 11 manipulation types across true Real and true Fake samples.
- **Assessed deployment readiness**: 16.24 MB model, 15.5 ms CPU latency, ~332 MB process RSS.
- **Built and deployed the FastAPI web application** with 7 endpoints and 5 frontend pages.
- **Honestly documented what remains unmeasured**: GPU latency/VRAM, ROC-AUC on held-out test set, demographic fairness.

### Milestone 6 — Final Report & Consolidation

- Compiled this Final Technical Report covering the complete M1→M6 development lifecycle.
- Synthesised findings from all milestones into a coherent narrative: from the original ViT+FFT proposal through the MobileNetV3-Large pivot, through the shortcut-learning discovery and partial mitigation, to the deployed web application.
- Documented all challenges (shortcut learning, preprocessing mismatch, domain shift, optimizer bug, computational constraints, non-face handling, and unaudited demographic/ethical bias) and how each was addressed or, where unresolved, honestly scoped for future work.
- Provided actionable recommendations for future work, grounded in the specific failure modes discovered during evaluation.

---

## 19. References

1. Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Niessner, M. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
2. Shiohara, K., Yamasaki, T., et al. (2023). *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. NeurIPS Datasets and Benchmarks Track, 2023.
3. Gong, L. Y., & Li, X. J. (2024). *A Contemporary Survey on Deepfake Detection: Datasets, Algorithms, and Challenges*. Electronics, 13(19), 3863.
4. Zi, B., Chang, M., Chen, J., Ma, X., & Jiang, Y.-G. (2020). *WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection*. Proceedings of the 28th ACM International Conference on Multimedia (ACM MM 2020).
5. Howard, A., et al. (2019). *Searching for MobileNetV3*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
6. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
7. Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
8. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
9. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s (ConvNeXt)*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
10. Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. Proceedings of the International Conference on Machine Learning (ICML), 2019.

---

## 20. Team Declaration

We certify that all team members have actively contributed to the design, development, training, evaluation, and documentation of this project across all six milestones. Each member has reviewed the contents of this final technical report, understands the work presented, and agrees with the submitted report.

| Team Member | Primary Contributions Across M1–M6 | Signature |
|---|---|---|
| **Rohit** | Problem definition lead (M1); Dual-Stream Spatial-Frequency Fusion architecture development & FFT forensic extraction (M3); Training stability & benchmark evaluation (M5) | Rohit |
| **Raunak** | Literature review & benchmark analysis lead (M1); MobileNetV3-Large model development, training & testing — primary architecture (M3–M4); Cross-domain error analysis & shortcut-learning root-cause investigation (M5) | Raunak |
| **Vishakha** | Research findings & comparative analysis lead (M1); EfficientNet-B2 model development, video pipeline & Grad-CAM explainability (M3); Pipeline & presentation lead, deployment readiness assessment (M5) | Vishakha |
| **Aman** | Baseline performance & evaluation strategy lead (M1); Pipeline optimisation, evaluation scripting & DataLoader hardware integration (M3); Preprocessing & transfer learning, test dataset documentation (M5) | Aman |
| **Somendu** | Data research & presentation lead (M1); Hyperparameter search, experiment tracking & diagram visualisation (M3); Explainability verification, robustness testing & operational constraints documentation (M5) | Somendu |

---

*This report was compiled as Deliverable 2 for the DS & AI Lab Project, Group 11.*
*It covers the complete development lifecycle from Milestone 1 (problem definition) through Milestone 6 (final consolidation).*

# Deep Learning-Based Human Face Authenticity Detection

Detect authentic vs AI-generated/manipulated human faces in images and videos using a hybrid deep learning pipeline (EfficientNet-B4 + Vision Transformer), robust preprocessing, and explainable AI (Grad-CAM).

---

## Abstract

The rapid evolution of generative AI has enabled highly realistic synthetic facial media, increasing risks of misinformation, impersonation, identity theft, and digital fraud. This project proposes a deep learning-based human face authenticity detection framework that combines local texture learning and global contextual understanding through a hybrid EfficientNet-B4 and Vision Transformer architecture.  
To improve practical usability, the pipeline integrates face detection and alignment preprocessing, compression-aware robustness strategies, cross-dataset evaluation, and Grad-CAM heatmap explainability.  
The system is designed for reliable real-world deployment across both image and video inputs, with benchmark-driven evaluation on datasets such as FaceForensics++, Celeb-DF, and DFDC.

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip` / virtual environment (`venv` or conda)
- CUDA-enabled GPU (recommended for training)

### 1. Clone the repository

```bash
git clone https://github.com/Vishakharoy1/Group-11-DS-and-AI-Lab-Project.git
cd Group-11-DS-and-AI-Lab-Project
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not added yet, install the required packages after your training/inference modules are finalized.

### 4. Run training / inference

Add your scripts first (for example under `src/`), then run:

```bash
python src/train.py
python src/evaluate.py
python src/infer.py --input path/to/image_or_video
```

---

## Project Scope (Milestone 1)

- Detect fake vs authentic human faces in images and videos.
- Improve cross-dataset generalization for unseen manipulation techniques.
- Improve robustness against compression and low-quality media.
- Provide explainability through Grad-CAM confidence heatmaps.
- Evaluate using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and confusion matrix.

---

## Proposed Architecture

The proposed framework combines:

- **Face Detection + Alignment** for standardized facial region extraction.
- **EfficientNet-B4** for local texture artifact learning.
- **Vision Transformer (ViT)** for global contextual feature modeling.
- **Feature Fusion + Classification Head** for final authenticity prediction.
- **Grad-CAM Explainability Layer** for visual decision interpretation.

---

## Benchmark Datasets

- **FaceForensics++ (FF++)**
- **Celeb-DF**
- **DeepFake Detection Challenge (DFDC)**
- **WildDeepfake** (for real-world robustness studies)

---

## Evaluation Strategy

Typical split strategy (as defined in the milestone plan):

- **Training:** FaceForensics++
- **Validation:** Celeb-DF
- **Testing:** DFDC

Primary metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## Current Repository Structure

```text
Group-11-DS-and-AI-Lab-Project/
|
├── doc/
│   └── Milestone-1/
│       ├── Milestone-1-Report.md
│       └── Team-Contribution-Tracker.md
|
└── README.md
```

---

## Team Contributions (Milestone 1)

| Team Member | Role |
| --- | --- |
| Rohit | Project Objectives & Problem Definition Lead |
| Raunak | Literature Review & Benchmark Analysis Lead |
| Vishakha | Research Findings & Comparative Analysis Lead |
| Aman | Baseline Performance & Evaluation Strategy Lead |

Detailed responsibilities are documented in `doc/Milestone-1/Team-Contribution-Tracker.md`.

---

## Documentation

- Milestone report: `doc/Milestone-1/Milestone-1-Report.md`
- Team contribution tracker: `doc/Milestone-1/Team-Contribution-Tracker.md`

---

## Opportunities for Improvement

- Better generalization to unseen deepfake generation methods.
- Higher robustness on heavily compressed/low-quality media.
- Stronger explainability and user trust via better heatmap consistency.
- Efficient deployment for real-time video-scale inference.

---

## License

License information will be added in upcoming milestones.


# Phase 1 — ConvNeXt V2 Deepfake Detection Baseline

> Rough work directory. Research-grade implementation of Phase 1.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset
Place images in this structure inside `rohit/dataset/`:
```
dataset/
    train/
        real/     ← real face images
        fake/     ← AI-generated faces
    val/
        real/
        fake/
    test/
        real/
        fake/
```

### 3. Train (all 3 stages run automatically)
```bash
cd rohit/
python run_train.py
```

### 4. Evaluate separately (optional)
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

### 5. Monitor training
```bash
tensorboard --logdir logs/
```

---

## Architecture
```
Face Image [224×224]
      │
ConvNeXt V2 (pretrained, timm)
      │  convnext_base.fb_in22k_ft_in1k
      │
Global Average Pooling
      │
Dropout(0.3)
      │
Linear(1024 → 2)
      │
real / fake
```

## Fine-Tuning Stages
| Stage | Frozen | Epochs | LR (head) | LR (backbone) |
|-------|--------|--------|-----------|----------------|
| 1 — head only | backbone | 5 | 1e-4 | — |
| 2 — last stage | all except last stage | 10 | 1e-5 | 1e-5 |
| 3 — full model | nothing | 15 | 1e-5 | 5e-6 |

## Outputs
| File | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Best model (by val AUC) |
| `outputs/roc_curve.png` | ROC curve on test set |
| `outputs/confusion_matrix.png` | Confusion matrix |
| `outputs/classification_report.txt` | Full per-class report |
| `logs/` | TensorBoard logs |

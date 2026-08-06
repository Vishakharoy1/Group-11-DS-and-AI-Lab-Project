# MobileNetV3-Large Hyperparameter Optimization Plan

## 1. Dataset Understanding & Analysis
- **Dataset Path**: `rohit/dataset/`
- **Class Structure**: Binary classification (`fake` = 1, `real` = 0)
- **Directory Breakdown**:
  - `train/`: 21,000 real + 21,000 fake = 42,000 images (includes sample `train/fake/ADM_4.png`)
  - `val/`: 4,500 real + 4,500 fake = 9,000 images
  - `test/`: 4,500 real + 4,500 fake = 9,000 images
  - **Total Dataset Size**: 60,000 images (balanced 50% real / 50% fake across all splits)
- **Image Specifications**:
  - Format: PNG / JPG, RGB 3-channel
  - Resolution: 256x256 (Resized to 224x224 for MobileNetV3 input)
- **Augmentation Pipeline**:
  - Random Horizontal Flip
  - Color Jitter (Brightness 0.2, Contrast 0.2)
  - Forensic JPEG Compression Simulation (Quality range 75–100)
  - Normalization: ImageNet Mean `[0.485, 0.456, 0.406]` & Std `[0.229, 0.224, 0.225]`

---

## 2. Hardware & Environment Setup
- **Hardware**: Apple Silicon MacBook Pro M4 Pro
- **Acceleration**: PyTorch Metal Performance Shaders (`torch.device('mps')`)
- **Worker Threads**: `num_workers=8` (Optimized for 14 CPU cores on M4 Pro unified memory)
- **Python Environment**: `rohit/venv/bin/python` (PyTorch 2.13.0 + timm)

---

## 3. Baseline Model Architecture
- **Model**: `mobilenetv3_large_100` (Pretrained on ImageNet-1k)
- **Parameter Breakdown**:
  - **Total Parameters**: `4,204,594` (~4.20 Million)
  - **Backbone Parameters**: `4,202,032`
  - **Classifier Head Parameters**: `2,562` ($1280 \times 2 + 2$)
- **Classifier Head**: Linear classifier (`1280` in-features -> `2` classes) with Dropout layer & CrossEntropyLoss


---

## 4. Priority 2: Hyperparameter Optimization Strategy

We will systematically experiment with each specified hyperparameter while controlling for other variables to measure exact impacts on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Hyperparameters to Experiment:
1. **Learning Rate**: `1e-4`, `5e-4`, `1e-3`, `5e-3`
2. **Batch Size**: `32`, `64`, `128`
3. **Weight Decay**: `0.0`, `0.01`, `0.05`, `0.1`
4. **Dropout Rate**: `0.0`, `0.2`, `0.3`, `0.5`
5. **Optimizer**: `Adam` vs `AdamW`
6. **Learning Rate Scheduler**: `CosineAnnealingLR` vs `ReduceLROnPlateau`
7. **Label Smoothing**: `0.0`, `0.05`, `0.10`

### Baseline Configuration:
| Parameter | Baseline Value |
|---|---|
| Backbone | `mobilenetv3_large_100` |
| Image Size | 224x224 |
| Learning Rate | `1e-3` |
| Batch Size | `64` |
| Weight Decay | `0.01` |
| Dropout Rate | `0.2` |
| Optimizer | `AdamW` |
| LR Scheduler | `CosineAnnealingLR` |
| Label Smoothing | `0.0` |
| Epochs | 10 per run |

---

## 5. Experiment Tracking & Comparison Matrix Plan

Each experiment run will automatically record:
- **Train Loss & Accuracy**
- **Validation Loss & Accuracy**
- **Test Loss, Accuracy, Precision, Recall, F1-Score, and ROC-AUC**
- **Training Time per Epoch**

### Comparison Table Schema:
| Exp # | Hyperparameter Varied | Setting | Val Acc (%) | Val Loss | Test Acc (%) | Test F1 | Test AUC | Best Model Path |
|---|---|---|---|---|---|---|---|---|
| E0 | Baseline | Baseline | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_baseline.pth` |
| E1a | Learning Rate | 1e-4 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_lr_1e4.pth` |
| E1b | Learning Rate | 5e-4 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_lr_5e4.pth` |
| E1c | Learning Rate | 1e-3 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_lr_1e3.pth` |
| E1d | Learning Rate | 5e-3 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_lr_5e3.pth` |
| E2a | Batch Size | 32 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_bs_32.pth` |
| E2b | Batch Size | 64 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_bs_64.pth` |
| E2c | Batch Size | 128 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_bs_128.pth` |
| E3a | Weight Decay | 0.0 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_wd_0.pth` |
| E3b | Weight Decay | 0.01 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_wd_0.01.pth` |
| E3c | Weight Decay | 0.05 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_wd_0.05.pth` |
| E3d | Weight Decay | 0.1 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_wd_0.1.pth` |
| E4a | Dropout Rate | 0.0 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_drop_0.0.pth` |
| E4b | Dropout Rate | 0.2 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_drop_0.2.pth` |
| E4c | Dropout Rate | 0.3 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_drop_0.3.pth` |
| E4d | Dropout Rate | 0.5 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_drop_0.5.pth` |
| E5a | Optimizer | Adam | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_opt_adam.pth` |
| E5b | Optimizer | AdamW | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_opt_adamw.pth` |
| E6a | Scheduler | CosineAnnealing | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_sched_cosine.pth` |
| E6b | Scheduler | ReduceLROnPlateau | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_sched_plateau.pth` |
| E7a | Label Smoothing | 0.0 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_ls_0.0.pth` |
| E7b | Label Smoothing | 0.05 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_ls_0.05.pth` |
| E7c | Label Smoothing | 0.10 | TBD | TBD | TBD | TBD | TBD | `checkpoints/exp_ls_0.10.pth` |

---

## 6. Implementation Architecture & Script Files
- `rohit/mobilenetv3large/plan.md`: Hyperparameter optimization plan (this document).
- `rohit/mobilenetv3large/config.yaml`: Base configuration file for MobileNetV3-Large experiments.
- `rohit/mobilenetv3large/train_mobilenet.py`: Flexible standalone trainer supporting dynamic hyperparameter overrides.
- `rohit/mobilenetv3large/run_experiments.py`: Orchestrator script to execute all hyperparameter variations, record logs, and build the summary report.
- `rohit/mobilenetv3large/report.md`: Final detailed hyperparameter optimization report with comparison tables and insights.

---

## 7. Deliverables & Expected Output
1. Full execution of all ablation experiments on MacBook M4 Pro GPU (`mps`).
2. Final hyperparameter tuning summary report (`rohit/mobilenetv3large/report.md`).
3. Automated comparison table with Test Accuracy, Loss, F1, and AUC metrics.
4. Identification of the optimal hyperparameter combination for MobileNetV3-Large deepfake detection.
# Deepfake Detection Project: Progress & Setup Summary

This document summarizes the current status of the project, details the codebase created, lists the datasets extracted, and outlines the hardware optimization implemented for your **Apple M4 Pro (48GB RAM)**.

---

## 1. Files Used to Train and Run the Model

The codebase is stored in the `rohit/` folder and is fully structured. Here are the files that work together to train your model:

| File / Folder Path | Type | Purpose | Key Details |
| :--- | :--- | :--- | :--- |
| [`configs/config.yaml`](file:///Users/syngenta/Desktop/ds_ai/rohit/configs/config.yaml) | Configuration | Central hyperparameters file | Configured for Apple M4 Pro (Batch size: 64, Workers: 8, MPS enabled). Specifies 3-stage learning rate & unfreezing schedule. |
| [`src/utils.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/src/utils.py) | Python Script | Utility helper functions | Seed initialization, logging setup, and GPU/MPS auto-detection. |
| [`src/dataset.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/src/dataset.py) | Python Script | Data loading & transforms | Integrates PyTorch `ImageFolder`. Implements light, forensic-aware augmentations (JPEG compression simulation, brightness/contrast jitter, flips). |
| [`src/model.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/src/model.py) | Python Script | Classifier architecture | Defines the `ConvNeXtV2Classifier` wrapping `timm`'s pretrained backbone with custom Dropout + Linear head. Provides unfreezing wrappers for staged fine-tuning. |
| [`src/train.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/src/train.py) | Python Script | Training engine | Runs individual epochs, tracks training/validation loss, calculates metrics (F1, AUC, accuracy), logs to TensorBoard, and saves the best model. |
| [`src/evaluate.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/src/evaluate.py) | Python Script | Evaluation & Visualization | Evaluates the model on test splits, prints metrics, saves the ROC Curve plot, Confusion Matrix heatmap, and classification report. |
| [`run_train.py`](file:///Users/syngenta/Desktop/ds_ai/rohit/run_train.py) | Python Script | Single Entry Point | Automates training across all three staged learning rates (Stage 1: Head-only $\rightarrow$ Stage 2: Head + Last stage $\rightarrow$ Stage 3: Full backbone), then runs the final evaluation. |
| [`requirements.txt`](file:///Users/syngenta/Desktop/ds_ai/rohit/requirements.txt) | Dependency | Python package setup | Pinned dependencies including `timm`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, and `tensorboard`. |

---

## 2. All Work Completed (So Far)

### A. Core Architecture Setup (Phase 1 Baseline)
We implemented Phase 1 of your research plan:
- **Backbone**: Pretrained `convnext_base.fb_in22k_ft_in1k` from the `timm` library.
- **Head**: Replaced the original classifier head with `Dropout(0.3) + Linear(1024 -> 2)`.
- **Target**: RGB classification (Real vs. Fake).

### B. M4 Pro Hardware Optimizations
We customized the environment and configurations specifically for your **Apple M4 Pro MacBook Pro** (14 CPU cores, 48GB Unified RAM):
1. **GPU Acceleration**: Updated `src/utils.py` to auto-detect and run on Apple Silicon's **MPS (Metal Performance Shaders)** backend.
2. **Batch Size**: Increased batch size from `32` to `64` to exploit the spacious 48GB unified RAM.
3. **Data Loading**: Bumped `num_workers` to `8` to leverage the 10 performance cores, speeding up pre-processing.
4. **Memory Settings**: Disabled pin memory (`pin_memory: false`) because unified memory architectures don't benefit from standard GPU page-locking.
5. **Precision**: Disabled standard CUDA mixed precision (`mixed_precision: false`) as `GradScaler` is not supported on MPS, letting the hardware run optimization natively.

### C. Dataset Extraction & Inventory
We successfully extracted several large image datasets from compressed `.tar` files into `rohit/tar_dataset/`.

*   **Real Class (30,000 images)**:
    *   `Real/`: 30,000 real faces ✅
*   **Fake/Generated Class (~571,440 images)**:
    *   `ADM/`: 30,000 images (Ablated Diffusion Model) ✅
    *   `DDIM/`: 30,000 images (Denoising Diffusion Implicit Models) ✅
    *   `DDPM/`: 30,000 images (Denoising Diffusion Probabilistic Models) ✅
    *   `DiffSwap/`: 31,440 images (Diffusion Face Swap) ✅
    *   `LDM/`: 30,000 images (Latent Diffusion Model) ✅
    *   `PNDM/`: 30,000 images (Pseudo Numerical Methods for Diffusion Models) ✅
    *   `Inpaint/images/`: 30,000 images (Inpainted/manipulated faces) ✅
    *   `SDv15_DS0.3/`, `SDv15_DS0.5/`, `SDv15_DS0.7/`: 90,000 images (Stable Diffusion v1.5 with varied denoising strengths) ✅
    *   `SDv21_DS0.3/`, `SDv21_DS0.5/`, `SDv21_DS0.7/`: 90,000 images (Stable Diffusion v2.1 with varied denoising strengths) ✅
    *   `stable_diffusion_v_1_5_text2img_p3g7/aligned/`: 30,000 images ✅
    *   `stable_diffusion_v_1_5_text2img_p4g5/aligned/`: 30,000 images ✅
    *   `stable_diffusion_v_1_5_text2img_p5g3/aligned/`: 30,000 images ✅
    *   `stable_diffusion_v_2_1_text2img_p0g5/aligned/`: 30,000 images ✅
    *   `stable_diffusion_v_2_1_text2img_p1g7/aligned/`: 30,000 images ✅
    *   `stable_diffusion_v_2_1_text2img_p2g3/aligned/`: 30,000 images ✅
*   **Fake (Unknown Source) Class (36,460 images)**:
    *   `Wild/images_256/`: 36,460 images. Extracted by combining `Wild.tar00` and `Wild.tar01`. Useful for out-of-distribution evaluation. ✅
*   **Skipped/Non-Image Tars**:
    *   `MM_CelebA.tar`: Checked and skipped because it contained metadata/text descriptions, rather than actual images.

---

## 3. Next Step

We need to create a **subset extraction script** to sample images from these directories and construct your balanced dataset (e.g., 20,000 to 60,000 images total, split 1:1 real vs. fake) into the final training layout (`rohit/dataset/train/`, `rohit/dataset/val/`, `rohit/dataset/test/`). 

Let me know when you are ready to build this data splitting script.

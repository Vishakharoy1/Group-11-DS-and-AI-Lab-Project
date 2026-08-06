# Training Notebooks — Usage Instructions

This project's final architecture is **MobileNetV3-Large** (selected in
Milestone 3; the original EfficientNet-B4 + Vision Transformer proposal
from Milestone 1 was not used). Two Kaggle notebooks produce the
checkpoints used throughout this project and the local web app:

| Notebook | Produces | Status |
|---|---|---|
| `final-mobilenet (1).ipynb` | `mobilenetv3_best.pth` — the main face-authenticity model | Complete, fully trained |
| `cross-domain.ipynb` | `mobilenetv3_cross_domain.pth` — a separate model for non-face/general images | Not yet run to completion (dataset paths only verified, training never executed) |

(For the local web app that loads these checkpoints, see
`webapp/backend/README.md`.)

---

## 1. `final-mobilenet (1).ipynb` — the main face model

### 1.1 Attaching datasets on Kaggle

In the notebook's right sidebar, click **+ Add Input** and attach these
three datasets:

| Dataset | Kaggle slug | Used as |
|---|---|---|
| Real vs AI Generated Faces Dataset | `philosopher0808/real-vs-ai-generated-faces-dataset` | `REAL_DIR` — FFHQ real face images (70,000 available) |
| Stable Diffusion Dataset | `mohannadaymansalah/stable-diffusion-dataaaaaaaaa` | `FAKE_DIR` — AI-generated fake face images (9,001 available) |
| CelebA Dataset | `jessicali9530/celeba-dataset` | `REAL_DIR_CELEBA` — additional HD real photos, used only in **Stage 3** to fix modern-photo shortcut learning (see Section 1.3) |

The CelebA dataset is auto-detected: if it isn't attached, Stage 3 is
skipped automatically (`INCLUDE_CELEBA = REAL_DIR_CELEBA.exists()`) and
the notebook produces a 2-stage model instead of the intended 3-stage
one — attach it if you want the model this project actually deploys.

Also enable **Settings → Internet → ON** — the notebook installs
`retina-face` via pip early on, which needs internet access. If it fails
to install, face detection silently falls back to a center-crop (the
notebook prints a warning; this does not stop the run).

### 1.2 Running the notebook — cell order

Run top-to-bottom (**Run All**, or **Restart & Run All** if the kernel
has stale state from a previous partial run):

| Cell | What it does |
|---:|---|
| 1 | Imports, configuration, hyperparameters (`SEED=42`, `IMG_SIZE=224`, `BATCH_SIZE=128`, `IMAGES_PER_CLASS=15000`), dataset paths |
| 2 | Installs `retina-face` (needs internet) |
| 3 | Defines `crop_and_align_face()` — RetinaFace crop with center-crop fallback |
| 4 | Builds the dataset DataFrame: loads FFHQ + Stable Diffusion (+ CelebA-HD if attached), samples down to `IMAGES_PER_CLASS`, shuffles |
| 5 | 80:10:10 stratified train/val/test split (`SEED=42`) |
| 6 | Transforms (training augmentation incl. `ChannelShift`, `JPEGCompression`, Gaussian noise/blur) + `DataLoader`s |
| 7 | Builds MobileNetV3-Large (ImageNet-pretrained backbone, 2-class classifier head) |
| 8 | **Stage 1**: 3 epochs, classifier head only, backbone frozen, LR `3e-4` |
| 9 | **Stage 2**: 7 epochs, unfreezes last 25% of backbone (blocks 12–16 of 17), LR `1e-5` |
| 10 | **Stage 3**: 3 epochs, full-model unfreeze, LR `5e-6`, adds CelebA-HD — **skipped automatically if CelebA isn't attached** |
| 11 | Final evaluation: loads the best checkpoint, runs it on the held-out test set, prints the classification report + confusion matrix |
| 12 | Defines `get_gradcam()` — Grad-CAM heatmap + layer-activation visualization function |
| 13 | Interactive `ipywidgets` file-upload cell (optional, for manual spot-checks inside the notebook) |
| 14 | Loads the saved checkpoint and runs a prediction on the uploaded image (run after Cell 13) |

Each stage's training loop only saves a new checkpoint when it beats the
previous best validation accuracy — Stage 1's and Stage 2's own
intermediate weights are **not** kept once a later stage overwrites the
file at `MODEL_SAVE_PATH`. If you need to compare stages against the test
set individually (not just validation accuracy), save each stage's
checkpoint under a different filename before letting the next stage
overwrite it.

### 1.3 Reproducing results

1. Attach all three datasets (Section 1.1) — CelebA is required to
   reproduce the actual 3-stage model this project deploys.
2. **Restart & Run All.**
3. Download `/kaggle/working/mobilenetv3_best.pth` from the Kaggle
   Output tab — this is exactly what the local web app
   (`webapp/backend/`) expects in its `output/` folder.

Expected results on this dataset composition (verified, from an actual
run): **99.63% test accuracy** on the 2,401-image held-out test set,
**99.71%** best validation accuracy. See `doc/Milestone-5/Milestone5.md`
Section 4 for the full breakdown (classification report, confusion
matrix, per-stage validation accuracy).

### 1.4 What this notebook does *not* save to disk

Unlike `cross-domain.ipynb` below, this notebook does **not** call
`plt.savefig()` anywhere — the confusion matrix and Grad-CAM figures are
only shown inline (`plt.show()`) during the Kaggle session, not written
to `/kaggle/working/`. The **only file this notebook saves** is the
checkpoint itself (`mobilenetv3_best.pth`). If you need the confusion
matrix or Grad-CAM images as files, either add your own `plt.savefig()`
calls before re-running, or extract them from the notebook's own saved
output (the `.ipynb` file embeds the last-rendered figure as inline
image data even without an explicit save — this is how the confusion
matrix image in `doc/Milestone-5/images/confusion_matrix_best_model.png`
was obtained).

---

## 2. `cross-domain.ipynb` — the general/non-face model

**Status: not yet run to completion.** Only the dataset-path-check cell
has ever produced output; no training has actually run. Attaching
datasets and running this notebook is still an open task.

### 2.1 Attaching datasets on Kaggle

| Dataset | Kaggle slug | Used as |
|---|---|---|
| AI Image with Nano Banana 2.0 vs Real Image | `ahnuf05/ai-imagewith-nano-banana-2-0-vs-real-image` | Real + Fake — AI art vs. real photos |
| CIFAKE | `birdy654/cifake-real-and-ai-generated-synthetic-images` | Real + Fake — SDXL-generated vs. real CIFAR-10 objects |
| CrossDomain | `rahulshetty1020/crossdomain` | Real + Fake — small hand-collected set (only ~10 real images) |
| Places365 (2 scenes) | `nickj26/places2-mit-dataset` | Real — indoor/outdoor scene photos, capped at 10,000 |
| Artifact | `awsaf49/artifact-dataset` | Fake — AI-generated scenes (DALL-E, Stable Diffusion, Midjourney), capped at 8,000 |

Run **Cell 2b (Verify all dataset paths)** first after attaching — it
prints which paths actually resolve. As of the last check, only the
first three datasets above were confirmed attached; Places365 and
Artifact still need to be attached before a full run.

### 2.2 Running the notebook — cell order

| Cell | What it does |
|---:|---|
| 1 | Imports |
| 2 | Configuration: dataset paths, `IMG_SIZE=224`, `BATCH_SIZE=32`, per-directory image caps |
| 2b | **Verify dataset paths** — run this first to confirm what's actually attached |
| 3 | Collect image paths into a labeled DataFrame |
| 4 | Transforms (crop, flip, colour jitter, blur, random erasing) |
| 5 | 85:15 train/val split with a class-balancing `WeightedRandomSampler` |
| 6 | Builds MobileNetV3-Large |
| 7 | Training helper functions |
| 8 | **Stage 1**: 3 epochs, frozen backbone |
| 9 | **Stage 2**: 7 epochs, last 25% of backbone unfrozen |
| 10 | Plots + saves training curves |
| 11 | Per-domain validation accuracy breakdown |
| 12 | Confusion matrix + classification report, saved as an image |
| 13 | Interactive `ipywidgets` upload-and-test cell (optional) |

This notebook, unlike the main one, does **not** use RetinaFace or any
face cropping — it trains directly on whole images, since it's meant for
general (non-face) content.

### 2.3 Reproducing results

Not yet reproducible end-to-end — Places365 and Artifact still need to
be attached, and the full training run (Cells 3–12) has never been
executed. Once run:

1. Attach all five datasets (Section 2.1), confirm with Cell 2b.
2. **Restart & Run All.**
3. Download everything under `/kaggle/working/output/` — the checkpoint
   plus the CSV/PNG artifacts described below.

### 2.4 What the output files mean

All written to `/kaggle/working/output/`:

| File | Contents |
|---|---|
| `mobilenetv3_cross_domain.pth` | Model checkpoint (state dict + val accuracy + class list) |
| `training_curves.png` | Train/val accuracy per epoch across both stages |
| `domain_accuracy.csv` | Validation accuracy broken down by data source (Nano Banana, CIFAKE, CrossDomain, Places365, Artifact) |
| `confusion_matrix.png` | Validation-set confusion matrix (Real vs. Fake) |

---

## 3. Where these outputs get used

Both checkpoints are consumed by the local web app's `ModelRegistry`
(`webapp/backend/app/model.py`), which loads whichever files it finds in
`output/` and skips missing ones gracefully. See `webapp/backend/README.md`
for how to run the app itself, and `doc/Milestone-5/Milestone5.md` for
the full evaluation of `mobilenetv3_best.pth` (Sections 2–9).

# Training Notebook — Usage Instructions

Covers `end-to-end-pipeline.ipynb`: how to attach the datasets on Kaggle,
run the notebook cell-by-cell, reproduce results, and what each output
file means. (For the local web app, see `webapp/backend/README.md`.)

## 1. Attaching datasets on Kaggle

Open the notebook in a Kaggle Notebook environment, then in the right
sidebar click **+ Add Input** and attach these three datasets:

| Dataset | Kaggle slug | Used as |
|---|---|---|
| Real vs AI Generated Faces Dataset | `philosopher0808/real-vs-ai-generated-faces-dataset` | `REAL_DIR` / `FAKE_DIR` — main face training data |
| Stable Diffusion Dataset | `mohannadaymansalah/stable-diffusion-dataaaaaaaaa` | `FAKE_DIR` — AI-generated faces |
| AI Image with Nano Banana 2.0 vs Real Image Dataset | `ahnuf05/ai-imagewith-nano-banana-2-0-vs-real-image` | `EXTRA_DATASET` — cross-domain data, merged into train/val/test |

After attaching, verify the mounted paths under `/kaggle/input/` match
what `REAL_DIR` / `FAKE_DIR` / `EXTRA_DATASET` expect in **Section 1
(Imports & Configuration)** — Kaggle sometimes mounts datasets under a
slightly different folder name than the slug; update the `Path(...)`
values there if so.

Also enable **Settings → Internet → ON** — Section 0 installs
`retina-face` via pip, which needs internet access.

## 2. Running the notebook — cell order

Run top-to-bottom with **Run All**, or **Restart & Run All** if the
kernel already has stale state (see the *"NameError: evaluate is not
defined"* pitfall below). Section numbers match the notebook's own
markdown headers:

| Section | What it does | Notes |
|---|---|---|
| 0 | Install `retina-face` | Needs internet |
| 1 | Imports & Configuration | Sets `HPARAMS`, dataset paths, all toggle flags |
| 2 | Build Dataset DataFrame | Loads/splits/tags data by domain (`face_main` vs `nano_banana`), runs stratification + leakage checks |
| 3 | Face Detection & Alignment | RetinaFace crop, cached to `faces_cropped/` — one-time, resumable |
| 4 | Robust Transformation Pipeline | Training augmentation + visual sample check |
| 5 | Dataset & DataLoaders | — |
| 6 | Model — MobileNetV3-Large | — |
| 7 | Training Functions | Defines `train_one_epoch`, `validate`, `train_stage` — **no output when run, this is expected** (see pitfall below) |
| 8 | Stage 1 & 2 Training | Main model training — this is where progress actually prints |
| 9 | Final Evaluation | Defines + runs `evaluate()` on the test set |
| 9b | Augmentation Ablation Training | Trains the no-augmentation comparison model. Gated by `TRAIN_NOAUG_MODEL` (default `True`) |
| 10 | Grad-CAM Explainability | Correct vs. incorrect prediction examples |
| 11 | Robustness Evaluation | Tint/blur/noise/jpeg corruption table |
| 12b | Image Manipulation Testing | Full 11-manipulation set |
| 12c | Cross-Domain Testing | `face_main` vs `nano_banana` accuracy split |
| 12 | Hyperparameter Sweep | 24-experiment sweep. Gated by `RUN_SWEEP` (default `True`) — takes real GPU time |
| 12d | Retrain With Best Hyperparameters | Gated by `RETRAIN_WITH_BEST_HPARAMS` and `RUN_SWEEP` |
| 14–20 | Upload & Test (Priority 1–6) | Interactive `ipywidgets` upload cells — optional, for manual spot-checks inside the notebook |

### Common pitfall: "NameError: name 'evaluate' is not defined"

Section 7 only **defines** functions — it produces no visible output when
run, which is normal, not a sign it was skipped. If a later cell (e.g.
the Section 12 sweep, which calls `evaluate()`) throws this error, it
means the kernel was restarted and Section 9 (which defines `evaluate`)
was never re-run before jumping to the later cell. Fix: **Restart & Run
All** from the top.

## 3. Toggle flags (Section 1)

| Flag | Default | Effect if `True` |
|---|---|---|
| `DO_FACE_PREPROCESSING` | `True` | Runs RetinaFace cropping (Section 3) |
| `ROBUST_TRAIN_AUG` | `True` | Uses the augmented training pipeline (Section 4) for the main model |
| `RUN_ROBUSTNESS_EVAL` | `True` | Runs Section 11's corruption table |
| `TRAIN_NOAUG_MODEL` | `True` | Runs Section 9b — a full second Stage 1+2 training pass, roughly doubles total training time |
| `RUN_SWEEP` | `True` | Runs the 24-experiment sweep (Section 12) — several hours |
| `RETRAIN_WITH_BEST_HPARAMS` | `True` | Runs Section 12d — a third full training pass, only meaningful if `RUN_SWEEP` also ran |

Set any of these to `False` to skip that stage and save GPU time — a full
top-to-bottom run with everything enabled trains **3 full models plus 24
short sweep experiments**.

## 4. Reproducing results

1. Attach all three datasets (Section 1 above).
2. Leave all toggle flags at their defaults for a full reproduction, or
   set `RUN_SWEEP = False` and `TRAIN_NOAUG_MODEL = False` for a faster
   run of just the main model.
3. **Restart & Run All.**
4. Download `/kaggle/working/mobilenetv3_best1.pth` (and `_noaug.pth` /
   `_tuned.pth` if those stages ran) plus the `/kaggle/working/outputs/`
   folder — these are exactly what the local web app
   (`webapp/backend/`) expects in its `output/` folder.

## 5. What the output files mean

All written to `/kaggle/working/outputs/` during the run:

| File | Contents |
|---|---|
| `mobilenetv3_best1.pth` | Main model checkpoint (state dict + val accuracy) |
| `mobilenetv3_noaug.pth` | No-augmentation comparison model (Section 9b) |
| `mobilenetv3_tuned.pth` | Swept-hyperparameters model (Section 12d) |
| `confusion_matrix.png` | Test-set confusion matrix (Section 9) |
| `preprocessing_samples.png` | Sample RetinaFace-cropped faces (Section 3) |
| `augmentation_samples.png` | Sample train-time augmented images (Section 4) |
| `manipulation_samples.png` | Sample manipulated images (Section 12b) |
| `cross_domain_samples.png` | Sample `nano_banana` domain images (Section 12c) |
| `gradcam_correct_*.png` / `gradcam_incorrect_*.png` | Grad-CAM heatmaps, split by correct/incorrect prediction (Section 10) |
| `robustness_results.csv` | Corruption-accuracy table (Section 11) |
| `manipulation_results.csv` | Full 11-manipulation accuracy table (Section 12b) |
| `augmentation_ablation_results.csv` | With-vs-without-augmentation comparison (Section 9b) |
| `cross_domain_results.csv` | `face_main` vs. `nano_banana` accuracy (Section 12c) |
| `sweep_comparison.csv` | All 24 hyperparameter experiments (Section 12) — only if `RUN_SWEEP` ran |
| `hparam_baseline_vs_tuned.csv` | Baseline vs. tuned comparison (Section 12d) — only if that stage ran |

# Milestone 5 — Model Evaluation & Analysis

**Group 11 · Deep Learning-Based Human Face Authenticity Detection**
**Deadline:** 6 August 2026

This report documents the final held-out evaluation of the M4 model
checkpoint, addresses the two critical issues flagged in the M4 faculty
review (shortcut learning on real images, and the preprocessing mismatch
between the training notebook and the UI backend), and covers robustness,
explainability, error analysis, and deployment readiness ahead of the
viva.

---

## 1. Introduction & Objectives

*Owner: Vishakha*

### 1.1 Background — The Milestone 4 Checkpoint

Milestone 4 delivered a binary Real-vs-AI-Generated face classifier built
on a **MobileNetV3-Large** backbone, trained with a two-stage transfer
learning strategy (Stage 1: frozen backbone, classifier head only; Stage
2: fine-tuning) on a combined dataset of FFHQ real faces, Stable
Diffusion-generated faces, and the Nano Banana 2.0 cross-domain dataset
(~24,000 primary training images, plus ~32,000 additional images merged
in from Nano Banana 2.0's own train/val/test splits).

The resulting checkpoint, `mobilenetv3_best.pth`, was selected via a
24-experiment hyperparameter sweep and reported the following headline
numbers in the M4 report:

| Metric | Reported Value |
|---|---|
| Best Validation Accuracy | 99.30% |
| Test Accuracy | 99.06% |
| Precision / Recall / F1 (Real) | 0.9900 / 0.9929 / 0.9914 |
| Precision / Recall / F1 (AI-generated) | 0.9914 / 0.9879 / 0.9896 |
| Selected Optimizer | AdamW |
| Selected LR / Batch Size | 5×10⁻⁴–1×10⁻³ / 64–128 |
| Selected Weight Decay / Dropout | 0.05–0.10 / 0.0–0.2 |
| Selected Scheduler / Label Smoothing | CosineAnnealingLR / 0.0 |

On paper, this checkpoint appeared close to solved. Notably, M4's own
"Future Improvements" list (Section 9.5, item 6) explicitly called for
*"a real-time inference application that can classify uploaded images
through a web or mobile interface"* — the deployment work undertaken as
part of this milestone.

### 1.2 Motivation for Milestone 5

Despite the strong headline metrics above, the M4 faculty review Q&A
session identified two critical, unresolved issues that are **not**
reflected in the M4 report's own numbers (both raised verbally during the
review, not previously documented in writing):

1. **Real images being misclassified as fake due to shortcut learning** —
   the model appears to key on incidental image properties (e.g. high
   resolution, vibrant colour, outdoor/HD scenes) rather than genuine
   forgery artifacts, meaning the 99%+ test accuracy does not necessarily
   reflect real-world robustness.
2. **A preprocessing mismatch between the training notebook and the
   deployed UI backend** — the face-crop/alignment step and channel-order
   handling used at inference time were not verified to be identical to
   what the model was trained and evaluated on.

This gap between a near-perfect held-out test score and demonstrably
fragile real-world behaviour is the central problem Milestone 5 exists to
diagnose and document.

### 1.3 Objectives of Milestone 5

1. Conduct a final evaluation on a strictly held-out test set, reporting
   Accuracy, Precision, Recall, F1, and ROC-AUC (Section 4).
2. Investigate and document the root cause of the shortcut-learning
   misclassification issue (Section 5).
3. Verify and document whether the deployed inference pipeline's
   preprocessing (face detection/alignment, resize, normalization)
   actually matches the training notebook's — the Priority 1 task for
   this milestone (Section 1 recap here; findings documented alongside
   the robustness work).
4. Quantify robustness under real-world image manipulations — colour
   tints, JPEG compression, blur, and noise (Section 6).
5. Verify explainability via Grad-CAM, confirming the model attends to
   facial regions rather than background/shortcut cues (Section 6).
6. Assess deployment readiness — latency, model size, and quantization
   potential (Section 9).
7. Compile all findings into a viva-ready report and presentation.

### 1.4 Scope

**In scope:** rigorous evaluation of the existing M4 checkpoint(s) on
held-out data, root-cause analysis of the two flagged critical issues,
robustness/explainability/error analysis, and targeted fixes to the
inference pipeline where the evaluation surfaces a genuine bug (e.g. the
preprocessing parity check). A local web application was also built
during this milestone to support interactive testing of the checkpoint(s)
outside the notebook environment, directly addressing M4's own suggested
future work.

**Out of scope:** retraining the model from scratch or changing its
architecture, unless doing so is required to correct a critical issue
identified during evaluation.

---

## 2. Evaluation Setup & Test Dataset

*Owner: Aman*

*(To be filled in — describe the test set: total images, real vs. fake
ratio, sources (FFHQ, Stable Diffusion, Nano Banana / cross-domain),
confirm zero data leakage, describe the Kaggle GPU environment used for
evaluation.)*

---

## 3. Metric Selection & Justification

*Owner: Aman / Raunak*

*(To be filled in — justify each metric chosen: Accuracy, Precision,
Recall, F1, ROC-AUC.)*

---

## 4. Quantitative Performance & Benchmarking

*Owner: Rohit*

*(To be filled in — final held-out test set results: Accuracy, Precision,
Recall, F1, ROC-AUC; confusion matrix, ROC curve, Precision-Recall curve;
classification report; Stage 1 vs. Stage 1+2 vs. Stage 1+2+3 comparison.)*

---

## 5. Comprehensive Error Analysis

*Owner: Raunak*

*(To be filled in — root cause analysis of real-images-predicted-as-fake
(shortcut learning), dataset distribution audit, out-of-distribution
testing, categorized false positive / false negative error analysis.)*

---

## 6. Model Robustness & Interpretability

*Owner: Somendu*

*(To be filled in — Grad-CAM heatmap gallery, robustness stress test
table (manipulation testing — see Section 6 notes below), OOD test
results.)*

### Robustness stress test data (manipulation testing)

*Owner: Vishakha*

Predictions before vs. after each manipulation (green tint, blue tint,
JPEG compression, Gaussian blur, Gaussian noise, etc.), generated via the
local web app's `/robustness` endpoint running `mobilenetv3_manipulations.pth`
against two known-labelled test images — one true Real, one true Fake
(sourced from the notebook's own `gradcam_correct_*`/`gradcam_incorrect_*`
Grad-CAM sample images, cropped to just the face panel to avoid the
composite-figure crop artifact). Baseline preprocessing for both runs:
RetinaFace was unavailable on the machine that generated this data
(falls back to center-crop — see Section 7 for the operational
consequence of this); both test images were already tightly face-cropped,
so this fallback does not distort the result here.

**Table 6.1 — True label: Real**

| Manipulation | Prediction | Real % | Fake % |
|---|---|---|---|
| original | Real | 56.4 | 43.6 |
| green_tint | Real | 96.8 | 3.2 |
| blue_tint | Real | 67.2 | 32.9 |
| brightness | Real | 69.2 | 30.8 |
| contrast | Real | 78.4 | 21.6 |
| gaussian_blur | Real | 83.7 | 16.3 |
| motion_blur | Real | 64.3 | 35.6 |
| jpeg | Real | 70.4 | 29.6 |
| resize | Real | 76.5 | 23.5 |
| crop | Real | 92.1 | 7.9 |
| noise | Real | 62.8 | 37.2 |

**Table 6.2 — True label: Fake**

| Manipulation | Prediction | Real % | Fake % |
|---|---|---|---|
| original | Real ❌ | 100.0 | 0.0 |
| green_tint | Real ❌ | 100.0 | 0.0 |
| blue_tint | Real ❌ | 100.0 | 0.0 |
| brightness | Real ❌ | 100.0 | 0.0 |
| contrast | Real ❌ | 99.8 | 0.2 |
| gaussian_blur | Real ❌ | 97.9 | 2.1 |
| motion_blur | Real ❌ | 93.1 | 6.9 |
| jpeg | Real ❌ | 99.9 | 0.1 |
| resize | Real ❌ | 96.8 | 3.2 |
| crop | Real ❌ | 99.9 | 0.1 |
| noise | Real ❌ | 100.0 | 0.0 |

**Observations:**
- On the true-Real sample, the model correctly held "Real" across all 11
  manipulations — the manipulation-specialized checkpoint does not flip
  under any single corruption on this example, indicating genuine
  robustness rather than a fragile/borderline call (confidence dips
  under `brightness`/`motion_blur`/`noise` but never crosses 50%).
- On the true-Fake sample, the model incorrectly predicted "Real" across
  **all 11** manipulations, including the unmanipulated original — this
  is a single-sample result, not a claim about aggregate accuracy (the
  full `manipulation_results.csv` from the training notebook, based on a
  much larger evaluation subset, should be used for the report's
  headline robustness numbers; this table is a targeted qualitative
  before/after illustration, not the primary metric). It is nonetheless
  a concrete, reproducible example worth investigating alongside
  Raunak's shortcut-learning root-cause analysis (Section 5) — this
  specific Fake sample being called "Real" with 100% confidence even
  before any manipulation is applied is consistent with the
  real-image-shortcut-learning failure mode, just in the opposite
  direction (a Fake image exhibiting whatever property the model
  associates with "Real").

*(Add your own manually-uploaded test images and their before/after
numbers here to broaden this beyond two examples — use the "Show
detailed breakdown (for report)" toggle in the Manipulation Robustness
Testing section of the web app.)*

---

## 7. Model Limitations & Operational Constraints

*Owner: Somendu*

*(To be filled in — which image types fail (colour tints, OOD),
computational requirements (VRAM, latency), ethical concerns, false
positive rate on real images.)*

---

## 8. Actionable Insights & Potential Improvements

*Owner: Somendu*

*(To be filled in — short-term: FAKE_THRESHOLD tuning, ChannelShift
augmentation. Long-term: larger backbone, more diverse training data,
adversarial training, frequency-domain analysis.)*

---

## 9. Deployment Readiness Assessment

*Owner: Vishakha*

### 9.1 Model Size

`mobilenetv3_best1.pth` is **45.3 MB** on disk (FP32 state dict). At this
size the checkpoint comfortably fits in memory on virtually any CPU or
edge deployment target; the bottleneck for deployment is latency, not
storage.

### 9.2 Latency — Preliminary CPU Benchmark

**Official GPU/CPU latency numbers are Rohit's Section 4 deliverable**
(100-image benchmark, both GPU and CPU) and should be used as the
report's headline figures once available. The numbers below are a
preliminary, CPU-only benchmark run on a different machine (desktop
Intel i7-7700 @ 3.6GHz, 4 cores/8 threads, no GPU) purely to sanity-check
where time is actually spent in the deployed pipeline — useful context,
not a substitute for Rohit's controlled measurement.

| Measurement | Result |
|---|---|
| Raw model forward pass (PyTorch, MKL + oneDNN CPU acceleration active) | **15.8 ms** average (10-run benchmark, `torch.no_grad()`) |
| Full `/predict` request end-to-end (face-crop, model forward, Grad-CAM backward pass + heatmap render, PNG+base64 encode, HTTP round-trip) | **~2.2 s** average (5-run benchmark) |

The ~140× gap between the raw forward pass and the full request is the
important finding here: **the model itself is not the bottleneck.** The
overhead comes from Grad-CAM (a full backward pass plus matplotlib
heatmap colormap generation) and image encoding, not the MobileNetV3
inference. This matters for deployment planning — a Grad-CAM-free
prediction endpoint (forward pass + softmax only, no explainability
overlay) would be close to the 15.8 ms figure, while any interactive UI
that always renders Grad-CAM should budget for multi-second responses on
CPU-only hardware. The Manipulation Robustness Testing endpoint compounds
this further, since it runs 11 sequential forward passes per request.

### 9.3 VRAM Usage

*(To be filled in — GPU VRAM usage from Rohit's Section 4 benchmarking
environment; MobileNetV3-Large's ~5.4M parameters imply a small
footprint, but the actual peak VRAM depends on batch size and whether
Grad-CAM's backward-pass activations are retained.)*

### 9.4 Quantization Potential

MobileNetV3-Large was explicitly designed for mobile/edge efficiency
(depthwise-separable convolutions, Hardswish activations), making it a
strong quantization candidate:

- **FP16:** halves the checkpoint to ~22.6 MB with typically negligible
  accuracy loss on GPU inference — the simplest, lowest-risk win.
- **INT8 (post-training static or dynamic quantization):** could bring
  the checkpoint to roughly 11–12 MB, well-suited to CPU/edge/mobile
  deployment; PyTorch's native quantization tooling supports
  MobileNetV3's op set directly. Expected accuracy trade-off would need
  to be measured empirically (not done in this milestone) before
  committing to it for production.
- Given the Section 9.2 finding that the model forward pass is already
  fast relative to the rest of the pipeline, **quantizing the model
  alone would not meaningfully improve the end-to-end latency users
  actually experience** — optimizing Grad-CAM rendering (e.g. making it
  optional, or caching/downsizing the heatmap) would have far more
  impact than quantization for this specific deployment.

### 9.5 Accuracy vs. Speed Trade-off Summary

*(To be filled in once Rohit's Section 4 numbers land — combine his
GPU/CPU accuracy-preserving latency figures with the CPU-only findings
above to state a final recommendation: e.g. GPU for interactive
Grad-CAM-enabled use, CPU acceptable for prediction-only/batch use.)*

---

## 10. Summary & Conclusion

*Owner: Vishakha*

*(To be filled in last, once Sections 1–9 are complete — summarize
evaluation highlights, compare against original M4 objectives, formal
sign-off statement.)*

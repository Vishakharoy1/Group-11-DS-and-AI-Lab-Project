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
JPEG compression, Gaussian blur, Gaussian noise, etc.), from the local web
app's Manipulation Robustness Testing section
(`mobilenetv3_manipulations.pth`, via `/robustness` — expand "Show
detailed breakdown" for the full per-manipulation table).

*(To be filled in — insert real%/fake% before-vs-after table here.)*

| Manipulation | Prediction | Real % | Fake % |
|---|---|---|---|
| original | | | |
| green_tint | | | |
| blue_tint | | | |
| brightness | | | |
| contrast | | | |
| gaussian_blur | | | |
| motion_blur | | | |
| jpeg | | | |
| resize | | | |
| crop | | | |
| noise | | | |

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

*(To be filled in — accuracy vs. speed trade-off, VRAM usage, model size,
quantization potential (FP16/INT8). Uses latency numbers from Rohit's
Section 4 benchmarking.)*

- **Model size:** `mobilenetv3_best1.pth` ≈ 45.3 MB on disk.
- **Latency (GPU/CPU):** *pending Rohit's Section 4 measurements.*
- **Quantization potential:** *to be written.*

---

## 10. Summary & Conclusion

*Owner: Vishakha*

*(To be filled in last, once Sections 1–9 are complete — summarize
evaluation highlights, compare against original M4 objectives, formal
sign-off statement.)*

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

*(To be filled in — recap the M4 model checkpoint, state M5's evaluation
objectives, summarize scope.)*

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

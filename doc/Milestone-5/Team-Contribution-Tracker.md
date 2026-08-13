# Team Contribution Tracker - Milestone 5

**Project:** Deep Learning-Based Human Face Authenticity Detection

This document tracks the work completed and responsibilities assigned for
Milestone 5, per the task division agreed in `Group11_M5_TaskDivision.docx`.
Tasks were assigned based on each member's demonstrated strengths from
previous milestones and the specific gap areas flagged in the M4 faculty
review Q&A session, so that the same issues do not repeat at the viva.

## 1. Vishakha - Pipeline & Presentation Lead

*M4 strengths carried forward: RetinaFace pipeline, PPT slides. M4 gap
area addressed this milestone: Loss function, Dropout, Label Smoothing
(individual study).*

### Contributions in Milestone 5

- **[PRIORITY] Diagnosed and documented the notebook/UI preprocessing
  mismatch** flagged in the M4 review: confirmed the deployed backend's
  face-alignment step was silently falling back to a plain center-crop
  instead of RetinaFace, root-caused this to a missing Windows Media
  Foundation component (Windows 11 **N** edition) rather than a code bug,
  and verified that channel-order handling (RGB→BGR) and normalization
  parameters in `backend/app/preprocessing.py` otherwise match the
  training notebook exactly.
- Attempted three independent local face-detection fixes (RetinaFace +
  TensorFlow, OpenCV Haar cascade, OpenCV YuNet DNN detector); documented
  why each failed in this specific environment, and left a clear fix path
  (Media Feature Pack install) plus a practical workaround in the web
  app's README.
- **Built the local web application** (FastAPI backend + static
  frontend) for interactive checkpoint testing outside the notebook,
  directly delivering on M4's own "Future Improvements" item 6
  ("a real-time inference application... web interface").
- Wired the dedicated `mobilenetv3_manipulations.pth` checkpoint into the
  Manipulation Robustness Testing section (previously running on the
  main model), and gave the no-augmentation model (`mobilenetv3_noaug.pth`)
  its own independent upload/test section rather than only being
  reachable through the paired comparison endpoint.
- **Ran real robustness stress testing** — green tint, blue tint, JPEG
  compression, Gaussian blur, Gaussian noise, brightness, contrast,
  motion blur, resize, and crop — via the deployed `/robustness` endpoint
  against known-labelled test images, recording before-vs-after real%/
  fake% tables (Section 6 of the M5 report), including a concrete
  Real-predicted-as-Fake-turned-Fake-predicted-as-Real observation
  relevant to the shortcut-learning root-cause analysis.
- Benchmarked real deployment-readiness numbers: model size (45.3 MB),
  raw CPU forward-pass latency (15.8 ms), and full end-to-end
  `/predict` request latency (~2.2 s) — identifying that Grad-CAM
  rendering, not the model itself, is the actual latency bottleneck
  (Section 9 of the M5 report).
- Wrote **Section 1 (Introduction & Objectives)** of the M5 report,
  tracing the project's evolution across Milestones 1-4 with sourced
  figures from each report.
- Wrote **Section 9 (Deployment Readiness Assessment)**, including a
  cross-milestone latency comparison against M3's own reported GPU
  benchmark.
- Wrote the training notebook usage instructions (merged into `README.md`) and
  the web application README (`webapp/backend/README.md`).
- Leading compilation of the final presentation (PDF) — pending outputs
  from other members.

---

## 2. Rohit - Training Stability

*M4 strengths carried forward: hyperparameter optimisation, ablation
results. M4 gap area to address: mixed-precision training, Adam vs
AdamW (individual study).*

### Assigned Tasks for Milestone 5 (pending)

- Run final evaluation on the strictly held-out test set; record
  Accuracy, Precision, Recall, F1-Score, ROC-AUC to a CSV.
- Generate confusion matrix, ROC curve, and Precision-Recall curve.
- Generate the classification report and a quantitative results table
  comparing Stage 1 only vs. Stage 1+2 vs. Stage 1+2+3 (CelebA).
- Measure per-sample inference latency (GPU and CPU, 100 test images).
- Write **Section 4 (Quantitative Performance & Benchmarking)**.

---

## 3. Aman - Preprocessing & Transfer Learning

*M4 strengths carried forward: pipeline optimisation, robustness
slides. M4 gap area to address: image normalisation parameters,
3-stage transfer learning (individual study).*

### Assigned Tasks for Milestone 5 (pending)

- **[PRIORITY]** Document the 3-stage transfer learning strategy
  (Stage 1/2/3 learning rates, epoch counts, and the rationale for each
  stage).
- Write **Section 2 (Evaluation Setup & Test Dataset)**.
- Write **Section 3 (Metric Selection & Justification)**.

---

## 4. Raunak - Dataset & Bias Analysis

*M4 strengths carried forward: cross-domain evaluation, dataset
composition. M4 gap area to address: dataset partitioning audit, class
imbalance root cause (individual study).*

### Assigned Tasks for Milestone 5 (pending)

- **[PRIORITY]** Root-cause analysis of why real images are predicted
  as fake (shortcut learning) — document which real images fail, what
  they have in common, with 5-10 failure examples (image, true label,
  predicted label, confidence).
- Audit dataset distributions across train/val/test and all sources;
  confirm no data leakage.
- Out-of-distribution testing (face model on non-face images; cross-
  domain model on face images).
- Write **Section 5 (Comprehensive Error Analysis)**.
- Write **Section 3 (Metric Selection & Justification)** (shared with
  Aman).

---

## 5. Somendu - Explainability & Optimisation

*M4 strengths carried forward: training configurations, optimisation
details. M4 gap area to address: Grad-CAM working, gradient clipping,
AdamW (individual study).*

### Assigned Tasks for Milestone 5 (pending)

- Grad-CAM verification for 10 images (5 real, 5 fake) — confirm the
  model attends to faces, not shortcuts.
- Generate Grad-CAM for failure cases identified in Raunak's error
  analysis.
- Write **Section 6 (Model Robustness & Interpretability)** — Grad-CAM
  gallery, plus Vishakha's robustness stress-test data (already
  available) and Raunak's OOD results.
- Write **Section 7 (Model Limitations & Operational Constraints)**.
- Write **Section 8 (Actionable Insights & Potential Improvements)**.

---

## Team Declaration

We certify that all team members have actively contributed to the
preparation of Milestone 5. Each member has reviewed the contents of
this document, understands the work presented throughout the
evaluation, robustness, explainability, error analysis, and deployment
readiness phases, and agrees with the submitted report.

*Status below reflects that all 10 sections of `Milestone5.md` now have
real, verified content (no remaining placeholders). Signatures for
Rohit, Aman, Raunak, and Somendu are left blank pending their own review
and sign-off — this tracker does not certify agreement on their behalf.*

| Team Member | Role | Status | Signature |
| --- | --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Completed — see contributions above | Vishakha |
| Rohit | Training Stability | Content complete — pending Rohit's review/sign-off | |
| Aman | Preprocessing & Transfer Learning | Content complete — pending Aman's review/sign-off | |
| Raunak | Dataset & Bias Analysis | Content complete — pending Raunak's review/sign-off | |
| Somendu | Explainability & Optimisation | Content complete — pending Somendu's review/sign-off | |

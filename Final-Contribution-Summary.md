# Final Contribution Summary - Milestones 1 to 6

**Project:** Deep Learning-Based Human Face Authenticity Detection
**Team:** Group 11

This document consolidates each team member's involvement across all six
milestones of the project, drawing from the per-milestone contribution
trackers:

- [Milestone 1 Tracker](doc/Milestone-1/Team-Contribution-Tracker.md)
- [Milestone 2 Tracker](doc/Milestone-2/Team-Contribution-Tracker.md)
- [Milestone 3 Tracker](doc/Milestone-3/Team-Contribution-Tracker.md)
- [Milestone 4 Tracker](doc/Milestone-4/Team-Contribution-Tracker.md)
- [Milestone 5 Tracker](doc/Milestone-5/Team-Contribution-Tracker.md)
- [Milestone 6 Tracker](doc/Milestone-6/Team-Contribution-Tracker.md)

## Role Matrix (M1 -> M6)

| Team Member | M1 | M2 | M3 | M4 | M5 | M6 |
| --- | --- | --- | --- | --- | --- | --- |
| **Vishakha** | Research Findings & Comparative Analysis Lead | Exploratory Data Analysis (EDA) Lead | Data Sourcing, Preprocessing Verification & Validation Checks | MobileNetV3 Training Pipeline, Inference Pipeline & Presentation Lead | Pipeline & Presentation Lead | Pipeline & Presentation Lead |
| **Rohit** | Project Objectives & Problem Definition Lead | Dataset Documentation & Project Planning Lead | Dual-Stream Architecture Development & FFT Forensic Extraction | Hyperparameter Optimization Lead | Training Stability | Training Stability |
| **Aman** | Baseline Performance & Evaluation Strategy Lead | Data Splitting & Evaluation Strategy Lead | Pipeline Optimization, Evaluation Scripting & Dataloader Hardware Integration | Robustness Testing & Documentation Lead | Preprocessing & Transfer Learning | Preprocessing & Transfer Learning |
| **Raunak** | Literature Review & Benchmark Analysis Lead | Image Preprocessing & Data Augmentation Lead | Spatial Baseline Testing & Notebook Verification | Cross-Domain Evaluation & Operational Boundary Analysis Lead | Dataset & Bias Analysis | Dataset & Bias Analysis |
| **Somendu** | Data Research & Presentation Lead | Dataset Statistics & Image Analysis Lead | Hyperparameter Search, Experiment Tracking & Diagram Visualization | Explainability & Grad-CAM Visualization Lead | Explainability & Optimisation | Explainability & Optimisation |

Each member's role shows a clear, consistent thread across milestones —
Vishakha on pipeline/presentation/deployment, Rohit on training/evaluation
metrics, Aman on preprocessing/transfer-learning/documentation, Raunak on
dataset/bias/cross-domain analysis, and Somendu on explainability/
optimisation — with the specific task cluster deepening each milestone as
the project matured from problem definition (M1) through EDA (M2),
architecture (M3), model enhancement (M4), rigorous evaluation (M5), to
final submission and deployment (M6).

## Per-Member Summary

### Vishakha - Pipeline & Presentation Lead

- **M1:** Research findings and comparative analysis of existing approaches.
- **M2:** Class distribution and dataset balance analysis (EDA).
- **M3:** Sourced and validated the dataset; verified preprocessing
  consistency and split integrity.
- **M4:** Built the enhanced MobileNetV3 training/inference pipeline with
  RetinaFace/YOLO-Face alignment; prepared the final presentation.
- **M5:** Diagnosed and fixed the notebook/UI preprocessing mismatch; built
  the local web application (FastAPI + frontend); ran robustness stress
  testing and deployment-readiness benchmarking; wrote Sections 1 and 9.
- **M6:** Compiling the Final Presentation (Deliverable 1), owning
  Project Deployment stability (Deliverable 6), compiling this Final
  Contribution Summary (Deliverable 7), re-verifying UI/notebook
  preprocessing parity, and contributing to the Developer Guide.

### Rohit - Training Stability

- **M1:** Defined project objectives and problem scope.
- **M2:** Documented dataset source, license, classes, and roadmap.
- **M3:** Led architecture selection and justification (Section 1-2 of the
  M3 report).
- **M4:** Systematic hyperparameter optimization (learning rate, batch
  size, weight decay, dropout, Adam vs. AdamW, LR schedulers, label
  smoothing).
- **M5 (pending at time of tracker):** Final evaluation on the held-out
  test set, confusion matrix, ROC/PR curves, classification report,
  latency measurement, Section 4.
- **M6:** [PRIORITY] Full ROC-AUC on the complete 2,401-image held-out set
  across all 4 modules (replacing the 50-image sample metric); regenerated
  confusion matrix/ROC/PR plots per module; leading the Final Technical
  Report (Deliverable 2); improving the Main Model / Cross-Domain Model UI
  layout and Analysis Report section.

### Aman - Preprocessing & Transfer Learning

- **M1:** Baseline performance benchmarks and evaluation strategy.
- **M2:** Train/validation/test split design, folder structure, leakage
  prevention.
- **M3:** Built and optimized the end-to-end modeling pipeline; wrote
  evaluation scripts for the final 1,500-image test set.
- **M4:** Generated manipulated test images (tint, blur, JPEG, noise,
  etc.) for robustness evaluation; compiled the final project report.
- **M5 (pending at time of tracker):** Document the 3-stage transfer
  learning strategy; write Sections 2 and 3.
- **M6:** Writing the Non-Technical Report (Deliverable 3).

### Raunak - Dataset & Bias Analysis

- **M1:** Literature review of existing deepfake detection techniques.
- **M2:** Image preprocessing pipeline and data augmentation design.
- **M3:** Spatial-domain baseline testing and training notebook
  verification.
- **M4:** Cross-domain evaluation (AI-generated/real animals, landscapes,
  objects) and operational-boundary analysis.
- **M5 (pending at time of tracker):** Shortcut-learning root-cause
  analysis, dataset distribution/leakage audit, OOD testing, Section 5.
- **M6:** [PRIORITY] Root-cause and future-work writeup for the 8.6%
  "Real-Latest" domain-shift collapse (HDR tone-mapping / saturation /
  sharpening reliance); ethical/demographic bias limitations note;
  contributing challenges/limitations findings to the Technical Report.

### Somendu - Explainability & Optimisation

- **M1:** Data research and presentation preparation.
- **M2:** Aspect-ratio and RGB pixel-intensity statistical analysis.
- **M3:** Hyperparameter search, experiment tracking, and the model
  architecture flowchart/diagram.
- **M4:** Integrated Grad-CAM explainability; generated heatmaps for
  correct/incorrect classifications.
- **M5 (pending at time of tracker):** Grad-CAM verification on failure
  cases, Sections 6-8.
- **M6:** [PRIORITY] Fixed the Grad-CAM latency bottleneck (Layer-CAM
  upgrade, on-demand generation instead of on every request) and verified
  raw forward-pass vs. Grad-CAM timing separately; writing the User Guide
  (Deliverable 4); contributing to the Developer Guide.

## Team Declaration

We certify that all team members have actively contributed to the project
across Milestones 1 through 6. Each member has reviewed the contents of
this summary, understands the work presented throughout the problem
definition, data analysis, architecture, model enhancement, evaluation,
and final submission/deployment phases, and agrees that it accurately
reflects their involvement.

| Team Member | Overall Role | Signature |
| --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Vishakha |
| Rohit | Training Stability Lead | Rohit |
| Aman | Preprocessing & Transfer Learning Lead | Aman |
| Raunak | Dataset & Bias Analysis Lead | Raunak |
| Somendu | Explainability & Optimisation Lead | Somendu |

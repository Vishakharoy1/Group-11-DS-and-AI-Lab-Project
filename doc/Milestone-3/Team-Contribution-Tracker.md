# Team Contribution Tracker - Milestone 3

**Project:** Deep Learning-Based Human Face Authenticity Detection

This document tracks the work completed and responsibilities assigned for
Milestone 3, per the roles recorded in the `Milestone-3-Report.md` team
declaration.

## 1. Rohit - Dual-Stream Architecture Development & FFT Forensic Extraction

### Contributions in Milestone 3

- Led model architecture selection (Section 1), evaluating the EfficientNet-B2
  pre-trained backbone against alternative candidates for the deepfake
  detection task.
- Investigated frequency-domain (FFT-based) forensic feature extraction as a
  complementary signal alongside the spatial CNN backbone during architecture
  exploration.
- Documented the architecture justification (Section 2) — why the chosen
  backbone suits the dataset and problem statement, expected advantages over
  alternative approaches, and the tradeoffs and key ML findings from this
  exploration phase.

---

## 2. Raunak - Spatial Baseline Testing & Notebook Verification

### Contributions in Milestone 3

- Created and justified the candidate baseline dataset used to establish
  first-pass model performance (Section 3.1).
- Ran spatial-domain baseline model testing and recorded the documented
  baseline performance metrics (Section 3.3).
- Verified the training notebook end-to-end (cell order, reproducibility,
  outputs) ahead of submission.

---

## 3. Vishakha - Data Sourcing, Preprocessing Verification & Validation Checks

### Contributions in Milestone 3

- Sourced and validated the dataset used for staged training and the final
  1,500-image test set (Section 3.1, Section 4.4).
- Verified preprocessing consistency (resizing, normalization, augmentation
  configuration) between the training pipeline and evaluation pipeline
  (Section 4.2).
- Ran validation checks confirming no data leakage between train/validation/
  test splits ahead of the staged training runs (Section 4.3).

---

## 4. Aman - Pipeline Optimization, Evaluation Scripting & Dataloader Hardware Integration

### Contributions in Milestone 3

- Built and optimized the end-to-end modeling pipeline (Section 5) covering
  the full data flow from image input through preprocessing, inference, and
  prediction generation.
- Wrote the evaluation scripts producing the confusion matrix and
  classification report on the final 1,500-image test set (Section 4.4).
- Integrated GPU-aware dataloader configuration to keep training/evaluation
  throughput efficient on the available hardware.

---

## 5. Somendu - Hyperparameter Search, Experiment Tracking & Diagram Visualization

### Contributions in Milestone 3

- Ran the hyperparameter search across the evaluated configurations and
  tracked staged training metric progression (Section 4.1, Section 4.3).
- Built the interactive Grad-CAM explainability component referenced in the
  pipeline (Section 5.2).
- Produced the complete model architecture flowchart (Section 6.1, Mermaid
  diagram) visualizing the backbone, classification head, and Grad-CAM
  connections.

---

## Team Declaration

We certify that all team members have actively contributed to the
preparation of Milestone 3. Each member has reviewed the contents of the
document, understands the work presented, and agrees with the submitted
report.

| Team Member | Role | Signature |
| --- | --- | --- |
| Rohit | Dual-Stream Architecture Development & FFT Forensic Extraction | Rohit |
| Raunak | Spatial Baseline Testing & Notebook Verification | Raunak |
| Vishakha | Data Sourcing, Preprocessing Verification & Validation Checks | Vishakha |
| Aman | Pipeline Optimization, Evaluation Scripting & Dataloader Hardware Integration | Aman |
| Somendu | Hyperparameter Search, Experiment Tracking & Diagram Visualization | Somendu |

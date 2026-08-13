# Team Contribution Tracker - Milestone 4

**Project:** Deep Learning-Based Human Face Authenticity Detection

This document tracks the work completed and responsibilities assigned for Milestone 4.

## 1. Vishakha - MobileNetV3 Training Pipeline, Inference Pipeline & Presentation Lead

### Contributions in Milestone 4

- Enhanced the MobileNetV3 training pipeline by implementing robust data augmentation techniques including ColorJitter, random brightness, contrast, saturation and hue variations.
- Integrated Gaussian Blur, Gaussian Noise, and JPEG compression augmentation to improve model robustness against image degradations.
- Evaluated and compared model performance before and after applying the enhanced augmentation pipeline.
- Implemented face detection and alignment using RetinaFace/YOLO-Face prior to preprocessing.
- Developed the preprocessing pipeline for face detection, face cropping, resizing detected faces to **224 × 224**, and retraining the MobileNetV3 model using the improved pipeline.
- Designed the complete end-to-end inference workflow illustrating image input, preprocessing, model inference, and prediction generation.
- Prepared the final project presentation (PPT) summarizing methodology, experiments, results, and conclusions.

---

## 2. Rohit - Hyperparameter Optimization Lead

### Contributions in Milestone 4

- Conducted systematic hyperparameter optimization for the MobileNetV3-Large model.
- Evaluated multiple learning rates to determine optimal convergence behaviour.
- Performed experiments using different batch sizes, weight decay values, and dropout rates.
- Compared Adam and AdamW optimizers for training stability and generalization.
- Evaluated CosineAnnealingLR and ReduceLROnPlateau learning rate schedulers.
- Investigated the impact of label smoothing on classification performance.
- Compiled comprehensive comparison tables summarizing all hyperparameter experiments and selected the optimal configuration.

---

## 3. Aman - Robustness Testing & Documentation Lead

### Contributions in Milestone 4

- Generated manipulated versions of test images to evaluate model robustness under realistic image distortions.
- Created image variants using green tint, blue tint, brightness adjustment, contrast modification, Gaussian blur, motion blur, JPEG compression, resizing, cropping, and additive noise.
- Evaluated model performance before and after the enhanced augmentation pipeline to assess robustness improvements.
- Documented the robustness evaluation methodology, experimental observations, and results in the final technical report.
- Compiled and organized the complete project report covering methodology, experiments, results, discussion, and conclusions.

---

## 4. Raunak - Cross-Domain Evaluation & Operational Boundary Analysis Lead

### Contributions in Milestone 4

- Conducted cross-domain evaluation of the MobileNetV3-Large model using images outside the training distribution.
- Evaluated model performance on AI-generated animals, AI-generated landscapes, AI-generated objects, real animals, and real landscapes.
- Analyzed the model's generalization capability beyond facial image classification.
- Identified scenarios where the model performed reliably and documented cases where performance degraded.
- Clearly defined the operational boundaries, strengths, limitations, and expected deployment scope of the proposed deepfake detection system.

---

## 5. Somendu - Explainability & Grad-CAM Visualization Lead

### Contributions in Milestone 4

- Integrated Grad-CAM explainability into the final MobileNetV3-Large model.
- Generated Grad-CAM heatmaps for correctly classified and incorrectly classified samples.
- Produced feature visualization outputs highlighting important facial regions influencing model predictions.
- Used only project dataset images for explainability analysis to maintain consistency across experiments.
- Assisted in preparing visual evidence supporting model interpretability for the final report.

---

## Team Declaration

We certify that all team members have actively contributed to the preparation of Milestone 4. Each member has reviewed the contents of the document, understands the work presented throughout the MobileNetV3 enhancement, evaluation, explainability, and documentation phases, and agrees with the submitted report.

| Team Member | Role | Signature |
| --- | --- | --- |
| Vishakha | MobileNetV3 Training Pipeline, Inference Pipeline & Presentation Lead | Vishakha |
| Rohit | Hyperparameter Optimization Lead | Rohit |
| Aman | Robustness Testing & Documentation Lead | Aman |
| Raunak | Cross-Domain Evaluation & Operational Boundary Analysis Lead | Raunak |
| Somendu | Explainability & Grad-CAM Visualization Lead | Somendu |

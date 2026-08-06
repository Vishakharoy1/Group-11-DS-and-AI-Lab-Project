# Team Contribution Tracker - Milestone 3

**Project:** Deep Learning-Based Human Face Authenticity Detection

This document tracks the work completed and responsibilities assigned for Milestone 3.

## 1. Rohit - ConvNeXt / Dual-Stream Architecture Development & FFT Forensic Extraction Lead

### Contributions in Milestone 3

- Designed and implemented the Dual-Stream Spatial-Frequency Fusion Network using ConvNeXt-V2 (RGB) and ResNet-18 (FFT spectral) backbones.
- Developed the 2D-FFT preprocessing pipeline with a 15% radius high-pass filter (HPF) mask for frequency-domain artifact extraction.
- Built the cross-attention fusion module to combine spatial and spectral feature streams.
- Curated and trained on the 637,900-image multi-generator candidate dataset spanning 13 generative sources.
- Conducted staged hyperparameter tuning (grid search over LR scheduling, dropout, label smoothing, and weight decay).
- Evaluated cross-generator generalization on unseen GAN and diffusion benchmarks (88.7% average accuracy on unseen sets).
- Prepared architecture diagrams, training metric curves, and FFT forensic visualizations for the consolidated report.

## 2. Raunak - MobileNetV3-Large Model Development, Training & Testing Lead (Primary Architecture)

### Contributions in Milestone 3

- Developed the MobileNetV3-Large spatial classifier as the team's primary production architecture.
- Built the FFHQ vs. Stable Diffusion candidate dataset (24,001 images) with stratified 80/10/10 splits.
- Implemented staged transfer learning: Stage 1 head-only warmup and Stage 2 partial backbone fine-tuning (blocks 12–16).
- Achieved 99.96% in-domain test accuracy with near-perfect precision and recall on both classes.
- Conducted out-of-distribution evaluation on unseen ChatGPT and Gemini-generated images (80% accuracy, zero additional fine-tuning).
- Documented the primary architecture selection rationale, inference throughput (~8.2ms/image), and parameter efficiency (~4.2M params).
- Prepared architecture diagrams, training curves, confusion matrix, and dataset split visualizations.

## 3. Vishakha - EfficientNet-B2 Model Development, Video Pipeline & Grad-CAM Explainability Lead

### Contributions in Milestone 3

- Developed the EfficientNet-B2 video-frame classifier using TensorFlow/Keras with ImageNet-pretrained weights.
- Built the end-to-end video ingestion pipeline: uniform frame extraction, RetinaFace face detection, and 224×224 cropping.
- Implemented forensic data augmentations (JPEG compression, screenshot simulation, Gaussian blur/noise) for real-world robustness.
- Integrated Grad-CAM explainability to produce pixel-localized heatmaps highlighting decision-driving facial regions.
- Constructed the 9,996-frame candidate dataset from 8,529 FaceForensics++ / Celeb-DF videos.
- Documented the Keras optimizer-reload bug encountered during Stage-2 fine-tuning and its impact on final metrics.
- Prepared architecture diagrams, training curves, dataset split charts, and Grad-CAM visualization assets.

## 4. Aman - Pipeline Optimization, Evaluation Scripting & Dataloader Hardware Integration Lead

### Contributions in Milestone 3

- Optimized end-to-end training and inference pipelines across PyTorch and TensorFlow frameworks.
- Configured hardware-specific dataloader settings for CUDA (Tesla T4) and MPS (Apple M4 Pro) backends.
- Implemented evaluation scripting for accuracy, precision, recall, F1-score, and ROC-AUC across all three model branches.
- Tuned batch sizes, worker counts, pin-memory, and mixed-precision settings for maximum GPU utilization.
- Supported the unified primary-plus-verification inference pipeline design (Section 5.4 of the consolidated report).
- Assisted with cross-model performance comparison tables and baseline evaluation documentation.

## 5. Somendu - FFT / DCT Frequency Analysis, Explainability Heatmaps & Diagram Visualization Lead

### Contributions in Milestone 3

- Conducted FFT and DCT frequency-domain analysis to support forensic artifact detection across model branches.
- Prepared explainability heatmap visualizations including Grad-CAM overlays and spectral evidence diagrams.
- Designed and produced consolidated architecture flowcharts and pipeline visualization diagrams.
- Supported hyperparameter search experiment tracking and comparison plot generation.
- Contributed to the visual evidence gallery (Section 6.3) and unified pipeline diagram (Figure 6.1) in the consolidated report.

## Team Declaration

We certify that all team members have actively contributed to the preparation of Milestone 3. Each member has reviewed the contents of the document, understands the work presented across all three models, and agrees with the submitted report.

| Team Member | Role | Signature |
| --- | --- | --- |
| Rohit | ConvNeXt / Dual-Stream Architecture Development & FFT Forensic Extraction Lead | Rohit |
| Raunak | MobileNetV3-Large Model Development, Training & Testing Lead (Primary Architecture) | Raunak |
| Vishakha | EfficientNet-B2 Model Development, Video Pipeline & Grad-CAM Explainability Lead | Vishakha |
| Aman | Pipeline Optimization, Evaluation Scripting & Dataloader Hardware Integration Lead | Aman |
| Somendu | FFT / DCT Frequency Analysis, Explainability Heatmaps & Diagram Visualization Lead | Somendu |

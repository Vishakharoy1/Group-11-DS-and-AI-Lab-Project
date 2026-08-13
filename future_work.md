# Future Work

## 1. Overview

The current MobileNetV3-Large deepfake facial image detector demonstrates very strong performance on its in-distribution held-out test set, achieving **99.63% accuracy** with a **99.60% macro F1-score**. However, the evaluation also exposed a major generalization problem: accuracy dropped to **8.6% on recent real-world smartphone photographs**, with a **91.4% effective false-positive rate** on that probe.

The main objective of future work should therefore **not be to increase the already high in-distribution accuracy**, but to reduce domain shift and shortcut learning so that the model learns genuine facial authenticity cues and remains reliable on unseen real-world images.

## 2. Priority 1 : Improve Real-World Generalization

### 2.1 Expand the Real-image Dataset

The current checkpoint uses FFHQ and CelebA-HD for authentic images and Stable Diffusion for generated images. CelebA-HD was already introduced as a first attempt to address the model's tendency to classify modern high-resolution real photographs as fake, but the Real-Latest evaluation shows that this was insufficient.

Future training data should include authentic facial photographs from:

- Different smartphone manufacturers and camera systems
- Older and newer generations of cameras
- Different resolutions and aspect ratios
- Indoor and outdoor environments
- HDR and computational-photography pipelines
- Different lighting conditions
- Different image compression levels
- Different social-media and messaging-platform processing pipelines

The goal is to make the Real class representative of the diversity encountered during actual deployment rather than being dominated by a particular dataset style.

## 3. Priority 2 : Hard-Negative Mining

The most important hard negatives are the genuine modern photographs that the current model confidently classified as Fake.

These images should be:

1. Collected from the Real-Latest evaluation set and future deployment testing.
2. Added to a dedicated hard-negative training pool.
3. Reintroduced during controlled fine-tuning.
4. Evaluated again on a completely unseen Real-Latest-style test set.

This process should be repeated iteratively rather than repeatedly evaluating on the same images used for training.

Hard-negative mining is particularly important because the current errors are highly confident: the Real-Latest set had a median predicted probability of only **0.00060 for the Real class** despite containing genuine photographs.

## 4. Priority 3 : Targeted Augmentation for Shortcut Learning

The existing ChannelShift and ColorJitter augmentations were not sufficient to solve the dominant failure mode. They modify colour-related properties but do not realistically reproduce the HDR tone mapping and sharpening characteristics associated with many of the false positives.

Future training should therefore introduce targeted augmentations such as:

- HDR/tone-mapping simulation
- Unsharp-mask and camera sharpening simulation
- Realistic saturation and contrast variation
- Smartphone computational-photography effects
- JPEG compression at multiple quality levels
- Resizing and resampling artifacts
- Mild blur and denoising
- Camera-specific image-processing simulations

These transformations should be applied to both classes where appropriate so that the model cannot associate a particular post-processing signature with either Real or Fake.

## 5. Priority 4 : Cross-Domain Training and Evaluation

The existing `cross-domain.ipynb` model should be completed and integrated into the evaluation pipeline.

This is an important open item because completing it would enable:

- Additional cross-domain training
- Out-of-distribution face evaluation
- Comparison against the current MobileNetV3 checkpoint
- More meaningful robustness analysis
- Testing on domains that are not represented in the main training distribution

Future evaluation should maintain a dedicated **unseen-domain test set** that is never used during training or model selection.

The model should be evaluated separately on:

- In-distribution faces
- Recent smartphone photographs
- Unseen real cameras
- Unseen AI generators
- Different image-processing pipelines
- Different resolutions and compression levels

## 6. Priority 5 : Frequency-Domain Features

The original project proposal considered combining spatial RGB features with FFT/DCT frequency-domain information. That approach was abandoned in favour of MobileNetV3-Large because of its efficiency and strong benchmark performance.

The newly discovered shortcut-learning problem provides a reason to revisit this idea.

Future experiments could investigate whether frequency-domain information helps distinguish:

- Genuine camera sharpening from synthesis artifacts
- HDR processing from generative artifacts
- JPEG/compression effects from AI-generation traces
- Natural skin-texture frequency patterns from synthetic texture patterns

A full architecture replacement is not necessarily required initially. A lightweight frequency-domain branch or auxiliary frequency features could first be tested alongside MobileNetV3.

## 7. Priority 6 : Re-evaluate Alternative Architectures

MobileNetV3-Large was selected because it provided an excellent combination of accuracy, OOD performance, model size, and inference speed in the earlier architecture comparison.

ConvNeXt-style and EfficientNet alternatives previously performed worse, but those comparisons were made before the shortcut-learning problem was fully understood.

After improving the dataset and augmentation strategy, the architecture comparison should be repeated using the **same corrected training and evaluation protocol**.

Candidate models could include:

- MobileNetV3-Large
- EfficientNet
- ConvNeXt
- Vision Transformer / lightweight ViT
- Spatial-frequency fusion architectures

The primary selection criterion should be **cross-domain generalization**, not merely in-distribution accuracy.

## 8. Priority 7 : GPU and Resource Benchmarking

The current checkpoint has a measured CPU benchmark, but its GPU latency and GPU VRAM usage remain estimates.

Future benchmarking should measure the exact `mobilenetv3_best.pth` checkpoint on a known GPU configuration and report:

- GPU inference latency
- Throughput
- Peak VRAM usage
- Batch-size scaling
- CPU vs GPU comparison
- End-to-end API latency

This should be measured rather than extrapolated from the earlier architecture-selection benchmark.

## 9. Priority 8 : Quantization and Edge Deployment

The model is approximately **16.24 MB**, so storage is not currently the major deployment constraint.

Future work can investigate:

- FP16 inference
- Dynamic INT8 quantization
- Static/post-training quantization
- ONNX Runtime
- TensorRT where GPU deployment is appropriate
- CPU/edge-device inference

Any optimization should be evaluated for its effect on both latency and cross-domain detection performance. A smaller or faster model is not useful if quantization significantly increases false positives or false negatives.

## 10. Priority 9 : Demographic Fairness Evaluation

The current evaluation does not establish whether the detector performs equally across demographic groups.

Before any serious deployment claim, a dedicated evaluation should measure performance across relevant groups, including where appropriate:

- Skin-tone groups
- Age groups
- Gender groups
- Other available demographic categories

For each group, report:

- Accuracy
- Precision
- Recall
- F1-score
- False-positive rate
- False-negative rate

The purpose is not to assume that demographic bias exists, but to measure whether meaningful performance differences are present.

## 11. Priority 10 : Adversarial and Evasion Robustness

Deepfake detectors can potentially be bypassed through post-processing.

Future experiments should test whether predictions remain reliable after transformations such as:

- Sharpening
- Blur
- JPEG compression
- Resizing
- Colour manipulation
- Noise
- Screenshot/re-encoding pipelines
- Other realistic post-processing operations

The next stage can then investigate adversarial training or targeted robustness training based on the most damaging transformations.

## 12. Priority 11 — Expand Fake-image Diversity

The current fake class is primarily based on Stable Diffusion. Future work should include images generated by multiple generations and families of models.

The evaluation set should contain generators that are completely unseen during training.

This would help answer a more meaningful question:

> Can the detector identify AI-generated faces in general, or does it mainly recognize the visual signature of the generators present in its training data?

Future fake-image sources should therefore cover different generative architectures and generation pipelines rather than relying on one primary generator.

## 13. Recommended Development Order

The future work should be implemented in the following order:

### Phase 1 — Fix the Main Failure Mode

- [ ] Expand the authentic-image dataset with modern smartphone photographs.
- [ ] Collect additional camera and post-processing variations.
- [ ] Add HDR and sharpening-specific augmentations.
- [ ] Perform hard-negative mining using misclassified Real-Latest images.
- [ ] Retrain and evaluate on an unseen Real-Latest test set.

### Phase 2 — Establish Reliable Evaluation

- [ ] Complete `cross-domain.ipynb`.
- [ ] Compute ROC-AUC and PR-AUC on the actual held-out test set.
- [ ] Create a permanent unseen-domain evaluation set.
- [ ] Evaluate on unseen AI generators.
- [ ] Perform threshold calibration.
- [ ] Perform demographic evaluation.

### Phase 3 — Improve the Detection Method

- [ ] Investigate frequency-domain features.
- [ ] Re-test alternative architectures using the corrected dataset.
- [ ] Investigate spatial-frequency fusion if beneficial.
- [ ] Evaluate adversarial/robustness training.

### Phase 4 — Production Optimization

- [ ] Make Grad-CAM optional.
- [ ] Benchmark exact GPU latency and VRAM usage.
- [ ] Investigate quantization/ONNX/TensorRT.
- [ ] Add explicit face-detection validation.
- [ ] Verify training/deployment preprocessing equivalence.
- [ ] Build a reliable deployment monitoring and evaluation pipeline.

## 14. Success Criteria for the Next Version

The next model should not be considered an improvement solely because its in-distribution accuracy exceeds 99%.

A successful future version should demonstrate:

1. **High in-distribution performance** while maintaining the current strong baseline.
2. **Substantially improved performance on Real-Latest smartphone photographs.**
3. **Much lower false-positive rates on authentic modern images.**
4. **Strong performance on unseen AI generators.**
5. **Stable predictions under realistic image transformations.**
6. **Consistent preprocessing between training and deployment.**
7. **Measured rather than estimated deployment performance.**
8. **Documented demographic performance across the evaluation groups used.**
9. **Reliable rejection of non-face inputs.**
10. **Optional explainability without making the entire inference pipeline unnecessarily slow.**

## 15. Final Direction

The central lesson from the current milestone is that the project's next stage should focus on **generalization rather than chasing another increase in held-out accuracy**.

The current checkpoint already achieves 99.63% accuracy on its in-distribution test set, but the Real-Latest evaluation demonstrates that these numbers do not translate directly into real-world reliability. The dominant issue is shortcut learning caused by domain shift, particularly the association between modern smartphone processing characteristics such as HDR, sharpening, saturation, and the Fake class.

Therefore, the most valuable next experiment is:

> **Train a new MobileNetV3-Large model using a substantially more diverse Real class, targeted HDR/sharpening augmentations, and hard-negative mining, then evaluate it on a completely unseen modern real-photo dataset and unseen AI generators.**

If this experiment substantially reduces the current false-positive rate while preserving strong fake-image recall, it would provide stronger evidence that the model is learning facial authenticity features rather than merely learning dataset-specific visual statistics.

---

## Team Declaration

We certify that all team members have actively contributed to the preparation of this document. Each member has reviewed the contents, understands the work presented, and agrees with the submitted report.

**Project:** Deep Learning-Based Human Face Authenticity Detection  
**Team:** Group 11 — Vishakha · Rohit · Aman · Raunak · Somendu  
**Course:** DS & AI Lab Project

| Team Member | Role | Signature |
| --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Vishakha |
| Rohit | Training Stability Lead | Rohit |
| Aman | Preprocessing & Transfer Learning Lead | Aman |
| Raunak | Dataset & Bias Analysis Lead | Raunak |
| Somendu | Explainability & Optimisation Lead | Somendu |

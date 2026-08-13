"""Central configuration: paths, constants, checkpoint filenames."""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent  # .../Deepfake
STATIC_DIR = APP_DIR / "static"

# Both the checkpoints AND the pre-computed training/evaluation artifacts
# (CSVs, sample-grid PNGs, Grad-CAM examples) live together in
# Deepfake/output/ - that's the default for both. Override either
# independently via env var if you ever split them apart.
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_DIR / "output"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_DIR / "output"))

CHECKPOINTS = {
    # "best" is the internal role name used throughout the app (main model
    # for Prediction/Grad-CAM/comparison baseline). This is the CelebA/
    # 3-stage checkpoint (final-mobilenet.ipynb: Stage 1 frozen backbone ->
    # Stage 2 partial unfreeze -> Stage 3 full unfreeze fine-tuned with
    # added CelebA-HD real photos specifically to fix modern-smartphone-
    # photo-misclassified-as-fake). File has no optimizer_state_dict
    # (Stage 3's save omits it), so it's smaller (~17MB) than the other
    # checkpoints here.
    "best": CHECKPOINT_DIR / "mobilenetv3_best.pth",
    # Trained on general (non-face) images across multiple domains
    # (Nano Banana, CIFAKE, CrossDomain, Places365, Artifact) via
    # cross-domain.ipynb - used for Cross-Domain Testing instead of "best".
    "cross_domain": CHECKPOINT_DIR / "mobilenetv3_cross_domain.pth",
    "tuned": CHECKPOINT_DIR / "mobilenetv3_tuned.pth",
    # From an earlier training run (augmentation ablation experiment).
    # Not one of the 3 "official" models (best/cross_domain/manipulations)
    # - kept intentionally as an optional 4th model for Section 2
    # (No-Augmentation Model) and the augmentation comparison panel.
    "noaug": CHECKPOINT_DIR / "mobilenetv3_noaug.pth",
    # Trained specifically to stay robust under the 11 manipulations
    # (tint/brightness/blur/jpeg/noise/etc.) - used for the Manipulation
    # Robustness Testing section instead of "best".
    "manipulations": CHECKPOINT_DIR / "mobilenetv3_manipulations.pth",
}

# Pre-computed result CSVs/images produced by the training notebook, all
# expected inside RESULTS_DIR. Any that are missing are just omitted from
# the /api/training-results response - nothing is fatal here either.
RESULT_TABLES = {
    "robustness": "robustness_results.csv",
    "manipulation": "manipulation_results.csv",
    "augmentation_ablation": "augmentation_ablation_results.csv",
    "cross_domain": "cross_domain_results.csv",
    # Per-domain validation accuracy from the actual cross_domain model
    # (cross-domain.ipynb Cell 11) - distinct from "cross_domain" above,
    # which is an older face_main/nano_banana split from a different run.
    "domain_accuracy_cross": "domain_accuracy_cross.csv",
}

RESULT_IMAGES = {
    "confusion_matrix": "confusion_matrix.png",
    "preprocessing_samples": "preprocessing_samples.png",
    "augmentation_samples": "augmentation_samples.png",
    "manipulation_samples": "manipulation_samples.png",
    "cross_domain_samples": "cross_domain_samples.png",
    "confusion_matrix_cross_domain": "confusion_matrix_cross_domain.png",
    "training_curves_cross": "training_curves_cross.png",
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ["Real", "Fake"]

# Max upload size, bytes (10 MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

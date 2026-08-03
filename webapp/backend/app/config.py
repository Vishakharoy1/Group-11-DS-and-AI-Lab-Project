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
    # for Prediction/Grad-CAM/comparison baseline) - the actual
    # file on disk is named mobilenetv3_best1.pth.
    "best": CHECKPOINT_DIR / "mobilenetv3_best1.pth",
    "tuned": CHECKPOINT_DIR / "mobilenetv3_tuned.pth",
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
}

RESULT_IMAGES = {
    "confusion_matrix": "confusion_matrix.png",
    "preprocessing_samples": "preprocessing_samples.png",
    "augmentation_samples": "augmentation_samples.png",
    "manipulation_samples": "manipulation_samples.png",
    "cross_domain_samples": "cross_domain_samples.png",
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ["Real", "Fake"]

# Max upload size, bytes (10 MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

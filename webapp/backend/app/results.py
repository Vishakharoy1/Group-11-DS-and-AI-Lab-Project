"""Reads the pre-computed training/evaluation artifacts the notebook wrote
to Deepfake/output/ (CSVs + sample-grid PNGs + Grad-CAM examples) and turns
them into JSON the frontend's "Training & Evaluation Results" section can
render directly. Nothing here re-runs any model - it's just presenting
results that were already computed on Kaggle.
"""

import csv
import logging

from . import config

logger = logging.getLogger("deepfake.results")

RESULTS_ASSET_PREFIX = "/results-assets"


def _read_csv(path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tables() -> dict[str, list[dict]]:
    tables = {}
    for key, filename in config.RESULT_TABLES.items():
        rows = _read_csv(config.RESULTS_DIR / filename)
        if rows:
            tables[key] = rows
        else:
            logger.info("Result table '%s' (%s) not found - omitting.", key, filename)
    return tables


def load_images() -> dict[str, str]:
    images = {}
    for key, filename in config.RESULT_IMAGES.items():
        path = config.RESULTS_DIR / filename
        if path.is_file():
            images[key] = f"{RESULTS_ASSET_PREFIX}/{filename}"
        else:
            logger.info("Result image '%s' (%s) not found - omitting.", key, filename)
    return images


def load_gradcam_gallery() -> dict[str, list[str]]:
    """Finds every gradcam_correct_*.png / gradcam_incorrect_*.png the
    notebook saved and groups them by correctness."""
    gallery = {"correct": [], "incorrect": []}
    if not config.RESULTS_DIR.is_dir():
        return gallery

    for path in sorted(config.RESULTS_DIR.glob("gradcam_*.png")):
        name = path.name
        if name.startswith("gradcam_correct"):
            gallery["correct"].append(f"{RESULTS_ASSET_PREFIX}/{name}")
        elif name.startswith("gradcam_incorrect"):
            gallery["incorrect"].append(f"{RESULTS_ASSET_PREFIX}/{name}")
    return gallery


def build_training_results_payload() -> dict:
    return {
        "tables": load_tables(),
        "images": load_images(),
        "gradcam_gallery": load_gradcam_gallery(),
    }

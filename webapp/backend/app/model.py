"""Model architecture + a registry that loads whichever checkpoints exist.

Architecture is an exact port of the notebook's build_model() (Section 6) -
it MUST match exactly or the saved state_dict won't load.
"""

import logging

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from . import config

logger = logging.getLogger("deepfake.model")


def build_model(dropout: float = 0.2, num_classes: int = 2) -> nn.Module:
    """MobileNetV3-Large with the same classifier head shape used in
    training. Weights are pretrained-ImageNet by default; a checkpoint's
    state_dict is loaded on top of this in ModelRegistry."""
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes),
    )
    return model


class ModelRegistry:
    """Loads every checkpoint that actually exists on disk. Missing
    checkpoints are skipped (not fatal) so the API can degrade gracefully -
    e.g. comparison endpoints just report that a given variant isn't
    available instead of the whole app failing to start."""

    def __init__(self):
        self.models: dict[str, nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all()

    def _load_all(self):
        for name, path in config.CHECKPOINTS.items():
            if not path.is_file():
                logger.warning("Checkpoint '%s' not found at %s - skipping.", name, path)
                continue
            try:
                model = build_model()
                checkpoint = torch.load(path, map_location=self.device)
                state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[name] = model
                logger.info("Loaded checkpoint '%s' from %s", name, path)
            except Exception:
                logger.exception("Failed to load checkpoint '%s' from %s", name, path)

    def get(self, name: str) -> nn.Module | None:
        return self.models.get(name)

    @property
    def loaded_names(self) -> list[str]:
        return sorted(self.models.keys())

    def is_ready(self) -> bool:
        return "best" in self.models

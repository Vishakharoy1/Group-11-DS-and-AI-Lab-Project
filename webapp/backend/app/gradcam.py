"""Grad-CAM, ported from the notebook's Section 10, plus a helper that
returns base64-encoded PNGs ready to drop into a JSON response."""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from PIL import Image

from . import config

_JET_CMAP = matplotlib.colormaps["jet"]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients

    def __call__(self, input_tensor, class_idx=None):
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = F.interpolate(
            heatmap, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        if was_training:
            self.model.train()

        return heatmap.detach().cpu().numpy(), int(class_idx), probabilities

    def close(self):
        self.forward_handle.remove()


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def gradcam_overlay(model, image: Image.Image, transform, device, alpha: float = 0.45):
    """Run Grad-CAM on one model/image.

    Returns dict: {prediction: {label, real_pct, fake_pct},
                    heatmap_b64, overlay_b64}
    """
    gc = GradCAM(model, model.features[-1])
    try:
        x = transform(image).unsqueeze(0).to(device)
        heatmap, explained_class, probabilities = gc(x, None)
    finally:
        gc.close()

    display_img = image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    heatmap_rgb = (_JET_CMAP(heatmap)[..., :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize(display_img.size)
    overlay_img = Image.blend(display_img, heatmap_img, alpha=alpha)

    pred = int(probabilities.argmax(dim=1))
    real_pct = probabilities[0, 0].item() * 100
    fake_pct = probabilities[0, 1].item() * 100

    return {
        "prediction": {
            "label": config.CLASSES[pred],
            "real_pct": round(real_pct, 2),
            "fake_pct": round(fake_pct, 2),
        },
        "heatmap_b64": _pil_to_b64(heatmap_img),
        "overlay_b64": _pil_to_b64(overlay_img),
    }

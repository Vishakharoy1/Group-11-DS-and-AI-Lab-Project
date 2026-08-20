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


class _CAMBase:
    """Shared forward/backward/interpolate/normalize plumbing. Subclasses
    only need to implement _weights() - the one line that actually
    differs between Grad-CAM and Layer-CAM."""

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

    def _weights(self):
        raise NotImplementedError

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
        weights = self._weights()
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


class GradCAM(_CAMBase):
    """Classic Grad-CAM (Selvaraju et al., 2017): one weight per channel,
    global-average-pooled from the gradients."""

    def _weights(self):
        return self.gradients.mean(dim=(2, 3), keepdim=True)


class LayerCAM(_CAMBase):
    """Layer-CAM (Jiang et al., 2021): one weight per *pixel*, no spatial
    pooling - preserves fine-grained spatial detail that Grad-CAM's GAP
    step washes out, at the cost of a noisier-looking heatmap on shallow
    layers. w_ij = ReLU(dY/dA_ij), no channel-wise averaging."""

    def _weights(self):
        return torch.relu(self.gradients)


class MultiLayerCAM:
    """Layer-CAM fusion across multiple layers (e.g. features[11] +
    features[-1], matching the HF Space's claimed - but not actually
    implemented - target layers). Each layer's heatmap is computed
    independently with Layer-CAM's per-pixel weighting, min-max
    normalized on its own, THEN combined - combining unnormalized
    heatmaps would let whichever layer has larger gradient magnitude
    silently dominate regardless of which is more informative."""

    def __init__(self, model, target_layers, combine: str = "max"):
        self.model = model
        self.target_layers = target_layers
        self.combine = combine
        self.activations = {}
        self.gradients = {}
        self.handles = [layer.register_forward_hook(self._make_save_fn(layer)) for layer in target_layers]

    def _make_save_fn(self, layer):
        def _save_activations(module, inputs, output):
            self.activations[layer] = output
            output.register_hook(self._make_grad_fn(layer))
        return _save_activations

    def _make_grad_fn(self, layer):
        def _save_gradients(gradients):
            self.gradients[layer] = gradients
        return _save_gradients

    def __call__(self, input_tensor, class_idx=None):
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # One backward pass triggers every registered hook - not one pass
        # per layer, so fusion costs about the same latency as single-layer.
        logits[0, class_idx].backward()

        per_layer_maps = []
        for layer in self.target_layers:
            weights = torch.relu(self.gradients[layer])
            heatmap = (weights * self.activations[layer]).sum(dim=1, keepdim=True)
            heatmap = torch.relu(heatmap)
            heatmap = F.interpolate(
                heatmap, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / (heatmap.max() + 1e-8)
            per_layer_maps.append(heatmap)

        stacked = torch.stack(per_layer_maps)
        fused = stacked.amax(dim=0) if self.combine == "max" else stacked.mean(dim=0)

        fused = fused - fused.min()
        fused = fused / (fused.max() + 1e-8)

        if was_training:
            self.model.train()

        return fused.detach().cpu().numpy(), int(class_idx), probabilities

    def close(self):
        for h in self.handles:
            h.remove()


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_CAM_CLASSES = {"gradcam": GradCAM, "layercam": LayerCAM}


def gradcam_overlay(model, image: Image.Image, transform, device, alpha: float = 0.45, method: str = "gradcam"):
    """Run Grad-CAM, Layer-CAM, or fused multi-layer Layer-CAM on one
    model/image.

    method: "gradcam" (default, unchanged behaviour), "layercam"
    (single-layer, features[-1]), or "layercam_fused" (features[11] +
    features[-1], element-wise max combine).

    Returns dict: {prediction: {label, real_pct, fake_pct},
                    heatmap_b64, overlay_b64}
    """
    if method == "layercam_fused":
        gc = MultiLayerCAM(model, [model.features[11], model.features[-1]], combine="max")
    else:
        cam_cls = _CAM_CLASSES.get(method, GradCAM)
        gc = cam_cls(model, model.features[-1])
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

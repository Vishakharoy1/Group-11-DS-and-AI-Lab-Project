from __future__ import annotations


import argparse
import io
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from webapp.backend.app.preprocessing import crop_and_align_face
except ImportError:
    try:
        from app.preprocessing import crop_and_align_face
    except ImportError:
        def crop_and_align_face(image_pil: Image.Image, padding: float = 0.2) -> Tuple[Image.Image, str]:
            w, h = image_pil.size
            short = min(w, h)
            left = (w - short) // 2
            top = (h - short) // 2
            cropped = image_pil.crop((left, top, left + short, top + short))
            return cropped, "center_crop_fallback"

logger = logging.getLogger("standalone_noise_detector")


def get_srm_filters() -> torch.Tensor:
    """Builds a 24-channel bank of spatial high-pass and noise residual filters."""
    filters = []

    # 1st order linear difference filters (horizontal, vertical, diagonals)
    k1 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    k3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    k4 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

    # 2nd order linear derivative filters
    k5 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32) / 4.0
    k6 = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]], dtype=np.float32) / 4.0

    # 3rd order SRM high-pass kernels (5x5 padded or 3x3)
    k7 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0
    k8 = np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]], dtype=np.float32) / 4.0

    # 5x5 SRM edge & residual kernels
    k9 = np.zeros((5, 5), dtype=np.float32)
    k9[2, :] = [-1, 2, -6, 2, -1]
    k9 /= 12.0

    k10 = np.zeros((5, 5), dtype=np.float32)
    k10[:, 2] = [-1, 2, -6, 2, -1]
    k10 /= 12.0

    k11 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12.0

    k12 = np.zeros((5, 5), dtype=np.float32)
    k12[1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0

    # Sobel X & Y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

    # Laplacian of Gaussian
    log_3x3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4.0

    kernel_list = [k1, k2, k3, k4, k5, k6, k7, k8, sobel_x, sobel_y, log_3x3]

    # Pad 3x3 kernels to 5x5 for unified convolution tensor
    padded_kernels = []
    for k in kernel_list:
        if k.shape == (3, 3):
            pk = np.pad(k, ((1, 1), (1, 1)), mode='constant')
        else:
            pk = k
        padded_kernels.append(pk)

    padded_kernels.extend([k9, k10, k11, k12])

    # Replicate or fill up to 24 channels
    while len(padded_kernels) < 24:
        idx = len(padded_kernels) % len(kernel_list)
        padded_kernels.append(padded_kernels[idx])

    stacked = np.stack(padded_kernels[:24], axis=0) # [24, 5, 5]
    stacked = np.expand_dims(stacked, axis=1)        # [24, 1, 5, 5]
    return torch.from_numpy(stacked)


def extract_spatial_noise_residuals(image_tensor: torch.Tensor) -> torch.Tensor:
    """Applies 24 SRM filters and median residual extraction to input image tensor [B, 3, H, W]."""
    # Convert RGB to Grayscale
    gray = 0.299 * image_tensor[:, 0:1, :, :] + 0.587 * image_tensor[:, 1:2, :, :] + 0.114 * image_tensor[:, 2:3, :, :]

    srm_filters = get_srm_filters().to(device=image_tensor.device, dtype=image_tensor.dtype)
    residuals = F.conv2d(gray, srm_filters, padding=2) # [B, 24, H, W]

    # Median residual
    pad_gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    patches = pad_gray.unfold(2, 3, 1).unfold(3, 3, 1)
    median = patches.contiguous().view(gray.size(0), 1, gray.size(2), gray.size(3), 9).median(dim=-1)[0]
    med_res = torch.abs(gray - median)

    residuals[:, 0:1, :, :] = med_res
    return residuals


class NoiseEvidenceCNN(nn.Module):
    """Convolutional network operating on 24-channel spatial noise residuals."""

    def __init__(self, in_channels: int = 24, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(embed_dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, noise_view: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(noise_view)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)


class StandaloneNoiseDetector:
    """Wrapper class for preprocessing, noise residual extraction, and prediction."""

    def __init__(self, device: str = "auto", fake_threshold: float = 0.5):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.fake_threshold = fake_threshold
        self.model = NoiseEvidenceCNN(in_channels=24, embed_dim=128).to(self.device)
        self.model.eval()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image], img_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, str]:
        """Loads, aligns face, and prepares RGB tensor [1, 3, H, W]."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a file path or PIL Image instance.")

        cropped_face, align_status = crop_and_align_face(img, padding=0.2)
        resized = cropped_face.resize(img_size, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, align_status

    def predict_image(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Runs spatial noise artifact analysis on an image."""
        tensor, align_status = self.preprocess_image(image_input, img_size=(224, 224))

        with torch.no_grad():
            noise_residuals = extract_spatial_noise_residuals(tensor)
            logits = self.model(noise_residuals)
            probs = F.softmax(logits, dim=1).squeeze(0)

            fake_prob = float(probs[1].item()) if probs.numel() > 1 else float(probs[0].item())
            real_prob = 1.0 - fake_prob

            # Calculate physical noise evidence metrics
            res_energy = float(torch.mean(noise_residuals ** 2).item())
            patch_vars = torch.var(noise_residuals, dim=(2, 3))
            noise_std = float(torch.std(patch_vars).item())

            confidence = float(abs(fake_prob - real_prob))
            decision = "likely_ai_generated" if fake_prob >= self.fake_threshold else "likely_real_camera"

        return {
            "prediction": "Fake" if decision == "likely_ai_generated" else "Real",
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "decision": decision,
            "confidence": round(confidence, 4),
            "noise_variance_std": round(noise_std, 4),
            "srm_residual_energy": round(res_energy, 4),
            "face_alignment_used": align_status,
        }

    def predict_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        p = Path(dir_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Directory not found: {p}")
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        results = []
        for img_path in p.iterdir():
            if img_path.suffix.lower() in valid_exts:
                try:
                    res = self.predict_image(img_path)
                    res["file"] = str(img_path)
                    results.append(res)
                except Exception as e:
                    results.append({"file": str(img_path), "error": str(e)})
        return results


def main():
    parser = argparse.ArgumentParser(description="Standalone Noise & Pixel Artifact Evidence Branch Deepfake Detector")
    parser.add_argument("path", type=str, help="Path to input image file or directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake decision threshold (default: 0.50)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()
    detector = StandaloneNoiseDetector(device=args.device, fake_threshold=args.threshold)

    path = Path(args.path)
    if path.is_file():
        res = detector.predict_image(path)
        res["file"] = str(path)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n=======================================================")
            print(f"  File             : {res['file']}")
            print(f"  P(AI Generated)  : {res['fake_probability']:.2%}")
            print(f"  P(Real Camera)   : {res['real_probability']:.2%}")
            print(f"  Decision         : {res['decision']}")
            print(f"  Confidence       : {res['confidence']:.2%}")
            print(f"  SRM Energy       : {res['srm_residual_energy']:.4f}")
            print(f"  Noise Std        : {res['noise_variance_std']:.4f}")
            print("=======================================================\n")
    elif path.is_dir():
        results = detector.predict_directory(path)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nAnalyzed {len(results)} images in directory: {path}")
            print(f"{'File':<40} {'P(Fake)':<10} {'Decision':<20}")
            print("-" * 75)
            for r in results:
                if "error" in r:
                    print(f"{r['file']:<40} ERROR: {r['error']}")
                else:
                    print(f"{Path(r['file']).name:<40} {r['fake_probability']:<10.1%} {r['decision']:<20}")
            print("-" * 75 + "\n")
    else:
        print(f"Error: Path {path} is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()

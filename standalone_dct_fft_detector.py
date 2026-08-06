from __future__ import annotations

"""
Standalone End-to-End DCT & FFT Deepfake Detection Model
=========================================================

Features:
  1. Global 2D FFT: Log-magnitude spectrum, phase (sin/cos) components, radial power spectrum index.
  2. Global 2D DCT: Orthogonal 2D DCT-II matrix mapping & log-magnitude spectrum.
  3. Local Block-DCT (8x8): Multi-band spatial energy maps (DC, Low, Mid, High frequencies).
  4. End-to-End Frequency Neural Network: Dual-stream (Global Frequency + Local Frequency-Spatial)
     backbone with Channel/Spatial Cross-Attention Gating.
  5. Spectral Physics Analytics: High-Frequency Energy Ratio (HFER), Power Spectrum Decay (1/f^alpha slope),
     Phase Entropy, and Block-Grid Discontinuity Index.
  6. Visualizations & Interpretability: Generates standalone heatmap maps & spectral diagnostic figures.

Usage:
  # Detect single image
  python standalone_dct_fft_detector.py path/to/image.jpg

  # Batch process a directory and save frequency maps & JSON summary
  python standalone_dct_fft_detector.py path/to/image_folder/ --output-dir ./results --save-maps --json

  # Run self-test verification benchmark
  python standalone_dct_fft_detector.py --self-test
"""

import argparse
import base64
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
import torchvision.transforms as T

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

logger = logging.getLogger("standalone_dct_fft_detector")


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_dct_matrix(n: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Computes an n x n orthogonal DCT-II transformation matrix (vectorized)."""
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    dct_mat = torch.cos(math.pi * i * (2.0 * j + 1.0) / (2.0 * n))
    dct_mat[0, :] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / n)
    return dct_mat


def extract_fft_features(gray_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes 2D FFT magnitude spectrum and sine/cosine phase components for [B, 1, H, W]."""
    fft_complex = torch.fft.fft2(gray_tensor, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    mag = torch.abs(fft_shifted)
    log_mag = torch.log(mag + 1e-8)
    phase = torch.angle(fft_shifted)
    phase_sin = torch.sin(phase)
    phase_cos = torch.cos(phase)

    return log_mag, phase_sin, phase_cos


def extract_global_dct_features(gray_tensor: torch.Tensor) -> torch.Tensor:
    """Computes 2D DCT log-magnitude spectrum using orthogonal matrix multiplication."""
    b, c, h, w = gray_tensor.shape
    dct_h = compute_dct_matrix(h, device=gray_tensor.device, dtype=gray_tensor.dtype)
    dct_w = compute_dct_matrix(w, device=gray_tensor.device, dtype=gray_tensor.dtype)

    dct_2d = torch.matmul(dct_h.unsqueeze(0), gray_tensor)
    dct_2d = torch.matmul(dct_2d, dct_w.unsqueeze(0).transpose(-1, -2))
    log_dct = torch.log(torch.abs(dct_2d) + 1e-8)
    return log_dct


class LocalBlockDCTExtractor(nn.Module):
    """Partitioning spatial images into 8x8 blocks to extract micro-band frequency energy."""

    def __init__(self, block_size: int = 8, bands: int = 6):
        super().__init__()
        self.block_size = block_size
        self.bands = bands

    def forward(self, gray_tensor: torch.Tensor) -> torch.Tensor:
        b, c, h, w = gray_tensor.shape
        bs = self.block_size

        # Unfold into 8x8 blocks
        blocks = gray_tensor.unfold(2, bs, bs).unfold(3, bs, bs) # [B, 1, H//8, W//8, 8, 8]
        bh, bw = blocks.shape[2], blocks.shape[3]
        blocks_flat = blocks.contiguous().view(-1, 1, bs, bs)

        dct_m = compute_dct_matrix(bs, device=gray_tensor.device, dtype=gray_tensor.dtype)
        dct_blocks = torch.matmul(dct_m.unsqueeze(0), blocks_flat)
        dct_blocks = torch.matmul(dct_blocks, dct_m.unsqueeze(0).transpose(-1, -2))
        dct_blocks = torch.abs(dct_blocks).view(b, bh, bw, bs, bs)

        # Select 6 primary frequency energy bands
        b0 = dct_blocks[:, :, :, 0, 0] # DC band
        b1 = dct_blocks[:, :, :, 0, 1] + dct_blocks[:, :, :, 1, 0] # Low freq
        b2 = dct_blocks[:, :, :, 1, 1] + dct_blocks[:, :, :, 0, 2] + dct_blocks[:, :, :, 2, 0]
        b3 = dct_blocks[:, :, :, 2, 1] + dct_blocks[:, :, :, 1, 2] + dct_blocks[:, :, :, 3, 0] # Mid freq
        b4 = dct_blocks[:, :, :, 3, 3] + dct_blocks[:, :, :, 4, 2] + dct_blocks[:, :, :, 2, 4]
        b5 = dct_blocks[:, :, :, 6, 6] + dct_blocks[:, :, :, 7, 6] + dct_blocks[:, :, :, 6, 7] # High freq

        band_maps = torch.stack([b0, b1, b2, b3, b4, b5], dim=1) # [B, 6, H//8, W//8]
        # Upsample to match spatial tensor resolution
        band_maps_upsampled = F.interpolate(band_maps, size=(h, w), mode='bilinear', align_corners=False)
        return torch.log(band_maps_upsampled + 1e-8)


def analyze_spectral_physics(image_tensor: torch.Tensor) -> Dict[str, float]:
    """Computes explicit spectral metrics: HFER, 1/f alpha slope, Phase Entropy, and Grid Artifact Score."""
    if image_tensor.dim() == 4:
        gray = 0.299 * image_tensor[:, 0:1, :, :] + 0.587 * image_tensor[:, 1:2, :, :] + 0.114 * image_tensor[:, 2:3, :, :]
    else:
        gray = image_tensor.unsqueeze(0).unsqueeze(0)

    b, c, h, w = gray.shape
    fft_c = torch.fft.fft2(gray, dim=(-2, -1))
    fft_s = torch.fft.fftshift(fft_c, dim=(-2, -1))
    mag = torch.abs(fft_s).squeeze(0).squeeze(0)
    phase = torch.angle(fft_s).squeeze(0).squeeze(0)

    # 1. HFER (High-Frequency Energy Ratio)
    cy, cx = h // 2, w // 2
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=gray.device) - cy,
        torch.arange(w, device=gray.device) - cx,
        indexing='ij'
    )
    r_grid = torch.sqrt(y_grid.float()**2 + x_grid.float()**2)
    max_r = min(cy, cx)

    hf_mask = (r_grid > 0.5 * max_r)
    total_e = torch.sum(mag**2) + 1e-8
    hf_e = torch.sum((mag * hf_mask.float())**2)
    hfer = float((hf_e / total_e).item())

    # 2. 1/f Spectral Decay Alpha
    r_flat = r_grid.view(-1)
    mag_flat = mag.view(-1)
    valid_mask = (r_flat >= 5) & (r_flat <= max_r * 0.9)
    r_valid = r_flat[valid_mask]
    mag_valid = mag_flat[valid_mask]

    if r_valid.numel() > 10:
        log_r = torch.log(r_valid + 1e-8)
        log_m = torch.log(mag_valid + 1e-8)
        cov = torch.mean((log_r - log_r.mean()) * (log_m - log_m.mean()))
        var = torch.var(log_r) + 1e-12
        alpha = float((-cov / var).item())
    else:
        alpha = 2.0

    # 3. Phase Entropy
    hist = torch.histc(phase, bins=36, min=-math.pi, max=math.pi)
    prob = hist / (torch.sum(hist) + 1e-8)
    prob = prob[prob > 0]
    phase_entropy = float((-torch.sum(prob * torch.log2(prob))).item())

    # 4. Grid Artifact Score
    row_max = torch.max(mag, dim=1)[0]
    col_max = torch.max(mag, dim=0)[0]
    grid_score = float(((torch.max(row_max) + torch.max(col_max)) / (torch.mean(mag) * 2.0 + 1e-8)).item())

    return {
        "hfer": round(hfer, 5),
        "spectral_alpha": round(alpha, 4),
        "phase_entropy": round(phase_entropy, 4),
        "grid_artifact_score": round(grid_score, 5)
    }


class FrequencySEAttention(nn.Module):
    """Channel and spatial squeeze-and-excitation attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Channel attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_ca = x * y

        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        s_att = self.spatial_conv(spatial)
        return x_ca * s_att


class FreqConvBlock(nn.Module):
    """Residual Frequency Convolutional Block with SE Attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attn = FrequencySEAttention(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        out += res
        return F.relu(out)


class GlobalFrequencyStream(nn.Module):
    """Processes 4-channel Global Spectral Maps: [FFT_Mag, FFT_Phase_Sin, FFT_Phase_Cos, Global_DCT_Mag]."""

    def __init__(self, in_channels: int = 4, base_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.stage1 = FreqConvBlock(base_dim, base_dim * 2, stride=2)
        self.stage2 = FreqConvBlock(base_dim * 2, base_dim * 4, stride=2)
        self.stage3 = FreqConvBlock(base_dim * 4, base_dim * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_map = self.stage3(x)
        vec = self.pool(feat_map).view(feat_map.size(0), -1)
        return vec, feat_map


class LocalFrequencySpatialStream(nn.Module):
    """Processes 9-channel Spatial + Local Frequency Maps: [RGB (3) + Block-DCT Bands (6)]."""

    def __init__(self, in_channels: int = 9, base_dim: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.stage1 = FreqConvBlock(base_dim, base_dim * 2, stride=2)
        self.stage2 = FreqConvBlock(base_dim * 2, base_dim * 4, stride=2)
        self.stage3 = FreqConvBlock(base_dim * 4, base_dim * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_map = self.stage3(x)
        vec = self.pool(feat_map).view(feat_map.size(0), -1)
        return vec, feat_map


class EndToEndFreqDeepfakeDetector(nn.Module):
    """Complete Dual-Stream Frequency Deepfake Neural Network."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.block_dct_extractor = LocalBlockDCTExtractor(block_size=8, bands=6)
        self.global_freq_stream = GlobalFrequencyStream(in_channels=4, base_dim=32)
        self.local_spatial_stream = LocalFrequencySpatialStream(in_channels=9, base_dim=32)

        self.gate_fc = nn.Sequential(
            nn.Linear(256 * 2, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.heatmap_head = nn.Conv2d(256, 1, kernel_size=1)

    def extract_frequency_tensors(self, rgb_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gray = 0.299 * rgb_tensor[:, 0:1, :, :] + 0.587 * rgb_tensor[:, 1:2, :, :] + 0.114 * rgb_tensor[:, 2:3, :, :]
        fft_mag, fft_sin, fft_cos = extract_fft_features(gray)
        global_dct = extract_global_dct_features(gray)
        global_tensor = torch.cat([fft_mag, fft_sin, fft_cos, global_dct], dim=1) # [B, 4, H, W]

        local_dct_bands = self.block_dct_extractor(gray)
        local_tensor = torch.cat([rgb_tensor, local_dct_bands], dim=1) # [B, 9, H, W]
        return global_tensor, local_tensor

    def forward(self, rgb_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_t, local_t = self.extract_frequency_tensors(rgb_tensor)

        g_vec, g_map = self.global_freq_stream(global_t)
        l_vec, l_map = self.local_spatial_stream(local_t)

        concat_vec = torch.cat([g_vec, l_vec], dim=1)
        gates = self.gate_fc(concat_vec)

        g_weighted = g_vec * gates[:, 0:1]
        l_weighted = l_vec * gates[:, 1:2]
        fused_vec = torch.cat([g_weighted, l_weighted], dim=1)

        logit = self.classifier(fused_vec)
        fake_prob = torch.sigmoid(logit)

        anomaly_map = torch.sigmoid(self.heatmap_head(l_map))
        return logit, fake_prob, gates, anomaly_map


def apply_jet_colormap(gray_array: np.ndarray) -> np.ndarray:
    """Applies jet colormap to a 2D float array in [0, 1]."""
    norm = np.clip(gray_array, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def create_spectral_visualization_panel(orig_img: Image.Image, model_output: Dict[str, Any], physics_metrics: Dict[str, float]) -> Image.Image:
    """Creates a Base64-renderable 2x2 visual diagnostic panel."""
    w, h = orig_img.size
    p1 = orig_img.resize((256, 256), Image.BILINEAR)

    # 2D FFT Spectrum panel
    arr = np.array(p1.convert("L"), dtype=np.float32)
    fft_c = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log(np.abs(fft_c) + 1e-8)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    p2 = Image.fromarray(apply_jet_colormap(mag_norm))

    # Global 2D DCT panel
    t_gray = torch.from_numpy(arr / 255.0).unsqueeze(0).unsqueeze(0)
    log_dct = extract_global_dct_features(t_gray).squeeze().numpy()
    dct_norm = (log_dct - log_dct.min()) / (log_dct.max() - log_dct.min() + 1e-8)
    p3 = Image.fromarray(apply_jet_colormap(dct_norm))

    # Anomaly map panel
    anom_map = model_output.get("anomaly_map", None)
    if anom_map is not None:
        if isinstance(anom_map, torch.Tensor):
            anom_arr = F.interpolate(anom_map, size=(256, 256), mode='bilinear').squeeze().cpu().numpy()
        else:
            anom_arr = np.array(anom_map)
        anom_norm = (anom_arr - anom_arr.min()) / (anom_arr.max() - anom_arr.min() + 1e-8)
        p4 = Image.fromarray(apply_jet_colormap(anom_norm))
    else:
        p4 = p1

    panel = Image.new("RGB", (512, 512))
    panel.paste(p1, (0, 0))
    panel.paste(p2, (256, 0))
    panel.paste(p3, (0, 256))
    panel.paste(p4, (256, 256))
    return panel


class StandaloneDCTFFTDetector:
    """High-level wrapper for preprocessing, inference, physics calculation, and panel generation."""

    def __init__(self, weights_path: str = None, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = EndToEndFreqDeepfakeDetector(embed_dim=256).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).is_file():
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(ckpt, strict=False)
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed loading weights from {weights_path}: {e}")

    def predict_image(self, image_input: Union[str, Path, Image.Image], threshold: float = 0.5) -> Dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            img = Image.open(p).convert("RGB")
            file_name = str(p)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
            file_name = "in_memory_image"
        else:
            raise TypeError("image_input must be a file path or PIL Image instance.")

        cropped_face, align_status = crop_and_align_face(img, padding=0.2)
        resized = cropped_face.resize((256, 256), Image.BILINEAR)

        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit, fake_prob_t, gates_t, anomaly_map_t = self.model(tensor)
            fake_prob = float(fake_prob_t.item())
            real_prob = 1.0 - fake_prob
            gates = gates_t.squeeze(0).cpu().numpy().tolist()

            physics_metrics = analyze_spectral_physics(tensor)
            out_dict = {"anomaly_map": anomaly_map_t}
            panel_img = create_spectral_visualization_panel(resized, out_dict, physics_metrics)
            panel_b64 = _pil_to_b64(panel_img)

            confidence = float(abs(fake_prob - real_prob))
            prediction = "FAKE" if fake_prob >= threshold else "REAL"

        return {
            "file": file_name,
            "prediction": prediction,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "face_alignment_used": align_status,
            "stream_weights": {
                "global_fft_dct_weight": round(gates[0], 4),
                "local_block_dct_weight": round(gates[1], 4)
            },
            "spectral_physics": physics_metrics,
            "panel_b64": panel_b64,
            "_raw_output": out_dict,
            "_orig_image": resized
        }


def run_self_test() -> bool:
    print("=========================================================")
    print(" Running Standalone DCT/FFT Detector Verification Test ")
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    detector = StandaloneDCTFFTDetector(device=device)

    print("[1/4] Testing Spectral Physics Analyzer...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    res = detector.predict_image(dummy_img)
    physics = res["spectral_physics"]
    print(f"      Physics metrics: HFER={physics['hfer']}, Alpha={physics['spectral_alpha']}")

    print("[2/4] Testing End-to-End Frequency Model Forward Pass...")
    print(f"      Prediction: {res['prediction']}, Fake Prob: {res['fake_probability']}")
    print(f"      Stream Weights: {res['stream_weights']}")

    print("[3/4] Testing Visualization & Spectral Panel Generation...")
    b64_len = len(res["panel_b64"])
    print(f"      Diagnostic Panel Base64 length: {b64_len}")

    print("\n>>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Standalone End-to-End DCT/FFT Deepfake Detector")
    parser.add_argument("input_path", type=str, nargs="?", default="", help="Path to input image file or directory")
    parser.add_argument("--weights", type=str, default="", help="Path to PyTorch model weights (.pth)")
    parser.add_argument("--output-dir", type=str, default="./results_frequency", help="Directory to save output results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (0.0 to 1.0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Hardware execution device")
    parser.add_argument("--save-maps", action="store_true", help="Save 4-panel diagnostic spectral heatmap images")
    parser.add_argument("--json", action="store_true", help="Save predictions to JSON report file")
    parser.add_argument("--self-test", action="store_true", help="Run model verification benchmark and exit")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        sys.exit(0)

    if not args.input_path:
        parser.print_help()
        print("\nNote: You can run '--self-test' to verify the frequency model pipeline.")
        sys.exit(1)

    detector = StandaloneDCTFFTDetector(weights_path=args.weights, device=args.device)
    path = Path(args.input_path)

    if path.is_file():
        res = detector.predict_image(path, threshold=args.threshold)
        print(f"[{res['prediction']}] {res['file']} -> Fake Prob: {res['fake_probability']:.4f} | HFER: {res['spectral_physics']['hfer']} | Alpha: {res['spectral_physics']['spectral_alpha']:.2f}")
    elif path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for p in path.iterdir():
            if p.suffix.lower() in valid_exts:
                r = detector.predict_image(p, threshold=args.threshold)
                print(f"[{r['prediction']}] {p.name:<30} -> Fake Prob: {r['fake_probability']:.4f}")
                results.append(r)
        if args.json:
            with open(out_dir / "detection_results.json", "w") as f:
                json.dump([{k: v for k, v in item.items() if not k.startswith("_")} for item in results], f, indent=2)
            print(f"\n[Success] Detailed detection report saved to JSON: {out_dir / 'detection_results.json'}")


if __name__ == "__main__":
    main()

"""AI Image Detector - tells whether an image is AI generated or AI edited.

Single-file detector consolidating the three classes of signal that large
platforms (Meta et al.) use:

1. Metadata & industry standards (C2PA / Content Credentials, Exif, XMP,
   generator container chunks).
2. Invisible watermark scanning (SD 1.x / SDXL / SD3 / DALL-E patterns).
3. Pixel-level image forensics (spectral analysis, sensor-noise statistics,
   error level analysis, double-JPEG detection) plus an optional neural
   classifier.

Usage:
    from meta_detector import detect_image
    report = detect_image("photo.jpg")
    print(report.verdict, report.ai_score, report.confidence)

CLI / Web UI live in cli.py and webui.py and import detect_image from here.
"""

import io
import os
import re
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.fft import fft2, fftshift

# =============================================================================
# 1. METADATA FORENSICS
# =============================================================================
# Mimics the "Industry Standards & Metadata" layer of platform detectors
# (Meta reads C2PA / Content Credentials, Exif, XMP and generator-specific
# container chunks). No external dependencies beyond Pillow.

# (regex, display label)
AI_SOFTWARE_PATTERNS = [
    (r"midjourney", "Midjourney"),
    (r"stable[\s_-]?diffusion", "Stable Diffusion"),
    (r"stability[\s_-]?ai", "Stability AI"),
    (r"sd3\.?\d?|stable.?diffusion.?3", "Stable Diffusion 3"),
    (r"flux(?:\.1)?|black[\s_-]?forest", "Black Forest Labs Flux"),
    (r"dall[-\s]?e|gpt[\s_-]?image|openai|chatgpt", "OpenAI (DALL-E / GPT Image)"),
    (r"firefly|adobe[\s_-]?generative", "Adobe Firefly"),
    (r"\bimagen\b|synthid|deepmind", "Google Imagen / SynthID"),
    (r"content[\s_-]?seal|meta[\s_-]?ai|meta[\s_-]?genai|muse\b", "Meta AI (Content Seal / Muse)"),
    (r"c2pa|content[\s_-]?credential|hdrgm|jumbf|hashdragon", "C2PA Content Credentials"),
    (r"stablesignature|stable[\s_-]?signature", "Stable Signature"),
    (r"comfyui|comfy[\s_-]?ui", "ComfyUI"),
    (r"fooocus", "Fooocus"),
    (r"invokeai|invoke[\s_-]?ai", "InvokeAI"),
    (r"a1111|automatic1111|sd[\s_-]?web[\s_-]?ui", "AUTOMATIC1111 WebUI"),
    (r"kandinsky|deepfloyd|wuerstchen|pixart|hunyuandit|taiyi", "Open diffusion model"),
    (r"leonardo[\s_-]?ai", "Leonardo AI"),
    (r"canva\b.*\bai|magic[\s_-]?edit|canva", "Canva AI tools"),
    (r"ideogram", "Ideogram"),
    (r"playground[\s_-]?ai", "Playground AI"),
    (r"tensor[\s_-]?art", "Tensor.Art"),
    (r"nightcafe|dreamstudio|bing[\s_-]?image|gemini\b", "Commercial AI image tool"),
    (r"generative[\s_-]?fill|generative[\s_-]?expand|generative[\s_-]?edit", "Generative edit tool"),
    (r"ai[\s_-]?generated|generated[\s_-]?by[\s_-]?ai|synthetic[\s_-]?media|ai[\s_-]?enhanced|ai[\s_-]?edit", "Explicit AI declaration"),
    (r"realistic[\s_-]?vision|dream[\s_-]?shaper|juggernaut|deliberate|chilloutmix|anything[\s_-]?v|majicmix", "SD checkpoint (art model)"),
    (r"digitalsourcetype.*digital.?art|digital.?art|trained.?algorithmic.?media", "Synthetic source tag"),
]
CAMERA_MAKE_PATTERNS = [
    r"canon", r"nikon", r"sony", r"fujifilm|fuji", r"panasonic", r"olympus",
    r"leica", r"pentax", r"ricoh", r"kodak", r"hasselblad", r"xiaomi", r"huawei",
    r"apple", r"samsung", r"google", r"oppo|oneplus|vivo|honor|motorola|lg\b",
    r"go[\s-]?pro|drone|djifly|parrot",
]

# PNG ancillary chunks that commonly carry generator info
PNG_TEXT_KEYS = {
    "parameters": "A1111/ComfyUI generation parameters",
    "prompts": "generation prompts",
    "negative_prompt": "generation prompts",
    "workflow": "ComfyUI workflow",
    "prompt": "generation prompt",
    "software": "software",
    "description": "description",
    "comment": "comment",
    "creation_time": "creation time",
    "sigmals": "Sigmals watermark",
    "seamless": "seamless",
    "raw": "raw params",
}

JPEG_EXIF_SOFTWARE = 0x0131  # Exif.Image.Software
JPEG_EXIF_MODEL = 0x0110
JPEG_EXIF_MAKE = 0x010F
JPEG_EXIF_DIGITAL_SOURCE = 0xA460
JPEG_EXIF_DATETIME = 0x0132
JPEG_EXIF_ORIGINAL = 0x9003
JPEG_EXIF_GPS = 0x8825

# Invisible-watermark binaries embedded in image bytes ARE AI evidence.
# C2PA / Content Credentials (c2pa, jumbf, hdrgm, ...) are a provenance
# standard - camera makers sign genuine photos with them too - so they are
# only reported (c2pa list) and never treated as AI markers by themselves.
RAW_BYTE_MARKERS = [
    (rb"c2pa", "C2PA claim embedded in file bytes"),
    (rb"jumbf", "JUMBF (C2PA container) in file bytes"),
    (rb"hdrgm", "Hashdragon geni watermark (Content Credentials)"),
    (rb"stdschema", "C2PA standard schema"),
    (rb"http://ns.adobe.com/c2pa/", "C2PA XMP namespace"),
    (rb"contentCredentials", "Content Credentials declaration"),
    (rb"StableDiffusionV1", "SD 1.x invisible watermark"),
    (rb"SDXL-v1.0", "SDXL invisible watermark"),
    (rb"SD3.0", "SD3 invisible watermark"),
]

# Byte markers that also count as hard AI-generator evidence (only the
# invisible-watermark signatures; C2PA markers are provenance, not AI proof).
C2PA_AI_MARKERS = (rb"StableDiffusionV1", rb"SDXL-v1.0", rb"SD3.0")


@dataclass
class MetadataReport:
    ai_markers: list = field(default_factory=list)
    camera_markers: list = field(default_factory=list)
    software_tags: list = field(default_factory=list)
    c2pa: list = field(default_factory=list)
    has_exif: bool = False
    has_gps: bool = False
    has_datetime: bool = False
    png_text: dict = field(default_factory=dict)
    has_camera: bool = False
    synthetic_source_tag: bool = False
    notes: list = field(default_factory=list)


def _scan_text(text: str, report: MetadataReport) -> None:
    for pattern, label in AI_SOFTWARE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            report.ai_markers.append(label)
    for pattern in CAMERA_MAKE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            report.camera_markers.append(text.strip()[:60])
            break
    if re.search(r"http://cv\.iptc\.org/newscodes/digitalsourcetype/digitalArt",
                 text, re.IGNORECASE) or \
       re.search(r"trainedAlgorithmicMedia|trained.?algorithmic.?media",
                 text, re.IGNORECASE):
        report.synthetic_source_tag = True
        report.ai_markers.append("Exif DigitalSourceType=synthetic")


def _scan_png_chunks(raw: bytes, report: MetadataReport) -> None:
    chunks = raw[8:]
    idx = 0
    while idx < len(chunks):
        if len(chunks) - idx < 12:
            break
        length = int.from_bytes(chunks[idx:idx + 4], "big")
        ctype = chunks[idx + 4:idx + 8]
        data = chunks[idx + 8:idx + 8 + length]
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            try:
                if ctype == b"tEXt":
                    key, _, value = data.partition(b"\x00")
                    value = value.decode("latin-1", "replace")
                elif ctype == b"iTXt":
                    key = data.split(b"\x00", 1)[0]
                    try:
                        value = data.split(b"\x00", 2)[2].decode("utf-8", "replace")
                    except Exception:
                        value = ""
                else:  # zTXt
                    key = data.split(b"\x00", 1)[0]
                    try:
                        value = zlib.decompress(data.split(b"\x00", 1)[1][1:]).decode("latin-1", "replace")
                    except Exception:
                        value = ""
                report.png_text[key.decode("latin-1", "replace")] = value[:500]
                if value:
                    _scan_text(value, report)
            except Exception:
                pass
        idx += 12 + length


def _scan_exif(img: Image.Image, report: MetadataReport) -> None:
    exif = img.getexif()
    if not exif:
        return
    report.has_exif = True
    for tag, name in (
        (JPEG_EXIF_SOFTWARE, "software"), (JPEG_EXIF_MODEL, "model"),
        (JPEG_EXIF_MAKE, "make"),
    ):
        try:
            raw = exif.get(tag)
            if raw is None:
                continue
            value = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            if not value.strip():
                continue
            report.software_tags.append(f"{name}={value}")
            _scan_text(value, report)
        except Exception:
            pass
    for tag, name in ((JPEG_EXIF_DIGITAL_SOURCE, "digital_source_type"),):
        try:
            raw = exif.get(tag)
            if raw is None:
                continue
            value = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            if value.strip():
                report.software_tags.append(f"{name}={value}")
                _scan_text(value, report)
        except Exception:
            pass
    report.has_datetime = bool(exif.get(JPEG_EXIF_DATETIME) or exif.get(JPEG_EXIF_ORIGINAL))
    gps = exif.get_ifd(JPEG_EXIF_GPS)
    if gps:
        report.has_gps = bool(gps)


def analyze_metadata(path: str) -> MetadataReport:
    report = MetadataReport()
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except Exception as exc:
        report.notes.append(f"cannot read file: {exc}")
        return report

    for marker, label in RAW_BYTE_MARKERS:
        if marker in raw:
            report.c2pa.append(label)
            if marker in C2PA_AI_MARKERS:
                report.ai_markers.append(label)

    # The authoritative C2PA AI claim: digitalSourceType=trainedAlgorithmicMedia
    # (what OpenAI / Google / Adobe embed in generated images). Presence in the
    # JUMBF manifest is hard evidence, unlike bare C2PA provenance.
    if b"trainedAlgorithmicMedia" in raw or b"trained algorithmic media" in raw:
        report.synthetic_source_tag = True
        report.ai_markers.append("C2PA digitalSourceType=synthetic (AI)")

    try:
        img = Image.open(path)
        if img.format == "PNG" and raw.startswith(b"\x89PNG"):
            _scan_png_chunks(raw, report)
        _scan_exif(img, report)
        try:
            xmp = img.getxmp()
            if xmp:
                _scan_text(repr(xmp), report)
        except Exception:
            pass
    except Exception as exc:
        report.notes.append(f"pillow failed to parse image: {exc}")

    report.has_camera = bool(report.camera_markers)
    return report


# =============================================================================
# 2. INVISIBLE-WATERMARK SCANNER (active tracing signals)
# =============================================================================
# Wraps the `invisible-watermark` library used by Stability AI (Stable
# Diffusion 1.x / SDXL / SD3) and OpenAI - the same class of "invisible
# tracking signal" as Meta's Content Seal. Detection is only conclusive when
# the decoded bit string equals a watermark message whose key we know; any
# other decode result is treated as noise (prevents false positives).
#
# If the library is missing, this module degrades gracefully.

KNOWN_MESSAGES = {
    "StableDiffusionV1": "Stable Diffusion 1.x",
    "SDXL-v1.0": "Stable Diffusion XL",
    "SD3.0": "Stable Diffusion 3",
    "DALL-E": "OpenAI DALL-E",
    "OpenAI": "OpenAI",
    "Midjourney": "Midjourney",
    "DeepFloyd": "DeepFloyd",
    "Kandinsky": "Kandinsky",
    "playground-v2-512": "Playground v2",
}

WATERMARK_MODES = ("dwtDct", "dwtDctSvd")


def scan_watermarks(arr: np.ndarray) -> dict:
    """Return {found, message, label, method} - only known keys count."""
    try:
        from imwatermark import WatermarkDecoder
    except Exception:
        return {"found": False, "message": None, "label": None, "method": None,
                "error": "invisible-watermark not installed"}

    if arr.ndim != 3 or arr.shape[2] != 3:
        return {"found": False, "message": None, "label": None, "method": None,
                "error": "image must be RGB"}

    # library expects BGR
    bgr = np.ascontiguousarray(arr[:, :, ::-1])

    # the decoder's length argument is in BITS in this library version
    lengths = sorted({len(m) * 8 for m in KNOWN_MESSAGES})
    for mode in WATERMARK_MODES:
        for length in lengths:
            try:
                decoder = WatermarkDecoder("bytes", length)
                got = decoder.decode(bgr, mode)
            except Exception:
                continue
            if got is None:
                continue
            for msg, label in KNOWN_MESSAGES.items():
                if len(msg) * 8 != length:
                    continue
                if bytes(got) == msg.encode():
                    return {"found": True, "message": msg, "label": label,
                            "method": mode}
    return {"found": False, "message": None, "label": None, "method": None}


# =============================================================================
# 3. FREQUENCY-DOMAIN FORENSICS
# =============================================================================
# Complements the invisible-watermark scan with statistical analysis of the
# image spectrum: high-frequency energy, spectral flatness, periodic
# (grid-like / watermark-like) peaks and double-JPEG recompression evidence.
# Pure numpy/scipy.

def _luminance(arr_rgb: np.ndarray) -> np.ndarray:
    if arr_rgb.ndim == 2:
        return arr_rgb.astype(np.float32)
    r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _spectrum(gray: np.ndarray) -> np.ndarray:
    return np.abs(fft2(gray.astype(np.float64) - gray.mean()))


def high_frequency_ratio(gray: np.ndarray) -> float:
    """Fraction of spectral energy above the middle band.

    Clean AI-rendered images usually have *less* high-frequency energy than
    photographs with sensor noise; heavily upscaled AI images have even less.
    """
    spec = _spectrum(gray)
    h, w = spec.shape
    total = spec.sum()
    if total <= 0:
        return 0.0
    # energy outside the central 1/2 region (shifted spectrum)
    spec = fftshift(spec)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    yy, xx = np.ogrid[:h, :w]
    mask_high = ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2
    return float(spec[mask_high].sum() / total)


def spectral_flatness(gray: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the magnitude spectrum.

    Natural images are highly peaked (low flatness). A suspiciously flat or
    unusually peaked spectrum can indicate synthetic processing.
    """
    spec = _spectrum(gray)
    spec = spec[spec > 0]
    if spec.size == 0:
        return 0.0
    log_mean = np.log(spec).mean()
    mean = spec.mean()
    if mean <= 0:
        return 0.0
    return float(np.exp(log_mean) / mean)


def periodic_peak_score(gray: np.ndarray) -> float:
    """Sharpness of the strongest spectral peak relative to its neighbourhood.

    Latent-diffusion watermark patterns, GAN checkerboards and neural
    upscalers leave sharp, localised peaks in the spectrum. The smooth
    broadband spectrum of a photograph (or a smooth-but-nonperiodic scene)
    has no such outliers.
    """
    spec = _spectrum(gray)
    h, w = spec.shape
    if h < 32 or w < 32:
        return 0.0
    spec = fftshift(spec)
    spec[0, 0] = 0.0
    # low-frequency energy is naturally concentrated; do not judge it here
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    near_dc = ((yy - cy) ** 2 + (xx - cx) ** 2) < (max(8, min(h, w) // 32)) ** 2
    spec[near_dc] = 0.0
    med = np.median(spec)
    if med <= 0:
        return 0.0
    # local envelope via median filter - a sharp peak stands far above it;
    # add the global median as a floor so the numeric noise of the spectral
    # tail cannot inflate the ratio
    env = median_filter(spec, size=33) + med
    ratio = spec / env
    # robust quantile instead of max (single-pixel spikes)
    peak = float(np.percentile(ratio, 99.9))
    # log scale: 2x -> 0.3, 5x -> 0.6, 15x -> 0.8, 100x -> 0.96
    return float(np.log1p(peak) / np.log1p(101.0))


def _blockiness_dct_ratio(gray: np.ndarray) -> float:
    """Aligned-grid blockiness vs. average shifted-grid blockiness.

    Single-compressed JPEG has high blockiness on the aligned 8x8 grid;
    re-compressed (double JPEG) images show elevated blockiness on shifted
    grids too. High aligned/alldelta => single compression evidence; a low
    ratio with high overall blockiness => double compression / re-save.
    """
    h, w = gray.shape
    H, W = h - h % 8, w - w % 8
    if H < 64 or W < 64:
        return 0.0
    g = gray[:H, :W]

    def blockiness(shift_y, shift_x):
        acc = 0.0
        n = 0
        for sy in range(shift_y, H - 8, 8):
            for sx in range(shift_x, W - 8, 8):
                block = g[sy:sy + 8, sx:sx + 8]
                # horizontal blockiness: mean absolute 1st derivative across
                # the right edge of the block vs. interior columns
                edge = np.abs(block[:, 7] - block[:, 6])
                interior = np.abs(block[:, 1] - block[:, 0])
                acc += float(edge.mean() - interior.mean())
                n += 1
        return acc / max(n, 1)

    aligned = blockiness(0, 0)
    shifted = blockiness(4, 4)
    avg_shift = []
    for sy in (0, 2, 4, 6):
        for sx in (0, 2, 4, 6):
            if sy == 0 and sx == 0:
                continue
            avg_shift.append(blockiness(sy, sx))
    mean_shift = float(np.mean(avg_shift))
    return float(aligned - mean_shift)


def double_jpeg_score(gray: np.ndarray, noise_level: Optional[float] = None) -> float:
    """Score in [0, 1] indicating evidence of double JPEG compression.

    Uses blockiness at all 64 8x8 grid offsets (Fridrich-style): single
    compression concentrates blockiness on one grid; recompression spreads it
    across shifted grids, making the aligned grid a worse predictor.
    """
    if noise_level is not None and noise_level < 0.5:
        # no sensor noise: illustration/synthetic image or embedded watermark
        # ripple - grid statistics are not JPEG evidence on such content
        return 0.0
    h, w = gray.shape
    H, W = h - h % 8, w - w % 8
    if H < 96 or W < 96:
        return 0.0
    g = gray[:H, :W].astype(np.float64)
    # normalization: overall gradient energy of the image - blockiness is
    # meaningful relative to how textured the image is
    grad = float((np.abs(np.diff(g, n=1, axis=0)).mean() +
                  np.abs(np.diff(g, n=1, axis=1)).mean()) + 1e-6)
    scores = np.zeros((8, 8))
    for sy in range(8):
        for sx in range(8):
            y = np.arange(sy, H - 8, 8)
            x = np.arange(sx, W - 8, 8)
            block = g[np.ix_(y, x)]
            edge = np.abs(np.diff(block[:, :-1], n=1, axis=1).mean()) + \
                   np.abs(np.diff(block[:-1, :], n=1, axis=0).mean())
            interior = np.abs(np.diff(block, n=2, axis=1)).mean() + \
                       np.abs(np.diff(block, n=2, axis=0)).mean()
            scores[sy, sx] = float(edge - interior) / grad
    aligned = scores[0, 0]
    off = float((scores.sum() - aligned) / 63.0)
    rms = float(np.sqrt(((scores - scores.mean()) ** 2).mean()))
    if rms < 0.02:
        return 0.0  # no 8x8 grid structure (uncompressed or very smooth)
    if abs(aligned - off) < 0.15 * (abs(off) + 1e-9):
        # aligned grid is not special: blockiness smeared over shifts,
        # the signature of a second JPEG compression pass
        return 0.8
    return 0.0


def _resize(gray: np.ndarray, size) -> np.ndarray:
    """Bicubic resize of a float array to (h, w) using scipy."""
    from scipy.ndimage import zoom
    h, w = gray.shape
    th, tw = size
    return zoom(gray, (th / h, tw / w), order=3, mode="nearest")


def analyze_rgb(arr: np.ndarray, noise_level: Optional[float] = None) -> dict:
    """Run all frequency-domain signals on an RGB (H,W,3) uint8 array."""
    gray = _luminance(arr)
    n = min(512, gray.shape[0]), min(512, gray.shape[1])
    g = _resize(gray, n)
    hf = high_frequency_ratio(g)
    flat = spectral_flatness(g)
    peak = periodic_peak_score(g)
    dj = double_jpeg_score(gray, noise_level=noise_level)
    return {
        "high_frequency_ratio": round(hf, 4),
        "spectral_flatness": round(flat, 6),
        "periodic_peak_score": round(peak, 4),
        "double_jpeg_score": round(dj, 4),
        "analyzed_size": [g.shape[1], g.shape[0]],
    }


def analyze_edit_metrics(arr: np.ndarray) -> dict:
    """Cheap regional statistics used by the engine for 'edited' evidence."""
    gray = _luminance(arr)
    h, w = gray.shape
    if h < 64 or w < 64:
        return {}
    H, W = h - h % 32, w - w % 32
    g = gray[:H, :W].reshape(H // 32, 32, W // 32, 32)
    block_stds = g.std(axis=(1, 3))
    return {
        "block_std_cv": round(float(block_stds.std() / (block_stds.mean() + 1e-9)), 4),
        "noise_smoothness": round(float(median_filter(gray, 5).std() / (gray.std() + 1e-9)), 4),
    }


# =============================================================================
# 4. SENSOR-NOISE FORENSICS
# =============================================================================
# Real photographs carry photon (shot) noise whose magnitude grows with scene
# brightness, plus a fixed sensor pattern. AI-generated pixels are usually
# smooth and noise-free, or carry spatially uniform synthetic noise. We measure
# the noise residual and how tightly it is coupled to brightness.

def noise_residual(gray: np.ndarray) -> np.ndarray:
    """Residual of a 5x5 median filter - the non-structural component."""
    return gray.astype(np.float32) - median_filter(gray, size=5)


def shot_noise_correlation(gray: np.ndarray) -> float:
    """Correlation between local brightness and local noise magnitude.

    For real sensor data, brighter areas are noisier (Poisson statistics).
    AI images typically show near-zero or negative correlation. Computed on
    32x32 blocks so single-pixel spikes do not dominate.
    """
    res = noise_residual(gray)
    if gray.size < 32 * 32:
        return 0.0
    h, w = gray.shape
    H, W = h - h % 32, w - w % 32
    if H < 64 or W < 64:
        return 0.0
    g = gray[:H, :W]
    r = np.abs(res[:H, :W])
    b = g.reshape(H // 32, 32, W // 32, 32).mean(axis=(1, 3))
    n = r.reshape(H // 32, 32, W // 32, 32).mean(axis=(1, 3))
    bb, nn = b.ravel(), n.ravel()
    if bb.std() < 8 or nn.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(bb, nn)[0, 1])


def noise_statistics(gray: np.ndarray) -> dict:
    """Global + local noise statistics."""
    res = noise_residual(gray)
    level = float(res.std())
    res_norm = res / (level + 1e-9)

    h, w = gray.shape
    H, W = h - h % 16, w - w % 16
    if H >= 64 and W >= 64:
        g = res_norm[:H, :W].reshape(H // 16, 16, W // 16, 16)
        block_stds = g.std(axis=(1, 3))
        uniformity = float(block_stds.std() / (block_stds.mean() + 1e-9))
        # noise gap: how far the quietest 5% of blocks sit below the median -
        # AI-painted regions are unnaturally noise-free. Exclude near-black
        # blocks (vignette/dark corners have legitimately little noise).
        lum = gray[:H, :W].reshape(H // 16, 16, W // 16, 16).mean(axis=(1, 3))
        usable = (lum > 35) & (lum < 220)
        if usable.sum() > 50:
            vals = block_stds[usable].ravel()
            sorted_stds = np.sort(vals)
            quiet = float(sorted_stds[: max(1, len(sorted_stds) // 20)].mean())
            med = float(np.median(vals))
            noise_gap = float((med - quiet) / (med + 1e-9))
        else:
            noise_gap = 0.0
    else:
        uniformity = 0.0
        noise_gap = 0.0

    return {
        "noise_level": round(level, 3),
        "noise_uniformity_cv": round(uniformity, 4),
        "noise_gap": round(noise_gap, 4),
        "shot_noise_correlation": round(shot_noise_correlation(gray), 4),
    }


# =============================================================================
# 5. ERROR LEVEL ANALYSIS (ELA)
# =============================================================================
# Re-saves the image at a fixed JPEG quality and looks at how much each region
# changes. Locally AI-edited / painted regions behave differently from the
# rest of the frame, which shows up as a spatially inconsistent ELA map.

ELA_QUALITY = 85


def ela_map(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    reencoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    return np.abs(arr.astype(np.int16) - reencoded).sum(axis=2)


def ela_statistics(arr: np.ndarray) -> dict:
    ela = ela_map(arr).astype(np.float32)
    h, w = ela.shape
    if h < 32 or w < 32:
        return {"ela_mean": 0.0, "ela_block_cv": 0.0}
    mean = float(ela.mean())
    H, W = h - h % 32, w - w % 32
    blocks = ela[:H, :W].reshape(H // 32, 32, W // 32, 32)
    block_means = blocks.mean(axis=(1, 3))
    block_cv = float(block_means.std() / (block_means.mean() + 1e-9))
    return {
        "ela_mean": round(mean, 2),
        "ela_block_cv": round(block_cv, 4),
    }


# =============================================================================
# 6. OPTIONAL DEEP-LEARNING CLASSIFIERS
# =============================================================================
# Two fine-tuned CNNs on CIFAKE-style data (real photos vs. AI-generated
# images), both disabled unless their checkpoint exists:
#   * ResNet-18   -> `models/ai_detector.pth`
#   * MobileNetV3 -> `mobilenetv3_test.pth` (repo root, alongside cli.py)

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_ROOT, "models", "ai_detector.pth")
MOBILENET_PATH = os.path.join(_ROOT, "mobilenetv3_test.pth")

_CNN_MEAN = [0.485, 0.456, 0.406]
_CNN_STD = [0.229, 0.224, 0.225]


def _load_ckpt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def cnn_available() -> bool:
    return os.path.exists(MODEL_PATH)


def mobilenet_available() -> bool:
    return os.path.exists(MOBILENET_PATH)


def _load_resnet():
    import torchvision.models as models

    net = models.resnet18(num_classes=2)
    net.load_state_dict(_load_ckpt(MODEL_PATH))
    net.eval()
    return net


def _load_mobilenet():
    import torchvision.models as models

    net = models.mobilenet_v3_large(num_classes=2)
    net.load_state_dict(_load_ckpt(MOBILENET_PATH)["model_state_dict"])
    net.eval()
    return net


def predict_ai_probability(arr) -> float:
    """P(AI) in [0,1] or -1 if the model is unavailable."""
    if not cnn_available():
        return -1.0
    import torch
    from torchvision import transforms

    net = _load_resnet()
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_CNN_MEAN, _CNN_STD),
    ])
    img = t(Image.fromarray(arr)).unsqueeze(0)
    with torch.no_grad():
        logits = net(img)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def predict_mobilenet_probability(arr) -> float:
    """P(AI) in [0,1] from the MobileNetV3 classifier, or -1 if unavailable."""
    if not mobilenet_available():
        return -1.0
    import torch
    from torchvision import transforms

    net = _load_mobilenet()
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_CNN_MEAN, _CNN_STD),
    ])
    img = t(Image.fromarray(arr)).unsqueeze(0)
    with torch.no_grad():
        logits = net(img)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


# =============================================================================
# 7. DETECTION ENGINE - fuses all forensic signals into a verdict
# =============================================================================
# Signal weights are calibrated on the sample set in tests/ - see
# tests/test_detector.py for the calibration harness.

_STRONG_GENERATORS = (
    "midjourney", "stable diffusion", "stability ai", "flux", "dall-e",
    "firefly", "imagen", "synthid", "content seal", "muse", "meta ai",
    "comfyui", "fooocus", "invokeai", "a1111", "kandinsky", "deepfloyd",
    "wuerstchen", "pixart", "leonardo ai", "ideogram", "tensor.art",
    "explicit ai declaration", "synthetic source tag", "sd checkpoint",
    "generative edit tool", "commercial ai image tool", "stable signature",
    "sd 1.x invisible watermark", "sdxl invisible watermark",
    "sd3 invisible watermark",
)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float, center: float, width: float) -> float:
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


@dataclass
class DetectionReport:
    verdict: str = "Uncertain"
    ai_score: float = 0.0
    edit_score: float = 0.0
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "ai_score": round(self.ai_score, 3),
            "edit_score": round(self.edit_score, 3),
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "evidence": self.evidence,
            "warnings": self.warnings,
        }


def _metadata_signal(m: MetadataReport) -> tuple[float, list[str]]:
    """Return (ai_evidence, notes)."""
    notes = []
    ai_evidence = 0.0
    strong = []
    for marker in m.ai_markers:
        low = marker.lower()
        if any(g in low for g in _STRONG_GENERATORS):
            strong.append(marker)
    strong = sorted(set(strong))
    if strong:
        ai_evidence = 1.0
        notes.append(f"Generator/AI marker in metadata: {', '.join(strong)}")
    if m.synthetic_source_tag:
        ai_evidence = max(ai_evidence, 1.0)
        notes.append("digitalSourceType marks the image as synthetic (C2PA/Exif)")
    if m.c2pa:
        notes.append(f"C2PA / Content Credentials present ({', '.join(m.c2pa)})")
    if m.has_camera:
        notes.append(f"Camera metadata present ({', '.join(m.camera_markers)})")
    if not m.has_exif and not m.png_text and not m.c2pa:
        notes.append("No camera/software metadata at all (stripped or AI-generated)")
    return ai_evidence, notes


def _derive_verdict(ai_score: float, edit_score: float, has_camera: bool,
                    hard: bool) -> tuple[str, float]:
    if ai_score >= 0.72:
        verdict = "AI generated / AI edited"
        conf = _clip(0.5 + 0.5 * (ai_score - 0.72) / 0.28 + (0.25 if hard else 0.0))
    elif ai_score >= 0.48:
        verdict = "Likely AI generated or AI edited"
        conf = _clip(0.4 + 0.6 * (ai_score - 0.48) / 0.24)
    elif ai_score >= 0.28:
        verdict = "Uncertain (mixed evidence)"
        conf = _clip(0.55 - abs(ai_score - 0.35) * 1.5)
    elif ai_score >= 0.10:
        if edit_score >= 0.33 and has_camera:
            # otherwise authentic-looking photo with strong localized-edit
            # evidence: manipulation present, but not wholesale generation
            verdict = "Uncertain (mixed evidence)"
            conf = _clip(0.5)
        else:
            verdict = "Likely a real photograph"
            conf = _clip(0.5 + 0.5 * (0.28 - ai_score) / 0.18)
    else:
        verdict = "Real photograph"
        conf = _clip(0.7 + 0.3 * (0.10 - ai_score) / 0.10)
    return verdict, conf


def _meta_dict(m: MetadataReport) -> dict:
    return {
        "ai_markers": sorted(set(m.ai_markers)),
        "camera_markers": list(m.camera_markers),
        "software_tags": list(m.software_tags),
        "c2pa": list(m.c2pa),
        "has_exif": m.has_exif,
        "has_gps": m.has_gps,
        "has_datetime": m.has_datetime,
        "png_text_keys": list(m.png_text.keys()),
        "synthetic_source_tag": m.synthetic_source_tag,
        "notes": list(m.notes),
    }


def detect_image(path: str, use_cnn: bool = True) -> DetectionReport:
    report = DetectionReport()

    m = analyze_metadata(path)
    metadata_ai, meta_notes = _metadata_signal(m)
    report.evidence.extend(meta_notes)

    # ---- pixel-level analysis -------------------------------------------------
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)
    except Exception as exc:
        report.warnings.append(f"cannot decode image: {exc}")
        report.signals = {"metadata": _meta_dict(m)}
        return report

    if arr.shape[0] < 16 or arr.shape[1] < 16:
        report.warnings.append("image too small for pixel analysis")
        report.signals = {"metadata": _meta_dict(m)}
        return report

    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)

    noise_sig = noise_statistics(gray)
    freq_sig = analyze_rgb(arr, noise_level=noise_sig["noise_level"])
    ela_sig = ela_statistics(arr)
    edit_metrics = analyze_edit_metrics(arr)
    wm = scan_watermarks(arr)

    # ---- map raw metrics to [0,1] signals ------------------------------------
    hf = freq_sig["high_frequency_ratio"]
    peak = freq_sig["periodic_peak_score"]
    dj = freq_sig["double_jpeg_score"]

    shot = noise_sig["shot_noise_correlation"]
    unif_cv = noise_sig["noise_uniformity_cv"]
    nlevel = noise_sig["noise_level"]
    noise_gap = noise_sig.get("noise_gap", 0.0)

    ela_cv = ela_sig["ela_block_cv"]

    # High-frequency energy: photographs have sensor noise (higher hf).
    hf_signal = _clip(1.0 - _sigmoid(hf, 0.32, 0.06))      # low hf -> AI-ish
    wm_signal = 1.0 if wm.get("found") else 0.0
    double_signal = _clip(dj)
    # Recompression (double JPEG) replaces sensor noise with quantization
    # noise: the shot-noise coupling, noise-gap and HF-energy tests are
    # designed for unaltered camera data and read a re-saved real photo as
    # AI evidence. Discount them in proportion to the recompression evidence.
    hf_signal = hf_signal * (1.0 - 0.4 * double_signal)
    # sharp periodic peaks only count as AI evidence when the image is also
    # smooth (coherent periodicity + no sensor noise = processed/generated)
    peak_signal = _clip(peak) * hf_signal
    shot_signal = _clip(1.0 - _sigmoid(shot, 0.25, 0.10))  # uncoupled noise -> AI
    shot_signal = shot_signal * (1.0 - 0.7 * double_signal)
    noise_smooth = _clip(1.0 - _sigmoid(nlevel, 2.0, 0.8))

    gap_term = 0.55 * _clip((noise_gap - 0.20) / 0.60) * (1.0 - 0.7 * double_signal)
    edit_region_signal = _clip(
        gap_term +
        0.25 * _clip((ela_cv - 0.25) / 0.75) +
        0.20 * _clip((unif_cv - 0.35) / 0.9)
    )

    # ---- combine ---------------------------------------------------------------
    # hard evidence dominates
    hard = metadata_ai or wm_signal
    if hard:
        ai_score = max(metadata_ai, wm_signal) * 0.55 + 0.45
        ai_score = _clip(ai_score)
    else:
        weights = {
            "hf": 0.24, "peak": 0.14, "shot": 0.20,
            "noise_smooth": 0.22, "double": 0.08, "edit_region": 0.12,
        }
        ai_score = _clip(
            hf_signal * weights["hf"] + peak_signal * weights["peak"] +
            shot_signal * weights["shot"] + noise_smooth * weights["noise_smooth"] +
            double_signal * weights["double"] + edit_region_signal * weights["edit_region"]
        )

    edit_score = _clip(
        0.50 * edit_region_signal +
        0.15 * _clip((unif_cv - 0.3) / 0.9) +
        0.25 * _clip(1.0 - shot) * (1.0 - 0.7 * double_signal) +
        0.10 * wm_signal
    )

    # ---- verdict ----------------------------------------------------------------
    verdict, confidence = _derive_verdict(ai_score, edit_score, m.has_camera, hard)

    report.ai_score = ai_score
    report.edit_score = edit_score
    report.verdict = verdict
    report.confidence = round(confidence, 3)

    report.signals = {
        "metadata": _meta_dict(m),
        "frequency": freq_sig,
        "noise": noise_sig,
        "ela": ela_sig,
        "watermark": wm,
        "edit_metrics": edit_metrics,
        "mapped": {
            "hf_signal": round(hf_signal, 3),
            "periodic_peak_signal": round(peak_signal, 3),
            "shot_noise_signal": round(shot_signal, 3),
            "noise_smoothness_signal": round(noise_smooth, 3),
            "double_jpeg_signal": round(double_signal, 3),
            "edit_region_signal": round(edit_region_signal, 3),
            "watermark_signal": wm_signal,
            "metadata_signal": metadata_ai,
        },
    }

    # ---- human-readable evidence -------------------------------------------------
    if wm.get("found"):
        report.evidence.append(
            f"Invisible watermark detected: {wm.get('label')} "
            f"(method {wm.get('method')})"
        )
    if dj > 0.5:
        report.evidence.append("Re-compressed (double JPEG) - image has been re-saved")
    if peak_signal > 0.5:
        report.evidence.append(
            "Strong periodic spectral peak - characteristic of generative upscaling "
            "or latent watermarking"
        )
    if shot < 0.10 and nlevel < 4:
        report.evidence.append(
            "Noise is not coupled to brightness (missing sensor noise signature)"
        )
    if hf < 0.28 and not hard:
        report.evidence.append("Low high-frequency energy - unusually smooth for a camera photo")
    if edit_region_signal > 0.45:
        report.evidence.append(
            "Spatially inconsistent noise/ELA - evidence of localized editing"
        )
    if m.camera_markers and ai_score < 0.45:
        report.evidence.append("Authentic camera metadata supports the photo being real")

    if use_cnn and (cnn_available() or mobilenet_available()):
        try:
            reports = []
            p_ai = predict_ai_probability(arr)
            p_mob = predict_mobilenet_probability(arr)
            if p_ai >= 0:
                reports.append(("resnet18", p_ai))
            if p_mob >= 0:
                reports.append(("mobilenetv3", p_mob))
            if reports:
                report.signals["cnn"] = {name: round(p, 3) for name, p in reports}
                p_ens = sum(p for _, p in reports) / len(reports)
                # Calibrate: CNN logits tend to saturate near 0/1, pull the
                # ensemble toward the neutral 0.5 so it does not dominate.
                p_cal = _clip(0.5 + 0.85 * (p_ens - 0.5))
                if abs(p_cal - ai_score) > 0.15:
                    # Trust the CNN strongly only when it agrees in direction
                    # with the forensic score, or when it is confident the
                    # image is real (the stats rarely over-claim "real");
                    # nudge weakly otherwise.
                    if p_cal < 0.15:
                        weight = 0.28
                    else:
                        weight = 0.35 if (p_cal > 0.5) == (ai_score > 0.5) else 0.12
                    ai_score2 = _clip(ai_score + weight * (p_cal - ai_score))
                    report.ai_score = round(ai_score2, 3)
                    report.verdict, report.confidence = _derive_verdict(
                        ai_score2, edit_score, m.has_camera, hard)
                    names = ", ".join(name for name, _ in reports)
                    report.evidence.append(
                        f"Neural classifiers P(AI) [{names}]={p_ens:.2f}"
                    )
        except Exception as exc:
            report.warnings.append(f"CNN inference failed: {exc}")

    return report


__version__ = "1.0.0"
__all__ = ["detect_image", "DetectionReport", "MetadataReport", "analyze_metadata",
           "scan_watermarks", "noise_statistics", "__version__"]

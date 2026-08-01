"""The 11 manipulations from the notebook's Section 12b (Aman), ported
as-is so /robustness reproduces exactly what the notebook measures."""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

AMAN_MANIPULATIONS = [
    "original",
    "green_tint",
    "blue_tint",
    "brightness",
    "contrast",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
    "resize",
    "crop",
    "noise",
]


def apply_manipulation(img: Image.Image, mode: str) -> Image.Image:
    if mode == "original":
        return img
    if mode == "green_tint":
        arr = np.array(img).astype(np.float32)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.3, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if mode == "blue_tint":
        arr = np.array(img).astype(np.float32)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.3, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if mode == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.4)
    if mode == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.4)
    if mode == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    if mode == "motion_blur":
        kernel_size = 9
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        arr = np.array(img).astype(np.float32)
        if CV2_AVAILABLE:
            blurred = cv2.filter2D(arr, -1, kernel)
            return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
        return img.filter(ImageFilter.GaussianBlur(radius=3))  # fallback, no cv2
    if mode == "jpeg":
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=30)
        out.seek(0)
        return Image.open(out).convert("RGB")
    if mode == "resize":
        w, h = img.size
        small = img.resize((max(1, w // 3), max(1, h // 3)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)
    if mode == "crop":
        w, h = img.size
        cw, ch = int(w * 0.7), int(h * 0.7)
        left = (w - cw) // 2
        top = (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)
    if mode == "noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img

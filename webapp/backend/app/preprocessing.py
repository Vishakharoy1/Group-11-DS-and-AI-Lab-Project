"""Face crop/align + inference-time transform, ported from the notebook's
Section 3 (Face Detection & Alignment) and Section 4 (Robust Transforms /
build_val_transform). Inference ALWAYS uses the val_transform pipeline
(resize -> tensor -> normalize), never the training-time augmentation."""

import logging

import numpy as np
from PIL import Image
from torchvision import transforms

from . import config

logger = logging.getLogger("deepfake.preprocessing")

try:
    from retinaface import RetinaFace

    RETINA_AVAILABLE = True
except ImportError:
    RETINA_AVAILABLE = False
    logger.warning(
        "retina-face not installed - face detection will fall back to "
        "center-crop for every request. Install with `pip install retina-face` "
        "for real face alignment."
    )

# retinaface.detect_faces() rebuilds its TF model from scratch on every call
# when model=None (see its source: `if model is None: model = build_model()`)
# - roughly a 13-15s tax per request on this stack, independent of image
# size, and the single biggest contributor to requests looking hung. Build
# once at import time and pass it explicitly on every call.
_RETINA_MODEL = None
if RETINA_AVAILABLE:
    try:
        _RETINA_MODEL = RetinaFace.build_model()
    except Exception:
        logger.exception("Failed to pre-build RetinaFace model - falling back to per-call rebuild.")
        _RETINA_MODEL = None

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


FACE_PADDING = 0.20

# RetinaFace cost scales with pixel count, but the model only ever sees a
# 224x224 crop - detecting on a multi-megapixel original (common for phone
# photos >5MB) burns CPU for no accuracy gain and was slow enough on Render's
# free-tier CPU to look like preprocessing had hung. Cap the longest side
# before detection; the crop below is still done in this downscaled space.
MAX_DETECTION_DIM = 1600

val_transform = transforms.Compose(
    [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)


def _downscale_for_detection(image_pil: Image.Image, max_dim: int = MAX_DETECTION_DIM) -> Image.Image:
    longest = max(image_pil.width, image_pil.height)
    if longest <= max_dim:
        return image_pil
    scale = max_dim / longest
    new_size = (max(1, round(image_pil.width * scale)), max(1, round(image_pil.height * scale)))
    return image_pil.resize(new_size, Image.Resampling.LANCZOS)


def center_crop(image_pil: Image.Image) -> Image.Image:
    w, h = image_pil.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return image_pil.crop((left, top, left + crop_size, top + crop_size)).resize(
        (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
    )


def crop_and_align_face(image_pil: Image.Image, padding: float = FACE_PADDING) -> tuple[Image.Image, str]:
    """Detect the largest face with RetinaFace, pad, crop, resize.
    Returns (cropped_image, method) where method is "retinaface" or
    "center_crop_fallback" so callers/UI can show which path was used."""
    image_pil = _downscale_for_detection(image_pil)
    if RETINA_AVAILABLE:
        try:
            if CV2_AVAILABLE:
                img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                faces = RetinaFace.detect_faces(img_bgr, model=_RETINA_MODEL)
            else:
                faces = RetinaFace.detect_faces(np.array(image_pil), model=_RETINA_MODEL)

            if isinstance(faces, dict) and len(faces) > 0:
                largest_face = max(
                    faces.values(),
                    key=lambda f: (
                        (f["facial_area"][2] - f["facial_area"][0])
                        * (f["facial_area"][3] - f["facial_area"][1])
                    ),
                )
                x1, y1, x2, y2 = largest_face["facial_area"]
                face_w = x2 - x1
                face_h = y2 - y1
                x1 = max(0, int(x1 - padding * face_w))
                y1 = max(0, int(y1 - padding * face_h))
                x2 = min(image_pil.width, int(x2 + padding * face_w))
                y2 = min(image_pil.height, int(y2 + padding * face_h))
                cropped = image_pil.crop((x1, y1, x2, y2)).resize(
                    (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
                )
                return cropped, "retinaface"
        except Exception:
            logger.exception("RetinaFace failed on this image - falling back to center-crop.")

    return center_crop(image_pil), "center_crop_fallback"


def face_alignment_method() -> str:
    """What crop_and_align_face WOULD use right now, for the /health check."""
    return "retinaface" if RETINA_AVAILABLE else "center_crop_fallback"

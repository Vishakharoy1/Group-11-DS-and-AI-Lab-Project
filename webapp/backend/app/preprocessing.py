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

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


FACE_PADDING = 0.20

val_transform = transforms.Compose(
    [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)


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
    if RETINA_AVAILABLE:
        try:
            if CV2_AVAILABLE:
                img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                faces = RetinaFace.detect_faces(img_bgr)
            else:
                faces = RetinaFace.detect_faces(np.array(image_pil))

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

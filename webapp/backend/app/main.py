"""FastAPI app: routes + static frontend mount.

Run with:
    uvicorn app.main:app --reload --port 8000
from inside the backend/ directory.
"""

import base64
import io
import logging
import os
import tempfile

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, gradcam, manipulations, meta_detector, preprocessing, report, results
from .model import ModelRegistry
from .schemas import (
    CompareResponse,
    DocxReportRequest,
    HealthResponse,
    MetaDetectorResponse,
    PredictionResult,
    PredictResponse,
    RobustnessResponse,
    RobustnessRow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake.main")

app = FastAPI(title="Deepfake Face Detection API")

registry: ModelRegistry | None = None


@app.on_event("startup")
def on_startup():
    global registry
    registry = ModelRegistry()
    if not registry.is_ready():
        logger.warning(
            "No checkpoints loaded from %s - /predict, /robustness and "
            "/compare will return errors until a checkpoint is placed there "
            "(or CHECKPOINT_DIR is pointed at the right folder) and the "
            "server is restarted.",
            config.CHECKPOINT_DIR,
        )


async def _load_upload_image(file: UploadFile) -> tuple[Image.Image, bytes]:
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large (max 10 MB).")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB"), data
    except Exception:
        raise HTTPException(400, "Could not decode file as an image.")


def _run_meta_detector(data: bytes, suffix: str = "") -> dict:
    """Runs the forensic meta-detector on the raw uploaded bytes (metadata /
    watermark / spectral / sensor-noise forensics need the original file,
    not the re-encoded PIL image)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return meta_detector.detect_image(path, use_cnn=False).to_dict()
    finally:
        os.unlink(path)


def _require_model(name: str = "noaug"):
    if registry is None or registry.get(name) is None:
        raise HTTPException(
            503,
            f"Model '{name}' is not loaded. Place its checkpoint in "
            f"{config.CHECKPOINT_DIR} and restart the server.",
        )
    return registry.get(name)


def _predict_pct(model, image, transform):
    model.eval()
    x = transform(image).unsqueeze(0).to(registry.device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    pred = int(probs.argmax())
    return config.CLASSES[pred], probs[0].item() * 100, probs[1].item() * 100


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = registry.loaded_names if registry else []
    return HealthResponse(
        status="ok" if loaded else "no_models_loaded",
        loaded_models=loaded,
        face_alignment=preprocessing.face_alignment_method(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(model: str = "noaug", file: UploadFile = File(...)):
    """model: which loaded checkpoint to run - "noaug" (default/main),
    "best", "cross_domain", or "tuned", whichever are actually loaded.
    Defaults to "noaug"."""
    model_obj = _require_model(model)
    image, data = await _load_upload_image(file)

    cropped, method = preprocessing.crop_and_align_face(image)
    result = gradcam.gradcam_overlay(model_obj, cropped, preprocessing.val_transform, registry.device)
    meta = _run_meta_detector(data, os.path.splitext(file.filename or "")[1])

    return PredictResponse(
        prediction=PredictionResult(**result["prediction"]),
        gradcam_heatmap=result["heatmap_b64"],
        gradcam_overlay=result["overlay_b64"],
        face_alignment_used=method,
        meta_detector=MetaDetectorResponse(**meta),
    )


@app.post("/report", response_class=HTMLResponse)
async def generate_report(model: str = "noaug", file: UploadFile = File(...)):
    """Runs the same prediction + Grad-CAM pipeline as /predict, then
    renders the result as a standalone printable HTML forensic report
    instead of JSON."""
    model_obj = _require_model(model)
    filename = file.filename or "uploaded_image"
    image, _data = await _load_upload_image(file)

    cropped, method = preprocessing.crop_and_align_face(image)
    result = gradcam.gradcam_overlay(model_obj, cropped, preprocessing.val_transform, registry.device)

    input_buf = io.BytesIO()
    cropped.resize((config.IMG_SIZE, config.IMG_SIZE)).save(input_buf, format="PNG")
    input_b64 = base64.b64encode(input_buf.getvalue()).decode("ascii")

    checkpoint_filename = config.CHECKPOINTS.get(model, config.CHECKPOINTS["noaug"]).name

    html = report.build_report_html(
        input_image_b64=input_b64,
        filename=filename,
        model_name=checkpoint_filename,
        face_alignment_used=method,
        label=result["prediction"]["label"],
        real_pct=result["prediction"]["real_pct"],
        fake_pct=result["prediction"]["fake_pct"],
        heatmap_b64=result["heatmap_b64"],
        overlay_b64=result["overlay_b64"],
    )
    return HTMLResponse(content=html)


@app.post("/report/docx")
async def generate_docx_report(payload: DocxReportRequest):
    """Builds a downloadable .docx from analysis data already computed by
    the frontend (the Report page is the canonical rendering; this just
    reproduces the same data as a Word document)."""
    buf = report.build_docx(
        analysis_id=payload.analysis_id,
        generated_at_str=payload.generated_at,
        filename=payload.filename,
        file_type=payload.file_type,
        resolution=payload.resolution,
        file_size=payload.file_size,
        color_mode=payload.color_mode,
        model_label=payload.model_label,
        model_version=payload.model_version,
        label=payload.label,
        real_pct=payload.real_pct,
        fake_pct=payload.fake_pct,
        input_image_b64=payload.input_image_b64,
        overlay_b64=payload.overlay_b64,
    )
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{payload.analysis_id}_report.docx"'},
    )


@app.post("/robustness", response_model=RobustnessResponse)
async def robustness(file: UploadFile = File(...)):
    # Uses the "manipulations" checkpoint - trained specifically to stay
    # robust under these 11 corruptions - instead of "best".
    model = _require_model("manipulations")
    image, _data = await _load_upload_image(file)
    cropped, _method = preprocessing.crop_and_align_face(image)

    rows: list[RobustnessRow] = []
    for mode in manipulations.AMAN_MANIPULATIONS:
        manipulated = manipulations.apply_manipulation(cropped, mode)
        label, real_pct, fake_pct = _predict_pct(model, manipulated, preprocessing.val_transform)

        thumb = manipulated.copy()
        thumb.thumbnail((112, 112))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        thumb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        rows.append(
            RobustnessRow(
                manipulation=mode,
                label=label,
                real_pct=round(real_pct, 2),
                fake_pct=round(fake_pct, 2),
                thumbnail=thumb_b64,
            )
        )

    return RobustnessResponse(rows=rows)


_COMPARE_MODES = {
    "augmentation": ("best", "noaug", "With Augmentation", "Without Augmentation"),
    "hparams": ("best", "tuned", "Baseline HPARAMS", "Swept/Tuned HPARAMS"),
}


@app.post("/compare", response_model=CompareResponse)
async def compare(mode: str, file: UploadFile = File(...)):
    if mode not in _COMPARE_MODES:
        raise HTTPException(400, f"mode must be one of {list(_COMPARE_MODES)}")

    key_a, key_b, label_a, label_b = _COMPARE_MODES[mode]
    model_a = registry.get(key_a) if registry else None
    model_b = registry.get(key_b) if registry else None

    if model_a is None or model_b is None:
        missing = key_a if model_a is None else key_b
        return CompareResponse(
            mode=mode,
            available=False,
            reason=f"Checkpoint '{missing}' is not loaded - train/save it and restart the server.",
        )

    image, _data = await _load_upload_image(file)
    cropped, _method = preprocessing.crop_and_align_face(image)

    label_a_res, real_a, fake_a = _predict_pct(model_a, cropped, preprocessing.val_transform)
    label_b_res, real_b, fake_b = _predict_pct(model_b, cropped, preprocessing.val_transform)

    return CompareResponse(
        mode=mode,
        available=True,
        results={
            label_a: PredictionResult(label=label_a_res, real_pct=round(real_a, 2), fake_pct=round(fake_a, 2)),
            label_b: PredictionResult(label=label_b_res, real_pct=round(real_b, 2), fake_pct=round(fake_b, 2)),
        },
    )


@app.get("/api/training-results")
def training_results():
    """Pre-computed evaluation artifacts from the training notebook
    (Deepfake/output/) - result tables, sample-grid images, and the
    correct/incorrect Grad-CAM gallery. Not live inference."""
    return results.build_training_results_payload()


# Serve the raw CSV/PNG artifacts referenced by /api/training-results.
# Mounted before "/" so it isn't shadowed by the catch-all static mount.
if config.RESULTS_DIR.is_dir():
    app.mount(
        results.RESULTS_ASSET_PREFIX,
        StaticFiles(directory=str(config.RESULTS_DIR)),
        name="results-assets",
    )
else:
    logger.warning("RESULTS_DIR %s does not exist - /results-assets not mounted.", config.RESULTS_DIR)

# Static frontend - mounted last so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

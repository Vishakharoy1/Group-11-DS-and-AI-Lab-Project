"""Typed response models - also auto-generates the /docs Swagger UI."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    face_alignment: str


class PredictionResult(BaseModel):
    label: str
    real_pct: float
    fake_pct: float


class MetaDetectorResponse(BaseModel):
    """Output of the forensic meta-detector (metadata / watermark /
    spectral / sensor-noise analysis) run alongside the CNN model."""
    verdict: str
    ai_score: float
    edit_score: float
    confidence: float
    signals: dict
    evidence: list[str]
    warnings: list[str]


class PredictResponse(BaseModel):
    prediction: PredictionResult
    face_alignment_used: str
    meta_detector: MetaDetectorResponse


class GradcamResponse(BaseModel):
    prediction: PredictionResult
    gradcam_heatmap: str  # base64 PNG
    gradcam_overlay: str  # base64 PNG
    face_alignment_used: str


class RobustnessRow(BaseModel):
    manipulation: str
    label: str
    real_pct: float
    fake_pct: float
    thumbnail: str  # base64 PNG


class RobustnessResponse(BaseModel):
    rows: list[RobustnessRow]


class CompareResponse(BaseModel):
    mode: str
    available: bool
    reason: str | None = None
    results: dict[str, PredictionResult] = {}


class DocxReportRequest(BaseModel):
    analysis_id: str
    generated_at: str
    filename: str
    file_type: str
    resolution: str
    file_size: str
    color_mode: str
    model_label: str
    model_version: str = "v1.0"
    label: str
    real_pct: float
    fake_pct: float
    input_image_b64: str
    overlay_b64: str

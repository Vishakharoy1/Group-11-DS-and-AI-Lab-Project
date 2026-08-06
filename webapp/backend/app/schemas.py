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


class PredictResponse(BaseModel):
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

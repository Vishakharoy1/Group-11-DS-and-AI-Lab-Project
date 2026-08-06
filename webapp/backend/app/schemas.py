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


class FrequencyPhysicsMetrics(BaseModel):
    hfer: float
    spectral_alpha: float
    phase_entropy: float
    grid_artifact_score: float


class FrequencyStreamWeights(BaseModel):
    global_fft_dct_weight: float
    local_block_dct_weight: float


class FrequencyPredictResponse(BaseModel):
    prediction: str
    fake_probability: float
    real_probability: float
    confidence: float
    stream_weights: FrequencyStreamWeights
    spectral_physics: FrequencyPhysicsMetrics
    panel_b64: str
    face_alignment_used: str


class NoisePredictResponse(BaseModel):
    prediction: str
    fake_probability: float
    real_probability: float
    decision: str
    confidence: float
    noise_variance_std: float
    srm_residual_energy: float
    face_alignment_used: str


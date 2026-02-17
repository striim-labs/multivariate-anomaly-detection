"""
Pydantic request/response models for the TranAD scoring API.

All models use Pydantic v2 (BaseModel). FastAPI auto-generates
OpenAPI/JSON schemas from these definitions at /docs.
"""

from pydantic import BaseModel, Field, field_validator


# ── Nested response types ──────────────────────────────────────────


class AttributedDimension(BaseModel):
    """A single feature identified as contributing to an anomaly segment."""

    dim: int = Field(..., description="Feature index (0-based)", ge=0)
    label: str = Field(..., description="Human-readable feature label")
    mean_elevation: float = Field(
        ...,
        description="Mean elevation ratio (score / baseline) across segment",
    )
    contribution: float = Field(
        ...,
        description="Fraction of total excess score attributed to this feature",
        ge=0.0,
        le=1.0,
    )


class AnomalySegment(BaseModel):
    """A contiguous run of anomalous timesteps with root cause attribution."""

    segment_start: int = Field(
        ..., description="First anomalous timestep index (0-based)"
    )
    segment_end: int = Field(
        ..., description="Last anomalous timestep index (inclusive)"
    )
    segment_length: int = Field(
        ..., description="Number of anomalous timesteps", ge=1
    )
    peak_score: float = Field(
        ..., description="Maximum aggregated anomaly score in segment"
    )
    peak_timestamp: int = Field(..., description="Timestep index of peak score")
    mean_score: float = Field(
        ..., description="Mean aggregated anomaly score across segment"
    )
    attributed_dimensions: list[AttributedDimension] = Field(
        default_factory=list,
        description="Features driving the anomaly, ranked by severity",
    )


class TimestepResult(BaseModel):
    """Per-timestep anomaly score and binary prediction."""

    index: int = Field(
        ..., description="Timestep index within the submitted batch (0-based)"
    )
    score: float = Field(
        ..., description="Aggregated anomaly score (mean across features)"
    )
    is_anomaly: bool = Field(..., description="True if score exceeds threshold")


# ── Identifier types ─────────────────────────────────────────────


class DeviceIdentifier(BaseModel):
    """Store and device identifier pair."""

    store_id: int = Field(..., description="Store identifier", ge=1)
    device_id: int = Field(..., description="Device identifier within the store", ge=1)


# ── Request ────────────────────────────────────────────────────────


class ScoringRequest(BaseModel):
    """Request body for POST /score.

    Send raw (unnormalized) time series data for a specific store/device.
    The API normalizes data using stored training-time parameters.
    Minimum 10 timesteps (model window_size) required.
    """

    store_id: int = Field(
        ...,
        description="Store identifier",
        examples=[1],
        ge=1,
    )
    device_id: int = Field(
        ...,
        description="Device identifier within the store",
        examples=[1],
        ge=1,
    )
    data: list[list[float]] = Field(
        ...,
        description=(
            "Time series matrix: outer list is timesteps, inner list is features. "
            "Each inner list must have exactly n_features (38 for SMD) float values. "
            "Data should be raw/unnormalized."
        ),
    )
    include_per_timestep: bool = Field(
        default=False,
        description="If true, include per-timestep scores and predictions in response",
    )
    include_attribution: bool = Field(
        default=True,
        description="If true, include segment-level feature attribution",
    )
    scoring_mode: str = Field(
        default="phase2_only",
        description="Scoring mode: 'phase2_only' (reference) or 'averaged' (paper Eq. 13)",
    )

    @field_validator("data")
    @classmethod
    def validate_minimum_timesteps(cls, v):
        if len(v) < 10:
            raise ValueError(
                f"At least 10 timesteps required (window_size), got {len(v)}"
            )
        return v

    @field_validator("data")
    @classmethod
    def validate_consistent_features(cls, v):
        if not v:
            return v
        n_features = len(v[0])
        for i, row in enumerate(v):
            if len(row) != n_features:
                raise ValueError(
                    f"Inconsistent feature count: row 0 has {n_features} features, "
                    f"row {i} has {len(row)}"
                )
        return v

    @field_validator("scoring_mode")
    @classmethod
    def validate_scoring_mode(cls, v):
        if v not in ("phase2_only", "averaged"):
            raise ValueError(
                f"scoring_mode must be 'phase2_only' or 'averaged', got '{v}'"
            )
        return v


# ── Response ───────────────────────────────────────────────────────


class ScoringResponse(BaseModel):
    """Response body for POST /score."""

    store_id: int = Field(..., description="Echo of the requested store_id")
    device_id: int = Field(..., description="Echo of the requested device_id")
    n_timesteps: int = Field(..., description="Number of timesteps scored")
    n_features: int = Field(..., description="Number of features per timestep")
    threshold: float = Field(
        ..., description="Anomaly threshold used for predictions"
    )
    n_anomalies: int = Field(
        ..., description="Total timesteps flagged as anomalous", ge=0
    )
    anomaly_ratio: float = Field(
        ...,
        description="Fraction of timesteps flagged as anomalous",
        ge=0.0,
        le=1.0,
    )
    anomaly_segments: list[AnomalySegment] = Field(
        default_factory=list,
        description="Contiguous anomaly segments with attribution",
    )
    per_timestep: list[TimestepResult] | None = Field(
        default=None,
        description="Per-timestep scores (only if include_per_timestep=True in request)",
    )
    scoring_mode: str = Field(..., description="Scoring mode used")
    threshold_method: str = Field(
        ..., description="Method used to calibrate threshold"
    )


# ── Auxiliary endpoint responses ───────────────────────────────────


class ModelInfo(BaseModel):
    """Information about a loaded device model."""

    store_id: int
    device_id: int
    threshold: float
    threshold_method: str
    n_features: int
    window_size: int


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="'ready' or 'starting'")
    detector: str = Field(default="tranad")
    available_devices: list[DeviceIdentifier] = Field(default_factory=list)
    loaded_devices: list[DeviceIdentifier] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    """Response for GET /models."""

    devices: list[DeviceIdentifier] = Field(
        ..., description="Available device models"
    )


class ConfigResponse(BaseModel):
    """Response for GET /config."""

    available_devices: list[DeviceIdentifier]
    loaded_devices: list[DeviceIdentifier]
    device: str
    default_scoring_mode: str
    models: list[ModelInfo] = Field(default_factory=list)

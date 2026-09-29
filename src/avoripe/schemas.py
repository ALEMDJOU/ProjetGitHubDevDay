"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Example values: real photo T20_d05_001_a_3 (labelled ripening index 3, ripe first stage).
EXAMPLE_FEATURES = {
    "storage_group": "T20",
    "day": 5,
    "l_mean": 36.1304,
    "l_std": 8.8149,
    "a_mean": 6.0232,
    "a_std": 3.8616,
    "b_mean": 3.1052,
    "b_std": 5.3169,
    "dark_fraction": 0.2505,
    "fruit_fraction": 0.2318,
}


class AvocadoFeatures(BaseModel):
    """Storage conditions and colour statistics of one avocado photo."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_FEATURES})

    storage_group: Literal["T10", "T20", "Tam"] = Field(
        description="T10 = 10 °C, T20 = 20 °C (both 85% RH), Tam = ambient"
    )
    day: int = Field(ge=1, le=60, description="Days since the start of storage")
    l_mean: float = Field(ge=0, le=100, description="Mean Lab lightness of the skin")
    l_std: float = Field(ge=0, le=100, description="Std of Lab lightness")
    a_mean: float = Field(ge=-128, le=127, description="Mean Lab a* (green - / red +)")
    a_std: float = Field(ge=0, le=128, description="Std of Lab a*")
    b_mean: float = Field(ge=-128, le=127, description="Mean Lab b* (blue - / yellow +)")
    b_std: float = Field(ge=0, le=128, description="Std of Lab b*")
    dark_fraction: float = Field(ge=0, le=1, description="Share of skin pixels with L* < 30")
    fruit_fraction: float = Field(ge=0, le=1, description="Share of the image covered by fruit")


class Prediction(BaseModel):
    """Predicted ripeness class with per-class probabilities."""

    ripeness: str
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    """Liveness information."""

    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Basic runtime monitoring counters."""

    request_count: int
    mean_latency_ms: float
    prediction_distribution: dict[str, int]

"""Pydantic request/response models for the API.

FastAPI uses these classes to:
- validate every /predict request (wrong type, out-of-range value or unknown field -> 422);
- generate the interactive documentation at /docs, including a ready-to-run example.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Example request shown in /docs and used by the tests and the Monitor workflow.
# Values are copied from one real photo of the dataset: T20_d05_001_a_3
# (avocado #1, stored at 20 °C, day 5, labelled ripening index 3 = ripe first stage).
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

    # extra="forbid": a misspelled or unexpected field is rejected (422) instead of
    # being silently ignored. The example appears in the Swagger UI "Try it out" form.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_FEATURES})

    # Bounds below are the physical limits of each quantity (Lab colour space, fractions
    # between 0 and 1, a storage experiment of a few weeks), not the dataset min/max.
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

    ripeness: str  # most likely class, e.g. "ripe_first_stage"
    probabilities: dict[str, float]  # one probability per class, they sum to 1


class HealthResponse(BaseModel):
    """Liveness information."""

    status: str  # "ok" when the model is loaded, "degraded" otherwise
    model_loaded: bool
    version: str  # commit SHA baked into the Docker image ("dev" when run locally)


class MetricsResponse(BaseModel):
    """Basic runtime monitoring counters."""

    request_count: int  # number of /predict calls since the service started
    mean_latency_ms: float  # average model inference time per request
    prediction_distribution: dict[str, int]  # how often each class was predicted

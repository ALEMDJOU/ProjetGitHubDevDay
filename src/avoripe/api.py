"""FastAPI service exposing the trained avocado ripeness model."""

from __future__ import annotations

import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from avoripe import __version__, config
from avoripe.schemas import AvocadoFeatures, HealthResponse, MetricsResponse, Prediction


class ServiceStats:
    """In-memory counters for the /metrics endpoint (reset on restart)."""

    def __init__(self) -> None:
        """Start with zero requests."""
        self.request_count = 0
        self.total_latency_ms = 0.0
        self.predictions: Counter[str] = Counter()

    def record(self, label: str, latency_ms: float) -> None:
        """Record one prediction and its latency."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.predictions[label] += 1


def model_path() -> Path:
    """Resolve the model location from MODEL_PATH (lets tests inject a temp model)."""
    return Path(os.getenv("MODEL_PATH", str(config.MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    path = model_path()
    app.state.model = joblib.load(path) if path.exists() else None
    app.state.stats = ServiceStats()
    yield


app = FastAPI(
    title="Avocado Ripeness API",
    description="Predicts the 5-stage ripening index of a Hass avocado.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and whether the model is loaded."""
    loaded = app.state.model is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict", response_model=Prediction)
def predict(features: AvocadoFeatures) -> Prediction:
    """Predict the ripeness class and per-class probabilities for one avocado."""
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.perf_counter()
    row = pd.DataFrame([features.model_dump()])[config.FEATURES]
    probabilities = model.predict_proba(row)[0]
    classes = [str(c) for c in model.classes_]
    label = classes[int(probabilities.argmax())]
    app.state.stats.record(label, (time.perf_counter() - start) * 1000)
    return Prediction(
        ripeness=label,
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probabilities, strict=True)},
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return request count, mean latency and the distribution of predicted classes."""
    stats: ServiceStats = app.state.stats
    mean = stats.total_latency_ms / stats.request_count if stats.request_count else 0.0
    return MetricsResponse(
        request_count=stats.request_count,
        mean_latency_ms=round(mean, 3),
        prediction_distribution=dict(stats.predictions),
    )

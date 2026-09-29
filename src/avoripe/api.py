"""FastAPI service exposing the trained avocado ripeness model.

Endpoints:
- GET  /         -> redirects to /docs (interactive Swagger UI)
- GET  /health   -> liveness, model status and deployed commit SHA
- POST /predict  -> ripeness class + per-class probabilities for one avocado
- GET  /metrics  -> request count, mean latency, distribution of predicted classes

Run locally with `make run` (http://localhost:7860/docs). In Docker, the port comes from
the PORT variable (Render sets it; Hugging Face and local runs use 7860).
"""

from __future__ import annotations

import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from avoripe import __version__, config
from avoripe.schemas import AvocadoFeatures, HealthResponse, MetricsResponse, Prediction


class ServiceStats:
    """In-memory counters for the /metrics endpoint (reset on restart)."""

    # Deliberately minimal monitoring: no database, no external service. Counters live
    # in the process memory and start again from zero whenever the container restarts.

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
    # Docker sets MODEL_PATH=/app/models/model.joblib; tests point it at a tiny model
    # trained in tmp_path; otherwise we use the default models/model.joblib.
    return Path(os.getenv("MODEL_PATH", str(config.MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    # Loading once (not per request) keeps /predict fast. If the file is missing the
    # service still starts: /health then reports "degraded" and /predict returns 503,
    # which is easier to diagnose than a container that crashes on boot.
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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness, whether the model is loaded and the deployed commit SHA."""
    # GIT_SHA is baked into the image at build time (see Dockerfile and deploy.yml).
    # The Deploy workflow polls this endpoint until it reports the new SHA, which proves
    # the freshly built image, not an old one, is live.
    loaded = app.state.model is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=os.getenv("GIT_SHA", "dev"),
    )


@app.post("/predict", response_model=Prediction)
def predict(features: AvocadoFeatures) -> Prediction:
    """Predict the ripeness class and per-class probabilities for one avocado."""
    # By the time we get here, FastAPI has already validated the body with AvocadoFeatures:
    # invalid input never reaches the model (the client gets a 422 instead).
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    # The Pipeline expects a DataFrame with the same columns, in the same order, as in
    # training. One request = one row.
    row = pd.DataFrame([features.model_dump()])[config.FEATURES]
    probabilities = model.predict_proba(row)[0]

    # classes_ holds the class names in the same order as the probability columns.
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
    # Avoid a division by zero before the first prediction.
    mean = stats.total_latency_ms / stats.request_count if stats.request_count else 0.0
    return MetricsResponse(
        request_count=stats.request_count,
        mean_latency_ms=round(mean, 3),
        prediction_distribution=dict(stats.predictions),
    )

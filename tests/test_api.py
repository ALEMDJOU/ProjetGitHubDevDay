"""Tests for the FastAPI service, using a tiny model built from real rows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from avoripe import config
from avoripe.api import app
from avoripe.schemas import EXAMPLE_FEATURES


@pytest.fixture()
def client(tiny_model_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """API client with MODEL_PATH pointing at the temporary model."""
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    with TestClient(app) as test_client:
        yield test_client


def real_payload(real_df: pd.DataFrame, row: int = 0) -> dict:
    """Build a /predict payload from one real dataset row."""
    record = real_df.loc[row, config.FEATURES].to_dict()
    record["day"] = int(record["day"])
    return record


def test_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """/health reports ok, the loaded model and the deployed commit SHA."""
    monkeypatch.setenv("GIT_SHA", "abc1234")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "version": "abc1234"}


def test_predict_happy_path(client: TestClient, real_df: pd.DataFrame) -> None:
    """/predict returns a known class and probabilities summing to 1."""
    response = client.post("/predict", json=real_payload(real_df))
    assert response.status_code == 200
    body = response.json()
    assert body["ripeness"] in config.CLASS_NAMES.values()
    assert body["ripeness"] in body["probabilities"]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_schema_example_is_valid(client: TestClient) -> None:
    """The documented example payload is accepted."""
    assert client.post("/predict", json=EXAMPLE_FEATURES).status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [("storage_group", "fridge"), ("day", 0), ("dark_fraction", 1.5), ("l_mean", "green")],
)
def test_predict_invalid_input(
    client: TestClient, real_df: pd.DataFrame, field: str, value: object
) -> None:
    """Out-of-range or wrongly typed fields are rejected with 422."""
    payload = real_payload(real_df) | {field: value}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_missing_field(client: TestClient, real_df: pd.DataFrame) -> None:
    """A missing feature is rejected with 422."""
    payload = real_payload(real_df)
    payload.pop("b_mean")
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_counts_predictions(client: TestClient, real_df: pd.DataFrame) -> None:
    """/metrics reflects the number and classes of predictions served."""
    labels = [client.post("/predict", json=real_payload(real_df, i)).json() for i in range(3)]
    body = client.get("/metrics").json()
    assert body["request_count"] == 3
    assert body["mean_latency_ms"] > 0
    assert sum(body["prediction_distribution"].values()) == 3
    assert set(body["prediction_distribution"]) == {p["ripeness"] for p in labels}


def test_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a model, /health is degraded and /predict returns 503."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.post("/predict", json=EXAMPLE_FEATURES).status_code == 503

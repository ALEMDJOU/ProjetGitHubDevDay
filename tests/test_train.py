"""Tests for the training pipeline and the evaluation quality gate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from avoripe import config, evaluate, train


def test_pipeline_fit_predict(real_df: pd.DataFrame) -> None:
    """The pipeline fits on real rows and predicts known class names."""
    model = train.build_pipeline().set_params(classifier__n_estimators=10)
    model.fit(real_df[config.FEATURES], real_df["label"])
    preds = model.predict(real_df[config.FEATURES].head(5))
    assert set(preds) <= set(config.CLASS_NAMES.values())
    assert model.predict_proba(real_df[config.FEATURES].head(1)).shape[1] == len(model.classes_)


def test_compute_metrics_perfect() -> None:
    """Identical predictions give a macro F1 of 1 and a diagonal matrix."""
    y = pd.Series(["underripe", "ripe_first_stage", "overripe"])
    metrics = evaluate.compute_metrics(y, y)
    assert metrics["f1_macro"] == 1.0
    assert sum(map(sum, metrics["confusion_matrix"])) == 3


def test_get_min_f1_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIN_F1 overrides the default threshold; empty falls back to default."""
    monkeypatch.setenv("MIN_F1", "0.42")
    assert evaluate.get_min_f1() == 0.42
    monkeypatch.setenv("MIN_F1", "")
    assert evaluate.get_min_f1() == config.DEFAULT_MIN_F1


@pytest.fixture()
def tmp_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_csv: Path) -> None:
    """Redirect data, model and metrics paths into tmp_path."""
    monkeypatch.setattr(config, "DATA_PATH", raw_csv)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics" / "metrics.json")
    monkeypatch.setattr(config, "REPORT_PATH", tmp_path / "metrics" / "report.md")


@pytest.mark.usefixtures("tmp_paths")
def test_train_and_evaluate_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() trains, evaluate() writes metrics.json + report.md and the gate passes at 0."""
    train.main()
    assert config.MODEL_PATH.exists()
    monkeypatch.setenv("MIN_F1", "0")
    metrics, passed = evaluate.evaluate()
    assert passed
    saved = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    assert saved["f1_macro"] == metrics["f1_macro"]
    assert "Quality gate" in config.REPORT_PATH.read_text(encoding="utf-8")


@pytest.mark.usefixtures("tmp_paths")
def test_quality_gate_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """An impossible MIN_F1 makes the evaluate CLI exit with code 1."""
    train.main()
    monkeypatch.setenv("MIN_F1", "1.01")
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1

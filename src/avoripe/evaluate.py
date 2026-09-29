"""Evaluate the saved model on the held-out split and enforce the quality gate."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from avoripe import config
from avoripe.data import load_data, split_data


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return accuracy, macro precision/recall/F1 and the confusion matrix."""
    labels = list(config.CLASS_NAMES.values())
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def get_min_f1() -> float:
    """Read the quality-gate threshold (macro F1) from the MIN_F1 env var."""
    raw = os.getenv("MIN_F1", "").strip()
    return float(raw) if raw else config.DEFAULT_MIN_F1


def render_report(metrics: dict, min_f1: float, passed: bool) -> str:
    """Render the metrics as a Markdown report for the GitHub job summary."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    lines = [
        "## 🥑 Model evaluation",
        "",
        f"**Quality gate:** {status} (macro F1 `{metrics['f1_macro']}` vs `MIN_F1` `{min_f1}`)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "n_test"]:
        lines.append(f"| {key} | {metrics[key]} |")
    labels = metrics["labels"]
    lines += ["", "**Confusion matrix** (rows = true, columns = predicted)", ""]
    lines.append("| true \\ pred | " + " | ".join(labels) + " |")
    lines.append("|---" * (len(labels) + 1) + "|")
    for label, row in zip(labels, metrics["confusion_matrix"], strict=True):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(metrics: dict, report: str) -> None:
    """Write metrics.json and report.md into the metrics directory."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config.REPORT_PATH.write_text(report, encoding="utf-8")


def evaluate(model_path: Path | None = None) -> tuple[dict, bool]:
    """Score the model on the test split; return metrics and gate result."""
    model_path = model_path or config.MODEL_PATH
    _, x_test, _, y_test = split_data(load_data())
    model = joblib.load(model_path)
    metrics = compute_metrics(y_test, model.predict(x_test))
    min_f1 = get_min_f1()
    passed = metrics["f1_macro"] >= min_f1
    metrics["min_f1"] = min_f1
    metrics["gate_passed"] = passed
    write_outputs(metrics, render_report(metrics, min_f1, passed))
    return metrics, passed


def main() -> None:
    """CLI entry point: exit with code 1 when the quality gate fails."""
    metrics, passed = evaluate()
    # Plain ASCII on the console (Windows terminals choke on emoji); Markdown goes to report.md.
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "n_test"]:
        print(f"{key:>16}: {metrics[key]}")
    print(f"Report written to {config.REPORT_PATH}")
    if not passed:
        print(f"Quality gate failed: f1_macro={metrics['f1_macro']} < {metrics['min_f1']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

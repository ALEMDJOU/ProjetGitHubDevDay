"""Evaluate the saved model on the held-out split and enforce the quality gate.

Run with `make evaluate` (or `python -m avoripe.evaluate`). It writes:
- metrics/metrics.json: all metrics + the gate decision (uploaded as a CI artifact);
- metrics/report.md:    a Markdown table appended to the GitHub Actions job summary.

Quality gate: if macro F1 < MIN_F1 the process exits with code 1. In CI this turns the
"Train & Evaluate" job red, so no model artifact is uploaded and nothing is deployed.
"""

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

# Metrics shown on the console and in the Markdown table, in this order.
SUMMARY_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "n_test"]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return accuracy, macro precision/recall/F1 and the confusion matrix."""
    # "macro" = average of the per-class scores, every class weighted equally.
    # It is stricter than plain accuracy when some stages have fewer photos.
    labels = list(config.CLASS_NAMES.values())
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "labels": labels,
        # Rows = true class, columns = predicted class, both in ripening order 1..5.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def get_min_f1() -> float:
    """Read the quality-gate threshold (macro F1) from the MIN_F1 env var."""
    # In CI, MIN_F1 comes from the workflow input or the MIN_F1 repository variable.
    # An empty value (e.g. input left blank) falls back to the default in config.py.
    raw = os.getenv("MIN_F1", "").strip()
    return float(raw) if raw else config.DEFAULT_MIN_F1


def render_report(metrics: dict, min_f1: float, passed: bool) -> str:
    """Render the metrics as a Markdown report for the GitHub job summary."""
    status = "✅ PASSED" if passed else "❌ FAILED"

    # Header + gate decision + metrics table.
    lines = [
        "## 🥑 Model evaluation",
        "",
        f"**Quality gate:** {status} (macro F1 `{metrics['f1_macro']}` vs `MIN_F1` `{min_f1}`)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in SUMMARY_KEYS:
        lines.append(f"| {key} | {metrics[key]} |")

    # Confusion matrix as a Markdown table: one header row, then one row per true class.
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

    # Same data + same seed as train.py -> exactly the same held-out avocados.
    _, x_test, _, y_test = split_data(load_data())
    model = joblib.load(model_path)
    metrics = compute_metrics(y_test, model.predict(x_test))

    # Apply the quality gate and record the decision next to the metrics.
    min_f1 = get_min_f1()
    passed = metrics["f1_macro"] >= min_f1
    metrics["min_f1"] = min_f1
    metrics["gate_passed"] = passed

    # Outputs are written even when the gate fails, so the job summary shows why.
    write_outputs(metrics, render_report(metrics, min_f1, passed))
    return metrics, passed


def main() -> None:
    """CLI entry point: exit with code 1 when the quality gate fails."""
    metrics, passed = evaluate()

    # Plain ASCII on the console (Windows terminals choke on emoji); Markdown goes to report.md.
    for key in SUMMARY_KEYS:
        print(f"{key:>16}: {metrics[key]}")
    print(f"Report written to {config.REPORT_PATH}")

    # Non-zero exit code = failed CI step = no artifact = no deployment.
    if not passed:
        print(f"Quality gate failed: f1_macro={metrics['f1_macro']} < {metrics['min_f1']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""All paths, constants and column names used across the project."""

from __future__ import annotations

from pathlib import Path

# Paths are relative to the repository root (where the Makefile lives).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "avocado_features.csv"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = METRICS_DIR / "metrics.json"
REPORT_PATH = METRICS_DIR / "report.md"

RANDOM_STATE = 42
# 5 folds -> the held-out fold is ~20% of the avocados.
N_SPLITS = 5
DEFAULT_MIN_F1 = 0.75

# Column used to keep every photo of the same avocado on one side of the split.
GROUP_COLUMN = "sample"
TARGET_COLUMN = "ripening_index"
CATEGORICAL_FEATURES = ["storage_group"]
NUMERIC_FEATURES = [
    "day",
    "l_mean",
    "l_std",
    "a_mean",
    "a_std",
    "b_mean",
    "b_std",
    "dark_fraction",
    "fruit_fraction",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

STORAGE_GROUPS = ["T10", "T20", "Tam"]
# 5-stage Ripening Index from Xavier, Rodrigues & Silva (2024).
CLASS_NAMES = {
    1: "underripe",
    2: "breaking",
    3: "ripe_first_stage",
    4: "ripe_second_stage",
    5: "overripe",
}

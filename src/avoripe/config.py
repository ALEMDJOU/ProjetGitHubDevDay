"""All paths, constants and column names used across the project.

Every other module imports its settings from here, so changing a path, a feature or the
quality-gate default only ever happens in this one file.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# ROOT_DIR is the repository root: this file lives in <root>/src/avoripe/config.py,
# so we go up two parent directories. All other paths are built from it, which makes
# the commands work no matter which directory they are launched from.
ROOT_DIR = Path(__file__).resolve().parents[2]

# Real feature table (one row per real photo), committed in the repo. See data/README.md.
DATA_PATH = ROOT_DIR / "data" / "raw" / "avocado_features.csv"

# Outputs of the pipeline. Both folders are git-ignored: CI uploads them as artifacts.
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
MODEL_PATH = MODELS_DIR / "model.joblib"  # fitted scikit-learn Pipeline
METRICS_PATH = METRICS_DIR / "metrics.json"  # machine-readable metrics
REPORT_PATH = METRICS_DIR / "report.md"  # Markdown report for the GitHub job summary

# ---------------------------------------------------------------------------
# Reproducibility and evaluation
# ---------------------------------------------------------------------------
# Fixed seed: the split and the random forest give identical results on every run and OS.
RANDOM_STATE = 42

# 5 folds -> the held-out fold is ~20% of the avocados.
N_SPLITS = 5

# Quality gate: minimum macro F1 a model needs to be deployed. It is overridden by the
# MIN_F1 environment variable (repository variable or workflow input) when set.
DEFAULT_MIN_F1 = 0.75

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
# Avocado ID. Each avocado is photographed many times, so the split groups on this column
# to keep every photo of the same fruit on one side (train OR test), never both.
GROUP_COLUMN = "sample"

# Label to predict: the 5-stage ripening index assigned by the dataset authors (1..5).
TARGET_COLUMN = "ripening_index"

# Input features of the model. The API accepts exactly these fields (see schemas.py).
CATEGORICAL_FEATURES = ["storage_group"]  # storage condition, one-hot encoded
NUMERIC_FEATURES = [
    "day",  # day of the experiment the photo was taken
    "l_mean",  # mean CIE Lab lightness of the skin (darker = riper)
    "l_std",  # spread of lightness (spots, uneven colour)
    "a_mean",  # mean a*: negative = green, positive = red/brown
    "a_std",
    "b_mean",  # mean b*: positive = yellow, drops as the skin turns dark
    "b_std",
    "dark_fraction",  # share of skin pixels with L* < 30 (almost black skin)
    "fruit_fraction",  # share of the image covered by the fruit (size proxy)
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Allowed values of storage_group: T10 = 10 °C, T20 = 20 °C (both 85% RH), Tam = ambient.
STORAGE_GROUPS = ["T10", "T20", "Tam"]

# 5-stage Ripening Index from Xavier, Rodrigues & Silva (2024).
# Keys are the values stored in the CSV; values are the class names returned by the API.
CLASS_NAMES = {
    1: "underripe",
    2: "breaking",
    3: "ripe_first_stage",
    4: "ripe_second_stage",
    5: "overripe",
}

"""Load, validate and split the avocado ripeness feature table.

Public functions:
- load_data():  read data/raw/avocado_features.csv, validate it, add a readable label.
- split_data(): return (x_train, x_test, y_train, y_test) with a leak-free split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from avoripe import config


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate_data(df: pd.DataFrame) -> None:
    """Check required columns, missing values and allowed label/group values."""
    # 1. Every column the pipeline needs must be present.
    required = [config.GROUP_COLUMN, config.TARGET_COLUMN, *config.FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # 2. No empty cells: the model cannot handle NaN and we never impute invented values.
    if df[required].isna().any().any():
        raise DataValidationError("Dataset contains missing values in required columns")

    # 3. Labels must be one of the 5 official ripening stages (1..5).
    bad_labels = set(df[config.TARGET_COLUMN]) - set(config.CLASS_NAMES)
    if bad_labels:
        raise DataValidationError(f"Unknown ripening index values: {sorted(bad_labels)}")

    # 4. Storage group must be one of the three conditions of the experiment.
    bad_groups = set(df["storage_group"]) - set(config.STORAGE_GROUPS)
    if bad_groups:
        raise DataValidationError(f"Unknown storage groups: {sorted(bad_groups)}")


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV, validate it and add a human-readable `label` column."""
    # The default path is resolved at call time (not at import time) so tests can
    # monkeypatch config.DATA_PATH and point the whole pipeline at a small fixture file.
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    validate_data(df)

    # The model is trained on class names ("ripe_first_stage"...) rather than 1..5,
    # so predictions come out readable in the API without any extra mapping.
    df["label"] = df[config.TARGET_COLUMN].map(config.CLASS_NAMES)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split by avocado, so no fruit appears in both train and test."""
    # Why not train_test_split? Each avocado is photographed on both sides, every day.
    # A random row split would put near-identical photos of the same fruit in train AND
    # test, and the score would look better than it really is (data leakage).
    #
    # StratifiedGroupKFold solves both needs at once:
    # - "Group":      all photos of one avocado (same `sample`) stay in the same fold;
    # - "Stratified": each fold keeps roughly the same class proportions.
    # We only use the first of the 5 folds as the test set (~20% of the avocados).
    splitter = StratifiedGroupKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    groups = df[config.GROUP_COLUMN]
    train_idx, test_idx = next(splitter.split(df, df[config.TARGET_COLUMN], groups=groups))

    train, test = df.iloc[train_idx], df.iloc[test_idx]
    # Only the model features go into X; the readable class name is the target y.
    return train[config.FEATURES], test[config.FEATURES], train["label"], test["label"]

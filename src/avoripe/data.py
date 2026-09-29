"""Load, validate and split the avocado ripeness feature table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from avoripe import config


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate_data(df: pd.DataFrame) -> None:
    """Check required columns, missing values and allowed label/group values."""
    required = [config.GROUP_COLUMN, config.TARGET_COLUMN, *config.FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df[required].isna().any().any():
        raise DataValidationError("Dataset contains missing values in required columns")
    bad_labels = set(df[config.TARGET_COLUMN]) - set(config.CLASS_NAMES)
    if bad_labels:
        raise DataValidationError(f"Unknown ripening index values: {sorted(bad_labels)}")
    bad_groups = set(df["storage_group"]) - set(config.STORAGE_GROUPS)
    if bad_groups:
        raise DataValidationError(f"Unknown storage groups: {sorted(bad_groups)}")


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV, validate it and add a human-readable `label` column."""
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    validate_data(df)
    df["label"] = df[config.TARGET_COLUMN].map(config.CLASS_NAMES)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split by avocado, so no fruit appears in both train and test."""
    # A plain row split would leak: each avocado is photographed on many days.
    # The first fold of a 5-fold StratifiedGroupKFold is a ~20% stratified, grouped test set.
    splitter = StratifiedGroupKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    groups = df[config.GROUP_COLUMN]
    train_idx, test_idx = next(splitter.split(df, df[config.TARGET_COLUMN], groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    return train[config.FEATURES], test[config.FEATURES], train["label"], test["label"]

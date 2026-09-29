"""Tests for data loading, validation and splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from avoripe import config
from avoripe.data import DataValidationError, load_data, split_data


def test_load_data_adds_readable_label(raw_csv: Path) -> None:
    """Labels are mapped from the numeric ripening index to class names."""
    df = load_data(raw_csv)
    assert set(df["label"]) <= set(config.CLASS_NAMES.values())


def test_missing_file_raises(tmp_path: Path) -> None:
    """A clear error is raised when the CSV does not exist."""
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_missing_column_raises(raw_csv: Path) -> None:
    """Dropping a required feature column fails validation."""
    df = pd.read_csv(raw_csv).drop(columns=["l_mean"])
    df.to_csv(raw_csv, index=False)
    with pytest.raises(DataValidationError, match="l_mean"):
        load_data(raw_csv)


def test_unknown_label_raises(raw_csv: Path) -> None:
    """A ripening index outside 1..5 fails validation."""
    df = pd.read_csv(raw_csv)
    df.loc[0, config.TARGET_COLUMN] = 9
    df.to_csv(raw_csv, index=False)
    with pytest.raises(DataValidationError, match="ripening index"):
        load_data(raw_csv)


def test_unknown_storage_group_raises(raw_csv: Path) -> None:
    """A storage group outside T10/T20/Tam fails validation."""
    df = pd.read_csv(raw_csv)
    df.loc[0, "storage_group"] = "T99"
    df.to_csv(raw_csv, index=False)
    with pytest.raises(DataValidationError, match="storage groups"):
        load_data(raw_csv)


def test_missing_values_raise(raw_csv: Path) -> None:
    """Missing values in required columns fail validation."""
    df = pd.read_csv(raw_csv)
    df.loc[0, "day"] = None
    df.to_csv(raw_csv, index=False)
    with pytest.raises(DataValidationError, match="missing"):
        load_data(raw_csv)


def test_split_is_deterministic_and_grouped(real_df: pd.DataFrame) -> None:
    """Same split every run, and no avocado appears on both sides."""
    x_train, x_test, _, _ = split_data(real_df)
    x_train2, x_test2, _, _ = split_data(real_df)
    assert list(x_test.index) == list(x_test2.index)
    assert list(x_train.index) == list(x_train2.index)
    train_ids = set(real_df.loc[x_train.index, config.GROUP_COLUMN])
    test_ids = set(real_df.loc[x_test.index, config.GROUP_COLUMN])
    assert train_ids.isdisjoint(test_ids)
    assert list(x_train.columns) == config.FEATURES

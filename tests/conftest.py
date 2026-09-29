"""Shared fixtures: small subsets of the real dataset and a tiny trained model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from avoripe import config
from avoripe.data import load_data
from avoripe.train import build_pipeline

# Test fixture: real rows for a handful of avocados per storage group.
AVOCADOS_PER_GROUP = 8


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    """A small, real subset of the dataset (all photos of a few avocados)."""
    df = load_data()
    samples = (
        df.groupby("storage_group")[config.GROUP_COLUMN]
        .apply(lambda s: s.drop_duplicates().head(AVOCADOS_PER_GROUP))
        .tolist()
    )
    return df[df[config.GROUP_COLUMN].isin(samples)].reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_model_path(real_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fit the real pipeline on the fixture rows and save it to a temp file."""
    model = build_pipeline().set_params(classifier__n_estimators=20)
    model.fit(real_df[config.FEATURES], real_df["label"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture()
def raw_csv(real_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the fixture rows (without the derived label) to a temp CSV."""
    path = tmp_path / "data.csv"
    real_df.drop(columns=["label"]).to_csv(path, index=False)
    return path

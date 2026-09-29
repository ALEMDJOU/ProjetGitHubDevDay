"""Shared fixtures: small subsets of the real dataset and a tiny trained model.

Rule of this repo: tests only use REAL rows from data/raw/avocado_features.csv.
No synthetic data, no network, and no dependency on models/model.joblib, so the
whole suite runs in a few seconds on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from avoripe import config
from avoripe.data import load_data
from avoripe.train import build_pipeline

# Test fixture: real rows for a handful of avocados per storage group.
# 8 avocados x 3 groups ~= 24 fruits, all their photos: small but covers every stage.
AVOCADOS_PER_GROUP = 8


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    """A small, real subset of the dataset (all photos of a few avocados)."""
    df = load_data()
    # Take the first N distinct avocado IDs of each storage group...
    samples = (
        df.groupby("storage_group")[config.GROUP_COLUMN]
        .apply(lambda s: s.drop_duplicates().head(AVOCADOS_PER_GROUP))
        .tolist()
    )
    # ...and keep every photo of those avocados (whole fruits, so grouped splits work).
    return df[df[config.GROUP_COLUMN].isin(samples)].reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_model_path(real_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fit the real pipeline on the fixture rows and save it to a temp file."""
    # Same Pipeline as production, only with fewer trees so it trains in milliseconds.
    # The API tests load it through the MODEL_PATH environment variable.
    model = build_pipeline().set_params(classifier__n_estimators=20)
    model.fit(real_df[config.FEATURES], real_df["label"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture()
def raw_csv(real_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the fixture rows (without the derived label) to a temp CSV."""
    # Same format as the real CSV on disk: tests can corrupt a copy of it (drop a column,
    # add a bad label...) to check validation, without touching the committed dataset.
    path = tmp_path / "data.csv"
    real_df.drop(columns=["label"]).to_csv(path, index=False)
    return path

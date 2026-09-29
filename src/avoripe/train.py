"""Train the ripeness classifier and save it as a single scikit-learn Pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from avoripe import config
from avoripe.data import load_data, split_data


def build_pipeline() -> Pipeline:
    """Return the preprocessing + classifier pipeline (unfitted)."""
    # One-hot the storage group; tree models need no scaling for numeric features.
    preprocess = ColumnTransformer(
        [
            ("storage", OneHotEncoder(categories=[config.STORAGE_GROUPS]), ["storage_group"]),
            ("numeric", "passthrough", config.NUMERIC_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def save_model(model: Pipeline, path: Path | None = None) -> Path:
    """Persist the fitted pipeline with joblib and return its path."""
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def main() -> None:
    """Load data, fit the pipeline on the training split and save it."""
    start = time.perf_counter()
    x_train, _, y_train, _ = split_data(load_data())
    model = build_pipeline().fit(x_train, y_train)
    # Train in parallel, but serve single-threaded: a thread pool per 1-row request is slow.
    model.set_params(classifier__n_jobs=1)
    path = save_model(model)
    elapsed = time.perf_counter() - start
    print(f"Trained on {len(x_train)} rows in {elapsed:.1f}s -> {path}")


if __name__ == "__main__":
    main()

"""Train the ripeness classifier and save it as a single scikit-learn Pipeline.

Run with `make train` (or `python -m avoripe.train`). Output: models/model.joblib.
The saved Pipeline contains the preprocessing AND the classifier, so the API only has to
load one file and call predict_proba() on raw feature values.
"""

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
    # Preprocessing, applied column by column:
    # - storage_group (text) -> 3 one-hot columns. Categories are fixed explicitly so the
    #   encoding never depends on which groups happen to be in the training rows.
    # - numeric features pass through unchanged: tree models do not need scaling.
    preprocess = ColumnTransformer(
        [
            ("storage", OneHotEncoder(categories=[config.STORAGE_GROUPS]), ["storage_group"]),
            ("numeric", "passthrough", config.NUMERIC_FEATURES),
        ]
    )

    # Random forest chosen after comparing variants on the real split:
    # 100 trees with min_samples_leaf=5 scored the same macro F1 as 200 trees / leaf 2,
    # but the saved model is ~3x smaller (3.7 MB) and trains in about a second.
    classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,  # each leaf needs 5 photos: smoother, smaller trees
        n_jobs=-1,  # use all CPU cores while training
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def save_model(model: Pipeline, path: Path | None = None) -> Path:
    """Persist the fitted pipeline with joblib and return its path."""
    # Resolved at call time so tests can redirect config.MODEL_PATH to a temp folder.
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # compress=3: good size/speed trade-off, keeps the Docker image and artifact small.
    joblib.dump(model, path, compress=3)
    return path


def main() -> None:
    """Load data, fit the pipeline on the training split and save it."""
    start = time.perf_counter()

    # Only the training part of the split is used here; evaluate.py recomputes the exact
    # same split (fixed seed) and scores the model on the held-out avocados.
    x_train, _, y_train, _ = split_data(load_data())
    model = build_pipeline().fit(x_train, y_train)

    # Train in parallel, but serve single-threaded: starting a thread pool for every
    # one-row API request costs far more than the prediction itself.
    model.set_params(classifier__n_jobs=1)

    path = save_model(model)
    elapsed = time.perf_counter() - start
    print(f"Trained on {len(x_train)} rows in {elapsed:.1f}s -> {path}")


if __name__ == "__main__":
    main()

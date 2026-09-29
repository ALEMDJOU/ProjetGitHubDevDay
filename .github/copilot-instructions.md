# Copilot instructions: Avocado Ripeness MLOps

This repo demonstrates an automated MLOps pipeline with GitHub Actions. The model is small on
purpose; the pipeline is the product.

## Stack (fixed)

Python 3.11, pandas, scikit-learn, joblib, FastAPI, Uvicorn, Pydantic v2, pytest, Ruff, Docker,
GitHub Actions. Do not suggest PyTorch, TensorFlow, MLflow, DVC, Kubernetes or paid services.

## Data rules

- Only real data from `data/raw/avocado_features.csv` (derived from real photos, CC BY 4.0).
- Never generate, simulate or fabricate rows, not even in tests. Tests use real rows via the
  `real_df` fixture in `tests/conftest.py`.
- Split with `split_data()`: it is stratified and grouped by `sample` (one avocado never ends up
  in both train and test).

## Code conventions

- Type hints everywhere, a one-line docstring on every function, functions under ~25 lines.
- Paths, constants and column names live in `src/avoripe/config.py`; `RANDOM_STATE = 42`.
- The model is one scikit-learn `Pipeline` saved with joblib to `models/model.joblib`.
- Ruff, line length 100. Run `make lint test` before proposing a change.

## Workflows

- `ci.yml`: lint, tests (coverage >= 80%), Docker build + `/health` smoke test.
- `train.yml`: train, evaluate, quality gate on macro F1 (`MIN_F1`), upload artifact `model`.
- `deploy.yml`: after a successful train, push image to GHCR and files to the HF Space.
- `monitor.yml`: every 6 h, check `/health` and a known prediction; open an issue on failure.
- Keep minimal `permissions:`, `timeout-minutes` on every job, actions pinned to major versions.

## Never

- Never weaken a test or lower the quality gate to make CI green.
- Never write metrics or timings into docs unless they come from a real run.
- Never commit secrets; use GitHub Secrets (`HF_TOKEN`) and Variables (`HF_SPACE`,
  `HF_SPACE_URL`, `MIN_F1`).

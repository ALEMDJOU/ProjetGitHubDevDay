# CLAUDE.md

Instructions for Claude Code working in this repository. Read this file fully before doing anything, and re-read it when starting a new task.

## Project

**Avocado Ripeness MLOps** is a small, fully automated MLOps pipeline that classifies avocado ripeness (e.g. unripe / ripe / overripe, exact classes come from the dataset). It exists to support a 4-5 minute live demo at a GitHub Dev Day talk titled *"Accelerating MLOps: Automating the AI pipeline with GitHub Actions and Copilot"*.

The ML model is **not** the point. The point is that every change (code, data, hyperparameters) flows through one automated path:

`lint + tests -> train -> evaluate (quality gate) -> package (Docker) -> deploy -> monitor`

Everything must be simple, fast, reproducible by workshop participants, and readable by GitHub Copilot.

## Hard constraints (never violate)

1. **Real data only. No synthetic, simulated, generated or augmented-by-fabrication data**, not even for tests or examples. Tests may use a few real rows copied from the dataset or tiny inline fixtures clearly labeled as test fixtures, never fabricated datasets presented as data.
2. **No sensitive data.** Public, openly licensed dataset only. Record source URL, license and citation in `data/README.md`.
3. **No GPU, no paid cloud.** Full train + evaluate must run in under 2 minutes on a GitHub-hosted runner; the whole pipeline in under 10 minutes.
4. **Stack is fixed:** Python 3.11, pandas, scikit-learn, joblib, FastAPI, Uvicorn, pytest, Ruff, Docker, GitHub Actions. Do not add heavy frameworks (no PyTorch, TensorFlow, MLflow, DVC, Kubernetes) unless I explicitly ask.
5. **Deployment target is free:** Docker image on GHCR + Hugging Face Space (Docker SDK, port 7860). Never require paid services.
6. **Never invent numbers.** Do not write accuracy, F1, timings or speedups into the README, docs or slides unless they come from a real run you executed or a real workflow run I gave you. If unknown, leave a clearly marked `TODO(measure)`.
7. **Never commit secrets.** Tokens live in GitHub Secrets/Variables only.

## Phase 0: dataset selection (do this first, then stop and ask me)

Find a real, public **avocado ripeness** dataset that fits the constraints. Prefer a **small tabular** dataset (features such as firmness, color/hue, sound, weight, size, with a ripeness label) because it trains in seconds and can live in the repo. Image datasets are acceptable only if they can be reduced to a lightweight classical pipeline (no GPU) and stay small enough to keep CI fast.

Before writing any pipeline code:
- Search and verify: source, license (must permit redistribution or documented download), size, row count, column schema, class balance, missing values, duplicates.
- If the file is under about 5 MB and the license allows it, commit it to `data/raw/`. Otherwise download it in CI from a stable URL with a pinned **SHA-256 checksum** and cache it.
- Write `data/README.md` (source, license, citation, schema, checksum, class distribution).
- **Report your findings and the proposed target column and classes, then wait for my confirmation.** If no suitable real dataset is found, tell me. Do not fall back to generating data.

## Target repository layout

```text
avocado-ripeness-mlops/
├── .github/
│   ├── workflows/{ci.yml,train.yml,deploy.yml,monitor.yml}
│   ├── ISSUE_TEMPLATE/{bug_report.md,feature_request.md}
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   ├── dependabot.yml
│   └── copilot-instructions.md
├── data/{raw/,README.md}
├── src/avoripe/{__init__,config,data,train,evaluate,schemas,api}.py
├── tests/{conftest,test_data,test_train,test_api}.py
├── deploy/space_README.md      # HF Space front matter: sdk: docker, app_port: 7860
├── docs/{architecture.md,demo-script.md}
├── models/.gitkeep
├── metrics/.gitkeep
├── assets/                     # README images and demo GIF
├── Dockerfile  .dockerignore  .gitignore  Makefile
├── pyproject.toml              # ruff + pytest config
├── requirements.txt  requirements-dev.txt
├── LICENSE  README.md  CLAUDE.md
```

## Code conventions (Copilot-friendly)

- Type hints everywhere; every function has a one-line docstring.
- Short, single-purpose functions (aim for under 25 lines). Clear names, no clever metaprogramming.
- Put a clear comment above non-obvious blocks so Copilot completes well.
- All paths and constants in `config.py`; fixed `RANDOM_STATE = 42`; stratified split.
- Model is a single scikit-learn `Pipeline` (preprocessing + classifier) saved with joblib. Start with `RandomForestClassifier` or `LogisticRegression`; pick whichever is simple and clearly good on the real data.
- Ruff for lint and format (line length 100). Pin dependency versions in `requirements*.txt` after installing and testing them.

## Module contracts

- `data.py`: `load_data()`, `split_data()`; validates required columns and label values; raises clear errors.
- `train.py`: `build_pipeline()`, `main()`; writes `models/model.joblib`.
- `evaluate.py`: computes accuracy, macro precision, macro recall, macro F1 and a confusion matrix; writes `metrics/metrics.json` and `metrics/report.md` (Markdown table for the GitHub job summary); reads `MIN_F1` (macro F1) from the environment and **exits with code 1 if the gate fails**.
- `api.py`: FastAPI app with `GET /health`, `POST /predict` (validated Pydantic input of the feature fields, returns predicted class plus class probabilities), `GET /metrics` (request count, mean latency, class distribution of predictions). Model path comes from the `MODEL_PATH` env var so tests can inject a temporary model.
- `schemas.py`: Pydantic models with sensible field bounds and example values taken from the real dataset.

## Testing rules

- `pytest --cov=src --cov-fail-under=80`. Tests must run in a few seconds and need no network.
- API tests build a tiny model in a fixture (`tmp_path`) from a few real dataset rows; never depend on `models/model.joblib`.
- Cover: data validation errors, deterministic split, pipeline fit/predict, API happy path, invalid input (422), health endpoint.

## GitHub Actions requirements

| Workflow | Triggers | Must do |
|---|---|---|
| `ci.yml` | PR, push to `main` | Ruff check + format check, pytest with coverage, Docker build + run container + `curl /health`. Use `concurrency` with cancel-in-progress. |
| `train.yml` | push to `main` on `data/**` or training code, weekly cron, `workflow_dispatch` (input `min_f1`) | Train, evaluate with the quality gate, append `metrics/report.md` to `$GITHUB_STEP_SUMMARY` (with `if: always()`), upload `model.joblib` + `metrics.json` as artifact `model` (30 days). |
| `deploy.yml` | `workflow_run` of "Train & Evaluate" on success, `workflow_dispatch` (input `run_id`) | Download artifact, build and push image to GHCR (`sha` + `latest`), push files to the Hugging Face Space, smoke test the live `/health`. Job uses environment `production`. |
| `monitor.yml` | cron every 6 h, manual | Check `/health` and one known prediction; on failure open a GitHub issue with a link to the run. |

Rules for all workflows: minimal `permissions:` per workflow, `timeout-minutes` on jobs, actions pinned to major versions, pip cache enabled, no secrets echoed.

Secrets and variables to document (not to create): secret `HF_TOKEN` (environment `production`); variables `HF_SPACE`, `HF_SPACE_URL`, `MIN_F1`.

## Makefile targets

`make install`, `make lint`, `make format`, `make test`, `make train`, `make evaluate`, `make run` (Uvicorn on port 7860), `make docker`, `make all` (lint, test, train, evaluate). Participants must be able to reproduce everything with these.

## README.md: make it beautiful

When the pipeline works, write a **beautiful, professional, visually polished `README.md`**. It is the first thing the audience sees, so treat it as a product page. Requirements:

- **Hero section:** project name, one-line pitch, a short tagline about the talk, and a row of badges (CI, Train & Evaluate, Deploy, Python 3.11, license, "Open in Hugging Face Space").
- **Demo visual:** a screenshot or GIF of the live API/Space in `assets/` (leave a clearly marked placeholder with instructions if I have not provided the image yet; never fake a screenshot).
- **Why this project:** 3 to 4 short lines on the problem (manual ML steps) and the automated solution.
- **Architecture:** a Mermaid flowchart of `data -> CI -> train -> evaluate gate -> package -> deploy -> monitor`, plus a table mapping each stage to the workflow and tool.
- **Quick start** in under 5 commands (clone, venv, `make install`, `make train evaluate`, `make run`) and a `curl` example for `/predict` using **real feature values from the dataset** with the real response format.
- **The pipeline:** a table of the four workflows (trigger, what it does, outputs), and a short explanation of the **quality gate** and how `MIN_F1` changes it.
- **Results:** a table of the real metrics from `metrics/metrics.json` and the class list; a confusion matrix image only if generated from a real run. No invented numbers (`TODO(measure)` if not yet run).
- **How GitHub Copilot is used** in this repo (workflows, tests, docs, code review, onboarding) with 1 to 2 real example prompts.
- **Dataset** section: source, license, citation, class distribution.
- **Project structure** as a tree with one-line descriptions, kept short.
- **Configuration**: secrets and variables table.
- **Limitations and roadmap:** honest bullets (toy scale, data versioning, model registry, minimal monitoring).
- **Contributing and license.**
- Style: clear headings, concise text, tasteful emoji as section markers (not everywhere), tables instead of long paragraphs, collapsible `<details>` blocks for long content, consistent formatting, no walls of text, no broken links or images. Preview it mentally as rendered on GitHub and verify all relative links.

## Order of work

1. Phase 0: dataset selection and my confirmation.
2. Scaffold the repo tree, `pyproject.toml`, requirements, Makefile, `.gitignore`, `.dockerignore`.
3. `data.py`, `train.py`, `evaluate.py` + tests; run locally and report the real metrics.
4. `api.py`, `schemas.py` + tests, Dockerfile; verify `docker build` and `/health` locally.
5. `ci.yml`, then `train.yml`, then `deploy.yml`, then `monitor.yml`.
6. `.github/` files (templates, CODEOWNERS, Dependabot, `copilot-instructions.md`).
7. `docs/demo-script.md`: 5-minute demo run of show with timings and a fallback plan.
8. The beautiful `README.md`.
9. Final check: `make all`, fresh-clone test of the Quick start, list of every `TODO(measure)` left.

## How to work with me

- Work in small steps; after each step, show what changed and the exact command to verify it.
- Use Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, `test:`, `chore:`) and small, focused commits.
- **Commit attribution:** commits are authored by the repository owner. End every commit message with `Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>`. Never add a Claude or Anthropic co-author trailer or any "Generated with Claude" line to commits or PR descriptions.
- Ask me before making a choice that changes the dataset, target classes, stack or deployment target. Otherwise decide, state your assumption in one line, and continue.
- Be direct and practical: no filler, no long explanations unless asked.
- If something fails (tests, workflow, deploy), show the real error and fix the cause; do not weaken tests or the quality gate to make it pass.
- Anything I will show on stage must be verified by actually running it.
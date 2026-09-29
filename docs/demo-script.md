# Demo script: 5 minutes on stage

**Talk:** *Accelerating MLOps: Automating the AI pipeline with GitHub Actions and Copilot*

**One message:** every change (code, data, hyperparameters) goes through one automated path:
`lint + tests -> train -> evaluate (quality gate) -> package -> deploy -> monitor`.

## Before going on stage (T-30 min)

| Check | How |
|---|---|
| API is up | `curl -s $APP_URL/health` returns `"model_loaded":true` and the latest commit SHA as `version` |
| Green runs exist | Latest `CI`, `Train & Evaluate`, `Deploy` runs on `main` are green; keep their tabs open |
| A red gate run exists | One `Train & Evaluate` run dispatched with `min_f1 = 0.95` (fails); keep its tab open |
| Demo branch ready | Branch `demo/more-trees` with a one-line change in `src/avoripe/train.py`, PR **not** opened yet |
| Local fallback | `make run` works on the laptop; Docker Desktop running; `avoripe:local` image built |
| Tabs open, in order | 1 README · 2 PR page · 3 Actions (CI run) · 4 Train job summary · 5 red gate run · 6 GHCR package · 7 Render `/docs` · 8 Monitor workflow |
| Font size | Browser zoom 150%, terminal font 20pt |

## Run of show

| Time | Show | Say (short) |
|---|---|---|
| 0:00–0:30 | **README** hero + Mermaid diagram | "Real photos of 478 avocados, a small model, and a pipeline where nobody runs a notebook by hand." |
| 0:30–1:15 | **Open the PR** from `demo/more-trees` (Copilot writes the PR description). Actions tab: `CI` starts | "I change one hyperparameter. That's all I do by hand. Copilot drafts the PR; Actions takes over." |
| 1:15–2:00 | **CI run** (pre-recorded green run if the live one is still going): Ruff, pytest with coverage, Docker build + `/health` | "Lint, 22 tests, coverage gate at 80%, and the container must answer `/health` before anything merges." |
| 2:00–3:00 | **Train & Evaluate job summary**: metrics table + confusion matrix. Then the **red run** with `min_f1 = 0.95` | "Training takes seconds. The gate is macro F1 against `MIN_F1`. Raise the bar and the pipeline refuses to ship: no artifact, no deploy." |
| 3:00–4:00 | **Deploy**: `workflow_run` trigger, `production` environment, GHCR image tagged with the SHA, then the **live `/docs` on Render**: `/health` shows the commit SHA, run `/predict` with the example | "Only a model that passed the gate reaches production: the exact image from GHCR, verified by its commit SHA." |
| 4:00–4:40 | **Monitor** workflow: cron every 6 h, `/health` + a known real prediction, opens an issue on failure | "And if it breaks at 3 a.m., we get an issue with a link to the run, not a surprise." |
| 4:40–5:00 | Back to README "How Copilot is used" | "Copilot wrote the tests, the workflows and reviews every PR. Clone it; `make all` runs everything." |

### The one-line change for the demo branch

```diff
-        n_estimators=100,
+        n_estimators=150,
```

## Timings

Measured locally (Windows laptop, CPU): `make train` ≈ 1.5–6 s, `make evaluate` ≈ 3 s,
full test suite ≈ 7 s (pytest time).

| Workflow | Duration on GitHub runner |
|---|---|
| CI | Lint & test 28 s, then Docker build & smoke test 43 s (run 36566619977) |
| Train & Evaluate | 30 s job (run 36567592198) |
| Deploy (incl. Render rollout) | 157 s, until live `/health` reported the new SHA (run 36577128199) |
| Monitor | TODO(measure) |

If Deploy is longer than ~1 minute, **never wait for it live**: show the last green run and the
live API.

## Fallback plan

| If… | Then… |
|---|---|
| Wi-Fi is slow or GitHub is down | Use the pre-opened tabs of the green and red runs; say "this ran this morning". |
| A live run is still in progress | Switch to the pre-recorded run of the same workflow; come back at the end if time allows. |
| The Render service is down | Local: `make run`, then open `http://localhost:7860/docs`. |
| The laptop Python env is broken | `docker run --rm -p 7860:7860 avoripe:local` |
| Everything is down | Screenshots/GIF in `assets/` (TODO: record before the talk) and the README. |

## Commands to keep handy

```bash
# Live prediction with a real photo's features
curl -s -X POST "$APP_URL/predict" -H "Content-Type: application/json" \
  -d @tests/fixtures/known_sample.json

# Trigger a red gate run from the terminal (GitHub CLI)
gh workflow run "Train & Evaluate" -f min_f1=0.95
```

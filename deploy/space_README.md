---
title: Avocado Ripeness API
emoji: 🥑
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predicts the 5-stage ripening index of Hass avocados.
---

# 🥑 Avocado Ripeness API

FastAPI service deployed automatically by the GitHub Actions pipeline
(`lint + tests -> train -> evaluate gate -> package -> deploy -> monitor`).

- `GET /health`: liveness and model status
- `POST /predict`: storage conditions + skin colour statistics -> ripeness class + probabilities
- `GET /metrics`: request count, mean latency, predicted-class distribution
- `GET /docs`: interactive Swagger UI

Model trained on features extracted from the *'Hass' Avocado Ripening Photographic Dataset*
(Xavier, Rodrigues & Silva, 2024, Mendeley Data, DOI 10.17632/3xd9n945v8.1, CC BY 4.0).

Do not edit this Space by hand: every file here is overwritten by the `Deploy` workflow.

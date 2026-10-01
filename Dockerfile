FROM python:3.11-slim

# Commit SHA baked in at build time; /health reports it so deploys can be verified.
ARG GIT_SHA=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    GIT_SHA=${GIT_SHA}

# Hugging Face Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
# Skip the editable `-e .` line: src/ is copied below and found through PYTHONPATH,
# which keeps this dependency layer cached when only the code changes.
RUN grep -v '^-e ' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ src/
COPY models/model.joblib models/model.joblib

USER app
# Render injects PORT; Hugging Face and local runs use 7860.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.getenv('PORT', '7860'))"
CMD ["sh", "-c", "uvicorn avoripe.api:app --host 0.0.0.0 --port ${PORT:-7860}"]

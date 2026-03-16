# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.14
FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    gcc \
    libcairo2-dev \
    libffi-dev \
    libjpeg62-turbo-dev \
    libpango1.0-dev \
    pkg-config \
    poppler-utils \
    shared-mime-info \
    tesseract-ocr \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY pyproject.toml ./pyproject.toml

RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel \
    && python - <<'PY' > /tmp/requirements.txt
import tomllib
with open('pyproject.toml', 'rb') as fh:
    deps = tomllib.load(fh)['project']['dependencies']
print('gunicorn>=23.0.0')
for dep in deps:
    print(dep)
PY

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.txt

COPY --chown=appuser:appuser . .

RUN mkdir -p generated-scripts generated-flowcharts media flowchart_media uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --chdir src server:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 300"]

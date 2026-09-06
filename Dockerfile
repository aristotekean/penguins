# Un solo Dockerfile, dos targets: `api` y `jupyter`.
#
# Los dos instalan desde el MISMO pyproject.toml / uv.lock, así que comparten
# exactamente la misma versión de scikit-learn. Esa es la condición para que un
# .pkl escrito por Jupyter se pueda deserializar en FastAPI.

# --------------------------------------------------------------------------- #
# Base con uv
# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv-base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
# uv.lock* : si el lockfile existe se respeta; si no, uv resuelve en el build.
# Para builds 100% reproducibles: correr `uv lock` en el host y commitearlo.
COPY pyproject.toml uv.lock* ./

# --------------------------------------------------------------------------- #
# Dependencias
# --------------------------------------------------------------------------- #
FROM uv-base AS deps-api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

FROM uv-base AS deps-jupyter
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --group jupyter

# --------------------------------------------------------------------------- #
# Runtime · API de inferencia
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    MODELS_DIR="/models"

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app
COPY --from=deps-api /app/.venv /app/.venv
COPY src/ /app/src/
COPY api/ /app/api/

# /models existe en la imagen con dueño 1000: al crear el volumen vacío,
# Docker copia esos permisos, y el contenedor no-root puede leerlo.
RUN mkdir -p /models && chown -R 1000:1000 /models /app

USER app
EXPOSE 8025

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8025/health', timeout=4)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8025"]

# --------------------------------------------------------------------------- #
# Runtime · JupyterLab (entrenamiento)
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS jupyter

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/src" \
    MODELS_DIR="/models" \
    DATA_PATH="/workspace/data/penguins.csv" \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    UV_LINK_MODE=copy

RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m -s /bin/bash app

# uv queda disponible dentro del contenedor de desarrollo: `uv add <paquete>`
# actualiza pyproject.toml y uv.lock, y sincroniza el mismo .venv que ya está en PATH.
COPY --from=uv-base /usr/local/bin/uv /usr/local/bin/uv

WORKDIR /workspace
COPY --from=deps-jupyter /app/.venv /app/.venv
COPY pyproject.toml /workspace/pyproject.toml
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY notebooks/ /workspace/notebooks/
COPY data/ /workspace/data/

RUN mkdir -p /models && chown -R 1000:1000 /models /workspace /app

USER app
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.root_dir=/workspace"]

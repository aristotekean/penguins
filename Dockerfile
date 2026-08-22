# Stage 1: install dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies only (project is a flat script, not an installable package)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: runtime image without uv or build tooling
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app.py model_decisiontree.pkl model_logisticregression.pkl model_randomforest.pkl ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8025

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8025"]

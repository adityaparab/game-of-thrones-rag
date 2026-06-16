# Application image: serves either the FastAPI API or the Streamlit UI.
# Used by docker-compose (local) and by Railway (production).
# Production connects to an external Qdrant via QDRANT_URL — no DB in this image.
FROM python:3.12-slim AS base

# uv: fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install runtime deps only (no dev tools, no project, no ingest extra) — cached layer.
#    uv.lock is used if present (run `uv lock` to commit one); otherwise uv resolves here.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# 2) App source
COPY . .

# Default to the API; override APP=streamlit for the UI service.
ENV APP=api
EXPOSE 8000 8501

ENTRYPOINT ["bash", "docker/entrypoint.sh"]

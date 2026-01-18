FROM ghcr.io/astral-sh/uv:python3.13-alpine AS base

## Add application to app folder in container
# WORKDIR /app 

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY models ./models

# Cloud Run default port (suggested i google cloud)
# EXPOSE 8080

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]

## Suggested command to run container
# CMD ["sh", "-c", "exec uv run uvicorn mlops_project.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

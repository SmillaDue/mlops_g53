FROM ghcr.io/astral-sh/uv:python3.13-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY README.md README.md
COPY LICENSE LICENSE
COPY models/ models/
COPY reports/ reports/


WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "src/mlops_project/train.py"]

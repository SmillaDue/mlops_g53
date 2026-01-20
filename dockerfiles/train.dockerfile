FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    python3-dev pkg-config \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE
COPY models/ models/
COPY reports/ reports/
COPY configs/ configs/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync\

COPY scripts/entrypoint_train.sh /entrypoint_train.sh
RUN chmod +x /entrypoint_train.sh
ENTRYPOINT ["/entrypoint_train.sh"]
# ENTRYPOINT ["uv", "run", "src/mlops_project/train.py"]

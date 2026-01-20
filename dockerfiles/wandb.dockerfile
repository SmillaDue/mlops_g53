FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    python3-dev pkg-config \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install gcloud SDK for downloading data from GCS
RUN curl https://sdk.cloud.google.com | bash
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE
COPY configs/ configs/

# Create empty directories for data and models (will be populated at runtime)
RUN mkdir -p data/ models/ reports/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Copy entrypoint script for GCP data download
COPY scripts/entrypoint_train.sh /entrypoint_train.sh
RUN chmod +x /entrypoint_train.sh

# Set PYTHONPATH for module imports
ENV PYTHONPATH=/src

ENTRYPOINT ["/entrypoint_train.sh"]

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    python3-dev pkg-config \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy dependency metadata
COPY uv.lock pyproject.toml README.md LICENSE tasks.py ./

# Copy code
COPY src/ src/
COPY configs/ configs/

# Install deps
RUN uv sync --frozen --no-dev

EXPOSE 8080
ENTRYPOINT ["uvx", "invoke", "api"]
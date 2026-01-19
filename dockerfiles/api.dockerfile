FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran \
    python3-dev pkg-config \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata first (better Docker layer caching)
COPY uv.lock pyproject.toml README.md LICENSE ./ 

# Copy application code
COPY src/ src/

# Run everything from a predictable project directory inside the container
WORKDIR /app
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Start the API via invoke
ENTRYPOINT ["uvx", "invoke", "api"]
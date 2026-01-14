import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    pythonpath = f"PYTHONPATH={os.getcwd()}/src" if not WINDOWS else f"set PYTHONPATH={os.getcwd()}\\src &&"
    ctx.run(f"{pythonpath} uv run python -m {PROJECT_NAME}.data data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    pythonpath = f"PYTHONPATH={os.getcwd()}/src" if not WINDOWS else f"set PYTHONPATH={os.getcwd()}\\src &&"
    ctx.run(f"{pythonpath} uv run python -m {PROJECT_NAME}.train", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    pythonpath = f"PYTHONPATH={os.getcwd()}/src" if not WINDOWS else f"set PYTHONPATH={os.getcwd()}\\src &&"
    ctx.run(f"{pythonpath} uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run(f"{pythonpath} uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

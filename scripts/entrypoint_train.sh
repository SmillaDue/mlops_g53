#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GCS_DATA_URI:-}" ]]; then
  echo "Downloading data from ${GCS_DATA_URI}"
  mkdir -p /tmp/processed
  gsutil -m cp -r "${GCS_DATA_URI}/"* /tmp/processed/
  export DATA_DIR=/tmp/processed
  echo "DATA_DIR=${DATA_DIR}"
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
exec "$PYTHON_BIN" src/mlops_project/train.py "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GCS_DATA_URI:-}" ]]; then
  echo "Downloading data from ${GCS_DATA_URI}"
  mkdir -p /tmp/processed

  export GCS_DATA_URI
  uv run python - <<'PY'
import os
from urllib.parse import urlparse
from google.cloud import storage

uri = os.environ["GCS_DATA_URI"]
out_dir = "/tmp/processed"

u = urlparse(uri)
bucket_name = u.netloc
prefix = u.path.lstrip("/").rstrip("/") + "/"

client = storage.Client()
bucket = client.bucket(bucket_name)

count = 0
for blob in client.list_blobs(bucket_name, prefix=prefix):
    if blob.name.endswith("/"):
        continue
    rel = blob.name[len(prefix):]
    dest = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    blob.download_to_filename(dest)
    count += 1

print(f"Downloaded {count} files into {out_dir}")
PY

  export DATA_DIR=/tmp/processed
  echo "DATA_DIR=${DATA_DIR}"
fi

echo "Starting training via uv..."
exec uv run src/mlops_project/train.py "$@"
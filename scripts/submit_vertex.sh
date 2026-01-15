#!/usr/bin/env bash
set -e

PROJECT_ID=mlops-g53
REGION=europe-west1

gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="mlops-training-smoketest-$(date +%Y%m%d-%H%M%S)" \
  --config="gcp/vertex_job.yaml"
  
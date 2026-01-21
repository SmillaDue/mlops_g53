#!/usr/bin/env bash
set -e

PROJECT_ID=mlops-g53
REGION=europe-west1

# Check if sweep ID is provided
if [ -z "$1" ]; then
  echo "Usage: ./scripts/submit_sweep.sh <sweep-id> [count]"
  echo "Example: ./scripts/submit_sweep.sh fx2j4z7g 5"
  exit 1
fi

SWEEP_ID=$1
COUNT=${2:-3}  # Default to 3 runs if not specified

# Get WANDB_API_KEY from local wandb config
WANDB_KEY=$(uv run python -c "import wandb; print(wandb.api.api_key)" 2>/dev/null)
if [ -z "$WANDB_KEY" ]; then
  echo "Error: WANDB_API_KEY not found. Please login with 'wandb login'"
  exit 1
fi

echo "Submitting sweep agent for: brainy_mlops/smallcnn/$SWEEP_ID"
echo "Number of runs: $COUNT"

# Create temporary config file with sweep ID
TEMP_CONFIG=$(mktemp)
cat > "$TEMP_CONFIG" << EOF
serviceAccount: vertex-train-brains-sa@mlops-g53.iam.gserviceaccount.com

baseOutputDirectory:
  outputUriPrefix: gs://mlops-brain-tumor/runs/sweep-$SWEEP_ID

workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
    replicaCount: 1
    containerSpec:
      imageUri: europe-west1-docker.pkg.dev/mlops-g53/mlops-g53-repo/wandb:latest
      command:
        - "/bin/bash"
        - "-c"
        - |
          echo "Downloading data from GCS..."
          mkdir -p /data/processed
          gcloud storage cp -r gs://mlops-brain-tumor/data/processed/v1/* /data/processed/
          echo "Data downloaded successfully"
          ls -la /data/processed/
          echo "Starting wandb agent..."
          uv run wandb agent brainy_mlops/smallcnn/$SWEEP_ID --count $COUNT
      env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: DATA_DIR
          value: "/data/processed"
        - name: WANDB_API_KEY
          value: "$WANDB_KEY"
EOF

gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="wandb-sweep-$SWEEP_ID-$(date +%Y%m%d-%H%M%S)" \
  --config="$TEMP_CONFIG"

rm "$TEMP_CONFIG"
echo "Sweep job submitted successfully!"

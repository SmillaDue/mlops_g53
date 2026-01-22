import os
from pathlib import Path

import hydra
import torch
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops_project.data import brain_tumor
from mlops_project.utils import ensure_data_and_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# import subprocess
# subprocess.run(["dvc", "pull"], check=True)
# LOCAL_MODEL = Path("models/model.pth"


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base=None)
def evaluate(config: DictConfig) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Using device: {DEVICE}")

    # Get batch size from config
    batch_size = config.batch_size
    model_checkpoint = config.get("model_checkpoint", LOCAL_MODEL)
    print(f"Loading model from: {model_checkpoint}")

    model = hydra.utils.instantiate(config.model).to(DEVICE)
    # Load the saved model weights from checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    # Load the test dataset (training set is ignored with _)
    _, test_set = brain_tumor()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # Set model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    preds = []
    targets = []

    # Iterate over test batches
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            preds.append(y_pred.cpu())
            targets.append(target.cpu())

    # Concatenate all predictions and targets
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)

    # Calculate metrics
    test_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    test_precision = precision_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)
    test_recall = recall_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)
    test_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)

    # Print evaluation metrics
    print("\n" + "=" * 50)
    print("TEST EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    MODEL_BUCKET, MODEL_PREFIX, DATA_PREFIX, LOCAL_DATA, LOCAL_MODEL = ensure_data_and_model()
    evaluate()

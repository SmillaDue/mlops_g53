import os
import sys
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from wandb import config

IS_LINUX = sys.platform.startswith("linux")
if IS_LINUX:
    # Use a headless backend (no display required)
    import matplotlib

    matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

import subprocess

subprocess.run(["dvc", "pull"], check=True)


LOCAL_MODEL = Path("models/model.pth")


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base=None)
def visualize(config: DictConfig) -> None:
    """Visualize model predictions."""
    figure_name: str = "embeddings.png"
    model_checkpoint: Path = LOCAL_MODEL

    model = hydra.utils.instantiate(config.model).to(DEVICE)

    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    model.fc = torch.nn.Identity()
    batch_size = config.batch_size

    print("Loading test data for visualization...")
    test_images = torch.load("data/processed/test_images.pt", weights_only=False)
    test_target = torch.load("data/processed/test_targets.pt", weights_only=False)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=batch_size):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    print("Reducing dimensionality with PCA (if needed)...")
    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    print("Plotting embeddings...")
    plt.figure(figsize=(10, 10))
    for i in range(4):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")
    plt.close()


if __name__ == "__main__":
    visualize()

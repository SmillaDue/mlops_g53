import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import monai
import torch
from google.cloud import storage
from torch import Tensor
from torch.utils.data import Dataset

from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing, ensure_data_and_model, show_image_and_target

# subprocess.run(["dvc", "pull", "data"], check=True)

IS_LINUX = sys.platform.startswith("linux")
if IS_LINUX:
    # Use a headless backend (no display required)
    import matplotlib

    matplotlib.use("Agg")

IMG_SIZE = 256


class BrainTumorDataset(Dataset):
    """Brain tumor dataset for PyTorch.
    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    name: str = "BrainTumorDataset"

    def __init__(
        self,
        data_folder: str = "data/raw",
        train: bool = True,
        img_preprocess=ArrayPreprocessing(img_size=IMG_SIZE, crop_img=True),
        img_transforms=TensorsPreprocessing(),
        target_transform: monai.transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_preprocess = img_preprocess
        self.img_transforms = img_transforms
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""

        if self.train:
            data_folder = os.path.join(self.data_folder, "Training")
        else:
            data_folder = os.path.join(self.data_folder, "Testing")

        img_paths, target = [], []
        for i, folder in enumerate(sorted(os.listdir(data_folder))):
            class_folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(class_folder_path):
                files = [
                    os.path.join(class_folder_path, f)
                    for f in os.listdir(class_folder_path)
                    if os.path.isfile(os.path.join(class_folder_path, f))
                ]
                img_paths += files
                target += [i] * len(files)

        imgs = [self.img_preprocess(p) for p in img_paths]  # list of tensors (1,H,W)
        imgs = self.img_transforms(imgs)

        X = imgs
        y = torch.tensor(target, dtype=torch.long)  # (N,), long

        self.images = X
        self.target = y

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor. Transforms are applied when loading data."""
        img, target = self.images[idx], self.target[idx]
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]


def dataset_statistics(
    IMG_SIZE: int = 256,
    transform_type: Literal["crop", "no-crop"] = "crop",
    datadir: str = "data/raw",
    seed: int | None = 37,
    img_name: str = "cropped_images",
    plot: bool = True,
) -> None:
    """
    Compute dataset statistics.
    Seed is for reproducible image selection.
    Images are from training set and saved to reports/figures/dataset/
    """
    info = f"""
Parameters:
IMG_SIZE: {IMG_SIZE}
seed: {seed}
        """
    print(info)

    if seed is not None:
        torch.manual_seed(seed)

    if transform_type == "crop":
        array_preprocessing = ArrayPreprocessing(img_size=IMG_SIZE, crop_img=True)
    else:
        array_preprocessing = ArrayPreprocessing(img_size=IMG_SIZE, crop_img=False)

    train_dataset = BrainTumorDataset(data_folder=datadir, train=True, img_preprocess=array_preprocessing)
    test_dataset = BrainTumorDataset(data_folder=datadir, train=False)

    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    if plot == True:
        n_total = train_dataset.images.shape[0]
        N = min(25, n_total)
        idx = torch.randperm(n_total)[:N]

        show_image_and_target(train_dataset.images[idx], train_dataset.target[idx], show=False)

        plt.suptitle("Random samples of images from training set", fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"reports/figures/dataset/{img_name}_class_all.png")
        plt.close()

        for i in range(4):
            indices = (train_dataset.target == i).nonzero(as_tuple=True)[0]
            n_samples = min(25, len(indices))
            selected_idx = indices[torch.randperm(len(indices))[:n_samples]]

            show_image_and_target(train_dataset.images[selected_idx], selected_idx, show=False)
            plt.suptitle(f"Random samples of images from training set - class {i}", fontsize=18, y=0.96)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"reports/figures/dataset/{img_name}_class_{i}.png")
            plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    train_label_counts = {i: count.item() for i, count in enumerate(train_label_distribution)}
    test_label_counts = {i: count.item() for i, count in enumerate(test_label_distribution)}

    plt.bar(torch.arange(4), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/dataset/train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(4), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/dataset/test_label_distribution.png")
    plt.close()

    return train_label_counts, test_label_counts


if __name__ == "__main__":
    MODEL_BUCKET, MODEL_PREFIX, DATA_PREFIX, LOCAL_DATA, LOCAL_MODEL = ensure_data_and_model()
    dataset_statistics()

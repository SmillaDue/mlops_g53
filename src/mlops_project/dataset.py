import os
from typing import Literal

import matplotlib.pyplot as plt
import monai
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from torch import Tensor
from torch.utils.data import Dataset

from mlops_project.utils import (
    CropImage,
    LoadImageFromCV,
    NormalizeImage,
    ToGrayCHW,
    describe_compose,
    show_image_and_target,
)

IMG_SIZE = 256
data_transforms = Compose(
    [
        LoadImageFromCV(),
        CropImage(),
        ToTensor(),
        EnsureChannelFirst(channel_dim=-1),
        ToGrayCHW(),
        Resize((IMG_SIZE, IMG_SIZE)),
        NormalizeImage(),
        ScaleIntensity(),
    ]
)

data_transforms_simple = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim=-1),
        ToGrayCHW(),
        Resize((IMG_SIZE, IMG_SIZE)),
        ScaleIntensity(),
    ]
)


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
        img_transform: monai.transforms.Transform | None = data_transforms,
        target_transform: monai.transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""

        if self.train:
            data_folder = os.path.join(self.data_folder, "Training")
        else:
            data_folder = os.path.join(self.data_folder, "Testing")

        images, target = [], []
        for i, folder in enumerate(os.listdir(data_folder)):
            class_folder_path = os.path.join(data_folder, folder)
            if os.path.isdir(class_folder_path):
                nb_files = [
                    f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))
                ]
                for file_path in nb_files:
                    img_path = os.path.join(class_folder_path, file_path)
                    images.append(self.img_transform(str(img_path)).float())
                    target.append(i)

        X = torch.stack(images, dim=0)  # (N,3,H,W) or (N,1,H,W)
        y = torch.tensor(target, dtype=torch.long)  # (N,), long
        self.images = X
        self.target = y

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor. Transforms are applied when loading data."""
        img, target = self.images[idx], self.target[idx]
        # if self.img_transform:
        #     img = self.img_transform(img)
        # if self.target_transform:
        #     target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]


def dataset_statistics(
    transform_type: Literal["simple", "crop"] = "simple",
    IMG_SIZE: int = 256,
    datadir: str = "data/raw",
    seed: int | None = 37,
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
transform_type: {transform_type}
datatransforms: \n{describe_compose(data_transforms) if transform_type == "crop" else describe_compose(data_transforms_simple)}
        """
    print(info)

    if seed is not None:
        torch.manual_seed(seed)

    if transform_type == "simple":
        train_dataset = BrainTumorDataset(data_folder=datadir, train=True, img_transform=data_transforms_simple)
        test_dataset = BrainTumorDataset(data_folder=datadir, train=False, img_transform=data_transforms_simple)
    else:
        train_dataset = BrainTumorDataset(data_folder=datadir, train=True, img_transform=data_transforms)
        test_dataset = BrainTumorDataset(data_folder=datadir, train=False, img_transform=data_transforms)

    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    n_total = train_dataset.images.shape[0]
    N = min(25, n_total)
    idx = torch.randperm(n_total)[:N]

    show_image_and_target(train_dataset.images[idx], train_dataset.target[idx], show=False)

    if transform_type == "simple":
        plt.suptitle("Random samples of images from training set", fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("reports/figures/dataset/braintumor_images_class_all.png")
    else:
        plt.suptitle("Random samples of processed images from training set", fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("reports/figures/dataset/braintumor_images_processed_class_all.png")
    plt.close()

    for i in range(4):
        indices = (train_dataset.target == i).nonzero(as_tuple=True)[0]
        n_samples = min(25, len(indices))
        selected_idx = indices[torch.randperm(len(indices))[:n_samples]]

        show_image_and_target(train_dataset.images[selected_idx], selected_idx, show=False)

        if transform_type == "simple":
            plt.suptitle(f"Random samples of images from training set - class {i}", fontsize=18, y=0.96)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"reports/figures/dataset/braintumor_images_class_{i}.png")

        else:
            plt.suptitle(f"Random samples of processed images from training set - class {i}", fontsize=18, y=0.96)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f"reports/figures/dataset/braintumor_images_processed_class_{i}.png")
        plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

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


if __name__ == "__main__":
    dataset_statistics(transform_type="crop")

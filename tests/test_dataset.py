import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from mlops_project.dataset import BrainTumorDataset, dataset_statistics
from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing
from PIL import Image


def test_dataset_loading(img_size: int = 256):
    raw_data_dir = "data/raw"
    train_dataset = BrainTumorDataset(
        data_folder=raw_data_dir,
        train=True,
        img_preprocess=ArrayPreprocessing(img_size=img_size),
        img_transforms=TensorsPreprocessing(),
    )
    test_dataset = BrainTumorDataset(
        data_folder=raw_data_dir,
        train=False,
        img_preprocess=ArrayPreprocessing(img_size=img_size),
        img_transforms=TensorsPreprocessing(),
    )

    assert len(train_dataset) == 5267, f"Expected 5267 training samples, got {len(train_dataset)}"
    assert len(test_dataset) == 1311, f"Expected 1311 testing samples, got {len(test_dataset)}"

    for dataset in [train_dataset, test_dataset]:
        for i, (sample_img, sample_label) in enumerate(dataset):
            assert isinstance(
                sample_img, torch.Tensor
            ), f"Expected image to be a torch.Tensor, got {type(sample_img)}, at index {i}"
            assert (
                sample_img.shape[0] == 1
            ), f"Expected image to have 1 channel (grayscale), got {sample_img.shape[0]}, at index {i}"
            assert sample_img.shape == torch.Size(
                [1, img_size, img_size]
            ), f"Expected image size to be 1x{img_size}x{img_size}, got {sample_img.shape}, at index {i}"

            assert (
                isinstance(sample_label, torch.Tensor)
                and sample_label.numel() == 1
                and sample_label.dtype == torch.long
                and 0 <= sample_label.item() < 4
            ), f"Expected label to be a 1-element tensor, dtype int and value < 4, got {sample_label} ({type(sample_label)}), at index {i}"


def test_dataset_statistics():
    expected_train_counts = {0: 1321, 1: 1339, 2: 1150, 3: 1457}
    expected_test_counts = {0: 300, 1: 306, 2: 405, 3: 300}

    train_label_distribution, test_label_distribution = dataset_statistics(plot=False)

    assert (
        train_label_distribution == expected_train_counts
    ), f"Expected training counts {expected_train_counts}, got {train_label_distribution}"
    assert (
        test_label_distribution == expected_test_counts
    ), f"Expected testing counts {expected_test_counts}, got {test_label_distribution}"

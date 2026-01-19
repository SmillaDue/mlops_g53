import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from mlops_project.data import brain_tumor, load_data
from PIL import Image


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data file not found")
def test_loading_of_processed_data(img_size: int = 256):
    train_dataset, test_dataset = brain_tumor()
    assert len(train_dataset) == 5267, f"Expected 5267 training samples, got {len(train_dataset)}"
    assert len(test_dataset) == 1311, f"Expected 1311 testing samples, got {len(test_dataset)}"

    for dataset in [train_dataset, test_dataset]:
        for i, (sample_img, sample_label) in enumerate(dataset):
            assert isinstance(
                sample_img, torch.Tensor
            ), f"Expected image to be a torch.Tensor, got {type(sample_img)}, at index {i}"
            assert (
                sample_img.ndim == 3
            ), f"Expected image to have 3 dimensions (C, H, W), got {sample_img.ndim}, at index {i}"
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

def test_data_processing():
    raw_data_dir = "data/raw"
    train_x, train_y, test_x, test_y, label_names = load_data(raw_data_dir)

    assert len(train_x) == 5267, f"Expected 5267 training samples, got {len(train_x)}"
    assert len(test_x) == 1311, f"Expected 1311 testing samples, got {len(test_x)}"
    assert len(train_y) == len(train_x), "Mismatch between number of training images and labels"
    assert len(test_y) == len(test_x), "Mismatch between number of testing images and labels"
    assert set(train_y).issubset({0, 1, 2, 3}), "Training labels should be in the range [0, 3]"
    assert set(test_y).issubset({0, 1, 2, 3}), "Testing labels should be in the range [0, 3]"
    assert len(label_names) == 4, f"Expected 4 label names, got {len(label_names)}"

    assert all(isinstance(path, str) for path in train_x), "All training image paths should be strings"
    assert all(isinstance(path, str) for path in test_x), "All testing image paths should be strings"
    
    for path in train_x + test_x:
        assert os.path.isfile(path), f"Image file does not exist: {path}"
        try:
            with Image.open(path) as img:
                img.verify()  # Verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            pytest.fail(f"Invalid image file: {path}, error: {e}")
            
def test_counts_match_filesystem(raw_dir = "data/raw"):
    raw = Path(raw_dir)
    train_root = raw / "Training"
    test_root = raw / "Testing"

    train_x, train_y, test_x, test_y, class_names = load_data(raw_dir)

    # Expected counts from filesystem (by class folder name)
    expected_train_counts = {
        cls: len([f for f in (train_root / cls).iterdir() if f.is_file()])
        for cls in class_names
    }
    expected_test_counts = {
        cls: len([f for f in (test_root / cls).iterdir() if f.is_file()])
        for cls in class_names
    }

    # Actual counts from returned labels (use folder name inferred from path)
    actual_train_counts = Counter(Path(p).parent.name for p in train_x)
    actual_test_counts = Counter(Path(p).parent.name for p in test_x)
    
    assert actual_train_counts == expected_train_counts
    assert actual_test_counts == expected_test_counts
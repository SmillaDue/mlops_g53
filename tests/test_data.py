from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
import random
import matplotlib.pyplot as plt
from mlops_project.data import brain_tumor
import numpy as np
import pytest
import os

@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data file not found")
def test_data_loading(img_size: int = 256):
    train_dataset, test_dataset = brain_tumor()
    assert len(train_dataset) == 5267, f"Expected 5267 training samples, got {len(train_dataset)}"
    assert len(test_dataset) == 1311, f"Expected 1311 testing samples, got {len(test_dataset)}"

    for dataset in [train_dataset, test_dataset]:
        for i, (sample_img, sample_label) in enumerate(dataset):
            assert isinstance(sample_img, torch.Tensor), f"Expected image to be a torch.Tensor, got {type(sample_img)}, at index {i}"
            assert sample_img.ndim == 3, f"Expected image to have 3 dimensions (C, H, W), got {sample_img.ndim}, at index {i}"
            assert sample_img.shape[0] == 3, f"Expected image to have 3 channels, got {sample_img.shape[0]}, at index {i}"
            assert sample_img.shape == torch.Size([3, img_size, img_size]), f"Expected image size to be 3x{img_size}x{img_size}, got {sample_img.shape}, at index {i}"
            
            assert (
                isinstance(sample_label, torch.Tensor)
                and sample_label.numel() == 1
                and sample_label.dtype == torch.long
                and 0 <= sample_label.item() < 4
                    ), f"Expected label to be a 1-element tensor, dtype int and value < 4, got {sample_label} ({type(sample_label)}), at index {i}"


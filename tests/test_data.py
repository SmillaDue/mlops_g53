import os.path

import pytest
import torch
from mlops_project.data import MyDataset
from torch.utils.data import Dataset


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)

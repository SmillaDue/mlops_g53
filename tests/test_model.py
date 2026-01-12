# tests/test_model.py
import pytest
from mlops_project.model import SmallCNN, DeepCNN, DenseNetModel
import torch

@pytest.mark.parametrize("model", [SmallCNN, DeepCNN, DenseNetModel])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(model, batch_size: int, num_classes: int = 4) -> None:
    model = model(num_classes=num_classes)
    x = torch.randn(batch_size, 1, 256, 256)
    y = model(x)
    assert y.shape == (batch_size, num_classes), f'Output shape is not ({batch_size}, {num_classes}) but {y.shape}'
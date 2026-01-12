import torch
from monai.networks.nets import DenseNet121
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    model = DenseNetModel(num_classes=4)
    model = SmallCNN(num_classes=4)
    model = DeepCNN(num_classes=4)
    
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(type(model))
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f'Output shape: {y.shape}')

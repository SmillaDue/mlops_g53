import torch
from monai.networks.nets import DenseNet121
from torch import nn


class SmallCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 16,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel),

            nn.Conv2d(c1, c2, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel: int = 2,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = channels

        def conv_block(cin: int, cout: int) -> nn.Sequential:
            layers = [nn.Conv2d(cin, cout, kernel_size, padding=padding)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            layers += [nn.ReLU(), nn.MaxPool2d(pool_kernel)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(in_channels, c1),
            conv_block(c1, c2),
            nn.Conv2d(c2, c3, kernel_size, padding=padding),
            nn.BatchNorm2d(c3) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class DenseNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        spatial_dims: int = 2,
        pretrained: bool = False,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.net = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            pretrained=pretrained,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

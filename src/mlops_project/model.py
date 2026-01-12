from torch import nn
import torch

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    model = DenseNetModel(num_classes=4)
    
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(type(model))
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f'Output shape: {y.shape}')  

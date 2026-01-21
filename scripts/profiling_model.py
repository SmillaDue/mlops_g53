import torch
from mlops_project.model import DeepCNN, DenseNetModel, SmallCNN
from mlops_project.train import train
from torch.profiler import ProfilerActivity, profile

batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Profiling of the forward step using the baseline DenseNet model")
print("Device:", DEVICE)
print("batch size:", batch_size)

model = DenseNetModel().to(DEVICE)
data = torch.randn(batch_size, 1, 256, 256)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=False,  # <- turn off
    profile_memory=False,  # <- turn off
    with_stack=False,  # <- turn off
) as prof:
    train.__wrapped__(cfg)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

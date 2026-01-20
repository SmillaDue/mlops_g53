from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from mlops_project.train import train
from torch.profiler import ProfilerActivity, profile

# Clear Hydra state (important if rerunning in same process)
GlobalHydra.instance().clear()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
    cfg = compose(config_name="default_config")  # no .yaml

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=False,  # <- turn off
    profile_memory=False,  # <- turn off
    with_stack=False,  # <- turn off
) as prof:
    train.__wrapped__(cfg)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

import os
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from google.cloud import storage
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing

# --- paths / config ---
PROJECT_ROOT = Path(os.getcwd())
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_BUCKET = "mlops-brain-tumor"
MODEL_OBJECT = "models/final_model.pth"
LOCAL_MODEL = Path("/tmp/model.pth")

TRAIN_ROOT = Path("data/raw/Testing")
OUT_CSV = Path("data_api/inference_features.csv")
N_PER_CLASS = 100
CLASS_DIRS = ["glioma", "meningioma", "notumor", "pituitary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def ensure_model():
    if LOCAL_MODEL.exists():
        return
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_OBJECT)
    blob.download_to_filename(str(LOCAL_MODEL))


def extract_features(img_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0

    return {
        "avg_brightness": avg_brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load hydra config (same idea as your API)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_config_dd")

    # Ensure weights exist locally
    ensure_model()

    # Build model once
    model = hydra.utils.instantiate(cfg.model).to(DEVICE)
    model.load_state_dict(torch.load(LOCAL_MODEL, map_location=DEVICE))
    model.eval()

    array_preprocessor = ArrayPreprocessing(img_size=256)
    tensor_preprocessor = TensorsPreprocessing()

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "filename", "prediction", "target", "avg_brightness", "contrast", "sharpness"])

        for class_name in CLASS_DIRS:
            folder = TRAIN_ROOT / class_name
            files = iter_images(folder)[:N_PER_CLASS]

            for img_path in files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                feats = extract_features(img)

                # --- SAME preprocessing pipeline as API ---
                # (use file path because your preprocessor takes a path)
                img_array_processed = array_preprocessor(str(img_path))
                img_final = tensor_preprocessor(img_array_processed).to(DEVICE)

                # --- model prediction ---
                with torch.no_grad():
                    y_pred = model(img_final).cpu()
                pred_class = int(y_pred.softmax(dim=1).argmax(dim=1).item())

                now = datetime.now(timezone.utc).isoformat()

                writer.writerow(
                    [
                        now,
                        str(img_path.as_posix()),
                        pred_class,
                        class_name,
                        feats["avg_brightness"],
                        feats["contrast"],
                        feats["sharpness"],
                    ]
                )

    print("Saved:", OUT_CSV.resolve())


if __name__ == "__main__":
    main()

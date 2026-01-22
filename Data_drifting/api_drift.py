# from __future__ import annotations

# import base64
# import os
# import re
# from contextlib import asynccontextmanager
# from enum import Enum
# from http import HTTPStatus
# from pathlib import Path

# import anyio
# import cv2
# import hydra
# import torch
# from fastapi import FastAPI, File, HTTPException, Request, UploadFile
# from fastapi.responses import FileResponse, JSONResponse
# from google.cloud import storage
# from hydra import compose, initialize_config_dir
# from hydra.core.global_hydra import GlobalHydra
# from omegaconf import OmegaConf
# from pydantic import BaseModel

# from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing

# PROJECT_ROOT = Path(os.getcwd())  # adjust if needed
# CONFIG_DIR = PROJECT_ROOT / "configs/"  # where the yaml's are
# MODEL_BUCKET = "mlops-brain-tumor"
# MODEL_OBJECT = "models/final_model.pth"
# LOCAL_MODEL = Path("/tmp/model.pth")

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# def ensure_model():
#     if LOCAL_MODEL.exists():
#         return

#     client = storage.Client()
#     bucket = client.bucket(MODEL_BUCKET)
#     blob = bucket.blob(MODEL_OBJECT)
#     blob.download_to_filename(str(LOCAL_MODEL))


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Important if the app reloads (uvicorn --reload) or tests import multiple times
#     GlobalHydra.instance().clear()

#     with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
#         cfg = compose(config_name="default_config")  # no .yaml

#     ensure_model()
#     app.state.cfg = cfg
#     print("Hydra config loaded")
#     yield
#     print("Shutting down")


# app = FastAPI(lifespan=lifespan)


# @app.get("/")
# def read_root():
#     """Health check."""
#     response = {
#         "message": "Welcome to the model inference API!",
#         "status-code": 200,
#     }
#     return response


# @app.post("/inference/")
# async def inference(
#     request: Request,
#     data: UploadFile = File(...),
# ):
#     """
#     API does inference by
#     1. User uploads an image for inference

#     Then the following is executed:

#     2. transform/preprocess the uploaded image
#     3. load in a model
#     4. predict
#     5. return pct for each class? just the class? Depends on the intended user."""

#     config = request.app.state.cfg

#     # Save uploaded file
#     content = await data.read()
#     with open("image.png", "wb") as f:
#         f.write(content)

#     # Read and process image(s)
#     img = cv2.imread("image.png")
#     if img is None:
#         raise HTTPException(status_code=400, detail="Invalid image")

#     array_preprocessor = ArrayPreprocessing(img_size=256)  # <-- change to use parameter set in config file
#     tensor_preprocessor = TensorsPreprocessing()

#     img_array_processed = array_preprocessor("image.png")
#     cv2.imwrite("image_preprocessed.png", img_array_processed)

#     img_final = tensor_preprocessor(img_array_processed)

#     # move input to same device as model
#     img_final = img_final.to(DEVICE)

#     # Load model
#     model = hydra.utils.instantiate(config.model).to(DEVICE)
#     model.load_state_dict(torch.load(LOCAL_MODEL, map_location=DEVICE))

#     # Make prediction
#     model.eval()

#     with torch.no_grad():
#         y_pred = model(img_final).cpu()

#     pred = y_pred.softmax(dim=1).tolist()

#     return {
#         "prediction": pred,
#         "image_url": "/inference/image_preprocessed.png",
#     }


# @app.get("/inference/image_preprocessed.png")
# def get_preprocessed_image():
#     path = Path("image_preprocessed.png")

#     if not path.exists():
#         raise HTTPException(status_code=404, detail="Preprocessed image not found")

#     return FileResponse(
#         "image_preprocessed.png",
#         media_type="image/png",
#         filename="image_preprocessed.png",
#     )


from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from google.cloud import storage
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing

PROJECT_ROOT = Path(os.getcwd())
CONFIG_DIR = PROJECT_ROOT / "configs/"
MODEL_BUCKET = "mlops-brain-tumor"
MODEL_OBJECT = "models/final_model.pth"
LOCAL_MODEL = Path("/tmp/model.pth")

# CSV “database”
DATA_DIR = PROJECT_ROOT / "data_api"
CSV_PATH = DATA_DIR / "inference_features.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

FEATURE_COLUMNS = ["avg_brightness", "contrast", "sharpness"]


def ensure_model():
    if LOCAL_MODEL.exists():
        return
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_OBJECT)
    blob.download_to_filename(str(LOCAL_MODEL))


def extract_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Same style as your MNIST example, but for BGR images."""
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


def init_csv_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # No spaces after commas -> easier parsing later
        path.write_text("time,filename,prediction,target,avg_brightness,contrast,sharpness\n")


def append_row_csv(
    *,
    csv_path: Path,
    now: str,
    filename: str | None,
    prediction: int,
    target: str | None,
    feats: dict[str, float],
):
    # write "None" as empty cell for better missing handling
    filename_out = "" if filename is None else filename
    target_out = "" if target is None else target

    with open(csv_path, "a") as f:
        f.write(
            f"{now},{filename_out},{prediction},{target_out},"
            f"{feats['avg_brightness']},{feats['contrast']},{feats['sharpness']}\n"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_config_dd")

    ensure_model()
    init_csv_db(CSV_PATH)

    app.state.cfg = cfg
    app.state.csv_path = CSV_PATH

    print("Hydra config loaded")
    print(f"✅ CSV database at: {CSV_PATH.resolve()}")
    yield
    print("Shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the model inference API!", "status-code": 200}


@app.post("/inference/")
async def inference(request: Request, data: UploadFile = File(...), target: str | None = None):
    config = request.app.state.cfg
    csv_path: Path = request.app.state.csv_path

    # Save upload (still overwrites; OK for exercise; can make unique later)
    content = await data.read()
    Path("image.png").write_bytes(content)

    img = cv2.imread("image.png")
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # --- Feature extraction for drift monitoring ---
    feats = extract_features(img)

    # --- Preprocess + model inference ---
    array_preprocessor = ArrayPreprocessing(img_size=256)
    tensor_preprocessor = TensorsPreprocessing()

    img_array_processed = array_preprocessor("image.png")
    cv2.imwrite("image_preprocessed.png", img_array_processed)

    img_final = tensor_preprocessor(img_array_processed).to(DEVICE)

    model = hydra.utils.instantiate(config.model).to(DEVICE)
    model.load_state_dict(torch.load(LOCAL_MODEL, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        y_pred = model(img_final).cpu()

    pred_probs = y_pred.softmax(dim=1).tolist()
    pred_class = int(torch.tensor(pred_probs).argmax().item())  # integer class

    # --- Append to CSV “database” ---
    now = datetime.now(timezone.utc).isoformat()
    append_row_csv(
        csv_path=csv_path,
        now=now,
        filename=data.filename,
        prediction=pred_class,
        target=target,
        feats=feats,
    )

    return {
        "prediction": pred_probs,
        "prediction_class": pred_class,
        "target": target,
        "features": feats,
        "csv_db": str(csv_path),
        "image_url": "/inference/image_preprocessed.png",
    }


@app.get("/inference/image_preprocessed.png")
def get_preprocessed_image():
    path = Path("image_preprocessed.png")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preprocessed image not found")
    return FileResponse(str(path), media_type="image/png", filename="image_preprocessed.png")

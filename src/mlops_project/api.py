from __future__ import annotations

import base64
import os
import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from pathlib import Path

import anyio
import cv2
import hydra
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google.cloud import storage
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel

from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing

PROJECT_ROOT = Path(os.getcwd())  # adjust if needed
CONFIG_DIR = PROJECT_ROOT / "configs/"  # where the yaml's are
MODEL_BUCKET = "mlops-brain-tumor"
MODEL_OBJECT = "models/final_model.pth"
LOCAL_MODEL = Path("/tmp/model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def ensure_model():
    """
    Ensure that the trained model weights exist locally.

    If the model file is not present at LOCAL_MODEL, it is downloaded
    from the configured Google Cloud Storage bucket.
    """
    if LOCAL_MODEL.exists():
        return

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_OBJECT)
    blob.download_to_filename(str(LOCAL_MODEL))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle hook.

    - Clears any existing Hydra state (important for reloads/tests).
    - Loads the Hydra configuration.
    - Ensures the trained model is available locally.
    - Stores the configuration on `app.state` for later use.
    """

    # Important if the app reloads (uvicorn --reload) or tests import multiple times
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_config")  # no .yaml

    ensure_model()
    app.state.cfg = cfg
    print("Hydra config loaded")
    yield
    print("Shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """
    Root endpoint and health check.

    Returns a short description of the service and how to use it.
    """
    return {
        "service": "Model Inference API",
        "status": "ok",
        "description": (
            "This API performs image classification using a trained PyTorch model.\n\n"
            "Usage:\n"
            "POST /inference/\n"
            "  - multipart/form-data with a single file field named `data`\n"
            "  - the file must be an image\n\n"
            "The response contains the model's class probabilities.\n"
            "You can also retrieve the preprocessed image at:\n"
            "GET /inference/image_preprocessed.png"
        ),
    }


@app.post("/inference/")
async def inference(
    request: Request,
    data: UploadFile = File(...),
):
    """
    API does inference by
    1. User uploads an image for inference

    Then the following is executed:

    2. transform/preprocess the uploaded image
    3. load in a model
    4. predict
    5. return pct for each class? just the class? Depends on the intended user.
    """

    config = request.app.state.cfg

    # Save uploaded file
    content = await data.read()
    with open("image.png", "wb") as f:
        f.write(content)

    # Read and process image(s)
    img = cv2.imread("image.png")
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    array_preprocessor = ArrayPreprocessing(img_size=256)  # <-- change to use parameter set in config file
    tensor_preprocessor = TensorsPreprocessing()

    img_array_processed = array_preprocessor("image.png")
    cv2.imwrite("image_preprocessed.png", img_array_processed)

    img_final = tensor_preprocessor(img_array_processed)

    # Load model
    model = hydra.utils.instantiate(config.model).to(DEVICE)
    model.load_state_dict(torch.load(LOCAL_MODEL, map_location=DEVICE))

    # Make prediction
    model.eval()

    with torch.no_grad():
        y_pred = model(img_final).cpu()

    pred = y_pred.softmax(dim=1).tolist()

    return {
        "prediction": pred,
        "image_url": "/inference/image_preprocessed.png",
    }


@app.get("/inference/image_preprocessed.png")
def get_preprocessed_image():
    """
    Retrieve the most recently preprocessed image produced by the
    `/inference/` endpoint.
    """

    path = Path("image_preprocessed.png")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Preprocessed image not found")

    return FileResponse(
        "image_preprocessed.png",
        media_type="image/png",
        filename="image_preprocessed.png",
    )

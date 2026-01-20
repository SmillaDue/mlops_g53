from __future__ import annotations
import base64
import re
from enum import Enum
from http import HTTPStatus
from omegaconf import OmegaConf
import anyio
import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse   
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import hydra
import torch
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from fastapi import FastAPI
from pathlib import Path

from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing

PROJECT_ROOT = Path(os.getcwd())  # adjust if needed
CONFIG_DIR = PROJECT_ROOT / "configs/"               # where your yaml lives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Important if the app reloads (uvicorn --reload) or tests import multiple times
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_config")  # no .yaml

    app.state.cfg = cfg
    print("Hydra config loaded")
    yield
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """ Health check."""
    response = {
        "message": "Welcome to the model inference API!",
        "status-code": 200,
    }
    return response
    
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
    5. return pct for each class? just the class? Depends on the intended user."""

    config = request.app.state.cfg
    model_checkpoint = '/gcs/mlops-brain-tumor/models/model.pth'
    
    # Save uploaded file
    content = await data.read()
    with open("image.png", "wb") as f:
        f.write(content)

    # Read and process image(s)
    img = cv2.imread("image.png")
    if img is None:
        return {"message": "Invalid image", "status-code": 400}
    
    array_preprocessor = ArrayPreprocessing(img_size = 256) # <-- change to use parameter set in config file
    tensor_preprocessor = TensorsPreprocessing()
    
    img_array_processed = array_preprocessor('image.png')
    cv2.imwrite("image_preprocessed.png", img_array_processed)

    img_final = tensor_preprocessor(img_array_processed)
    
    #Load model
    model = hydra.utils.instantiate(config.model).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    
    #Make prediction
    model.eval()
    model(img_final)

    with torch.no_grad():
        y_pred = model(img_final).cpu()
    
    # Return the preprocessed image directly, and the prediction
    with open("image_preprocessed.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    pred = y_pred.softmax(dim=1).tolist()

    return {
    "prediction": pred,
    "image_url": "/inference/image_preprocessed.png",
}

@app.get("/inference/image_preprocessed.png")
def get_preprocessed_image():
    return FileResponse(
        "image_preprocessed.png",
        media_type="image/png",
        filename="image_preprocessed.png",
    )
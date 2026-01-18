from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from PIL import Image
import io

from mlops_project.models.model import BrainTumorNet  # adjust if needed

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

## NOTE: Assumes we have a trained model saved at this path (Maybe this should be pushed?)
MODEL_PATH = Path("models/model.pth")

NUM_CLASSES = 4  # adjust to your dataset
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]  # example

app = FastAPI(title="Brain Tumor Classifier")

# -------------------------------------------------------------------
# Model loading (happens ONCE at startup)
# -------------------------------------------------------------------

model = BrainTumorNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------------------------------------------
# Image preprocessing (must match training!)
# -------------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   # match your dataset
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }

import io
import torch
import functions_framework
from google.cloud import storage
from PIL import Image
from torchvision import transforms

from model import DenseNetModel

MODEL_BUCKET = "mlops-brain-tumor"
MODEL_BLOB = "models/final_model.pth"

DEVICE = torch.device("cpu")
model = None 

def load_model():
    global model
    if model is None:
        client = storage.Client()
        blob = client.bucket(MODEL_BUCKET).blob(MODEL_BLOB)
        state_dict = torch.load(io.BytesIO(blob.download_as_bytes()), map_location=DEVICE)

        model = DenseNetModel(num_classes=4, in_channels=1)
        model.load_state_dict(state_dict)
        model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

# @functions_framework.http
# def predict(request):
#     try:
#         image_bytes = request.files["file"].read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         x = preprocess(image).unsqueeze(0)

#         model = load_model()
#         with torch.no_grad():
#             out = model(x)
#             pred = int(out.argmax())

#         return {"prediction": pred}
#     except Exception as e:
#         return {"error": str(e)}, 500

import json  # add at top

# @functions_framework.http (THIS ONE WORKS)
# def predict(request):
#     try:
#         if "file" not in request.files:
#             return (json.dumps({"error": "No file part in request"}), 400, {"Content-Type": "application/json"})

#         image_bytes = request.files["file"].read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         x = preprocess(image).unsqueeze(0)

#         mdl = load_model()
#         with torch.no_grad():
#             out = mdl(x)

#         # Debug prints (shows in Cloud logs)
#         print("out type:", type(out))
#         print("out shape:", tuple(out.shape) if hasattr(out, "shape") else None)
#         print("out sample:", out[0].tolist() if hasattr(out, "__getitem__") else str(out))

#         pred = int(out.argmax(dim=1).item())  # safer than int(out.argmax())
        

#         resp = {"prediction": pred}
#         return (json.dumps(resp), 200, {"Content-Type": "application/json"})

#     except Exception as e:
#         return (json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"})

@functions_framework.http
def predict(request):
    try:
        if "file" not in request.files:
            return (json.dumps({"error": "No file part in request"}), 400, {"Content-Type": "application/json"})

        image_bytes = request.files["file"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image).unsqueeze(0)

        mdl = load_model()
        with torch.no_grad():
            out = mdl(x)

        # Debug prints (shows in Cloud logs)
        print("out type:", type(out))
        print("out shape:", tuple(out.shape) if hasattr(out, "shape") else None)
        print("out sample:", out[0].tolist() if hasattr(out, "__getitem__") else str(out))

        #pred = int(out.argmax(dim=1).item())  # safer than int(out.argmax())
        LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
        pred_idx = int(out.argmax(dim=1).item())
        pred_label = LABELS[pred_idx]

        resp = {"prediction": pred_idx, "label": pred_label}
        return (json.dumps(resp), 200, {"Content-Type": "application/json"})

    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"})




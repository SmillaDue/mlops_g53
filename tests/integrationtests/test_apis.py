from pathlib import Path

import cv2  # uses the same backend as the API
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlops_project.api import app



@pytest.fixture(autouse=True, scope='session')
def clean_generated_images():
    """
    Ensure that image artifacts created by the API
    do not leak between tests.
    """
    app.LOCAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    Path("image.png").unlink(missing_ok=True)
    Path("image_preprocessed.png").unlink(missing_ok=True)

    yield

    Path("image.png").unlink(missing_ok=True)
    Path("image_preprocessed.png").unlink(missing_ok=True)


def _make_png_bytes(size: int = 16) -> bytes:
    """
    Create a small valid PNG image as bytes (no PIL dependency).
    """

    img = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def test_root_returns_html():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Model Inference API" in resp.text
        assert "POST /inference/" in resp.text


def test_get_preprocessed_image_not_found():
    with TestClient(app) as client:
        # Ensure the file doesn't exist
        resp = client.get("/inference/image_preprocessed.png")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Preprocessed image not found"


def test_inference_rejects_invalid_image():
    with TestClient(app) as client:
        # Not an image -> cv2.imread should return None -> 400
        files = {"data": ("not_an_image.png", b"this is not a real image", "image/png")}
        resp = client.post("/inference/", files=files)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid image"


def test_inference_success_and_preprocessed_image_available():
    with TestClient(app) as client:
        png_bytes = _make_png_bytes()
        files = {"data": ("image.png", png_bytes, "image/png")}

        resp = client.post("/inference/", files=files)
        print(resp)
        assert resp.status_code == 200

        payload = resp.json()
        assert "prediction" in payload
        assert "probabilities" in payload

        probs = payload["probabilities"]
        assert set(probs.keys()) == {"glioma", "meningioma", "notumor", "pituitary"}

        # Preprocessed image should now exist
        img_resp = client.get("/inference/image_preprocessed.png")
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"].startswith("image/png")
        assert len(img_resp.content) > 0

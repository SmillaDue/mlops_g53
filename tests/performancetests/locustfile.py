import io
import random

import cv2
import numpy as np
from locust import HttpUser, between, task


def make_png_bytes(size: int = 64) -> bytes:
    """
    Create a small valid PNG image in memory.
    """
    img = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


class MyUser(HttpUser):
    """
    Locust user for the Model Inference API.
    """

    wait_time = between(1, 2)

    def on_start(self):
        # Pre-generate an image so we don't spend time creating it every request
        self.image_bytes = make_png_bytes()

    @task(1)
    def get_root(self) -> None:
        """Visit the root/health endpoint."""
        self.client.get("/")

    @task(5)
    def post_inference(self) -> None:
        """Upload an image and get predictions."""
        files = {
            "data": ("image.png", self.image_bytes, "image/png"),
        }
        self.client.post("/inference/", files=files)

    @task(2)
    def get_preprocessed_image(self) -> None:
        """
        Fetch the most recently preprocessed image.
        This may return 404 if no inference has run yet,
        which is acceptable in a load test.
        """
        self.client.get("/inference/image_preprocessed.png")

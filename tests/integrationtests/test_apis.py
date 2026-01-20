from fastapi.testclient import TestClient
from mlops_project.api import app


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Welcome to the model inference API!",
            "status-code": 200,
        }, f"{response.json()}"

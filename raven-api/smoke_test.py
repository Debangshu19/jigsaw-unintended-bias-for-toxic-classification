from fastapi.testclient import TestClient

from app import app


def main():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["ok"] is True

    response = client.post(
        "/predict-batch",
        json={
            "texts": [
                "Great article. Thanks for sharing this perspective.",
                "You are stupid and I hate this.",
            ]
        },
    )
    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert predictions[0]["label"] == "safe", predictions
    assert predictions[1]["label"] == "review", predictions
    print("Raven API smoke test passed")


if __name__ == "__main__":
    main()

from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"


def test_ingest_requires_api_key():
    client = TestClient(app)
    resp = client.post(
        "/ingest",
        json={
            "docs_dir": "./knowledge_base",
            "store_type": "chroma",
            "chunk_size": 800,
            "chunk_overlap": 120,
        },
    )
    assert resp.status_code == 401

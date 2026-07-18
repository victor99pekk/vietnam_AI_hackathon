from fastapi.testclient import TestClient

from kg_generator.api import DEMO_TEXT, app


client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_sample():
    response = client.get("/api/demo/sample")
    assert response.status_code == 200
    assert response.json()["text"] == DEMO_TEXT


def test_pipeline_run_returns_demo_contract(monkeypatch):
    # Keep the unit test independent of optional NLP/native dependencies;
    # production attempts the full pipeline by default.
    monkeypatch.setenv("KG_DEMO_USE_FULL_PIPELINE", "0")
    response = client.post("/api/pipeline/run", json={"text": DEMO_TEXT, "language": "vi", "extraction": "offline"})
    assert response.status_code == 200
    payload = response.json()
    assert {"metadata", "graph", "entities", "triples", "stats", "metrics"} <= payload.keys()
    assert payload["stats"]["num_nodes"] >= 1


def test_pipeline_rejects_blank_and_oversized_input():
    assert client.post("/api/pipeline/run", json={"text": "  "}).status_code == 422
    assert client.post("/api/pipeline/run", json={"text": "x" * 20_001}).status_code == 422

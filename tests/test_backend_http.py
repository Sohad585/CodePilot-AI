"""Real HTTP-level checks. Skipped unless FastAPI (and its TestClient) are installed."""

import pytest

pytest.importorskip("fastapi.testclient")
pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

from codepilot_ai.backend import main  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path)
    return TestClient(main.app)


def test_health_and_root(client):
    assert client.get("/health").json() == {"status": "healthy"}
    assert "CodePilot" in client.get("/").json()["message"]


def test_upload_then_analyze_round_trip(client, monkeypatch):
    ok = {"overall_success": True, "valid_syntax": True, "static_analysis": {}, "ai_analysis": {"success": True}}
    monkeypatch.setattr(main, "analyze_python_code", lambda code: ok)
    response = client.post("/upload", files={"file": ("a.py", b"x = 1\n", "text/x-python")})
    assert response.status_code == 200
    analyzed = client.post(f"/upload/{response.json()['upload_id']}/analyze")
    assert analyzed.status_code == 200 and analyzed.json()["analysis"] == ok


def test_validation_errors_are_422(client):
    assert client.post("/run", json={"task": "", "source_code": "x"}).status_code == 422
    assert client.post("/analyze", json={"source_code": "   "}).status_code == 422
    assert client.post("/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 400


def test_provider_failure_reaches_the_client_with_its_type(client, monkeypatch):
    failure = {"overall_success": False, "valid_syntax": True, "static_analysis": {},
               "ai_analysis": {"success": False, "error": "Groq rejected the API key", "error_type": "provider_auth",
                               "retryable": False}}
    monkeypatch.setattr(main, "analyze_python_code", lambda code: failure)
    response = client.post("/analyze", json={"source_code": "x = 1"})
    assert response.status_code == 502 and response.json()["detail"]["error_type"] == "provider_auth"

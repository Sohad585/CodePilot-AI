"""Backend handler tests (no HTTP layer, no network, no Docker, no API key)."""

import asyncio
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")

from pydantic import ValidationError  # noqa: E402

from codepilot_ai.backend import main  # noqa: E402

UUID = "123e4567-e89b-12d3-a456-426614174000"


class FakeUpload:
    def __init__(self, filename, data):
        self.filename, self._data, self.closed = filename, data, False

    async def read(self, size):
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk

    async def close(self):
        self.closed = True


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_ROOT", tmp_path)
    return tmp_path


def upload(filename, data):
    return asyncio.run(main.upload_python_file(FakeUpload(filename, data)))


# ---- health ------------------------------------------------------------------

def test_health():
    assert main.health() == {"status": "healthy"}


def test_health_details_never_exposes_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue123")
    monkeypatch.setattr(main, "check_docker_environment", lambda: {"success": False, "error": "no docker"})
    details = main.health_details()
    assert details["llm"]["api_key_configured"] is True and "supersecret" not in str(details)
    assert details["sandbox"] == {"available": False, "image": None, "error": "no docker"}


# ---- request models ------------------------------------------------------------

def test_request_validation():
    with pytest.raises(ValidationError):
        main.CodeAnalysisRequest(source_code="   ")
    with pytest.raises(ValidationError):
        main.WorkflowRequest(task="t", source_code="")
    assert main.WorkflowRequest(task="t", source_code="x=1", test_code="   ").test_code is None
    assert main.WorkflowRequest(task="t", source_code="x=1").source_filename == "solution.py"
    with pytest.raises(ValidationError):
        main.SandboxTestRequest(source_code="x", test_code="y", timeout=0)


# ---- upload ----------------------------------------------------------------------

def test_upload_success_stores_file_without_executing(uploads):
    result = upload("My App.py", b"raise SystemExit('must never run')\n")
    assert result["success"] and result["filename"] == "My App.py"
    stored = uploads / result["upload_id"] / "source.py"
    assert stored.read_text() == "raise SystemExit('must never run')\n"


@pytest.mark.parametrize(
    "filename, data, status",
    [("a.txt", b"x=1", 400), (None, b"x=1", 400), ("a.py", b"", 400), ("a.py", b"   \n", 400),
     ("a.py", b"\xff\xfe\x00", 400), ("a.py", b"x=1\x00", 400)],
)
def test_upload_rejections_leave_nothing_behind(uploads, filename, data, status):
    with pytest.raises(main.HTTPException) as info:
        upload(filename, data)
    assert info.value.status_code == status
    assert list(uploads.iterdir()) == []


def test_upload_too_large(uploads, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", 10)
    with pytest.raises(main.HTTPException) as info:
        upload("a.py", b"x" * 11)
    assert info.value.status_code == 413 and list(uploads.iterdir()) == []


def test_cleanup_old_uploads(uploads):
    old, fresh = uploads / "old", uploads / "fresh"
    old.mkdir(); fresh.mkdir()
    past = time.time() - 3 * 24 * 3600
    os.utime(old, (past, past))
    assert main.cleanup_old_uploads() == 1
    assert not old.exists() and fresh.exists()


# ---- analyze -----------------------------------------------------------------------

def ai_failure(error_type, **extra):
    return {"overall_success": False, "valid_syntax": True, "static_analysis": {"issues": []},
            "ai_analysis": {"success": False, "error": f"msg for {error_type}", "error_type": error_type,
                            "retryable": True, **extra}}


@pytest.mark.parametrize(
    "error_type, status",
    [("provider_unavailable", 503), ("provider_timeout", 504), ("provider_rate_limited", 429),
     ("provider_auth", 502), ("invalid_response", 502), ("configuration", 500), ("analysis_failed", 500)],
)
def test_analyze_maps_each_failure_to_its_own_status(monkeypatch, error_type, status):
    monkeypatch.setattr(main, "analyze_python_code", lambda code: ai_failure(error_type))
    with pytest.raises(main.HTTPException) as info:
        main.analyze_code(main.CodeAnalysisRequest(source_code="x = 1"))
    assert info.value.status_code == status
    detail = info.value.detail
    assert detail["error_type"] == error_type and detail["message"] == f"msg for {error_type}"
    assert detail["analysis"]["static_analysis"] == {"issues": []}   # static results are kept


def test_analyze_redacts_secrets_in_errors(monkeypatch):
    failure = ai_failure("provider_auth")
    failure["ai_analysis"]["error"] = "bad key gsk_SECRETSECRET123456"
    monkeypatch.setattr(main, "analyze_python_code", lambda code: failure)
    with pytest.raises(main.HTTPException) as info:
        main.analyze_code(main.CodeAnalysisRequest(source_code="x = 1"))
    assert "SECRETSECRET" not in str(info.value.detail)


def test_analyze_success_and_uploaded_analysis(uploads, monkeypatch):
    ok = {"overall_success": True, "valid_syntax": True, "static_analysis": {}, "ai_analysis": {"success": True}}
    monkeypatch.setattr(main, "analyze_python_code", lambda code: ok)
    assert main.analyze_code(main.CodeAnalysisRequest(source_code="x = 1")) == ok
    stored = upload("a.py", b"x = 1\n")
    result = main.analyze_uploaded_file(stored["upload_id"])
    assert result["success"] and result["analysis"] == ok


def test_analyze_uploaded_file_errors(uploads):
    for bad, status in (("not-a-uuid", 400), (UUID, 404)):
        with pytest.raises(main.HTTPException) as info:
            main.analyze_uploaded_file(bad)
        assert info.value.status_code == status


# ---- run ---------------------------------------------------------------------------

class FakeWorkflow:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.state = result, error, None

    def invoke(self, state):
        self.state = state
        if self.error:
            raise self.error
        return self.result


def test_run_builds_full_initial_state(monkeypatch):
    fake = FakeWorkflow({"status": "completed"})
    monkeypatch.setattr(main, "codepilot_workflow", fake)
    request = main.WorkflowRequest(task="t", source_code="x = 1", test_code="def test_a(): pass",
                                   source_filename="calc.py")
    assert main.run_workflow(request) == {"status": "completed"}
    state = fake.state
    assert state["current_code"] == "x = 1" and state["test_code"] and state["source_filename"] == "calc.py"
    # Every key the workflow reads must exist in the initial state.
    from codepilot_ai.workflow import CodePilotState
    assert set(state) == set(CodePilotState.__annotations__)


def test_run_contains_workflow_crash_and_redacts(monkeypatch):
    monkeypatch.setattr(main, "codepilot_workflow", FakeWorkflow(error=RuntimeError("x gsk_SECRETSECRET123456")))
    result = main.run_workflow(main.WorkflowRequest(task="t", source_code="x = 1"))
    assert result["status"] == "failed" and result["error_type"] == "workflow_error"
    assert "SECRETSECRET" not in result["error"]


# ---- sandbox endpoint ----------------------------------------------------------------

def test_sandbox_endpoint(monkeypatch):
    request = main.SandboxTestRequest(source_code="x = 1", test_code="def test_a(): pass")
    monkeypatch.setattr(main, "run_code_tests_in_sandbox", lambda **kw: {"success": True, "tests_passed": False})
    assert main.run_sandbox_tests(request)["tests_passed"] is False   # failing tests are still HTTP 200

    for error_type, status in (("sandbox_unavailable", 503), ("sandbox_timeout", 504), ("invalid_input", 422),
                               ("sandbox_error", 500)):
        monkeypatch.setattr(main, "run_code_tests_in_sandbox",
                            lambda error_type=error_type, **kw: {"success": False, "error_type": error_type,
                                                                 "error": "e"})
        with pytest.raises(main.HTTPException) as info:
            main.run_sandbox_tests(request)
        assert info.value.status_code == status and info.value.detail["error_type"] == error_type

"""Frontend tests for the modules that do not import Streamlit."""

import json

import httpx
import pytest

import api_client
import diff_view
import helpers


REAL_CLIENT = httpx.Client


def mock_backend(monkeypatch, handler):
    """Make api_client talk to ``handler(request) -> httpx.Response``."""
    monkeypatch.setattr(
        api_client.httpx, "Client",
        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )


# ---- the reported bug: 503 must not always mean "Groq is down" ------------------------

def test_plain_503_is_not_blamed_on_any_provider():
    error = api_client.describe_failure("Service Unavailable", status_code=503)
    assert "Groq" not in error.title + error.message and "Gemini" not in error.title + error.message
    assert error.kind == "error"


def test_503_with_provider_type_names_groq():
    error = api_client.describe_failure("Groq is temporarily unavailable (HTTP 503).",
                                        status_code=503, error_type="provider_unavailable")
    assert error.title == "Groq is temporarily unavailable" and error.retryable is True


def test_503_from_the_sandbox_is_a_sandbox_problem_not_groq():
    error = api_client.describe_failure("Docker Engine is not available.", status_code=503,
                                        error_type="sandbox_unavailable")
    assert "sandbox" in error.title.lower() and "Groq" not in error.title and error.retryable is False


@pytest.mark.parametrize("text", ["the unavailable overloaded timeout 504 503 words", "high demand"])
def test_words_in_text_do_not_trigger_provider_messages(text):
    error = api_client.describe_failure(text)
    assert error.kind == "error" and "Groq" not in error.title


def test_every_error_title_is_for_a_known_type_and_none_mentions_gemini():
    assert all("gemini" not in t.lower() and "google" not in t.lower() for t, _ in api_client.ERROR_TITLES.values())


def test_redaction_covers_groq_keys():
    out = api_client.redact("failed with gsk_abcdEFGH12345678 and api_key=zzzzzzzz")
    assert "gsk_abcd" not in out and "zzzzzzzz" not in out


# ---- HTTP error handling ----------------------------------------------------------------

def test_structured_backend_error_is_decoded(monkeypatch):
    body = {"detail": {"error_type": "provider_rate_limited", "message": "Groq's rate limit was reached (HTTP 429).",
                       "retryable": True, "analysis": {"static_analysis": {}}}}
    mock_backend(monkeypatch, lambda r: httpx.Response(429, json=body))
    with pytest.raises(api_client.ApiError) as info:
        api_client.analyze_upload("http://x", "id")
    error = info.value
    assert error.kind == "provider_rate_limited" and error.status_code == 429 and error.retryable
    assert "rate limit" in error.message and "static_analysis" not in error.technical


def test_validation_error_list_is_readable(monkeypatch):
    body = {"detail": [{"loc": ["body", "task"], "msg": "String should have at least 1 character"}]}
    mock_backend(monkeypatch, lambda r: httpx.Response(422, json=body))
    with pytest.raises(api_client.ApiError) as info:
        api_client.run_workflow("http://x", "", "x = 1")
    assert "task: String should have" in info.value.message and info.value.kind == "rejected"


def test_non_json_and_non_object_responses(monkeypatch):
    mock_backend(monkeypatch, lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(api_client.ApiError) as info:
        api_client.upload_python_file("http://x", "a.py", b"x")
    assert info.value.kind == "bad_response"
    mock_backend(monkeypatch, lambda r: httpx.Response(200, json=[1]))
    with pytest.raises(api_client.ApiError):
        api_client.upload_python_file("http://x", "a.py", b"x")


def test_connection_failures_are_friendly(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")
    mock_backend(monkeypatch, refuse)
    with pytest.raises(api_client.ApiError) as info:
        api_client.upload_python_file("http://x", "a.py", b"x")
    assert info.value.kind == "connection" and info.value.retryable


def test_health_check(monkeypatch):
    monkeypatch.setattr(api_client.httpx, "get", lambda *a, **k: httpx.Response(200, json={"status": "healthy"}))
    assert api_client.check_health("http://x") is True
    monkeypatch.setattr(api_client.httpx, "get", lambda *a, **k: httpx.Response(500, json={}))
    assert api_client.check_health("http://x") is False

    def boom(*a, **k):
        raise httpx.ConnectError("no")
    monkeypatch.setattr(api_client.httpx, "get", boom)
    assert api_client.check_health("http://x") is False


# ---- request payloads match the backend models -----------------------------------------------

def test_run_workflow_payload(monkeypatch):
    seen = {}

    def handler(request):
        seen["path"], seen["json"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={"status": "completed"})

    mock_backend(monkeypatch, handler)
    api_client.run_workflow("http://x/", "task", "x = 1", "tb", {"a": 1}, test_code="def test_a(): pass",
                            source_filename="calc.py")
    assert seen["path"] == "/run"
    assert seen["json"] == {"task": "task", "source_code": "x = 1", "error_context": "tb", "test_report": {"a": 1},
                            "test_code": "def test_a(): pass", "source_filename": "calc.py"}
    api_client.run_workflow("http://x", "task", "x = 1")
    assert seen["json"]["test_code"] is None


def test_run_workflow_requires_status(monkeypatch):
    mock_backend(monkeypatch, lambda r: httpx.Response(200, json={"foo": 1}))
    with pytest.raises(api_client.ApiError):
        api_client.run_workflow("http://x", "t", "x")


def test_upload_sends_multipart_and_requires_id(monkeypatch):
    seen = {}

    def handler(request):
        seen["ctype"] = request.headers["content-type"]
        return httpx.Response(200, json={"success": True, "upload_id": "u1"})

    mock_backend(monkeypatch, handler)
    assert api_client.upload_python_file("http://x", "a.py", b"x = 1")["upload_id"] == "u1"
    assert seen["ctype"].startswith("multipart/form-data")
    mock_backend(monkeypatch, lambda r: httpx.Response(200, json={"success": True}))
    with pytest.raises(api_client.ApiError):
        api_client.upload_python_file("http://x", "a.py", b"x")


def test_analyze_upload_returns_analysis(monkeypatch):
    mock_backend(monkeypatch, lambda r: httpx.Response(200, json={"analysis": {"valid_syntax": True}}))
    assert api_client.analyze_upload("http://x", "u1") == {"valid_syntax": True}
    mock_backend(monkeypatch, lambda r: httpx.Response(200, json={"success": True}))
    with pytest.raises(api_client.ApiError):
        api_client.analyze_upload("http://x", "u1")


# ---- workflow states ----------------------------------------------------------------------------

def test_failed_workflow_state_uses_state_error_type():
    state = {"status": "failed", "error": "Refactorer Agent failed: Groq rejected the API key (HTTP 401).",
             "error_type": "provider_auth", "retryable": False}
    error = api_client.error_from_workflow_state(state)
    assert error.kind == "provider_auth" and error.retryable is False and "Groq" in error.title


def test_verification_failure_is_not_retryable_and_not_a_provider_error():
    error = api_client.error_from_workflow_state(
        {"status": "failed", "error": "The modified code does not pass your tests (1 failed).",
         "error_type": "verification_failed", "retryable": False})
    assert error.kind == "verification_failed" and "Groq" not in error.title


def test_completed_and_clarification_states_are_not_errors():
    assert api_client.error_from_workflow_state({"status": "completed"}) is None
    assert api_client.error_from_workflow_state({"status": "needs_clarification"}) is None


def test_failed_state_without_details_still_reports():
    assert api_client.error_from_workflow_state({"status": "failed"}).message == "The workflow failed."


# ---- helpers ----------------------------------------------------------------------------------------

def test_validate_python_upload():
    assert helpers.validate_python_upload("a.py", b"x = 1\n") == ("x = 1\n", None)
    for name, data in (("a.txt", b"x"), ("a.py", b""), ("a.py", b"  \n"), ("a.py", b"\xff"), ("a.py", b"x\x00"),
                       ("a.py", b"x" * (helpers.MAX_UPLOAD_BYTES + 1))):
        code, error = helpers.validate_python_upload(name, data)
        assert code is None and error


def test_module_filename():
    assert helpers.module_filename("inventory.py") == "inventory.py"
    assert helpers.module_filename("my app-v2.py") == "solution.py"
    assert helpers.module_filename(None) == "solution.py"
    assert helpers.module_filename("../../evil.py") == "evil.py"


def test_parse_test_report():
    assert helpers.parse_test_report("") == (None, None)
    assert helpers.parse_test_report('{"a": 1}') == ({"a": 1}, None)
    assert helpers.parse_test_report("{")[1] and helpers.parse_test_report("[]")[1] and helpers.parse_test_report("{}")[1]


def test_compose_task_and_download_name():
    assert helpers.compose_task("debug", "").startswith("Debug this Python code")
    assert "extra" in helpers.compose_task("refactor", "extra")
    assert helpers.safe_download_name("my app!.py") == "my_app_modified.py"
    assert helpers.safe_download_name(None) == "source_modified.py"


def test_limits_match_the_backend():
    pytest.importorskip("fastapi")
    pytest.importorskip("langgraph")
    from codepilot_ai.backend import main

    def max_length(field):
        return next(m.max_length for m in main.WorkflowRequest.model_fields[field].metadata
                    if hasattr(m, "max_length"))

    assert max_length("test_code") == helpers.MAX_TEST_CODE_CHARS
    assert max_length("source_code") == helpers.MAX_RUN_CHARS
    assert max_length("error_context") == helpers.MAX_ERROR_CONTEXT_CHARS
    assert helpers.MAX_UPLOAD_BYTES == main.MAX_UPLOAD_SIZE


# ---- diff view -----------------------------------------------------------------------------------------

def test_diff_stats_and_unified_diff():
    assert diff_view.diff_stats("a\nb\n", "a\nc\nd\n") == (2, 1)
    assert diff_view.unified_diff("a\n", "a\n") == ""
    assert "+c" in diff_view.unified_diff("a\n", "a\nc\n")


def test_side_by_side_escapes_html_and_handles_edge_cases():
    html = diff_view.side_by_side_html("x = '<script>alert(1)</script>'\n", "x = 1\n")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "cp-diff" in diff_view.side_by_side_html("", "a\n")
    assert "unchanged" in diff_view.side_by_side_html("\n".join("l%d" % i for i in range(30)),
                                                      "\n".join("l%d" % i for i in range(30)) + "\nnew", 3)

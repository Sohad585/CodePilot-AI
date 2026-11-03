"""HTTP client for the CodePilot AI FastAPI backend.

Only endpoints that already exist are used:

    GET  /health
    POST /upload
    POST /upload/{upload_id}/analyze
    POST /run

Every failure is converted into an :class:`ApiError` carrying a message that
is safe to show to the user, so the Streamlit app never crashes on a bad
response or a timeout.

Failures are described from the ``error_type`` the backend reports (it comes
from the real exception or HTTP status of the failed step), never guessed from
words such as "503" or "unavailable" in the text.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

DEFAULT_API_URL = os.getenv("CODEPILOT_API_URL", "http://127.0.0.1:8000")

# Backend LLM calls may take a while (up to 160 s each, and a workflow can
# chain several agents), so reads get a generous timeout.
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0)


class ApiError(Exception):
    """A failed backend call, described in terms a user can act on."""

    def __init__(
        self,
        kind: str,
        title: str,
        message: str,
        *,
        status_code: int | None = None,
        technical: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.title = title
        self.message = message
        self.status_code = status_code
        self.technical = technical
        self.retryable = retryable


# --------------------------------------------------------------------------- #
# Redaction and error description
# --------------------------------------------------------------------------- #

_SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9_\-]{8,}"),  # Groq API keys
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # other sk- style keys
    re.compile(
        r"(?i)\b(api[_-]?key|token|authorization)(\"?\s*[:=]\s*\"?)([^\s\"&,}]+)"
    ),
)


def redact(text: str) -> str:
    """Mask anything that looks like an API key or token."""
    text = _SECRET_PATTERNS[0].sub("[redacted]", text)
    text = _SECRET_PATTERNS[1].sub("[redacted]", text)
    return _SECRET_PATTERNS[2].sub(r"\1\2[redacted]", text)


# error_type reported by the backend -> (title, default retryable).
# The backend's own message already says what went wrong, so it is shown as
# the body; the title only names the kind of problem.
ERROR_TITLES: dict[str, tuple[str, bool]] = {
    "provider_unavailable": ("Groq is temporarily unavailable", True),
    "provider_timeout": ("The AI request timed out", True),
    "provider_rate_limited": ("Groq rate limit reached", True),
    "provider_auth": ("Groq rejected the API key", False),
    "provider_bad_request": ("Groq rejected the request", False),
    "invalid_response": ("The AI returned an unusable answer", True),
    "configuration": ("Backend configuration problem", False),
    "analysis_failed": ("The analysis failed", False),
    "agent_failed": ("An agent failed", False),
    "supervisor_failed": ("The Supervisor could not plan the task", True),
    "workflow_error": ("The workflow crashed", False),
    "verification_failed": ("The code does not pass your tests", False),
    "sandbox_unavailable": ("The Docker sandbox is not available", False),
    "sandbox_timeout": ("The sandbox run timed out", False),
    "sandbox_error": ("The sandbox run failed", False),
    "invalid_input": ("Invalid input", False),
}


def describe_failure(
    text: str,
    *,
    status_code: int | None = None,
    technical: str | None = None,
    error_type: str | None = None,
    retryable: bool | None = None,
) -> ApiError:
    """Turn a backend error (text, status and error_type) into an ApiError."""
    text = redact((text or "").strip())
    detail = redact(technical if technical is not None else text)[:3000]

    if error_type in ERROR_TITLES:
        title, default_retryable = ERROR_TITLES[error_type]
        return ApiError(
            error_type,
            title,
            text or "The backend did not give a reason.",
            status_code=status_code,
            technical=detail,
            retryable=default_retryable if retryable is None else bool(retryable),
        )

    # The backend gave no error_type (an unexpected or non-CodePilot response).
    # Describe only what is known, and do not blame any particular service.
    if status_code == 500:
        return ApiError(
            "server",
            "Backend error (HTTP 500)",
            text or "The backend hit an unexpected error.",
            status_code=status_code,
            technical=detail,
            retryable=bool(retryable),
        )

    if status_code is not None and 400 <= status_code < 500:
        return ApiError(
            "rejected",
            f"Request rejected (HTTP {status_code})",
            text or "The backend rejected the request.",
            status_code=status_code,
            technical=detail,
            retryable=False,
        )

    return ApiError(
        "error",
        "Request failed"
        if status_code is None
        else f"Request failed (HTTP {status_code})",
        text or "The backend reported an unknown error.",
        status_code=status_code,
        technical=detail,
        retryable=True if retryable is None else bool(retryable),
    )


def _detail_text(detail: Any) -> str:
    """Extract readable text from a FastAPI ``detail`` value."""
    if detail is None:
        return ""

    if isinstance(detail, str):
        return detail

    if isinstance(detail, dict):
        for key in ("message", "error", "detail"):
            if detail.get(key):
                return str(detail[key])

        return json.dumps(detail, default=str)[:500]

    if isinstance(detail, list):  # Pydantic validation errors (HTTP 422)
        parts = []
        for item in detail:
            if isinstance(item, dict):
                location = ".".join(
                    str(part) for part in item.get("loc", []) if part != "body"
                )
                message = item.get("msg", "invalid value")
                parts.append(f"{location}: {message}" if location else str(message))
            else:
                parts.append(str(item))
        return "; ".join(parts)

    return str(detail)


def _error_from_response(response: httpx.Response) -> ApiError:
    try:
        body = response.json()
    except ValueError:
        body = None

    detail = body.get("detail") if isinstance(body, dict) else None
    text = _detail_text(detail) or response.text[:500]

    error_type = detail.get("error_type") if isinstance(detail, dict) else None
    retryable = detail.get("retryable") if isinstance(detail, dict) else None

    if isinstance(detail, dict):
        # Keep the technical view small: the static analysis that may ride
        # along with an AI failure is not an error detail.
        shown = {k: v for k, v in detail.items() if k not in ("analysis", "result")}
        raw = json.dumps(shown, indent=2, default=str)
    else:
        raw = text

    technical = f"HTTP {response.status_code} {response.reason_phrase}\n{raw}"

    return describe_failure(
        text,
        status_code=response.status_code,
        technical=technical,
        error_type=error_type if isinstance(error_type, str) else None,
        retryable=retryable if isinstance(retryable, bool) else None,
    )


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def _base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _request(method: str, base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Send a request and return the decoded JSON object, or raise ApiError."""
    base = _base_url(base_url)

    try:
        with httpx.Client(base_url=base, timeout=REQUEST_TIMEOUT) as client:
            response = client.request(method, path, **kwargs)

    except httpx.ConnectTimeout:
        raise ApiError(
            "connection",
            "Cannot reach the backend",
            f"Connecting to {base} timed out. Check the backend URL in the sidebar.",
            retryable=True,
        ) from None

    except httpx.ConnectError:
        raise ApiError(
            "connection",
            "Cannot reach the backend",
            f"Nothing is answering at {base}. Start the backend or fix the "
            "backend URL in the sidebar.",
            retryable=True,
        ) from None

    except httpx.TimeoutException:
        raise ApiError(
            "timeout",
            "The request timed out",
            "The backend did not answer in time. The AI model may be slow or "
            "overloaded — try again.",
            retryable=True,
        ) from None

    except (httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError):
        raise ApiError(
            "connection",
            "Invalid backend URL",
            f"'{base}' is not a valid URL. Use something like http://127.0.0.1:8000.",
        ) from None

    except httpx.HTTPError as error:
        raise ApiError(
            "connection",
            "Network error",
            "The connection to the backend failed unexpectedly.",
            technical=redact(f"{type(error).__name__}: {error}"),
            retryable=True,
        ) from None

    if response.status_code >= 400:
        raise _error_from_response(response)

    try:
        body = response.json()
    except ValueError:
        raise ApiError(
            "bad_response",
            "Unexpected backend response",
            "The backend answered with something that is not JSON.",
            status_code=response.status_code,
            technical=redact(response.text[:1000]),
        ) from None

    if not isinstance(body, dict):
        raise ApiError(
            "bad_response",
            "Unexpected backend response",
            "The backend answered with an unexpected JSON structure.",
            status_code=response.status_code,
            technical=redact(response.text[:1000]),
        )

    return body


def check_health(base_url: str) -> bool:
    """Return True when GET /health reports a healthy backend. Never raises."""
    try:
        response = httpx.get(f"{_base_url(base_url)}/health", timeout=HEALTH_TIMEOUT)
        return (
            response.status_code == 200 and response.json().get("status") == "healthy"
        )
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, AttributeError):
        return False


def upload_python_file(base_url: str, filename: str, content: bytes) -> dict[str, Any]:
    """POST /upload — store the file on the backend (it is never executed)."""
    body = _request(
        "POST",
        base_url,
        "/upload",
        files={"file": (filename, content, "text/x-python")},
    )

    if not body.get("upload_id"):
        raise ApiError(
            "bad_response",
            "Unexpected backend response",
            "The upload succeeded but the backend returned no upload ID.",
        )

    return body


def analyze_upload(base_url: str, upload_id: str) -> dict[str, Any]:
    """POST /upload/{upload_id}/analyze — returns the ``analysis`` object."""
    body = _request("POST", base_url, f"/upload/{upload_id}/analyze")
    analysis = body.get("analysis")

    if not isinstance(analysis, dict):
        raise ApiError(
            "bad_response",
            "Unexpected backend response",
            "The analysis response did not contain an analysis.",
            technical=redact(json.dumps(body, default=str)[:1000]),
        )

    return analysis


def run_workflow(
    base_url: str,
    task: str,
    source_code: str,
    error_context: str = "",
    test_report: dict[str, Any] | None = None,
    test_code: str = "",
    source_filename: str = "solution.py",
) -> dict[str, Any]:
    """POST /run — returns the final workflow state.

    The backend answers HTTP 200 even when the workflow fails, so callers must
    check ``state["status"]`` (see :func:`error_from_workflow_state`).

    ``test_code`` (optional pytest code) makes the backend run the final code
    against it inside the Docker sandbox.
    """
    state = _request(
        "POST",
        base_url,
        "/run",
        json={
            "task": task,
            "source_code": source_code,
            "error_context": error_context,
            "test_report": test_report,
            "test_code": test_code or None,
            "source_filename": source_filename,
        },
    )

    if "status" not in state:
        raise ApiError(
            "bad_response",
            "Unexpected backend response",
            "The workflow response did not contain a status.",
            technical=redact(json.dumps(state, default=str)[:1000]),
        )

    return state


def error_from_workflow_state(state: dict[str, Any]) -> ApiError | None:
    """Describe a failed workflow state, or return None if it did not fail."""
    if state.get("status") != "failed":
        return None

    error_type = state.get("error_type")
    retryable = state.get("retryable")

    return describe_failure(
        str(state.get("error") or "The workflow failed."),
        error_type=error_type if isinstance(error_type, str) else None,
        retryable=retryable if isinstance(retryable, bool) else None,
    )

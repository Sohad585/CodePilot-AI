"""Shared helpers for the LLM-backed agents.

Everything the agents used to copy-paste lives here:

* reading text out of a LangChain response,
* parsing the JSON object the model was asked to return,
* checking that returned code is valid Python,
* turning an exception into an accurate, user-safe error description.

Nothing in this module calls a provider or executes submitted code.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class LLMConfigurationError(ValueError):
    """The LLM provider is not configured (for example, no API key)."""


class LLMResponseError(ValueError):
    """The model answered, but not with a usable result."""


# --------------------------------------------------------------------------- #
# Secret redaction
# --------------------------------------------------------------------------- #

_SECRET_PATTERNS = (
    (re.compile(r"\bgsk_[A-Za-z0-9_\-]{8,}"), "[redacted]"),  # Groq keys
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[redacted]"),  # other sk- style keys
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer [redacted]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|authorization)(\"?\s*[:=]\s*\"?)([^\s\"&,}]+)"
        ),
        r"\1\2[redacted]",
    ),
)


def redact_secrets(text: str) -> str:
    """Mask anything that looks like an API key or token."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# --------------------------------------------------------------------------- #
# Reading model output
# --------------------------------------------------------------------------- #


def extract_text(response: Any) -> str:
    """Return the text of a LangChain chat response.

    Handles plain-string content and the list-of-blocks format some providers
    return. Raises :class:`LLMResponseError` when there is no text at all.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in (None, "text")
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        text = "\n".join(parts)
    else:
        raise LLMResponseError("The model returned an unsupported response format.")

    if not text.strip():
        raise LLMResponseError("The model returned an empty response.")

    return text


_FENCED = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)\n?```[ \t]*$", re.DOTALL)


def parse_json_object(text: str) -> dict:
    """Parse the JSON object the model was asked to return.

    Tolerates a surrounding Markdown code fence (with or without a language
    tag) and short prose before or after the object. Raises
    :class:`LLMResponseError` if no JSON object can be recovered.
    """
    text = text.strip()

    fenced = _FENCED.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")

        if start == -1 or end <= start:
            raise LLMResponseError(
                "The model's reply did not contain a JSON object."
            ) from None

        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise LLMResponseError(
                "Could not parse the model's reply as JSON."
            ) from None

    if not isinstance(value, dict):
        raise LLMResponseError("The model's reply was JSON, but not an object.")

    return value


def require_fields(result: dict, types: dict[str, type | tuple[type, ...]]) -> None:
    """Check that ``result`` has every field in ``types`` with the right type.

    ``bool`` is never accepted where ``int`` is required.
    """
    missing = [name for name in types if name not in result]
    if missing:
        raise LLMResponseError(
            "The model's reply is missing required fields: " + ", ".join(missing) + "."
        )

    for name, expected in types.items():
        value = result[name]
        expected_tuple = expected if isinstance(expected, tuple) else (expected,)

        wrong_type = not isinstance(value, expected_tuple)
        bool_as_int = isinstance(value, bool) and bool not in expected_tuple

        if wrong_type or bool_as_int:
            raise LLMResponseError(
                f"The model's reply has an invalid value for '{name}'."
            )


def validate_python_code(code: str, label: str) -> str:
    """Return ``code`` if it is non-empty, valid Python; otherwise raise.

    The code is parsed with :func:`ast.parse` and never executed.
    """
    if not code.strip():
        raise LLMResponseError(f"The model returned empty code for '{label}'.")

    try:
        ast.parse(code)
    except SyntaxError as error:
        raise LLMResponseError(
            f"The model's '{label}' is not valid Python "
            f"(line {error.lineno}: {error.msg}). The original code was kept."
        ) from None
    except (ValueError, RecursionError, MemoryError):
        raise LLMResponseError(
            f"The model's '{label}' could not be parsed as Python. "
            "The original code was kept."
        ) from None

    return code


def code_fence(source: str) -> tuple[str, str]:
    """Return ``(opening, closing)`` fence lines that ``source`` cannot close.

    Submitted code that contains ``` would otherwise end the fenced block in
    the prompt early.
    """
    longest = max((len(run) for run in re.findall(r"`+", source)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}python", ticks


# --------------------------------------------------------------------------- #
# Exception classification
# --------------------------------------------------------------------------- #

# error_type -> (message, retryable). ``{detail}`` is filled in when present.
_PROVIDER_MESSAGES = {
    "provider_unavailable": (
        "Groq is temporarily unavailable or could not be reached"
        "{status}. Your file and task are unchanged; try again in a moment.",
        True,
    ),
    "provider_timeout": (
        "The request to Groq timed out{status}. Try again; large files and "
        "broad tasks take longer.",
        True,
    ),
    "provider_rate_limited": (
        "Groq's rate limit was reached{status}. Wait a little and try again.",
        True,
    ),
    "provider_auth": (
        "Groq rejected the API key{status}. Check GROQ_API_KEY in the "
        "backend's .env file and restart the backend.",
        False,
    ),
    "provider_bad_request": (
        "Groq rejected the request{status}: {detail}",
        False,
    ),
}


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen and len(chain) < 6:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return chain


def _status_code(error: BaseException) -> int | None:
    for holder in (error, getattr(error, "response", None)):
        status = getattr(holder, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _provider_error_type(error: BaseException) -> str | None:
    """Map one exception to a provider error type, or None if it isn't one.

    Decisions use the exception's class and HTTP status, never its message
    text, so a "503" appearing in an unrelated message is not mistaken for a
    provider outage.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    status = _status_code(error)

    if names & {"APITimeoutError", "TimeoutException", "TimeoutError"} or status == 504:
        return "provider_timeout"

    if status == 429 or "RateLimitError" in names:
        return "provider_rate_limited"

    if status in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
        return "provider_auth"

    if (status is not None and status >= 500) or "InternalServerError" in names:
        return "provider_unavailable"

    if status is not None and 400 <= status < 500:
        return "provider_bad_request"

    if names & {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "NetworkError",
        "RemoteProtocolError",
    }:
        return "provider_unavailable"

    return None


def classify_exception(error: BaseException, context: str = "The AI request") -> dict:
    """Describe ``error`` accurately and safely.

    Returns a dict with ``error_type``, ``message`` (safe to show to users),
    ``retryable``, ``status_code`` and a redacted, truncated ``detail``.
    """
    chain = _exception_chain(error)
    detail = redact_secrets(str(error)).strip()[:500]

    for item in chain:
        if isinstance(item, LLMConfigurationError):
            return _info("configuration", redact_secrets(str(item)), False, None, detail)

        if isinstance(item, LLMResponseError):
            return _info("invalid_response", redact_secrets(str(item)), True, None, detail)

    for item in chain:
        error_type = _provider_error_type(item)

        if error_type is None:
            continue

        status = _status_code(item)
        template, retryable = _PROVIDER_MESSAGES[error_type]
        provider_detail = redact_secrets(str(item)).strip()[:300] or "no details"
        message = template.format(
            status=f" (HTTP {status})" if status else "",
            detail=provider_detail,
        )
        return _info(error_type, message, retryable, status, provider_detail)

    return _info(
        "analysis_failed",
        f"{context} failed unexpectedly ({type(error).__name__}): {detail}",
        False,
        None,
        detail,
    )


def _info(
    error_type: str,
    message: str,
    retryable: bool,
    status_code: int | None,
    detail: str,
) -> dict:
    return {
        "error_type": error_type,
        "message": message,
        "retryable": retryable,
        "status_code": status_code,
        "detail": detail,
    }


def error_result(error: BaseException, context: str = "The AI request") -> dict:
    """Build the standard failed-agent result for ``error``."""
    info = classify_exception(error, context)

    return {
        "success": False,
        "error": info["message"],
        "error_type": info["error_type"],
        "retryable": info["retryable"],
    }

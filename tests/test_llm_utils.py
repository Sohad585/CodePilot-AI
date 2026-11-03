import types

import pytest

from conftest import StatusError, named_error
from codepilot_ai.llm_utils import (
    LLMConfigurationError,
    LLMResponseError,
    classify_exception,
    code_fence,
    error_result,
    extract_text,
    parse_json_object,
    redact_secrets,
    require_fields,
    validate_python_code,
)


# ---- extract_text -----------------------------------------------------------

def test_extract_text_plain_string():
    assert extract_text(types.SimpleNamespace(content="hello")) == "hello"


def test_extract_text_block_list():
    content = [{"type": "text", "text": "a"}, {"type": "other"}, "b"]
    assert extract_text(types.SimpleNamespace(content=content)) == "a\nb"


@pytest.mark.parametrize("content", ["", "   ", [], None, 5])
def test_extract_text_rejects_empty_or_unsupported(content):
    with pytest.raises(LLMResponseError):
        extract_text(types.SimpleNamespace(content=content))


# ---- parse_json_object ------------------------------------------------------

def test_parse_plain_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json_with_language():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_one_line_fence_does_not_crash():
    # The old code did content.split("\n", 1)[1] and raised IndexError here.
    assert parse_json_object('```{"a": 1}```') == {"a": 1}


def test_parse_json_with_surrounding_prose():
    assert parse_json_object('Sure! {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}


@pytest.mark.parametrize("text", ["no json here", '{"a": ', "[1, 2]", '"str"'])
def test_parse_rejects_non_objects(text):
    with pytest.raises(LLMResponseError):
        parse_json_object(text)


# ---- validation -------------------------------------------------------------

def test_require_fields_rejects_missing_and_wrong_types():
    with pytest.raises(LLMResponseError, match="missing"):
        require_fields({"a": 1}, {"a": int, "b": str})
    with pytest.raises(LLMResponseError, match="invalid value"):
        require_fields({"a": "x"}, {"a": int})
    with pytest.raises(LLMResponseError):  # bool is not an int
        require_fields({"a": True}, {"a": int})
    require_fields({"a": True}, {"a": bool})


def test_validate_python_code():
    assert validate_python_code("x = 1\n", "c") == "x = 1\n"
    with pytest.raises(LLMResponseError, match="not valid Python"):
        validate_python_code("def broken(:\n", "c")
    with pytest.raises(LLMResponseError, match="empty"):
        validate_python_code("  \n", "c")


def test_code_fence_cannot_be_closed_by_submitted_code():
    opening, closing = code_fence("print('''```''')\n````\n")
    assert len(closing) >= 5 and opening.startswith(closing)
    assert code_fence("x = 1")[1] == "```"


# ---- redaction --------------------------------------------------------------

def test_redact_secrets():
    text = "key gsk_abcdEFGH12345678 and Bearer abcdefgh12345 and api_key=hunter2xyz"
    out = redact_secrets(text)
    assert "gsk_abcd" not in out and "abcdefgh12345" not in out and "hunter2xyz" not in out
    assert "GROQ_API_KEY is not set" == redact_secrets("GROQ_API_KEY is not set")


# ---- classify_exception: the reported bug ----------------------------------

@pytest.mark.parametrize(
    "error, expected, retryable",
    [
        (StatusError(503), "provider_unavailable", True),
        (StatusError(500), "provider_unavailable", True),
        (StatusError(504), "provider_timeout", True),
        (StatusError(429), "provider_rate_limited", True),
        (StatusError(401), "provider_auth", False),
        (StatusError(403), "provider_auth", False),
        (StatusError(400), "provider_bad_request", False),
        (StatusError(404), "provider_bad_request", False),
        (named_error("APITimeoutError"), "provider_timeout", True),
        (named_error("APIConnectionError"), "provider_unavailable", True),
        (named_error("RateLimitError"), "provider_rate_limited", True),
        (LLMConfigurationError("GROQ_API_KEY is not set."), "configuration", False),
        (LLMResponseError("bad json"), "invalid_response", True),
    ],
)
def test_classify_provider_errors_by_type(error, expected, retryable):
    info = classify_exception(error)
    assert info["error_type"] == expected
    assert info["retryable"] is retryable


def test_503_in_message_text_alone_is_not_a_provider_outage():
    for error in (ValueError("503 items found"), RuntimeError("service unavailable in my code")):
        info = classify_exception(error, context="Debugging")
        assert info["error_type"] == "analysis_failed"
        assert "Groq" not in info["message"]


def test_wrapped_provider_error_is_found_through_cause():
    try:
        try:
            raise StatusError(429)
        except StatusError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        assert classify_exception(outer)["error_type"] == "provider_rate_limited"


def test_error_messages_never_leak_keys():
    info = classify_exception(StatusError(401, "bad key gsk_SECRETSECRET123456"))
    assert "SECRETSECRET" not in str(info)
    result = error_result(RuntimeError("api_key=gsk_SECRETSECRET123456"))
    assert "SECRETSECRET" not in str(result)
    assert result["success"] is False and result["error_type"] == "analysis_failed"

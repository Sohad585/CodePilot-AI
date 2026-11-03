import sys
import types

import pytest

from codepilot_ai import llm_factory
from codepilot_ai.llm_utils import LLMConfigurationError


def test_missing_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        llm_factory.get_llm()


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(LLMConfigurationError):
        llm_factory.get_llm()


def test_unsupported_provider():
    with pytest.raises(LLMConfigurationError, match="Unsupported"):
        llm_factory.get_llm("gemini")


def test_uses_env_model_and_key_without_leaking(monkeypatch):
    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_groq", types.SimpleNamespace(ChatGroq=FakeChatGroq))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.setenv("GROQ_MODEL", "my/model")
    llm_factory.get_llm()
    assert captured["model"] == "my/model" and captured["api_key"] == "test-key-123"
    assert captured["temperature"] == 0

    monkeypatch.delenv("GROQ_MODEL")
    llm_factory.get_llm()
    assert captured["model"] == llm_factory.DEFAULT_GROQ_MODEL

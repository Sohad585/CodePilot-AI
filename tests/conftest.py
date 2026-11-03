"""Shared test helpers. No test needs a real API key or network access."""

import types

import pytest


class FakeLLM:
    """Stands in for ChatGroq: returns canned text, or raises an exception."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(content=self.reply)


@pytest.fixture
def fake_llm(monkeypatch):
    """Return ``install(reply=..., error=...)`` which patches every agent's get_llm."""

    def install(reply=None, error=None):
        llm = FakeLLM(reply, error)
        for module in (
            "ai_analyzer", "debugger", "refactorer", "supervisor", "tester",
        ):
            monkeypatch.setattr(
                f"codepilot_ai.agents.{module}.get_llm", lambda llm=llm: llm
            )
        return llm

    return install


class StatusError(Exception):
    """Mimics groq.APIStatusError: an exception carrying an HTTP status."""

    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


def named_error(name, message="boom", status_code=None):
    """Create an exception whose class has the given name (e.g. APITimeoutError)."""
    cls = type(name, (Exception,), {})
    error = cls(message)
    if status_code is not None:
        error.status_code = status_code
    return error

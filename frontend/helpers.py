"""Pure helper functions for the CodePilot AI frontend.

Nothing in here imports Streamlit, and nothing ever executes user code:
the uploaded file is only decoded, measured and displayed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Limits mirror the backend (src/codepilot_ai/backend/main.py).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # MAX_UPLOAD_SIZE for POST /upload
MAX_RUN_CHARS = 100_000  # WorkflowRequest.source_code max_length
MAX_ERROR_CONTEXT_CHARS = 20_000  # WorkflowRequest.error_context max_length
MAX_TEST_CODE_CHARS = 50_000  # WorkflowRequest.test_code max_length
# The backend allows 5,000 characters for `task`; the rest is reserved for the
# sentence that tells the Supervisor which action the user picked.
MAX_TASK_CHARS = 4_000

# Only used to keep the page responsive; the full file is always what is sent.
MAX_DISPLAY_LINES = 3_000

ACTION_LABELS = {
    "analyze": "Analyze",
    "debug": "Debug",
    "refactor": "Refactor",
}

# POST /run has no "mode" field: the Supervisor chooses its agents from the
# task text. These sentences carry the intent of the Debug / Refactor buttons.
ACTION_PROMPTS = {
    "debug": "Debug this Python code: find the bugs and fix them.",
    "refactor": (
        "Refactor this Python code to improve readability and "
        "maintainability while preserving its behavior."
    ),
}


def format_size(num_bytes: int) -> str:
    """Return a short human readable size such as '12.3 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} MB"  # pragma: no cover - loop always returns


def validate_python_upload(filename: str, data: bytes) -> tuple[str | None, str | None]:
    """Validate an uploaded file the same way the backend does.

    Returns ``(source_code, None)`` when the file is acceptable, otherwise
    ``(None, error_message)``.
    """
    name = Path(filename or "").name

    if Path(name).suffix.lower() != ".py":
        return None, "Only Python (.py) files are supported."

    if len(data) == 0:
        return None, "The file is empty."

    if len(data) > MAX_UPLOAD_BYTES:
        return None, (
            f"The file is {format_size(len(data))}; "
            f"the maximum size is {format_size(MAX_UPLOAD_BYTES)}."
        )

    try:
        code = data.decode("utf-8")
    except UnicodeDecodeError:
        return None, "The file must be valid UTF-8 text."

    if not code.strip():
        return None, "The file contains only whitespace."

    if "\x00" in code:
        return None, "The file is not a valid Python source file."

    return code, None


def compose_task(action: str, user_task: str) -> str:
    """Build the ``task`` string sent to POST /run for Debug or Refactor."""
    prompt = ACTION_PROMPTS[action]
    user_task = (user_task or "").strip()

    if not user_task:
        return prompt

    return f"{prompt}\n\nAdditional instructions from the user:\n{user_task}"


def module_filename(filename: str | None) -> str:
    """Return the file name the sandbox should use for the submitted code.

    The user's pytest code imports the code under test by module name, so the
    uploaded file's own name is kept when it is a valid module name (for
    example 'inventory.py'). Anything else falls back to 'solution.py'.
    """
    name = Path(filename or "").name

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}\.py", name):
        return name

    return "solution.py"


def safe_download_name(filename: str | None, suffix: str = "_modified") -> str:
    """Return a safe file name such as 'app_modified.py'."""
    stem = Path(filename or "").stem or "source"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "source"
    return f"{stem}{suffix}.py"


def parse_test_report(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the optional pytest report text box.

    Returns ``(report, None)``; ``(None, None)`` when the box is empty and
    ``(None, error_message)`` when the text is not a non-empty JSON object.
    """
    text = (text or "").strip()

    if not text:
        return None, None

    try:
        report = json.loads(text)
    except json.JSONDecodeError as error:
        return (
            None,
            f"The pytest report is not valid JSON ({error.msg}, line {error.lineno}).",
        )

    if not isinstance(report, dict) or not report:
        return None, "The pytest report must be a non-empty JSON object."

    return report, None

"""CodePilot AI — Streamlit frontend.

Run from the project root (with the FastAPI backend already running):

    uv run streamlit run frontend/app.py

This app never executes uploaded code. It only reads the file, shows it, and
sends it to the backend, where all execution stays inside Docker.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from api_client import (
    DEFAULT_API_URL,
    ApiError,
    analyze_upload,
    check_health,
    run_workflow,
    upload_python_file,
)
from helpers import (
    ACTION_LABELS,
    MAX_DISPLAY_LINES,
    MAX_ERROR_CONTEXT_CHARS,
    MAX_RUN_CHARS,
    MAX_TASK_CHARS,
    MAX_TEST_CODE_CHARS,
    MAX_UPLOAD_BYTES,
    compose_task,
    format_size,
    module_filename,
    parse_test_report,
    validate_python_upload,
)
from ui_results import (
    render_analysis,
    render_api_error,
    render_workflow_result,
)
from ui_theme import chip, chips, hero, inject_theme, section

st.set_page_config(
    page_title="CodePilot AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

FILE_STATE_DEFAULTS = {
    "file_key": None,  # name + hash of the file currently loaded
    "file_error": None,  # validation message for a rejected file
    "filename": None,
    "original_code": None,  # the uploaded source — never modified
    "raw_bytes": None,
    "upload": None,  # POST /upload response
    "upload_error": None,  # ApiError from the last upload attempt
}

RESULT_STATE_DEFAULTS = {
    "result": None,  # last successful analysis / workflow result
    "error": None,  # ApiError from the last request
    "last_request": None,  # lets the "Try again" button repeat a run
    "queued_request": None,
    "run_counter": 0,
}


def init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("backend_online", None)

    for defaults in (FILE_STATE_DEFAULTS, RESULT_STATE_DEFAULTS):
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)


def clear_results() -> None:
    for key, value in RESULT_STATE_DEFAULTS.items():
        st.session_state[key] = value


def reset_file_state() -> None:
    for key, value in FILE_STATE_DEFAULTS.items():
        st.session_state[key] = value
    clear_results()


def refresh_health() -> None:
    st.session_state.backend_online = check_health(st.session_state.api_url)


# --------------------------------------------------------------------------- #
# Upload handling
# --------------------------------------------------------------------------- #


def ensure_upload() -> str:
    """Return the backend upload ID, uploading the file first if needed."""
    if st.session_state.upload:
        return st.session_state.upload["upload_id"]

    response = upload_python_file(
        st.session_state.api_url,
        st.session_state.filename,
        st.session_state.raw_bytes,
    )
    st.session_state.upload = response
    st.session_state.upload_error = None
    return response["upload_id"]


def load_uploaded_file(uploaded) -> None:
    """Validate a newly selected file, keep the original, upload it once."""
    if uploaded is None:
        if st.session_state.file_key is not None:
            reset_file_state()
        return

    data = uploaded.getvalue()
    key = f"{uploaded.name}:{len(data)}:{hashlib.sha256(data).hexdigest()}"

    if key == st.session_state.file_key:
        return  # same file as the previous run

    reset_file_state()
    st.session_state.file_key = key

    code, error = validate_python_upload(uploaded.name, data)

    if error:
        st.session_state.file_error = error
        return

    st.session_state.original_code = code
    st.session_state.filename = Path(uploaded.name).name
    st.session_state.raw_bytes = data

    with st.spinner("Uploading to the backend…"):
        try:
            ensure_upload()
        except ApiError as error:
            st.session_state.upload_error = error
            if error.kind == "connection":
                st.session_state.backend_online = False


def render_file_panel() -> None:
    error = st.session_state.file_error

    if error:
        st.error(f"**File rejected.** {error}")
        return

    code = st.session_state.original_code

    if code is None:
        st.caption("No file loaded yet.")
        return

    line_count = len(code.splitlines())
    upload = st.session_state.upload

    chips(
        chip(st.session_state.filename, "info"),
        chip(format_size(len(st.session_state.raw_bytes))),
        chip(f"{line_count:,} lines"),
        chip("Uploaded to backend", "ok")
        if upload
        else chip("Not uploaded yet", "warn"),
    )

    if st.session_state.upload_error:
        render_api_error(st.session_state.upload_error)
        if st.button("Retry upload", key="retry_upload"):
            try:
                ensure_upload()
            except ApiError as exc:
                st.session_state.upload_error = exc
            st.rerun()

    st.markdown("**Original code** · read-only")
    lines = code.splitlines()
    st.code(
        "\n".join(lines[:MAX_DISPLAY_LINES]),
        language="python",
        line_numbers=True,
        # Fit short files; long files scroll inside a fixed-height panel.
        height=max(120, min(430, 26 * len(lines) + 32)),
    )

    if len(lines) > MAX_DISPLAY_LINES:
        st.caption(
            f"Showing the first {MAX_DISPLAY_LINES:,} of {len(lines):,} lines. "
            "The complete file is used for every action."
        )


# --------------------------------------------------------------------------- #
# Running actions
# --------------------------------------------------------------------------- #


def analyze_with_one_retry() -> dict:
    """Analyze the upload; if the backend lost the file, send it again once."""
    upload_id = ensure_upload()

    try:
        return analyze_upload(st.session_state.api_url, upload_id)
    except ApiError as error:
        if error.status_code != 404:
            raise

    st.session_state.upload = None
    return analyze_upload(st.session_state.api_url, ensure_upload())


def run_request(request: dict) -> None:
    action = request["action"]
    label = ACTION_LABELS[action]

    st.session_state.last_request = request
    st.session_state.result = None
    st.session_state.error = None

    original = st.session_state.original_code
    filename = st.session_state.filename

    test_report = None
    if action != "analyze":
        test_report, report_error = parse_test_report(request["test_report_text"])
        if report_error:
            st.session_state.error = ApiError(
                "validation", "Invalid pytest report", report_error
            )
            return

    with st.status(f"Running {label}…", expanded=True) as status:
        try:
            if action == "analyze":
                st.write("Analyzing with static checks and the AI reviewer…")
                analysis = analyze_with_one_retry()
                result = {"kind": "analysis", "action": label, "analysis": analysis}
                final_state = "complete"
            else:
                st.write(
                    "Running the supervisor and its agents. "
                    "This can take a minute or two…"
                )
                state = run_workflow(
                    st.session_state.api_url,
                    task=compose_task(action, request["task"]),
                    source_code=original,
                    error_context=request["error_context"],
                    test_report=test_report,
                    test_code=request["test_code"],
                    source_filename=module_filename(filename),
                )
                result = {
                    "kind": "workflow",
                    "action": label,
                    "state": state,
                    "original": original,
                    "filename": filename,
                }
                final_state = "error" if state.get("status") == "failed" else "complete"

            st.session_state.run_counter += 1
            result["run_id"] = str(st.session_state.run_counter)
            st.session_state.result = result
            status.update(label=f"{label} finished", state=final_state, expanded=False)

        except ApiError as error:
            st.session_state.error = error
            if error.kind == "connection":
                st.session_state.backend_online = False
            status.update(label=f"{label} failed", state="error", expanded=False)


def queue_retry() -> None:
    st.session_state.queued_request = st.session_state.last_request


def render_results() -> None:
    error = st.session_state.error
    result = st.session_state.result
    retryable = False

    if error is not None:
        render_api_error(error)
        retryable = error.retryable

    elif result is None:
        st.info(
            "Upload a Python file, then choose **Analyze**, **Debug** or **Refactor**."
        )
        return

    elif result["kind"] == "analysis":
        chips(chip(result["action"], "info"), chip("Completed", "ok"))
        render_analysis(result["analysis"])

    else:
        retryable = render_workflow_result(result)

    if retryable and st.session_state.last_request:
        st.button(
            "Try again",
            on_click=queue_retry,
            icon=":material/refresh:",
            key="try_again",
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

inject_theme()
init_state()

if st.session_state.backend_online is None:
    refresh_health()

with st.sidebar:
    st.markdown("### ✈️ CodePilot AI")
    st.text_input(
        "Backend URL",
        key="api_url",
        on_change=refresh_health,
        help="Where the FastAPI backend is running. Can also be set with the CODEPILOT_API_URL environment variable.",
    )
    chips(
        chip("Backend online", "ok")
        if st.session_state.backend_online
        else chip("Backend offline", "bad")
    )
    if st.button("Check connection", width="stretch"):
        refresh_health()
        st.rerun()

    st.divider()
    st.markdown("**Limits**")
    st.caption(
        f"- Python files only (`.py`), up to {format_size(MAX_UPLOAD_BYTES)}\n"
        f"- Debug and Refactor accept up to {MAX_RUN_CHARS:,} characters of code"
    )
    st.markdown("**Safety**")
    st.caption(
        "- This app never executes your code.\n"
        "- The backend only runs code if you supply pytest tests, and then only inside its Docker sandbox (no network, read-only, resource-limited).\n"
        "- Without tests, modified code is only syntax-checked and is marked as not verified.\n"
        "- Your original file is never changed.\n"
        "- API keys live only in the backend's `.env` and are never shown here."
    )

hero()

left, right = st.columns([1.25, 1], gap="large")

with left:
    section(1, "Upload", "Python files only · max 5 MB")
    uploaded_file = st.file_uploader(
        "Upload a Python file",
        type=["py"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="uploader",
    )
    load_uploaded_file(uploaded_file)
    render_file_panel()

has_code = st.session_state.original_code is not None
too_long = has_code and len(st.session_state.original_code) > MAX_RUN_CHARS

with right:
    section(2, "Describe the task")
    task = st.text_area(
        "Task description",
        key="task",
        max_chars=MAX_TASK_CHARS,
        height=140,
        placeholder="e.g. The function crashes when the list is empty — fix it without changing its signature.",
        label_visibility="collapsed",
    )

    with st.expander("Advanced options (optional)"):
        error_context = st.text_area(
            "Error message or traceback",
            key="error_context",
            max_chars=MAX_ERROR_CONTEXT_CHARS,
            height=110,
            help="Passed to the Debugger as extra context.",
        )
        test_report_text = st.text_area(
            "Pytest report (JSON)",
            key="test_report_text",
            height=110,
            help=(
                "Lets the Tester agent evaluate results. This app never runs "
                "tests itself; it only forwards the report to the backend."
            ),
        )
        test_code = st.text_area(
            "Pytest tests to verify the result (Python)",
            key="test_code",
            max_chars=MAX_TEST_CODE_CHARS,
            height=140,
            help=(
                "Optional. For Debug and Refactor, the backend runs these "
                "tests against the final code inside its Docker sandbox and "
                "reports whether they pass. Import your code by its file name, "
                "for example `from inventory import add_item` for inventory.py "
                "(`solution` if the file name is not a valid module name)."
            ),
        )

    section(3, "Run")
    col_analyze, col_debug, col_refactor = st.columns(3)
    clicked = {
        "analyze": col_analyze.button(
            "Analyze", icon=":material/search:", width="stretch", disabled=not has_code
        ),
        "debug": col_debug.button(
            "Debug",
            icon=":material/bug_report:",
            type="primary",
            width="stretch",
            disabled=not has_code or too_long,
        ),
        "refactor": col_refactor.button(
            "Refactor",
            icon=":material/auto_fix_high:",
            width="stretch",
            disabled=not has_code or too_long,
        ),
    }

    if too_long:
        st.warning(
            f"This file has more than {MAX_RUN_CHARS:,} characters, which is more than "
            "Debug and Refactor accept. Analyze still works."
        )
    else:
        st.caption(
            "**Analyze** reviews the file without changing it. **Debug** and "
            "**Refactor** run the agent workflow on your task and return modified code."
        )

request = st.session_state.queued_request
st.session_state.queued_request = None

if request is None:
    for action, was_clicked in clicked.items():
        if was_clicked:
            request = {
                "action": action,
                "task": task,
                "error_context": error_context,
                "test_report_text": test_report_text,
                "test_code": test_code,
            }
            break

if request is not None and has_code:
    run_request(request)

st.divider()
render_results()

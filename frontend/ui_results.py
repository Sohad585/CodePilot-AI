"""Rendering of backend results: analysis, workflow status, tests and code."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st
from api_client import ApiError, describe_failure, error_from_workflow_state
from diff_view import diff_stats, side_by_side_html, unified_diff
from helpers import safe_download_name
from ui_theme import chip, chips

STATUS_TONE = {"completed": "ok", "failed": "bad", "needs_clarification": "warn"}
STATUS_LABEL = {
    "completed": "Completed",
    "failed": "Failed",
    "needs_clarification": "Needs clarification",
}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def render_api_error(error: ApiError) -> None:
    """Show a friendly error with the (redacted) technical details folded away."""
    st.error(f"**{error.title}**\n\n{error.message}")

    if error.technical:
        with st.expander("Technical details"):
            st.code(error.technical, language="text", wrap_lines=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def render_analysis(analysis: dict[str, Any]) -> None:
    """Render the object returned by the analyzer (static + AI analysis)."""
    static = analysis.get("static_analysis") or {}
    issues = [i for i in (static.get("issues") or []) if isinstance(i, dict)]
    ai = analysis.get("ai_analysis") or {}
    valid = analysis.get("valid_syntax")

    if ai.get("success"):
        ai_label = "Complete"
    elif ai.get("error"):
        ai_label = "Failed"
    else:
        ai_label = "Skipped"

    col_syntax, col_issues, col_ai = st.columns(3)
    col_syntax.metric("Syntax", "Valid" if valid else "Invalid")
    col_issues.metric("Static issues", static.get("total_issues", len(issues)))
    col_ai.metric("AI review", ai_label)

    st.markdown("#### Static analysis")
    if issues:
        st.dataframe(
            [
                {
                    "Severity": str(issue.get("severity", "")).title(),
                    "Line": issue.get("line"),
                    "Type": issue.get("type", ""),
                    "Message": issue.get("message", ""),
                }
                for issue in issues
            ],
            hide_index=True,
            width="stretch",
        )
    elif valid:
        st.success(
            "The built-in static checks found no issues "
            "(syntax, bare except, eval/exec, wildcard imports)."
        )

    st.markdown("#### AI review")
    if ai.get("success"):
        st.markdown(ai.get("analysis") or "_The model returned an empty analysis._")
    elif ai.get("error"):
        render_api_error(
            describe_failure(
                str(ai["error"]),
                error_type=ai.get("error_type"),
                retryable=ai.get("retryable"),
            )
        )
    else:
        st.info(ai.get("message") or "The AI review was skipped.")


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #


def _pipeline_html(state: dict[str, Any]) -> str:
    status = state.get("status")
    selected = state.get("selected_agents") or []
    results = state.get("agent_results") or {}
    failed_index = state.get("current_agent_index", 0)

    steps: list[tuple[str, str]] = [
        ("Supervisor", "done" if state.get("supervision") else "failed")
    ]

    for index, name in enumerate(selected):
        if name in results:
            steps.append((name, "done"))
        elif status == "failed" and index == failed_index:
            steps.append((name, "failed"))
        else:
            steps.append((name, "pending"))

    if state.get("test_code"):
        verification = state.get("verification")

        if verification is None:
            steps.append(("Sandbox tests", "pending"))
        elif verification.get("ran") and verification.get("tests_passed"):
            steps.append(("Sandbox tests", "done"))
        else:
            steps.append(("Sandbox tests", "failed"))

    icons = {"done": "✓", "failed": "✗", "pending": "○"}
    parts = [
        f'<span class="cp-step {css}"><span>{icons[css]}</span>{html.escape(name)}</span>'
        for name, css in steps
    ]

    return (
        '<div class="cp-steps">'
        + '<span class="cp-arrow">→</span>'.join(parts)
        + "</div>"
    )


def _render_overview(state: dict[str, Any]) -> None:
    supervision = state.get("supervision") or {}
    results = state.get("agent_results") or {}

    plan = supervision.get("execution_plan") or []
    if plan:
        st.markdown("**Execution plan**")
        for number, step in enumerate(plan, start=1):
            st.markdown(f"{number}. {step}")

    if supervision.get("reason"):
        st.caption(f"Why these agents: {supervision['reason']}")

    for name in state.get("selected_agents") or []:
        result = results.get(name)

        if name not in ("Debugger", "Refactorer") or not isinstance(result, dict):
            continue

        with st.expander(f"{name} — what changed", expanded=True):
            if result.get("explanation"):
                st.markdown(result["explanation"])

            changes = result.get("changes_made") or []
            if changes:
                st.markdown("\n".join(f"- {change}" for change in changes))

    if "Analyzer" in results:
        st.caption("Analyzer results are in the Analysis tab.")
    if "Tester" in results:
        st.caption("Tester results are in the Tests tab.")


def render_test_results(agent_result: dict[str, Any]) -> None:
    """Render the Tester agent's evaluation of a pytest report."""
    evaluation = agent_result.get("evaluation") or {}

    if agent_result.get("corrections"):
        st.caption(
            "The pass/fail verdict and counts below were taken from the pytest "
            "report itself, because the AI's reading differed from it."
        )

    if evaluation.get("tests_passed"):
        st.success("All tests passed.")
    else:
        st.error(
            f"Tests did not pass (result: {evaluation.get('overall_result', 'unknown')})."
        )

    cols = st.columns(4)
    cols[0].metric("Total", evaluation.get("total_tests", 0))
    cols[1].metric("Passed", evaluation.get("passed_tests", 0))
    cols[2].metric("Failed", evaluation.get("failed_tests", 0))
    cols[3].metric("Errors", evaluation.get("errors", 0))

    details = [
        d for d in (evaluation.get("failed_test_details") or []) if isinstance(d, dict)
    ]
    if details:
        st.markdown("**Failed tests**")
        st.dataframe(
            [
                {"Test": d.get("test_name", ""), "Reason": d.get("reason", "")}
                for d in details
            ],
            hide_index=True,
            width="stretch",
        )

    if evaluation.get("recommendation"):
        st.info(f"**Recommendation:** {evaluation['recommendation']}")


def render_verification(verification: dict[str, Any]) -> None:
    """Render the result of running the user's pytest code in the sandbox."""
    if not verification.get("ran"):
        st.error(
            "The sandbox could not run your tests, so the code is **unverified**."
        )
    else:
        summary = verification.get("summary") or {}

        if verification.get("tests_passed"):
            st.success("All of your tests passed in the Docker sandbox.")
        else:
            st.error("Some of your tests did not pass in the Docker sandbox.")

        cols = st.columns(4)
        cols[0].metric("Total", summary.get("total", 0))
        cols[1].metric("Passed", summary.get("passed", 0))
        cols[2].metric("Failed", summary.get("failed", 0))
        cols[3].metric("Errors", summary.get("errors", 0))

        failures = [
            f for f in (verification.get("failures") or []) if isinstance(f, dict)
        ]
        if failures:
            st.markdown("**Failed tests**")
            st.dataframe(
                [
                    {
                        "Test": f.get("test_name", ""),
                        "Stage": f.get("stage") or "",
                        "Message": f.get("message") or f.get("error", ""),
                    }
                    for f in failures
                ],
                hide_index=True,
                width="stretch",
            )

        collection_errors = [
            c
            for c in (verification.get("collection_errors") or [])
            if isinstance(c, dict)
        ]
        if collection_errors:
            st.markdown("**Collection errors**")
            st.dataframe(
                [
                    {"File": c.get("nodeid", ""), "Message": c.get("message", "")}
                    for c in collection_errors
                ],
                hide_index=True,
                width="stretch",
            )

    output = verification.get("output")
    if output:
        with st.expander("Sandbox output"):
            st.code(output, language="text", wrap_lines=True)


def render_code_changes(original: str, modified: str, filename: str, key: str) -> None:
    """Side-by-side view, unified diff, final code and the download button."""
    changed = original != modified
    added, removed = diff_stats(original, modified)

    col_added, col_removed, col_download = st.columns([1, 1, 2])
    col_added.metric("Lines added", added)
    col_removed.metric("Lines removed", removed)

    if changed:
        col_download.download_button(
            "Download modified code",
            data=modified.encode("utf-8"),
            file_name=safe_download_name(filename),
            mime="text/x-python",
            icon=":material/download:",
            type="primary",
            on_click="ignore",
            key=f"download_{key}",
            width="stretch",
        )
    else:
        st.info(
            "The final code is identical to your original file, so there is "
            "nothing to diff or download."
        )

    side_tab, unified_tab, final_tab = st.tabs(
        ["Side by side", "Unified diff", "Final code"]
    )

    with side_tab:
        if changed:
            show_all = st.toggle("Show unchanged lines", key=f"full_{key}")
            st.html(side_by_side_html(original, modified, None if show_all else 3))
        else:
            st.caption("No differences.")

    with unified_tab:
        diff_text = unified_diff(
            original, modified, f"original/{filename}", f"modified/{filename}"
        )
        if diff_text:
            st.code(diff_text, language="diff", height=480)
        else:
            st.caption("No differences.")

    with final_tab:
        st.code(modified, language="python", line_numbers=True, height=480)


def render_workflow_result(result: dict[str, Any]) -> bool:
    """Render a /run result. Returns True when the user can retry it."""
    state: dict[str, Any] = result["state"]
    status = str(state.get("status", "unknown"))
    results = state.get("agent_results") or {}
    supervision = state.get("supervision") or {}
    final_code = state.get("final_code")

    chips(
        chip(result["action"], "info"),
        chip(STATUS_LABEL.get(status, status.title()), STATUS_TONE.get(status, "info")),
        *(
            [chip(f"Task: {supervision['task_type']}")]
            if supervision.get("task_type")
            else []
        ),
        *(
            [chip("Verified by tests", "ok")]
            if (state.get("verification") or {}).get("tests_passed")
            else [chip("Not verified by tests", "warn")]
            if status == "completed" and state.get("code_changed")
            else []
        ),
    )

    st.html(_pipeline_html(state))

    retryable = False
    error = error_from_workflow_state(state)

    if error is not None:
        render_api_error(error)
        retryable = error.retryable

    elif status == "needs_clarification":
        question = (
            supervision.get("clarification_question")
            or "Please describe the task in more detail."
        )
        st.warning(f"**The Supervisor needs more information.**\n\n{question}")
        st.caption("Refine the task description above and run the action again.")

    elif status == "completed" and not final_code:
        st.warning("The workflow completed but did not return any code.")

    for warning in state.get("warnings") or []:
        st.warning(str(warning))

    tab_names = ["Overview"]
    if "Analyzer" in results:
        tab_names.append("Analysis")
    if final_code:
        tab_names.append("Code changes")
    if "Tester" in results:
        tab_names.append("Tests")
    if state.get("verification"):
        tab_names.append("Verification")

    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Overview"]:
        _render_overview(state)

    if "Analysis" in tabs:
        with tabs["Analysis"]:
            render_analysis(results["Analyzer"])

    if "Code changes" in tabs:
        with tabs["Code changes"]:
            render_code_changes(
                result["original"], final_code, result["filename"], key=result["run_id"]
            )

    if "Tests" in tabs:
        with tabs["Tests"]:
            render_test_results(results["Tester"])

    if "Verification" in tabs:
        with tabs["Verification"]:
            render_verification(state["verification"])

    return retryable

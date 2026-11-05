from typing import TypedDict, Optional, Any

from codepilot_ai.agents.supervisor import supervise_task

from codepilot_ai.agents.analyzer import analyze_python_code
from codepilot_ai.agents.debugger import debug_python_code
from codepilot_ai.agents.refactorer import refactor_python_code
from codepilot_ai.agents.tester import evaluate_test_results
from codepilot_ai.llm_utils import classify_exception
from codepilot_ai.tools.docker_sandbox import run_code_tests_in_sandbox

from langgraph.graph import StateGraph, START, END

# At most this many failures / error lines from the sandbox are returned.
MAX_VERIFICATION_FAILURES = 20
MAX_VERIFICATION_OUTPUT_CHARS = 4_000


class CodePilotState(TypedDict):

    # User input
    task: str
    source_code: str
    source_filename: str
    test_code: Optional[str]

    # Execution context
    current_code: str
    error_context: str
    test_report: Optional[dict]

    # Supervisor output
    supervision: Optional[dict]
    selected_agents: list[str]

    # Agent execution
    current_agent_index: int
    agent_results: dict[str, Any]

    # Verification (only present when the user supplied pytest code)
    verification: Optional[dict]

    # Final output
    final_code: Optional[str]
    code_changed: bool
    warnings: list[str]
    error: Optional[str]
    error_type: Optional[str]
    retryable: bool
    status: str



def supervisor_node(state: CodePilotState) -> dict:
    """
    Run the Supervisor and update the workflow state.
    """

    result = supervise_task(
        task=state["task"],
        source_code=state["source_code"]
    )

    # Handle Supervisor failure
    if not result["success"]:
        return {
            "error": result["error"],
            "error_type": result.get("error_type", "supervisor_failed"),
            "retryable": result.get("retryable", False),
            "status": "failed"
        }

    supervision = dict(result["supervision"])

    # Handle tasks that need clarification
    if supervision["needs_clarification"]:
        return {
            "supervision": supervision,
            "selected_agents": [],
            "status": "needs_clarification"
        }

    selected_agents = list(supervision["selected_agents"])
    warnings = list(state.get("warnings") or [])
    tester_skipped = False

    # The Tester evaluates a pytest report. Without one it cannot do anything,
    # so find out now, before any LLM time is spent on the other agents.
    if "Tester" in selected_agents and state.get("test_report") is None:
        selected_agents.remove("Tester")
        tester_skipped = True
        warnings.append(
            "The Tester was skipped because no pytest report was provided. "
            "No test results were evaluated."
        )

    if not selected_agents:

        if tester_skipped:
            supervision["selected_agents"] = []
            supervision["needs_clarification"] = True
            supervision["clarification_question"] = (
                "This task only involves evaluating test results, which "
                "needs a pytest report (JSON). Provide one under Advanced "
                "options, or choose Analyze, Debug or Refactor."
            )

            return {
                "supervision": supervision,
                "selected_agents": [],
                "warnings": warnings,
                "status": "needs_clarification"
            }

        return {
            "supervision": supervision,
            "error": "The Supervisor did not select any agent.",
            "error_type": "supervisor_failed",
            "retryable": True,
            "status": "failed"
        }

    supervision["selected_agents"] = selected_agents

    return {
        "supervision": supervision,
        "selected_agents": selected_agents,
        "current_agent_index": 0,
        "warnings": warnings,
        "status": "ready"
    }


def _agent_failure(agent_name: str, result: dict) -> Optional[dict]:
    """
    Return state updates describing a failed agent result, or None if the
    agent succeeded.
    """

    if agent_name == "Analyzer":

        if "valid_syntax" not in result or "static_analysis" not in result:
            return {
                "error": "Analyzer returned an invalid result.",
                "error_type": "invalid_response",
                "retryable": True
            }

        ai_result = result.get("ai_analysis", {})

        # A skipped AI analysis (syntax error) has no "error" key and is fine.
        if (
            isinstance(ai_result, dict)
            and not ai_result.get("success", False)
            and "error" in ai_result
        ):
            return {
                "error": f"Analyzer: {ai_result['error']}",
                "error_type": ai_result.get("error_type", "analysis_failed"),
                "retryable": ai_result.get("retryable", False)
            }

        return None

    if not result.get("success", False):
        return {
            "error": (
                f"{agent_name} Agent failed: "
                f"{result.get('error', 'no details were reported')}"
            ),
            "error_type": result.get("error_type", "agent_failed"),
            "retryable": result.get("retryable", False)
        }

    return None


def _partial_code_updates(state: CodePilotState, failed_agent: str) -> dict:
    """
    Keep code produced by earlier agents when a later agent fails.

    The earlier agents' syntax-checked changes are returned as an unverified
    candidate with an explicit warning, so a valid fix is not silently lost.
    """

    source_code = state["source_code"]
    current_code = state.get("current_code") or source_code

    if current_code == source_code:
        return {}

    return {
        "final_code": current_code,
        "code_changed": True,
        "warnings": list(state.get("warnings") or []) + [
            f"The {failed_agent} Agent failed. The code shown contains only "
            "the changes made by earlier agents and is unverified."
        ]
    }


def agent_execution_node(state: CodePilotState) -> dict:
    """
    Execute the current agent and update the workflow state.

    A failed agent stops the workflow. Its result (including any partial
    information) is still recorded, and the real reason is propagated in
    ``error``, ``error_type`` and ``retryable``.
    """

    selected_agents = state["selected_agents"]
    current_index = state["current_agent_index"]

    if current_index >= len(selected_agents):
        return {"status": "running"}

    agent_name = selected_agents[current_index]
    current_code = state.get("current_code") or state["source_code"]

    try:

        if agent_name in {"Analyzer", "Debugger", "Refactorer"}:
            if not current_code.strip():
                raise ValueError(
                    f"Source code is required by the {agent_name} Agent."
                )

        # Analyzer Agent
        if agent_name == "Analyzer":
            result = analyze_python_code(current_code)

        # Debugger Agent
        elif agent_name == "Debugger":
            result = debug_python_code(
                current_code,
                state.get("error_context", "")
            )

        # Refactorer Agent
        elif agent_name == "Refactorer":
            result = refactor_python_code(current_code)

        # Tester Agent
        elif agent_name == "Tester":

            test_report = state.get("test_report")

            if test_report is None:
                raise ValueError("A pytest report is required by the Tester Agent.")

            result = evaluate_test_results(test_report)

        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

    except Exception as error:
        info = classify_exception(error, context=f"The {agent_name} Agent")

        return {
            "error": f"{agent_name} Agent failed: {info['message']}",
            "error_type": info["error_type"],
            "retryable": info["retryable"],
            "status": "failed",
            **_partial_code_updates(state, agent_name)
        }

    agent_results = dict(state.get("agent_results", {}))
    agent_results[agent_name] = result

    failure = _agent_failure(agent_name, result)

    if failure is not None:
        return {
            **failure,
            "agent_results": agent_results,
            "status": "failed",
            **_partial_code_updates(state, agent_name)
        }

    updates = {
        "agent_results": agent_results,
        "current_agent_index": current_index + 1,
        "status": "running"
    }

    # Only successful, syntax-checked agent output replaces the working code.
    if agent_name == "Debugger":
        updates["current_code"] = result["fixed_code"]

    elif agent_name == "Refactorer":
        updates["current_code"] = result["refactored_code"]

    return updates


def route_after_agent(state: CodePilotState) -> str:
    """
    Decide the next workflow step after agent execution.
    """

    if state.get("status") == "failed":
        return "end"

    if state["current_agent_index"] >= len(state["selected_agents"]):
        return "finalize"

    return "continue"


def _build_verification(sandbox_result: dict) -> dict:
    """
    Turn a sandbox result into the compact structure returned to the user.
    """

    if not sandbox_result.get("success", False):
        return {
            "ran": False,
            "executed_in": "docker_sandbox",
            "tests_passed": None,
            "error": sandbox_result.get("error", "The sandbox run failed."),
            "error_type": sandbox_result.get("error_type", "sandbox_error"),
            "output": _tail(sandbox_result)
        }

    report = sandbox_result.get("test_report", {})

    return {
        "ran": True,
        "executed_in": "docker_sandbox",
        "tests_passed": bool(sandbox_result.get("tests_passed")),
        "summary": report.get("summary", {}),
        "failures": report.get("failures", [])[:MAX_VERIFICATION_FAILURES],
        "collection_errors": report.get("collection_errors", [])[
            :MAX_VERIFICATION_FAILURES
        ],
        "output_truncated": bool(sandbox_result.get("output_truncated", False)),
        "output": _tail(sandbox_result),
        "error": None
    }


def _tail(sandbox_result: dict) -> str:
    """Combine and cap the sandbox's stdout and stderr."""

    text = "\n".join(
        part
        for part in (
            sandbox_result.get("stdout", ""),
            sandbox_result.get("stderr", "")
        )
        if part
    )

    if len(text) > MAX_VERIFICATION_OUTPUT_CHARS:
        return "... " + text[-MAX_VERIFICATION_OUTPUT_CHARS:]

    return text


def finalize_node(state: CodePilotState) -> dict:
    """
    Choose the final code and decide the final status.

    The original code is never modified: ``source_code`` stays as submitted
    and ``final_code`` is the working copy. Success is only claimed on
    evidence. When pytest code was supplied, the final code is run against it
    in the Docker sandbox and the status depends on the outcome. Without
    tests, the run completes but carries an explicit warning that the code
    was only syntax-checked.
    """

    source_code = state["source_code"]
    final_code = state.get("current_code") or source_code
    code_changed = final_code != source_code
    warnings = list(state.get("warnings") or [])
    agent_results = state.get("agent_results", {})

    updates = {
        "final_code": final_code,
        "code_changed": code_changed,
        "status": "completed"
    }

    test_code = state.get("test_code")

    if test_code:

        sandbox_result = run_code_tests_in_sandbox(
            source_code=final_code,
            test_code=test_code,
            source_filename=state.get("source_filename") or "solution.py"
        )

        verification = _build_verification(sandbox_result)
        updates["verification"] = verification

        if not verification["ran"]:
            updates.update({
                "status": "failed",
                "error": (
                    "The result could not be verified because the Docker "
                    f"sandbox run failed: {verification['error']}"
                ),
                "error_type": verification["error_type"],
                "retryable": False
            })

        elif not verification["tests_passed"]:
            summary = verification["summary"]
            subject = "The modified code" if code_changed else "Your code"
            updates.update({
                "status": "failed",
                "error": (
                    f"{subject} does not pass your tests "
                    f"({summary.get('failed', 0)} failed, "
                    f"{summary.get('errors', 0)} errors, "
                    f"{summary.get('passed', 0)} passed of "
                    f"{summary.get('total', 0)}). "
                    "The returned code is an unverified candidate; "
                    "your original file is unchanged."
                ),
                "error_type": "verification_failed",
                "retryable": False
            })

    elif code_changed:
        warnings.append(
            "The modified code was only checked for valid Python syntax. It "
            "was not run or tested. Review the diff and run your own tests "
            "before using it. (Provide pytest code under Advanced options to "
            "have it verified in the Docker sandbox.)"
        )

    elif {"Debugger", "Refactorer"} & set(agent_results):
        warnings.append("The agents returned your code unchanged.")

    if code_changed and "Tester" in agent_results:
        warnings.append(
            "The Tester evaluated the pytest report you provided, which "
            "describes the code you submitted, not the modified code."
        )

    updates["warnings"] = warnings

    return updates



def build_workflow():
    """
    Build and compile the CodePilot LangGraph workflow.

    Flow: supervisor -> agent_execution (repeated once per selected agent)
    -> finalize. A failed supervisor or agent ends the run immediately.
    """

    # Create the StateGraph
    workflow = StateGraph(CodePilotState)

    # Register workflow nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("agent_execution", agent_execution_node)
    workflow.add_node("finalize", finalize_node)

    # Define the starting point
    workflow.add_edge(START, "supervisor")

    # Connect Supervisor to Agent Execution
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: (
            "end"
            if state.get("status") in {"failed", "needs_clarification"}
            else "continue"
        ),
        {
            "continue": "agent_execution",
            "end": END
        }
    )

    # Connect Agent Execution using conditional routing
    workflow.add_conditional_edges(
        "agent_execution",
        route_after_agent,
        {
            "continue": "agent_execution",
            "finalize": "finalize",
            "end": END
        }
    )

    workflow.add_edge("finalize", END)

    # Compile the graph
    return workflow.compile()

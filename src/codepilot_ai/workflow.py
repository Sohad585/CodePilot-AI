from typing import TypedDict, Optional, Any

from codepilot_ai.agents.supervisor import supervise_task

from codepilot_ai.agents.analyzer import analyze_python_code
from codepilot_ai.agents.debugger import debug_python_code
from codepilot_ai.agents.refactorer import refactor_python_code
from codepilot_ai.agents.tester import evaluate_test_results

from langgraph.graph import StateGraph, START, END

class CodePilotState(TypedDict):

    # User input
    task: str
    source_code: str

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

    # Final output
    final_code: Optional[str]
    error: Optional[str]
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
            "status": "failed"
        }

    supervision = result["supervision"]

    # Handle tasks that need clarification
    if supervision["needs_clarification"]:
        return {
            "supervision": supervision,
            "selected_agents": [],
            "status": "needs_clarification"
        }

    selected_agents = supervision["selected_agents"]

    # Ensure that at least one agent was selected
    if not selected_agents:
        return {
            "supervision": supervision,
            "error": "The Supervisor did not select any agent.",
            "status": "failed"
        }

    return {
        "supervision": supervision,
        "selected_agents": selected_agents,
        "current_agent_index": 0,
        "status": "ready"
    }


def agent_execution_node(state: CodePilotState) -> dict:
    """
    Execute the current agent and update the workflow state.
    """

    selected_agents = state["selected_agents"]
    current_index = state["current_agent_index"]

    # Check whether all agents have been executed
    if current_index >= len(selected_agents):
        return {
            "status": "completed",
            "final_code": state.get("current_code", state["source_code"])
        }

    agent_name = selected_agents[current_index]

    current_code = state.get("current_code") or state["source_code"]

    try:

        # Analyzer Agent
        if agent_name == "Analyzer":

            if not current_code.strip():
                raise ValueError("Source code is required by the Analyzer Agent.")

            result = analyze_python_code(current_code)

            if (
                "valid_syntax" not in result
                or "static_analysis" not in result
            ):
                raise ValueError("Analyzer returned an invalid result.")

            ai_result = result.get("ai_analysis", {})

            if (
                isinstance(ai_result, dict)
                and not ai_result.get("success", False)
                and "error" in ai_result
            ):
                raise RuntimeError(
                    "Analyzer completed static analysis, but AI analysis failed."
                )

        # Debugger Agent
        elif agent_name == "Debugger":

            if not current_code.strip():
                raise ValueError("Source code is required by the Debugger Agent.")

            result = debug_python_code(
                current_code,
                state.get("error_context", "")
            )

            if result.get("success"):
                current_code = result["fixed_code"]

        # Refactorer Agent
        elif agent_name == "Refactorer":

            if not current_code.strip():
                raise ValueError("Source code is required by the Refactorer Agent.")

            result = refactor_python_code(current_code)

            if result.get("success"):
                current_code = result["refactored_code"]

        # Tester Agent
        elif agent_name == "Tester":

            test_report = state.get("test_report")

            if test_report is None:
                raise ValueError("A pytest report is required by the Tester Agent.")

            result = evaluate_test_results(test_report)

        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

        # Check the agent result
        if agent_name != "Analyzer" and not result.get("success", False):
            raise RuntimeError(f"{agent_name} Agent failed.")

        agent_results = dict(state.get("agent_results", {}))
        agent_results[agent_name] = result

        next_index = current_index + 1
        is_last_agent = next_index >= len(selected_agents)

        updates = {
            "current_code": current_code,
            "agent_results": agent_results,
            "current_agent_index": next_index,
            "status": "completed" if is_last_agent else "running"
        }

        if is_last_agent:
            updates["final_code"] = current_code or None

        return updates

    except Exception as error:

        return {
            "error": str(error),
            "status": "failed"
        }


def route_after_agent(state: CodePilotState) -> str:
    """
    Decide the next workflow step after agent execution.
    """

    if state.get("status") == "failed":
        return "end"

    if state["current_agent_index"] >= len(state["selected_agents"]):
        return "end"

    return "continue"



def build_workflow():
    """
    Build and compile the CodePilot LangGraph workflow.
    """

    # Create the StateGraph
    workflow = StateGraph(CodePilotState)

    # Register workflow nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("agent_execution", agent_execution_node)

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
            "end": END
        }
    )

    # Compile the graph
    return workflow.compile()

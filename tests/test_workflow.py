"""Workflow tests. Agents and the sandbox are mocked; no network or Docker."""

import pytest

pytest.importorskip("langgraph")

from codepilot_ai import workflow as wf  # noqa: E402

CODE = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
REFACTORED = "def add(a, b):\n    result = a + b\n    return result\n"


def initial_state(**overrides):
    state = {
        "task": "fix it", "source_code": CODE, "source_filename": "solution.py", "test_code": None,
        "current_code": CODE, "error_context": "", "test_report": None, "supervision": None,
        "selected_agents": [], "current_agent_index": 0, "agent_results": {}, "verification": None,
        "final_code": None, "code_changed": False, "warnings": [], "error": None, "error_type": None,
        "retryable": False, "status": "pending",
    }
    state.update(overrides)
    return state


def supervision(agents, **overrides):
    data = {"selected_agents": agents, "task_type": "multiple", "execution_plan": [], "reason": "r",
            "needs_clarification": False, "clarification_question": ""}
    data.update(overrides)
    return {"success": True, "supervision": data}


@pytest.fixture
def stub(monkeypatch):
    """Patch supervisor, agents and sandbox; returns a call log."""
    calls = []
    cfg = {
        "supervisor": supervision(["Debugger"]),
        "Analyzer": {"overall_success": True, "valid_syntax": True, "static_analysis": {"issues": []},
                     "ai_analysis": {"success": True, "analysis": "ok"}},
        "Debugger": {"success": True, "fixed_code": FIXED, "changes_made": ["c"], "explanation": "e"},
        "Refactorer": {"success": True, "refactored_code": REFACTORED, "changes_made": [], "explanation": ""},
        "Tester": {"success": True, "evaluation": {"tests_passed": True}},
        "sandbox": {"success": True, "tests_passed": True, "stdout": "", "stderr": "",
                    "test_report": {"summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0},
                                    "failures": [], "collection_errors": []}},
    }

    def make(name):
        def agent(*args, **kwargs):
            calls.append((name, args))
            return cfg[name]
        return agent

    monkeypatch.setattr(wf, "supervise_task", lambda **kw: cfg["supervisor"])
    monkeypatch.setattr(wf, "analyze_python_code", make("Analyzer"))
    monkeypatch.setattr(wf, "debug_python_code", make("Debugger"))
    monkeypatch.setattr(wf, "refactor_python_code", make("Refactorer"))
    monkeypatch.setattr(wf, "evaluate_test_results", make("Tester"))

    def sandbox(**kwargs):
        calls.append(("sandbox", kwargs))
        return cfg["sandbox"]

    monkeypatch.setattr(wf, "run_code_tests_in_sandbox", sandbox)
    return cfg, calls


def run(**overrides):
    return wf.build_workflow().invoke(initial_state(**overrides))


def test_debug_only_without_tests_completes_with_not_verified_warning(stub):
    state = run()
    assert state["status"] == "completed" and state["final_code"] == FIXED
    assert state["code_changed"] is True and state["source_code"] == CODE  # original preserved
    assert any("only checked for valid Python syntax" in w for w in state["warnings"])
    assert state["verification"] is None


def test_agents_run_once_in_order_and_chain_code(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision(["Analyzer", "Debugger", "Refactorer"])
    state = run()
    assert [c[0] for c in calls] == ["Analyzer", "Debugger", "Refactorer"]
    assert calls[2][1][0] == FIXED                      # Refactorer received the *fixed* code
    assert state["final_code"] == REFACTORED and state["source_code"] == CODE


def test_analysis_only_leaves_code_unchanged(stub):
    cfg, _ = stub
    cfg["supervisor"] = supervision(["Analyzer"])
    state = run()
    assert state["status"] == "completed" and state["final_code"] == CODE and state["code_changed"] is False
    assert state["warnings"] == []


def test_supervisor_failure_propagates_real_error(stub):
    cfg, calls = stub
    cfg["supervisor"] = {"success": False, "error": "Groq rejected the API key (HTTP 401).",
                         "error_type": "provider_auth", "retryable": False}
    state = run()
    assert state["status"] == "failed" and state["error_type"] == "provider_auth"
    assert "401" in state["error"] and calls == []


def test_needs_clarification_stops_before_agents(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision([], needs_clarification=True, clarification_question="What?")
    state = run()
    assert state["status"] == "needs_clarification" and calls == []


def test_empty_selection_fails(stub):
    cfg, _ = stub
    cfg["supervisor"] = supervision([])
    state = run()
    assert state["status"] == "failed" and state["error_type"] == "supervisor_failed"


def test_agent_failure_keeps_reason_partial_results_and_earlier_code(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision(["Debugger", "Refactorer"])
    cfg["Refactorer"] = {"success": False, "error": "Groq is temporarily unavailable (HTTP 503).",
                         "error_type": "provider_unavailable", "retryable": True}
    state = run()
    assert state["status"] == "failed" and state["error_type"] == "provider_unavailable"
    assert state["retryable"] is True and "Refactorer Agent failed" in state["error"]
    assert "Debugger" in state["agent_results"] and "Refactorer" in state["agent_results"]
    assert state["final_code"] == FIXED                  # the valid fix is not lost...
    assert any("unverified" in w for w in state["warnings"])  # ...but is labelled
    assert state["verification"] is None


def test_analyzer_ai_failure_is_not_reported_as_success(stub):
    cfg, _ = stub
    cfg["supervisor"] = supervision(["Analyzer", "Debugger"])
    cfg["Analyzer"] = {"overall_success": False, "valid_syntax": True, "static_analysis": {},
                       "ai_analysis": {"success": False, "error": "rate limited",
                                       "error_type": "provider_rate_limited", "retryable": True}}
    state = run()
    assert state["status"] == "failed" and state["error_type"] == "provider_rate_limited"
    assert "Debugger" not in state["agent_results"]


def test_analyzer_with_syntax_error_is_not_a_failure(stub):
    cfg, _ = stub
    cfg["supervisor"] = supervision(["Analyzer"])
    cfg["Analyzer"] = {"overall_success": True, "valid_syntax": False, "static_analysis": {},
                       "ai_analysis": {"success": False, "message": "skipped"}}
    assert run()["status"] == "completed"


def test_unexpected_exception_in_an_agent_is_contained(stub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(wf, "debug_python_code", boom)
    state = run()
    assert state["status"] == "failed" and "kaboom" in state["error"]


def test_tester_without_report_is_skipped_up_front(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision(["Debugger", "Tester"])
    state = run()
    assert [c[0] for c in calls] == ["Debugger"]
    assert any("Tester was skipped" in w for w in state["warnings"])


def test_tester_only_without_report_asks_for_one(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision(["Tester"])
    state = run()
    assert state["status"] == "needs_clarification" and calls == []
    assert "pytest report" in state["supervision"]["clarification_question"]


def test_tester_with_report_runs_and_warns_when_code_changed(stub):
    cfg, calls = stub
    cfg["supervisor"] = supervision(["Debugger", "Tester"])
    state = run(test_report={"summary": {"total": 1}})
    assert [c[0] for c in calls] == ["Debugger", "Tester"]
    assert any("describes the code you submitted" in w for w in state["warnings"])


def test_verification_pass_marks_verified(stub):
    _, calls = stub
    state = run(test_code="def test_x(): pass", source_filename="calc.py")
    assert state["status"] == "completed" and state["verification"]["tests_passed"] is True
    assert not any("only checked" in w for w in state["warnings"])
    sandbox_call = [c for c in calls if c[0] == "sandbox"][0][1]
    assert sandbox_call["source_code"] == FIXED and sandbox_call["source_filename"] == "calc.py"


def test_verification_failure_is_a_failed_run_with_candidate(stub):
    cfg, _ = stub
    cfg["sandbox"] = {"success": True, "tests_passed": False, "stdout": "boom", "stderr": "",
                      "test_report": {"summary": {"total": 2, "passed": 1, "failed": 1, "errors": 0},
                                      "failures": [{"test_name": "t"}], "collection_errors": []}}
    state = run(test_code="def test_x(): assert 0")
    assert state["status"] == "failed" and state["error_type"] == "verification_failed"
    assert "1 failed" in state["error"] and state["final_code"] == FIXED
    assert state["verification"]["failures"] == [{"test_name": "t"}]


def test_sandbox_unavailable_means_unverified_failure_not_success(stub):
    cfg, _ = stub
    cfg["sandbox"] = {"success": False, "error_type": "sandbox_unavailable",
                      "error": "Docker Engine is not available."}
    state = run(test_code="def test_x(): pass")
    assert state["status"] == "failed" and state["error_type"] == "sandbox_unavailable"
    assert state["verification"]["ran"] is False


def test_unchanged_code_from_debugger_is_flagged(stub):
    cfg, _ = stub
    cfg["Debugger"] = {"success": True, "fixed_code": CODE, "changes_made": [], "explanation": ""}
    state = run()
    assert state["code_changed"] is False and "returned your code unchanged" in " ".join(state["warnings"])

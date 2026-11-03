import json

from codepilot_ai.tools import code_reader, code_writer
from codepilot_ai.tools.diff_generator import generate_code_diff
from codepilot_ai.tools.pytest_runner import parse_pytest_report
from codepilot_ai.tools.static_analyzer import run_static_analysis


def test_static_analysis_valid_and_issues():
    clean = run_static_analysis("x = 1\n")
    assert clean["valid_syntax"] and clean["static_analysis"]["total_issues"] == 0

    risky = run_static_analysis("from os import *\ntry:\n    eval('1')\nexcept:\n    pass\n")
    messages = {i["message"] for i in risky["static_analysis"]["issues"]}
    assert {"Wildcard import detected.", "Bare except clause detected.",
            "Usage of eval() detected."} <= messages


def test_static_analysis_syntax_error_is_reported_not_raised():
    result = run_static_analysis("def broken(:\n")
    assert result["valid_syntax"] is False
    assert result["static_analysis"]["issues"][0]["line"] == 1


def test_diff_generator():
    same = generate_code_diff("a\n", "a\n")
    assert same["has_changes"] is False and same["summary"] == {"added_lines": 0, "removed_lines": 0}
    changed = generate_code_diff("a\nb\n", "a\nc\nd\n")
    assert changed["has_changes"] and changed["summary"] == {"added_lines": 2, "removed_lines": 1}
    assert generate_code_diff(1, "x")["success"] is False


def test_code_reader_writer_confine_to_project(tmp_path, monkeypatch):
    monkeypatch.setattr(code_reader, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(code_writer, "PROJECT_ROOT", tmp_path.resolve())
    (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / ".env").write_text("GROQ_API_KEY=secret", encoding="utf-8")

    assert code_reader.read_code_file("a.py")["content"] == "x = 1"
    assert code_reader.read_code_file("../etc/passwd")["success"] is False
    assert code_reader.read_code_file(".env")["success"] is False
    assert code_reader.read_code_file("missing.py")["error"] == "File not found."

    assert code_writer.write_code_file("a.py", "y")["success"] is False  # no overwrite
    assert code_writer.write_code_file("a.py", "y", overwrite=True)["success"] is True
    assert (tmp_path / "a.py").read_text() == "y"
    assert code_writer.write_code_file("../escape.py", "y")["success"] is False
    assert code_writer.write_code_file(".env", "y", overwrite=True)["success"] is False
    assert (tmp_path / ".env").read_text() == "GROQ_API_KEY=secret"


def _write_report(tmp_path, report):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_parse_report_passing(tmp_path):
    result = parse_pytest_report(_write_report(tmp_path, {
        "exitcode": 0, "duration": 0.1,
        "summary": {"passed": 2, "total": 2}, "collectors": [], "tests": [],
    }))
    assert result["success"] and result["tests_passed"] is True


def test_parse_report_failure_error_and_collection_error(tmp_path):
    report = {
        "exitcode": 1, "duration": 0.2,
        "summary": {"passed": 0, "failed": 1, "error": 1, "total": 2},
        "collectors": [{"nodeid": "bad.py", "outcome": "failed", "longrepr": "SyntaxError"}],
        "tests": [
            {"nodeid": "t.py::test_a", "outcome": "failed",
             "call": {"outcome": "failed", "crash": {"path": "t.py", "lineno": 3, "message": "assert 1 == 2"},
                      "traceback": [{"path": "t.py", "lineno": 3, "message": "AssertionError"}],
                      "longrepr": "x" * 10_000}},
            # A fixture error: outcome "error", failing stage "setup".
            {"nodeid": "t.py::test_b", "outcome": "error",
             "setup": {"outcome": "failed", "crash": {"message": "fixture boom"}, "traceback": []}},
        ],
    }
    result = parse_pytest_report(_write_report(tmp_path, report))
    assert result["tests_passed"] is False
    assert result["summary"]["failed"] == 1 and result["summary"]["errors"] == 1
    names = {f["test_name"]: f for f in result["failures"]}
    assert set(names) == {"test_a", "test_b"}            # the error test is no longer lost
    assert names["test_b"]["stage"] == "setup"
    assert len(names["test_a"]["details"]) < 5_000       # long tracebacks are capped
    assert result["collection_errors"][0]["nodeid"] == "bad.py"


def test_parse_report_rejects_bad_input(tmp_path):
    assert parse_pytest_report(str(tmp_path / "nope.json"))["success"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert parse_pytest_report(str(bad))["success"] is False
    assert parse_pytest_report(_write_report(tmp_path, {"summary": {"passed": -1}}))["success"] is False


def test_zero_tests_is_not_a_pass(tmp_path):
    result = parse_pytest_report(_write_report(tmp_path, {
        "exitcode": 5, "duration": 0, "summary": {"total": 0}, "collectors": [], "tests": []}))
    assert result["tests_passed"] is False

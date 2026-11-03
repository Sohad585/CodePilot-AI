"""Sandbox logic that needs no Docker: limits, validation, staging, cleanup."""

import io
import os

import pytest

from codepilot_ai.tools import docker_sandbox as sb


# ---- output limiting ------------------------------------------------------------

def read(data, limit=100):
    return sb._read_limited_output(io.BytesIO(data), limit)


def test_small_output_is_kept_whole():
    result = read(b"hello")
    assert result == {"output": "hello", "truncated": False, "total_bytes": 5}


def test_output_between_half_and_full_limit_is_not_duplicated():
    data = bytes(65 + (i % 26) for i in range(80))
    result = read(data)
    assert result["output"] == data.decode() and result["truncated"] is False


def test_large_output_keeps_head_and_tail_with_marker():
    result = read(b"A" * 500 + b"B" * 500)
    assert result["truncated"] is True and result["total_bytes"] == 1000
    assert result["output"].startswith("A" * 50) and result["output"].endswith("B" * 50)
    assert "output truncated" in result["output"]
    assert len(result["output"]) < 200


def test_invalid_utf8_does_not_crash():
    assert read(b"\xff\xfe")["output"]


# ---- validation ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["solution.py", "inventory.py", "tester.py", "_x1.py"])
def test_valid_source_filenames(name):
    assert sb.validate_source_filename(name)["success"] is True


@pytest.mark.parametrize(
    "name", ["../evil.py", "a/b.py", "my-app.py", "1abc.py", "x.txt", "", None,
             "json.py", "queue.py", "pytest.py", "conftest.py", "test_codepilot_user_tests.py"],
)
def test_invalid_source_filenames(name):
    result = sb.validate_source_filename(name)
    assert result["success"] is False and result["error_type"] == "invalid_input"


@pytest.mark.parametrize(
    "kwargs",
    [dict(relative_files="a.py", test_files=["a.py"]),
     dict(relative_files=["a.py"], test_files=[]),
     dict(relative_files=["a.py"], test_files=["a.py"], timeout=0),
     dict(relative_files=["a.py"], test_files=["a.py"], timeout=True),
     dict(relative_files=["../a.py"], test_files=["../a.py"]),
     dict(relative_files=["/etc/passwd"], test_files=["/etc/passwd"]),
     dict(relative_files=["a.py"], test_files=["b.py"]),
     dict(relative_files=["C:\\Windows\\file.py"], test_files=["C:\\Windows\\file.py"]),
     dict(relative_files=["C:/Windows/file.py"], test_files=["C:/Windows/file.py"]),
     dict(relative_files=["\\Windows\\file.py"], test_files=["\\Windows\\file.py"]),
     dict(relative_files=["\\\\server\\share\\f.py"], test_files=["\\\\server\\share\\f.py"]),
     dict(relative_files=["a\\..\\b.py"], test_files=["a\\..\\b.py"]),
     dict(relative_files=["a/../b.py"], test_files=["a/../b.py"]),
     dict(relative_files=["a.py", "/etc/passwd"], test_files=["a.py"]),
     dict(relative_files=["a.py", "C:\\x.py"], test_files=["a.py"]),
     dict(relative_files=["a.py"], test_files=["/etc/passwd"]),
     dict(relative_files=["a.py"], test_files=["C:\\Windows\\file.py"])],
)
def test_run_tests_rejects_bad_input_before_touching_docker(monkeypatch, kwargs):
    monkeypatch.setattr(sb, "check_docker_environment", lambda: pytest.fail("Docker must not be consulted"))
    result = sb.run_tests_in_sandbox("/tmp", **kwargs)
    assert result["success"] is False and result["error_type"] == "invalid_input"


def test_docker_unavailable_is_reported_with_type(monkeypatch):
    monkeypatch.setattr(sb, "check_docker_environment", lambda: {"success": False, "error": "no engine"})
    result = sb.run_tests_in_sandbox(str(sb.PROJECT_ROOT), ["pyproject.toml"], ["pyproject.toml"])
    assert result["error_type"] == "sandbox_unavailable" and result["tests_passed"] is None


def test_docker_cli_missing(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(sb.subprocess, "run", missing)
    assert "Docker CLI was not found" in sb.check_docker_environment()["error"]


# ---- host-side workspace safety ---------------------------------------------------

def test_workspace_must_be_inside_allowed_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(sb, "STAGING_ROOT", tmp_path / "staging")
    (tmp_path / "project").mkdir(); (tmp_path / "staging").mkdir(); (tmp_path / "other").mkdir()
    assert sb.validate_workspace(str(tmp_path / "project"))["success"]
    assert sb.validate_workspace(str(tmp_path / "staging"))["success"]
    assert sb.validate_workspace(str(tmp_path / "other"))["success"] is False
    assert sb.validate_workspace(str(tmp_path / "project" / ".." / "other"))["success"] is False


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "PROJECT_ROOT", tmp_path)
    (tmp_path / "ok.py").write_text("x = 1")
    return tmp_path


def test_create_temp_workspace_copies_only_selected_files(workspace):
    (workspace / "secret.txt").write_text("nope")
    result = sb.create_temp_workspace(str(workspace), ["ok.py"])
    try:
        assert result["success"] and sorted(p.name for p in result["workspace"].iterdir()) == ["ok.py"]
    finally:
        sb.shutil.rmtree(result["workspace"], ignore_errors=True)


@pytest.mark.parametrize("bad", [
    "../x.py", "/etc/passwd", ".env", ".git/config", "missing.py",
    "C:\\Windows\\file.py", "C:/Windows/file.py", "C:file.py", "\\Windows\\file.py",
    "\\\\server\\share\\f.py", "..\\x.py", "a/../ok.py", "a\\..\\ok.py",
])
def test_create_temp_workspace_rejects_unsafe_paths(workspace, bad):
    assert sb.create_temp_workspace(str(workspace), [bad])["success"] is False


def test_create_temp_workspace_rejects_symlinks_and_cleans_up(workspace):
    try:
        os.symlink("/etc/hostname", workspace / "link.py")
    except (OSError, NotImplementedError) as error:
        # Windows without Developer Mode/admin raises WinError 1314.
        pytest.skip(f"Symlink creation is unavailable on this system: {error}")
    before = set(os.listdir(sb.tempfile.gettempdir()))
    assert sb.create_temp_workspace(str(workspace), ["ok.py", "link.py"])["success"] is False
    leftovers = [d for d in set(os.listdir(sb.tempfile.gettempdir())) - before if d.startswith("codepilot-sandbox-")]
    assert leftovers == []   # the half-built staging directory was removed


def test_create_temp_workspace_enforces_size_limit(workspace):
    (workspace / "big.py").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    assert "size limit" in sb.create_temp_workspace(str(workspace), ["big.py"])["error"]


# ---- API entry point: staging and cleanup ---------------------------------------------

def test_run_code_tests_stages_files_then_removes_them(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "STAGING_ROOT", tmp_path / "staging")
    seen = {}

    def fake_run(workspace_path, relative_files, test_files, timeout):
        staged = sb.Path(workspace_path)
        seen.update(files=sorted(p.name for p in staged.iterdir()), args=(relative_files, test_files, timeout),
                    source=(staged / "calc.py").read_text())
        return {"success": True, "tests_passed": True}

    monkeypatch.setattr(sb, "run_tests_in_sandbox", fake_run)
    result = sb.run_code_tests_in_sandbox("x = 1", "def test_a(): pass", "calc.py", timeout=7)
    assert result["tests_passed"] is True
    assert seen["files"] == ["calc.py", sb.USER_TEST_FILENAME] and seen["args"][2] == 7 and seen["source"] == "x = 1"
    assert list((tmp_path / "staging").iterdir()) == []     # staging directory removed


def test_run_code_tests_cleans_up_even_when_the_run_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "STAGING_ROOT", tmp_path / "staging")
    monkeypatch.setattr(sb, "run_tests_in_sandbox", lambda **kw: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        sb.run_code_tests_in_sandbox("x = 1", "def test_a(): pass")
    assert list((tmp_path / "staging").iterdir()) == []


@pytest.mark.parametrize(
    "args", [("", "t"), ("x", "  "), ("x", "t", "../a.py"), ("x", "t", "json.py")]
)
def test_run_code_tests_input_validation(args, monkeypatch):
    monkeypatch.setattr(sb, "run_tests_in_sandbox", lambda **kw: pytest.fail("must not run"))
    result = sb.run_code_tests_in_sandbox(*args)
    assert result["success"] is False and result["error_type"] == "invalid_input"


@pytest.mark.parametrize(
    "path, unsafe",
    [("/etc/passwd", True), ("C:\\Windows\\file.py", True), ("C:/Windows/file.py", True),
     ("\\Windows\\file.py", True), ("\\\\server\\share\\f.py", True), ("C:file.py", True),
     ("../a.py", True), ("..\\a.py", True), ("a/../b.py", True), ("a\\..\\b.py", True),
     ("a.py", False), ("pkg/mod.py", False), ("pkg\\mod.py", False), ("..hidden.py", False)],
)
def test_is_unsafe_relative_path(path, unsafe):
    assert sb._is_unsafe_relative_path(path) is unsafe


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:\\Windows\\file.py", "\\\\server\\share\\f.py", "a\\..\\ok.py"])
def test_create_temp_workspace_rejects_unsafe_paths_with_invalid_path_error_and_no_leftovers(workspace, bad):
    before = set(os.listdir(sb.tempfile.gettempdir()))
    result = sb.create_temp_workspace(str(workspace), ["ok.py", bad])
    assert result["success"] is False and "Invalid file path" in result["error"]
    leftovers = [d for d in set(os.listdir(sb.tempfile.gettempdir())) - before if d.startswith("codepilot-sandbox-")]
    assert leftovers == []


def test_create_temp_workspace_keeps_valid_nested_relative_paths(workspace):
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "mod.py").write_text("y = 2")
    result = sb.create_temp_workspace(str(workspace), ["ok.py", "pkg/mod.py"])
    try:
        assert result["success"] and (result["workspace"] / "pkg" / "mod.py").read_text() == "y = 2"
    finally:
        sb.shutil.rmtree(result["workspace"], ignore_errors=True)

"""Real Docker sandbox tests.

They need a running Docker daemon and the sandbox image
(``docker build -t autodebug-sandbox:2 .``, or set CODEPILOT_SANDBOX_IMAGE).
They are skipped, never faked, when either is missing.
"""

import subprocess
from pathlib import Path

import pytest

from codepilot_ai.tools import docker_sandbox as sb

pytestmark = pytest.mark.docker

FIXTURES = Path(__file__).parent / "sandbox_fixtures"


@pytest.fixture(scope="module", autouse=True)
def require_docker():
    status = sb.check_docker_environment()
    if not status["success"]:
        pytest.skip(f"Docker sandbox not available: {status['error']}")


@pytest.fixture
def run_fixture(monkeypatch):
    # Fixture files live in tests/, which is inside the project root.
    def run(name, timeout=30):
        return sb.run_tests_in_sandbox(str(FIXTURES), [name], [name], timeout=timeout)
    return run


def leftover_containers():
    out = subprocess.run(["docker", "ps", "-a", "--filter", "name=codepilot-", "--format", "{{.Names}}"],
                         capture_output=True, text=True, timeout=20).stdout
    return [line for line in out.splitlines() if line.strip()]


def test_successful_execution(run_fixture):
    result = run_fixture("sample_success.py")
    assert result["success"] and result["tests_passed"] is True
    assert result["test_report"]["summary"]["passed"] == 3


def test_failing_tests_are_reported_not_hidden(run_fixture):
    result = run_fixture("sample_mixed.py")
    assert result["success"] is True and result["tests_passed"] is False
    summary = result["test_report"]["summary"]
    assert (summary["passed"], summary["failed"]) == (2, 1)
    assert result["test_report"]["failures"][0]["test_name"] == "test_failure"


def test_timeout_kills_and_reports(run_fixture):
    result = run_fixture("sample_timeout.py", timeout=3)
    assert result["success"] is False and result["error_type"] == "sandbox_timeout"
    assert result["tests_passed"] is None
    assert leftover_containers() == []


def test_huge_output_is_truncated(run_fixture):
    result = run_fixture("sample_large_output.py")
    assert result["output_truncated"] is True
    assert len(result["stdout"].encode()) <= sb.OUTPUT_LIMIT + 200
    assert result["tests_passed"] is False   # the fixture fails on purpose


def test_containers_are_removed_after_every_outcome(run_fixture):
    for name in ("sample_success.py", "sample_mixed.py"):
        run_fixture(name)
    assert leftover_containers() == []


def test_source_and_tests_run_end_to_end():
    source = "def add(a, b):\n    return a + b\n"
    tests = "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    result = sb.run_code_tests_in_sandbox(source, tests, "calc.py")
    assert result["success"] and result["tests_passed"] is True


def test_collection_errors_are_reported():
    result = sb.run_code_tests_in_sandbox("def broken(:\n", "from calc import x\n\ndef test_a():\n    pass\n", "calc.py")
    assert result["success"] and result["tests_passed"] is False
    assert result["test_report"]["summary"]["errors"] >= 1


@pytest.mark.parametrize(
    "probe, why",
    [
        ("import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)", "network must be disabled"),
        ("open('/etc/codepilot_probe', 'w').write('x')", "root filesystem must be read-only"),
        ("import os; open(os.path.expanduser('~/probe'), 'w')", "no writable home"),
    ],
)
def test_isolation_blocks_dangerous_operations(probe, why):
    tests = f"import pytest\n\ndef test_probe():\n    with pytest.raises(Exception):\n        {probe}\n"
    result = sb.run_code_tests_in_sandbox("x = 1\n", tests, "calc.py")
    assert result["success"] and result["tests_passed"] is True, why


def test_host_files_are_not_visible():
    tests = ("import os\n\ndef test_host():\n"
             "    assert not os.path.exists('/home/claude')\n"
             "    assert not os.path.exists('/mnt/user-data')\n"
             "    assert os.geteuid() == 10001\n")
    result = sb.run_code_tests_in_sandbox("x = 1\n", tests, "calc.py")
    assert result["tests_passed"] is True

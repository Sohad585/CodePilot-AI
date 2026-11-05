
import os
import re
import subprocess
import sys
import tempfile
import shutil
import threading
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath


# Docker sandbox configuration.
# Build the image with:  docker build -t autodebug-sandbox:2 .
SANDBOX_IMAGE = os.getenv("CODEPILOT_SANDBOX_IMAGE", "autodebug-sandbox:2")

MEMORY_LIMIT = "256m"
CPU_LIMIT = "0.5"
PID_LIMIT = 64

EXECUTION_TIMEOUT = 30
OUTPUT_LIMIT = 100_000

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Code submitted through the API is staged here (never mounted into Docker).
STAGING_ROOT = Path(tempfile.gettempdir()) / "codepilot_staging"

TRUNCATION_MARKER = b"\n[... output truncated by CodePilot ...]\n"


def _is_unsafe_relative_path(file_path: str) -> bool:
    """
    Return True if the path is not a plain, safe relative path.

    Checked with both POSIX and Windows rules so the result does not depend
    on the host OS. Rejects absolute, rooted, drive-qualified and UNC paths,
    and parent traversal (..) with either slash style.
    """

    posix = PurePosixPath(file_path)
    windows = PureWindowsPath(file_path)

    if posix.is_absolute() or windows.is_absolute():
        return True

    # Rooted ("\\dir\\f.py"), drive-relative ("C:f.py") and UNC paths.
    if windows.root or windows.drive:
        return True

    return ".." in posix.parts or ".." in windows.parts


def _normalize_relative_path(file_path: str) -> str:
    """Normalize a validated relative path to POSIX form for comparison."""

    return PureWindowsPath(file_path).as_posix()


def _read_limited_output(stream, limit: int = OUTPUT_LIMIT) -> dict:
    """
    Read a subprocess output stream without storing unlimited data.

    Keep:
    - First half of the output.
    - Last half of the output.

    All limits are measured in bytes.
    """

    half_limit = limit // 2

    first_part = bytearray()
    last_part = bytearray()
    total_bytes = 0

    while True:
        chunk = stream.read(8192)

        if not chunk:
            break

        total_bytes += len(chunk)

        # Keep the first half
        remaining_head_space = half_limit - len(first_part)

        if remaining_head_space > 0:
            first_part.extend(chunk[:remaining_head_space])

        # Keep the rolling last half
        last_part.extend(chunk)

        if len(last_part) > half_limit:
            del last_part[:-half_limit]

    # Reconstruct output without duplicating overlapping bytes
    if total_bytes <= half_limit:
        output = bytes(first_part)

    elif total_bytes <= limit:
        overlap = len(first_part) + len(last_part) - total_bytes
        output = bytes(first_part) + bytes(last_part[overlap:])

    else:
        output = bytes(first_part) + TRUNCATION_MARKER + bytes(last_part)

    return {
        "output": output.decode("utf-8", errors="replace"),
        "truncated": total_bytes > limit,
        "total_bytes": total_bytes
    }


def validate_workspace(workspace_path: str) -> dict:
    """
    Validate the workspace directory before using it.

    The workspace must:
    - Exist.
    - Be a directory.
    - Be inside the project directory or the sandbox staging directory.
    """

    try:
        workspace = Path(workspace_path).resolve()

        allowed_roots = (PROJECT_ROOT.resolve(), STAGING_ROOT.resolve())

        if not any(workspace.is_relative_to(root) for root in allowed_roots):
            return {
                "success": False,
                "error": (
                    "Workspace must be inside the project directory "
                    "or the sandbox staging directory."
                )
            }

        if not workspace.exists():
            return {
                "success": False,
                "error": "Workspace directory does not exist."
            }

        if not workspace.is_dir():
            return {
                "success": False,
                "error": "Workspace path must be a directory."
            }

        return {
            "success": True,
            "workspace": workspace
        }

    except (OSError, ValueError, RuntimeError) as error:
        return {
            "success": False,
            "error": f"Workspace validation failed: {error}"
        }


def check_docker_environment() -> dict:
    """
    Check whether Docker is available and the sandbox image exists.
    """

    try:
        # 1. Check Docker Engine
        docker_result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if docker_result.returncode != 0:
            return {
                "success": False,
                "error": "Docker Engine is not available."
            }

        # 2. Check sandbox image
        image_result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                SANDBOX_IMAGE
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if image_result.returncode != 0:
            return {
                "success": False,
                "error": f"Docker image '{SANDBOX_IMAGE}' was not found."
            }

        return {
            "success": True,
            "message": "Docker environment is ready.",
            "image": SANDBOX_IMAGE
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Docker command timed out."
        }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "Docker CLI was not found. Please install Docker Desktop."
        }

    except OSError as error:
        return {
            "success": False,
            "error": f"Docker environment check failed: {error}"
        }



def create_temp_workspace(
    workspace_path: str,
    relative_files: list[str]
) -> dict:
    """
    Create a temporary host workspace and copy only selected files.

    This directory is used as a staging area.
    Files are later copied into the size-limited Docker tmpfs.
    """

    validation = validate_workspace(workspace_path)

    if not validation["success"]:
        return validation

    source_workspace = validation["workspace"]

    blocked_names = {
        ".env",
        ".git",
        ".venv",
        "__pycache__"
    }

    max_file_size = 5 * 1024 * 1024
    max_total_size = 32 * 1024 * 1024

    temp_path = None

    try:
        temp_path = Path(
            tempfile.mkdtemp(prefix="codepilot-sandbox-")
        )

        total_size = 0

        for relative_file in relative_files:

            if (
                not isinstance(relative_file, str)
                or not relative_file.strip()
                or _is_unsafe_relative_path(relative_file)
            ):
                raise ValueError("Invalid file path.")

            relative_path = Path(relative_file)

            if any(part in blocked_names for part in relative_path.parts):
                raise ValueError("Access to restricted files is denied.")

            source_file = source_workspace / relative_path

            current_path = source_workspace

            for part in relative_path.parts:
                current_path = current_path / part

                if current_path.is_symlink():
                    raise ValueError("Symbolic links are not allowed.")

            if not source_file.is_file():
                raise ValueError(
                    f"File not found or invalid: {relative_file}"
                )

            file_size = source_file.stat().st_size

            if file_size > max_file_size:
                raise ValueError(
                    f"File exceeds the size limit: {relative_file}"
                )

            total_size += file_size

            if total_size > max_total_size:
                raise ValueError(
                    "Workspace exceeds the total size limit of 32 MB."
                )

            destination_file = temp_path / relative_path

            destination_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copyfile(source_file, destination_file)

        return {
            "success": True,
            "workspace": temp_path,
            "files_copied": len(relative_files),
            "total_size_bytes": total_size
        }

    except (OSError, ValueError, RuntimeError) as error:

        if temp_path is not None:
            shutil.rmtree(temp_path, ignore_errors=True)

        return {
            "success": False,
            "error": f"Could not prepare temporary workspace: {error}"
        }



def run_tests_in_sandbox(
    workspace_path: str,
    relative_files: list[str],
    test_files: list[str],
    timeout: int = EXECUTION_TIMEOUT
) -> dict:
    """
    Run selected pytest files inside an isolated Docker container.

    The container workspace uses a size-limited tmpfs (64 MB).
    Files are copied into the container using docker cp.
    The original project directory is never mounted.

    stdout and stderr are read concurrently.
    Each stream retains at most OUTPUT_LIMIT bytes.
    """

    # 1. Validate input types and values
    if (
        not isinstance(relative_files, (list, tuple))
        or not all(
            isinstance(file_path, str) and file_path.strip()
            for file_path in relative_files
        )
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "error": "relative_files must be a list of non-empty file paths."
        }

    if (
        not isinstance(test_files, (list, tuple))
        or not test_files
        or not all(
            isinstance(test_file, str) and test_file.strip()
            for test_file in test_files
        )
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "error": "test_files must be a non-empty list of valid file paths."
        }

    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "error": "Timeout must be a positive integer."
        }

    # 2. Validate selected file paths
    normalized_files = set()

    for file_path in relative_files:
        if _is_unsafe_relative_path(file_path):
            return {
                "success": False,
                "error_type": "invalid_input",
                "error": "Invalid file path."
            }

        normalized_files.add(_normalize_relative_path(file_path))

    for test_file in test_files:
        if _is_unsafe_relative_path(test_file):
            return {
                "success": False,
                "error_type": "invalid_input",
                "error": "Invalid test file path."
            }

        if _normalize_relative_path(test_file) not in normalized_files:
            return {
                "success": False,
                "error_type": "invalid_input",
                "error": (
                    "Test file was not included in copied files: "
                    f"{test_file}"
                )
            }

    # 3. Check Docker environment
    environment = check_docker_environment()

    if not environment["success"]:
        return {
            **environment,
            "error_type": "sandbox_unavailable",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None
        }

    # 4. Prepare the host staging workspace
    workspace_result = create_temp_workspace(
        workspace_path,
        relative_files
    )

    if not workspace_result["success"]:
        return {
            **workspace_result,
            "error_type": "invalid_input",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None
        }

    temp_workspace = workspace_result["workspace"]

    # Initialize cleanup variables
    container_id = None
    container_name = f"codepilot-{uuid.uuid4().hex}"

    process = None
    stdout_thread = None
    stderr_thread = None

    stdout_data = {}
    stderr_data = {}

    current_operation = "Docker sandbox execution"

    def collect_output():
        return {
            "stdout": stdout_data.get("output", ""),
            "stderr": stderr_data.get("output", ""),
            "stdout_truncated": stdout_data.get("truncated", False),
            "stderr_truncated": stderr_data.get("truncated", False),
            "output_truncated": (
                stdout_data.get("truncated", False)
                or stderr_data.get("truncated", False)
            ),
            "stdout_bytes": stdout_data.get("total_bytes", 0),
            "stderr_bytes": stderr_data.get("total_bytes", 0)
        }

    try:
        # 5. Create the isolated container
        current_operation = "Docker container creation"

        create_result = subprocess.run(
            [
                "docker", "create",
                "--name", container_name,
                "--network", "none",
                "--read-only",
                "--memory", MEMORY_LIMIT,
                "--memory-swap", MEMORY_LIMIT,
                "--cpus", CPU_LIMIT,
                "--pids-limit", str(PID_LIMIT),
                "--user", "10001:10001",
                "--security-opt", "no-new-privileges:true",
                "--cap-drop", "ALL",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--tmpfs",
                (
                    "/workspace:rw,noexec,nosuid,"
                    "size=64m,uid=10001,gid=10001,mode=0755"
                ),
                "--workdir", "/workspace",
                SANDBOX_IMAGE,
                "python3", "-c",
                "import time; time.sleep(3600)"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if create_result.returncode != 0:
            return {
                "success": False,
                "error_type": "sandbox_error",
                "execution_success": False,
                "report_parsed": False,
                "tests_passed": None,
                "error": (
                    "Could not create Docker container: "
                    f"{create_result.stderr[:OUTPUT_LIMIT]}"
                )
            }

        container_id = create_result.stdout.strip()

        if not container_id:
            return {
                "success": False,
                "error_type": "sandbox_error",
                "execution_success": False,
                "report_parsed": False,
                "tests_passed": None,
                "error": "Docker did not return a container ID."
            }

        # 6. Start the container
        current_operation = "Docker container startup"

        start_result = subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            text=True,
            timeout=10
        )

        if start_result.returncode != 0:
            return {
                "success": False,
                "error_type": "sandbox_error",
                "execution_success": False,
                "report_parsed": False,
                "tests_passed": None,
                "error": (
                    "Could not start Docker container: "
                    f"{start_result.stderr[:OUTPUT_LIMIT]}"
                )
            }


        # 7. Copy selected files into the limited tmpfs workspace
        # Use docker exec stdin instead of docker cp because
        # docker cp cannot write into this read-only container.

        current_operation = "Copying files into Docker container"

        copy_script = (
            "import pathlib, sys\n"
            "root = pathlib.Path('/workspace').resolve()\n"
            "target = (root / sys.argv[1]).resolve()\n"
            "if root not in target.parents:\n"
            "    raise SystemExit('Invalid destination path.')\n"
            "target.parent.mkdir(parents=True, exist_ok=True)\n"
            "target.write_bytes(sys.stdin.buffer.read())\n"
        )

        for relative_file in relative_files:
            source_file = temp_workspace / Path(relative_file)

            copy_result = subprocess.run(
                [
                    "docker", "exec", "-i",
                    container_id,
                    "python3", "-c",
                    copy_script,
                    Path(relative_file).as_posix()
                ],
                input=source_file.read_bytes(),
                capture_output=True,
                timeout=30
            )

            if copy_result.returncode != 0:
                error_message = copy_result.stderr.decode(
                    "utf-8",
                    errors="replace"
                )

                return {
                    "success": False,
                    "error_type": "sandbox_error",
                    "execution_success": False,
                    "report_parsed": False,
                    "tests_passed": None,
                    "error": (
                        "Could not copy files into Docker container: "
                        f"{error_message[:OUTPUT_LIMIT]}"
                    )
                }


        # 8. Run Pytest inside the container
        current_operation = "Pytest execution"

        pytest_command = [
            "docker", "exec",
            "-w", "/workspace",
            container_id,
            "python3", "-m", "pytest",
            "-p", "no:cacheprovider",
            "--json-report",
            "--json-report-file=/workspace/pytest_report.json",
            *test_files
        ]

        process = subprocess.Popen(
            pytest_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        stdout_thread = threading.Thread(
            target=lambda: stdout_data.update(
                _read_limited_output(process.stdout)
            ),
            daemon=True
        )

        stderr_thread = threading.Thread(
            target=lambda: stderr_data.update(
                _read_limited_output(process.stderr)
            ),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # 9. Wait for Pytest with the configured timeout
        try:
            process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:

            try:
                subprocess.run(
                    ["docker", "kill", container_id],
                    capture_output=True,
                    timeout=10
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

            try:
                process.kill()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass

            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            result = {
                "success": False,
                "error_type": "sandbox_timeout",
                "execution_success": False,
                "report_parsed": False,
                "tests_passed": None,
                "return_code": process.returncode,
                "error": (
                    "Pytest execution exceeded its allowed time limit."
                )
            }

            result.update(collect_output())
            return result

        # Wait for both output streams
        stdout_thread.join()
        stderr_thread.join()


        # 10. Retrieve the report from the container through stdout
        current_operation = "Retrieving Pytest report"

        report_path = temp_workspace / "pytest_report.json"

        report_result = subprocess.run(
            [
                "docker", "exec",
                container_id,
                "python3", "-c",
                (
                    "from pathlib import Path; "
                    "import sys; "
                    "p = Path('/workspace/pytest_report.json'); "
                    "sys.stdout.buffer.write(p.read_bytes())"
                )
            ],
            capture_output=True,
            timeout=30
        )

        if report_result.returncode != 0:
            result = {
                "success": False,
                "error_type": "sandbox_error",
                "execution_success": True,
                "report_parsed": False,
                "tests_passed": None,
                "return_code": process.returncode,
                "error": (
                    "Could not retrieve Pytest report: "
                    + report_result.stderr.decode(
                        "utf-8", errors="replace"
                    )[:OUTPUT_LIMIT]
                )
            }

            result.update(collect_output())
            return result

        report_path.write_bytes(report_result.stdout)

        # 11. Parse the report on the host
        current_operation = "Pytest report parsing"

        from codepilot_ai.tools.pytest_runner import parse_pytest_report

        parsed_report = parse_pytest_report(str(report_path))

        if not parsed_report.get("success", False):
            result = {
                "success": False,
                "error_type": "sandbox_error",
                "execution_success": True,
                "report_parsed": False,
                "tests_passed": None,
                "return_code": process.returncode,
                "error": parsed_report.get(
                    "error",
                    "Could not parse the Pytest report."
                )
            }

            result.update(collect_output())
            return result

        # 12. Return the parsed test result
        result = {
            "success": True,
            "execution_success": True,
            "report_parsed": True,
            "tests_passed": parsed_report["tests_passed"],
            "return_code": process.returncode,
            "test_report": parsed_report,
            "message": "Pytest execution and report parsing completed."
        }

        result.update(collect_output())
        return result

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_type": "sandbox_timeout",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None,
            "error": (
                f"{current_operation} exceeded its allowed time limit."
            )
        }

    except (OSError, ValueError, RuntimeError) as error:
        return {
            "success": False,
            "error_type": "sandbox_error",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None,
            "error": f"Docker sandbox execution failed: {error}"
        }

    finally:
        # 13. Stop any remaining local Docker CLI process
        try:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass

        finally:
            # 14. Remove the container using its unique name
            try:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass

            finally:
                # 15. Finish output readers
                for thread in (stdout_thread, stderr_thread):
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=5)

                # 16. Close process streams
                if process is not None:
                    for stream in (process.stdout, process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except OSError:
                                pass

                # 17. Remove host staging workspace
                shutil.rmtree(temp_workspace, ignore_errors=True)


# Module names that must not be shadowed inside the sandbox: a staged file
# called ``json.py`` or ``queue.py`` would break pytest itself.
_RESERVED_MODULE_NAMES = set(sys.stdlib_module_names) | {
    "pytest",
    "_pytest",
    "pluggy",
    "conftest",
    "pytest_jsonreport",
}

_SOURCE_FILENAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}\.py$")

USER_TEST_FILENAME = "test_codepilot_user_tests.py"


def validate_source_filename(source_filename: str) -> dict:
    """
    Check that ``source_filename`` is a safe module file name for the sandbox.

    It must be a plain ``name.py`` (no directories), a valid Python module
    name, and must not shadow the standard library, pytest, or the staged
    test file.
    """

    if not isinstance(source_filename, str) or not _SOURCE_FILENAME.match(
        source_filename
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "error": (
                "source_filename must be a plain Python module file name such "
                "as 'solution.py' (letters, digits and underscores only)."
            )
        }

    module_name = source_filename[: -len(".py")]

    if (
        module_name in _RESERVED_MODULE_NAMES
        or f"{module_name}.py" == USER_TEST_FILENAME
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "error": (
                f"'{source_filename}' cannot be used inside the sandbox because "
                "it shadows a standard-library or pytest module (or the staged "
                "test file). Rename the file."
            )
        }

    return {"success": True}


def run_code_tests_in_sandbox(
    source_code: str,
    test_code: str,
    source_filename: str = "solution.py",
    timeout: int = EXECUTION_TIMEOUT
) -> dict:
    """
    Run pytest ``test_code`` against ``source_code`` inside the Docker sandbox.

    Both strings are written to a private staging directory on the host,
    copied into the isolated container, and executed only there. The host
    never imports or runs them. The staging directory is always removed.

    The returned dict has the same shape as ``run_tests_in_sandbox``.
    """

    if (
        not isinstance(source_code, str)
        or not source_code.strip()
        or not isinstance(test_code, str)
        or not test_code.strip()
    ):
        return {
            "success": False,
            "error_type": "invalid_input",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None,
            "error": "source_code and test_code must be non-empty strings."
        }

    filename_check = validate_source_filename(source_filename)

    if not filename_check["success"]:
        return {
            **filename_check,
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None
        }

    try:
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT))

    except OSError as error:
        return {
            "success": False,
            "error_type": "sandbox_error",
            "execution_success": False,
            "report_parsed": False,
            "tests_passed": None,
            "error": f"Could not create the sandbox staging directory: {error}"
        }

    try:
        (staging_dir / source_filename).write_text(source_code, encoding="utf-8")
        (staging_dir / USER_TEST_FILENAME).write_text(test_code, encoding="utf-8")

        return run_tests_in_sandbox(
            workspace_path=str(staging_dir),
            relative_files=[source_filename, USER_TEST_FILENAME],
            test_files=[USER_TEST_FILENAME],
            timeout=timeout
        )

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

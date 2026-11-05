
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File

from pydantic import BaseModel, Field, field_validator


from codepilot_ai.agents.analyzer import analyze_python_code
from codepilot_ai.llm_factory import DEFAULT_GROQ_MODEL
from codepilot_ai.llm_utils import redact_secrets
from codepilot_ai.tools.docker_sandbox import (
    check_docker_environment,
    run_code_tests_in_sandbox,
)
from codepilot_ai.workflow import build_workflow


logger = logging.getLogger("codepilot_ai")


class CodeAnalysisRequest(BaseModel):
    source_code: str = Field(min_length=1, max_length=100_000)

    @field_validator("source_code")
    @classmethod
    def validate_source_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source code cannot be empty or whitespace only.")
        return value


class WorkflowRequest(BaseModel):
    task: str = Field(min_length=1, max_length=5_000)
    source_code: str = Field(min_length=1, max_length=100_000)
    error_context: str = Field(default="", max_length=20_000)
    test_report: Optional[dict[str, Any]] = None

    # Optional pytest code. When given, the final code is run against it
    # inside the Docker sandbox and the result decides success or failure.
    test_code: Optional[str] = Field(default=None, max_length=50_000)
    source_filename: str = Field(default="solution.py", max_length=100)

    @field_validator("task", "source_code")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be empty or whitespace only.")
        return value

    @field_validator("test_code")
    @classmethod
    def blank_test_code_means_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        return value


class SandboxTestRequest(BaseModel):
    source_code: str = Field(min_length=1, max_length=100_000)
    test_code: str = Field(min_length=1, max_length=50_000)
    source_filename: str = Field(default="solution.py", max_length=100)
    timeout: int = Field(default=30, ge=1, le=120)

    @field_validator("source_code", "test_code")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be empty or whitespace only.")
        return value


# Temporary storage for uploaded Python files
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "codepilot_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
UPLOAD_TTL_SECONDS = 24 * 60 * 60  # uploads are deleted after one day


def cleanup_old_uploads(
    max_age_seconds: int = UPLOAD_TTL_SECONDS,
    now: Optional[float] = None
) -> int:
    """
    Delete upload folders older than ``max_age_seconds``.

    Returns the number of folders removed. Problems with a single folder are
    ignored so cleanup can never break a request.
    """

    now = time.time() if now is None else now
    removed = 0

    try:
        children = list(UPLOAD_ROOT.iterdir())
    except OSError:
        return 0

    for child in children:
        try:
            if (
                child.is_dir()
                and not child.is_symlink()
                and now - child.stat().st_mtime > max_age_seconds
            ):
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue

    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_old_uploads()
    yield


app = FastAPI(
    title="CodePilot AI",
    description="Python code analysis, debugging and refactoring agents",
    version="0.3.0",
    lifespan=lifespan
)

# Compile the LangGraph workflow once when the application starts
codepilot_workflow = build_workflow()


# HTTP status for each failure type. The type comes from the actual
# exception / HTTP status of the failed step (see llm_utils.classify_exception)
# so "Groq is down" is only reported when Groq really failed.
ERROR_STATUS_CODES = {
    "provider_unavailable": 503,
    "provider_timeout": 504,
    "provider_rate_limited": 429,
    "provider_auth": 502,
    "provider_bad_request": 502,
    "invalid_response": 502,
    "configuration": 500,
    "analysis_failed": 500,
    "sandbox_unavailable": 503,
    "sandbox_timeout": 504,
    "sandbox_error": 500,
    "invalid_input": 422,
}


def _redact_deep(value: Any) -> Any:
    """Redact secrets in every string inside a JSON-like structure."""

    if isinstance(value, str):
        return redact_secrets(value)

    if isinstance(value, dict):
        return {key: _redact_deep(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_redact_deep(item) for item in value]

    return value


def _raise_failure(
    error_type: str,
    message: str,
    retryable: bool,
    **extra: Any
):
    """Raise an HTTPException whose detail says exactly what failed."""

    raise HTTPException(
        status_code=ERROR_STATUS_CODES.get(error_type, 500),
        detail={
            "error_type": error_type,
            "message": redact_secrets(str(message)),
            "retryable": retryable,
            **_redact_deep(extra)
        }
    )


def _raise_for_failed_analysis(result: dict):
    """Raise the HTTP error for an analysis whose AI step failed."""

    ai_result = result.get("ai_analysis") or {}

    _raise_failure(
        ai_result.get("error_type", "analysis_failed"),
        ai_result.get("error", "The analysis failed."),
        ai_result.get("retryable", False),
        # Static analysis still succeeded; keep it available to the caller.
        analysis=result
    )


@app.get("/")
def root():
    return {
        "message": "Welcome to CodePilot AI"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/health/details")
def health_details():
    """
    Report whether the LLM provider and the Docker sandbox are ready.

    Secrets are never included: only whether the API key is set.
    """

    sandbox = check_docker_environment()

    return {
        "status": "healthy",
        "llm": {
            "provider": "groq",
            "api_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "model": os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
        },
        "sandbox": {
            "available": bool(sandbox.get("success")),
            "image": sandbox.get("image"),
            "error": sandbox.get("error")
        }
    }


@app.post("/upload")
async def upload_python_file(file: UploadFile = File(...)):
    """
    Upload a Python file into an isolated temporary folder.
    The uploaded file is stored but never executed.
    """

    # 1. Validate file extension
    if not file.filename or Path(file.filename).suffix.lower() != ".py":
        raise HTTPException(
            status_code=400,
            detail="Only Python (.py) files are allowed."
        )

    cleanup_old_uploads()

    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_ROOT / upload_id

    try:
        upload_dir.mkdir(parents=True, exist_ok=False)

        destination = upload_dir / "source.py"
        total_size = 0

        # 2. Read the upload in chunks and enforce size limit
        with destination.open("wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum size is 5 MB."
                    )

                output_file.write(chunk)

        # 3. Reject empty files
        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # 4. Validate UTF-8 text
        try:
            content = destination.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be valid UTF-8 text."
            )

        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Python file cannot contain only whitespace."
            )

        if "\x00" in content:
            raise HTTPException(
                status_code=400,
                detail="Invalid Python source file."
            )

        return {
            "success": True,
            "upload_id": upload_id,
            "filename": Path(file.filename).name,
            "size_bytes": total_size,
            "message": "Python file uploaded successfully."
        }

    except HTTPException:
        # Remove incomplete or rejected uploads
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    except Exception as error:
        shutil.rmtree(upload_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {redact_secrets(str(error))}"
        )

    finally:
        await file.close()

@app.post("/upload/{upload_id}/analyze")
def analyze_uploaded_file(upload_id: str):
    """
    Analyze an uploaded Python file without modifying the original.
    """

    # 1. Validate upload ID format
    try:
        safe_upload_id = str(uuid.UUID(upload_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid upload ID."
        )

    # 2. Locate the uploaded file safely
    upload_dir = (UPLOAD_ROOT / safe_upload_id).resolve()

    if not upload_dir.is_relative_to(UPLOAD_ROOT.resolve()):
        raise HTTPException(
            status_code=400,
            detail="Invalid upload path."
        )

    source_path = upload_dir / "source.py"

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found."
        )

    # 3. Read the original source code
    try:
        source_code = source_path.read_text(encoding="utf-8")

    except (OSError, UnicodeDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not read uploaded Python file."
        )

    # 4. Analyze the source code
    result = analyze_python_code(source_code)

    if not result.get("overall_success", False):
        _raise_for_failed_analysis(result)

    return {
        "success": True,
        "upload_id": safe_upload_id,
        "filename": "source.py",
        "analysis": result,
        "message": "Uploaded file analyzed successfully."
    }

@app.post("/analyze")
def analyze_code(request: CodeAnalysisRequest):
    result = analyze_python_code(request.source_code)

    if not result.get("overall_success", False):
        _raise_for_failed_analysis(result)

    return result


@app.post("/run")
def run_workflow(request: WorkflowRequest):
    """
    Execute the CodePilot LangGraph workflow.

    The response is the final workflow state. HTTP 200 is returned even when
    the workflow fails: check ``status`` ("completed", "failed" or
    "needs_clarification"), and for failures ``error``, ``error_type`` and
    ``retryable``.
    """

    initial_state = {
        "task": request.task,
        "source_code": request.source_code,
        "source_filename": request.source_filename,
        "test_code": request.test_code,
        "current_code": request.source_code,
        "error_context": request.error_context,
        "test_report": request.test_report,
        "supervision": None,
        "selected_agents": [],
        "current_agent_index": 0,
        "agent_results": {},
        "verification": None,
        "final_code": None,
        "code_changed": False,
        "warnings": [],
        "error": None,
        "error_type": None,
        "retryable": False,
        "status": "pending"
    }

    try:
        result = codepilot_workflow.invoke(initial_state)
        return result

    except Exception as error:
        logger.exception("Workflow execution crashed")

        return {
            "status": "failed",
            "error": (
                "Workflow execution failed: "
                f"{redact_secrets(str(error))}"
            ),
            "error_type": "workflow_error",
            "retryable": False
        }


@app.post("/sandbox/test")
def run_sandbox_tests(request: SandboxTestRequest):
    """
    Run pytest code against Python source inside the Docker sandbox.

    The code is executed only inside the isolated container (no network,
    read-only filesystem, dropped capabilities, memory/CPU/PID limits).
    HTTP 200 means the sandbox ran, even if tests failed: check
    ``tests_passed``. Sandbox problems are reported as HTTP errors.
    """

    result = run_code_tests_in_sandbox(
        source_code=request.source_code,
        test_code=request.test_code,
        source_filename=request.source_filename,
        timeout=request.timeout
    )

    if not result.get("success", False):
        _raise_failure(
            result.get("error_type", "sandbox_error"),
            result.get("error", "The sandbox run failed."),
            False,
            result=result
        )

    return result


import tempfile
import uuid
from pathlib import Path

from typing import Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File

from pydantic import BaseModel, Field, field_validator


from codepilot_ai.agents.analyzer import analyze_python_code
from codepilot_ai.workflow import build_workflow





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

    @field_validator("task", "source_code")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be empty or whitespace only.")
        return value


app = FastAPI(
    title="CodePilot AI",
    description="Autonomous Code Debugging & Refactoring Agent",
    version="0.2.0"
)

# Temporary storage for uploaded Python files
UPLOAD_ROOT = Path(tempfile.gettempdir()) / "codepilot_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB

# Compile the LangGraph workflow once when the application starts
codepilot_workflow = build_workflow()


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
        import shutil
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    except Exception as error:
        import shutil
        shutil.rmtree(upload_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {error}"
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
        raise HTTPException(
            status_code=503,
            detail=result
        )

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
        raise HTTPException(
            status_code=503,
            detail=result
        )

    return result


@app.post("/run")
def run_workflow(request: WorkflowRequest):
    """
    Execute the CodePilot LangGraph workflow.
    """

    initial_state = {
        "task": request.task,
        "source_code": request.source_code,
        "current_code": request.source_code,
        "error_context": request.error_context,
        "test_report": request.test_report,
        "supervision": None,
        "selected_agents": [],
        "current_agent_index": 0,
        "agent_results": {},
        "final_code": None,
        "error": None,
        "status": "pending"
    }

    try:
        result = codepilot_workflow.invoke(initial_state)
        return result

    except Exception as error:
        return {
            "status": "failed",
            "error": f"Workflow execution failed: {error}"
        }

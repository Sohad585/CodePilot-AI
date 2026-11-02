
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_code_file(file_path: str) -> dict:
    """
    Read a file from inside the project directory.

    Security rules:
    - Only files inside the project directory can be read.
    - Sensitive files and directories are blocked.
    - The function does not execute the file.
    """

    try:
        # Resolve the requested path
        requested_path = (PROJECT_ROOT / file_path).resolve()

        # 1. Prevent access outside the project directory
        if not requested_path.is_relative_to(PROJECT_ROOT):
            return {
                "success": False,
                "error": "Access denied: File is outside the project directory."
            }

        # 2. Block sensitive files and directories
        blocked_names = {
            ".env",
            ".env.example",
            ".git",
            ".venv",
            "__pycache__"
        }

        relative_path = requested_path.relative_to(PROJECT_ROOT)

        if any(part in blocked_names for part in relative_path.parts):
            return {
                "success": False,
                "error": "Access denied: This file or directory is restricted."
            }

        # 3. Check that the path exists and is a file
        if not requested_path.exists():
            return {
                "success": False,
                "error": "File not found."
            }

        if not requested_path.is_file():
            return {
                "success": False,
                "error": "The requested path is not a file."
            }

        # 4. Read the file content
        content = requested_path.read_text(encoding="utf-8")

        return {
            "success": True,
            "file_path": str(relative_path),
            "content": content
        }

    except UnicodeDecodeError:
        return {
            "success": False,
            "error": "The file is not a valid UTF-8 text file."
        }

    except (OSError, ValueError, RuntimeError) as error:
        return {
            "success": False,
            "error": f"Could not read file: {error}"
        }

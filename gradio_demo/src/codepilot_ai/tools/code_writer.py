from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_code_file(
    file_path: str,
    content: str,
    overwrite: bool = False
) -> dict:
    """
    Write code content to a file inside the project directory.

    Security rules:
    - Only files inside the project directory can be modified.
    - Sensitive files and directories are blocked.
    - Existing files cannot be overwritten unless explicitly allowed.
    - Parent directories are not created automatically.
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

        # 3. Check that the parent directory exists
        if not requested_path.parent.is_dir():
            return {
                "success": False,
                "error": "Parent directory does not exist."
            }

        # 4. Prevent overwriting existing files
        if requested_path.exists() and not overwrite:
            return {
                "success": False,
                "error": "File already exists. Set overwrite=True to replace it."
            }

        # 5. Check that the path is not a directory
        if requested_path.exists() and not requested_path.is_file():
            return {
                "success": False,
                "error": "The requested path is not a file."
            }

        # 6. Write the content
        requested_path.write_text(
            content,
            encoding="utf-8"
        )

        return {
            "success": True,
            "file_path": str(relative_path),
            "message": "File written successfully."
        }

    except (OSError, ValueError, RuntimeError) as error:
        return {
            "success": False,
            "error": f"Could not write file: {error}"
        }
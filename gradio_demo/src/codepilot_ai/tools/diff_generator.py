
import difflib


def generate_code_diff(
    original_code: str,
    modified_code: str,
    original_filename: str = "original.py",
    modified_filename: str = "modified.py"
) -> dict:
    """
    Generate a unified diff between original and modified code.

    This function compares two code versions without executing
    or modifying any files.
    """

    # 1. Validate inputs
    if not isinstance(original_code, str) or not isinstance(modified_code, str):
        return {
            "success": False,
            "error": "Both code versions must be strings."
        }

    if not original_filename or not modified_filename:
        return {
            "success": False,
            "error": "Filenames cannot be empty."
        }

    # 2. Split code into lines
    original_lines = original_code.splitlines(keepends=True)
    modified_lines = modified_code.splitlines(keepends=True)

    # 3. Generate unified diff
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=original_filename,
        tofile=modified_filename,
        lineterm="\n"
    )

    diff_text = "".join(diff)

    # 4. Count added and removed lines
    added_lines = 0
    removed_lines = 0

    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            added_lines += 1
        elif line.startswith("-"):
            removed_lines += 1

    # 5. Return structured result
    return {
        "success": True,
        "has_changes": original_code != modified_code,
        "diff": diff_text,
        "summary": {
            "added_lines": added_lines,
            "removed_lines": removed_lines
        }
    }

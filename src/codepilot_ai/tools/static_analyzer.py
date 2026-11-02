import ast


def analyze_python_syntax(source_code: str) -> dict:
    """
    Check Python syntax without executing the submitted code.
    """

    try:
        ast.parse(source_code)

        return {
            "valid_syntax": True,
            "issues": []
        }

    except SyntaxError as error:
        return {
            "valid_syntax": False,
            "issues": [
                {
                    "type": "Syntax Error",
                    "message": error.msg,
                    "line": error.lineno,
                    "severity": "error"
                }
            ]
        }


def analyze_code_quality(source_code: str) -> dict:
    """
    Detect predefined code quality and potentially risky patterns.
    """

    issues = []

    try:
        syntax_tree = ast.parse(source_code)

    except SyntaxError as error:
        return {
            "success": False,
            "issues": [
                {
                    "type": "Syntax Error",
                    "message": error.msg,
                    "line": error.lineno,
                    "severity": "error"
                }
            ],
            "total_issues": 1
        }

    for node in ast.walk(syntax_tree):

        # Detect bare except clauses
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "type": "Code Quality",
                "message": "Bare except clause detected.",
                "line": node.lineno,
                "severity": "warning"
            })

        # Detect eval() and exec() usage
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    issues.append({
                        "type": "Potentially Risky Code",
                        "message": f"Usage of {node.func.id}() detected.",
                        "line": node.lineno,
                        "severity": "warning"
                    })

        # Detect wildcard imports
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    issues.append({
                        "type": "Code Quality",
                        "message": "Wildcard import detected.",
                        "line": node.lineno,
                        "severity": "warning"
                    })

    return {
        "success": True,
        "issues": issues,
        "total_issues": len(issues)
    }


def run_static_analysis(source_code: str) -> dict:
    """
    Run all static analysis checks and combine their results.
    """

    syntax_result = analyze_python_syntax(source_code)

    if not syntax_result["valid_syntax"]:
        return {
            "valid_syntax": False,
            "static_analysis": {
                "issues": syntax_result["issues"],
                "total_issues": len(syntax_result["issues"])
            }
        }

    quality_result = analyze_code_quality(source_code)

    return {
        "valid_syntax": True,
        "static_analysis": quality_result
    }
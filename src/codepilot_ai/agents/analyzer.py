
import ast

from codepilot_ai.agents.ai_analyzer import analyze_code_with_ai


def analyze_python_code(source_code: str) -> dict:
    """
    Perform static and AI-powered analysis on Python source code.

    Static analysis checks syntax and predefined risky patterns.
    AI analysis provides deeper code understanding and suggestions.

    The submitted code is never executed.
    """

    issues = []

    # 1. Check Python syntax
    try:
        syntax_tree = ast.parse(source_code)

    except SyntaxError as error:
        return {
            "valid_syntax": False,
            "static_analysis": {
                "issues": [
                    {
                        "type": "Syntax Error",
                        "message": error.msg,
                        "line": error.lineno,
                        "severity": "error"
                    }
                ],
                "total_issues": 1
            },
            "ai_analysis": {
                "success": False,
                "message": "AI analysis skipped because the Python code contains a syntax error."
            }
        }

    # 2. Perform static analysis
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

    static_result = {
        "issues": issues,
        "total_issues": len(issues)
    }

    # 3. Perform AI-powered analysis
    ai_result = analyze_code_with_ai(source_code)

    # 4. Combine both analysis results
    return {
        "valid_syntax": True,
        "static_analysis": static_result,
        "ai_analysis": ai_result
    }

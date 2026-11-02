
from codepilot_ai.agents.ai_analyzer import analyze_code_with_ai
from codepilot_ai.tools.static_analyzer import run_static_analysis


def analyze_python_code(source_code: str) -> dict:
    """
    Perform static and AI-powered analysis on Python source code.

    The submitted code is never executed.
    """

    # 1. Perform static analysis
    static_result = run_static_analysis(source_code)

    # 2. Skip AI analysis if syntax is invalid
    if not static_result["valid_syntax"]:
        return {
            "overall_success": True,
            "valid_syntax": False,
            "static_analysis": static_result["static_analysis"],
            "ai_analysis": {
                "success": False,
                "message": (
                    "AI analysis skipped because the Python code "
                    "contains a syntax error."
                )
            }
        }

    # 3. Perform AI-powered analysis
    ai_result = analyze_code_with_ai(source_code)

    # 4. Combine both results
    return {
        "overall_success": ai_result.get("success", False),
        "valid_syntax": True,
        "static_analysis": static_result["static_analysis"],
        "ai_analysis": ai_result
    }

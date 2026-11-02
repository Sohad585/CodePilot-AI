
import json
from pathlib import Path


def parse_pytest_report(report_path: str) -> dict:
    """
    Read a pytest JSON report and return structured test results.

    This function does not execute any code.

    success / report_parsed:
        Indicates whether the report was read and parsed successfully.

    tests_passed:
        Indicates whether all tests passed successfully.
    """

    try:
        # 1. Read the JSON report
        report_file = Path(report_path)

        if not report_file.is_file():
            return {
                "success": False,
                "report_parsed": False,
                "error": "Pytest JSON report file not found."
            }

        with report_file.open("r", encoding="utf-8") as file:
            report = json.load(file)

        if not isinstance(report, dict):
            return {
                "success": False,
                "report_parsed": False,
                "error": "Invalid JSON report format."
            }

        # 2. Validate report sections
        summary_data = report.get("summary", {})
        collectors = report.get("collectors", [])
        tests = report.get("tests", [])

        if (
            not isinstance(summary_data, dict)
            or not isinstance(collectors, list)
            or not isinstance(tests, list)
        ):
            return {
                "success": False,
                "report_parsed": False,
                "error": "Invalid Pytest report structure."
            }

        # 3. Extract summary information
        passed = summary_data.get("passed", 0)
        failed = summary_data.get("failed", 0)
        skipped = summary_data.get("skipped", 0)
        total = summary_data.get("total", 0)
        duration = report.get("duration", 0)
        return_code = report.get("exitcode", 1)

        count_values = (passed, failed, skipped, total)

        if (
            not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in count_values
            )
            or not isinstance(return_code, int)
            or isinstance(return_code, bool)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
        ):
            return {
                "success": False,
                "report_parsed": False,
                "error": "Invalid Pytest report summary values."
            }

        # 4. Extract collection errors
        errors = sum(
            1
            for collector in collectors
            if isinstance(collector, dict)
            and collector.get("outcome") == "failed"
        )

        # 5. Extract failed test details
        failures = []

        for test in tests:

            if not isinstance(test, dict):
                continue

            if test.get("outcome") != "failed":
                continue

            nodeid = test.get("nodeid", "")
            failure_stage = {}
            failure_stage_name = None

            # Find the stage that contains the failure
            for stage in ("setup", "call", "teardown"):
                stage_data = test.get(stage, {})

                if (
                    isinstance(stage_data, dict)
                    and stage_data.get("outcome") == "failed"
                ):
                    failure_stage = stage_data
                    failure_stage_name = stage
                    break

            crash = failure_stage.get("crash", {})
            traceback = failure_stage.get("traceback", [])

            if not isinstance(crash, dict):
                crash = {}

            if not isinstance(traceback, list):
                traceback = []

            # Format traceback information
            traceback_text = "\n".join(
                f"{item.get('path', '')}:{item.get('lineno', '')}: "
                f"{item.get('message', '')}"
                for item in traceback
                if isinstance(item, dict)
            )

            # Extract test name and file path
            test_name = nodeid.split("::")[-1] if isinstance(nodeid, str) else "Unknown"
            test_file = nodeid.split("::")[0] if isinstance(nodeid, str) else ""

            failures.append({
                "test_name": test_name,
                "file": crash.get("path", test_file),
                "line": crash.get("lineno", test.get("lineno")),
                "stage": failure_stage_name,
                "error": (
                    traceback[-1].get("message", "Test failed")
                    if traceback and isinstance(traceback[-1], dict)
                    else "Test failed"
                ),
                "message": crash.get("message", ""),
                "traceback": traceback_text,
                "details": failure_stage.get("longrepr", "")
            })

        # 6. Determine actual test status
        tests_passed = (
            return_code == 0
            and failed == 0
            and errors == 0
            and total > 0
        )

        # 7. Return structured results
        return {
            "success": True,
            "report_parsed": True,
            "tests_passed": tests_passed,
            "return_code": return_code,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "duration": round(duration, 3)
            },
            "failures": failures
        }

    except json.JSONDecodeError:
        return {
            "success": False,
            "report_parsed": False,
            "error": "Invalid JSON report file."
        }

    except (OSError, TypeError, AttributeError, ValueError) as error:
        return {
            "success": False,
            "report_parsed": False,
            "error": f"Could not parse pytest report: {error}"
        }

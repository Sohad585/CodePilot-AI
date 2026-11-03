# CodePilot AI

CodePilot AI reviews, debugs and refactors Python files with a small team of
LLM agents (Groq), coordinated by a LangGraph workflow. A FastAPI backend does
the work; a Streamlit app is the user interface. Optionally, code can be
**verified by your own pytest tests, run inside an isolated Docker sandbox**.

## What it does (and does not do)

| Capability | Status |
|---|---|
| Static checks (syntax, bare `except`, `eval`/`exec`, wildcard imports) | Implemented, no LLM |
| AI code review (Analyzer) | Implemented |
| AI bug fixing (Debugger) and refactoring (Refactorer) | Implemented; output must be valid Python (`ast.parse`) or it is rejected |
| Evaluating a pytest report (Tester) | Implemented; pass/fail and counts are taken from the report itself, the LLM only explains |
| Running **your** pytest tests against the result in Docker | Implemented, optional (`test_code`) |
| Writing tests for you, or loops that fix-and-retest until green | **Not implemented** |
| Proof that a refactor preserves behavior | Only if you supply tests; otherwise the result is marked *not verified* |

Without tests, modified code is only **syntax-checked**, never run, and the UI
says so. Success is only reported as "verified" when your tests pass in the sandbox.

## Architecture

```
Streamlit (frontend/)  --HTTP-->  FastAPI (src/codepilot_ai/backend/main.py)
                                      |
                                      v
                     LangGraph workflow (src/codepilot_ai/workflow.py)
        supervisor -> agent_execution (once per agent) -> finalize
                                      |                      |
                  agents/ (Groq LLM)  |                      +--> Docker sandbox
        Analyzer, Debugger, Refactorer, Tester, Supervisor         (only if test_code given)
```

* **Supervisor** picks agents from your task. Agents always run in the fixed
  order Analyzer -> Debugger -> Refactorer -> Tester, each at most once.
  Tester needs a pytest report; without one it is skipped with a warning.
* **A failed agent stops the run.** Its real error (`error`, `error_type`,
  `retryable`) and any partial results are returned. Code from earlier agents is
  kept as a labelled, unverified candidate.
* **Your original code is never modified**; `source_code` is returned untouched
  and `final_code` is the working copy.
* **Statuses:** `completed`, `failed`, `needs_clarification`.
* **Docker sandbox** (`tools/docker_sandbox.py`): no network, read-only root
  filesystem, all capabilities dropped, non-root user, memory/CPU/PID limits,
  size-limited tmpfs workspace, hard timeout, output truncation, and the
  container is always removed. Nothing from the host is mounted.
* **LLM:** all agents use `llm_factory.get_llm()` (Groq via `langchain-groq`).

### Error types

The backend reports *why* something failed, from the real exception or HTTP
status, never from words in a message:

`provider_unavailable` (503), `provider_timeout` (504), `provider_rate_limited`
(429), `provider_auth` (bad API key), `provider_bad_request`, `invalid_response`
(model answered with unusable output), `configuration` (e.g. missing key),
`analysis_failed`, `verification_failed`, `sandbox_unavailable`,
`sandbox_timeout`, `sandbox_error`, `invalid_input`, `workflow_error`.

## Installation

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker (only
needed for sandbox verification).

```bash
uv sync
cp .env.example .env        # then edit .env
```

### Environment variables

Set in `.env` (never commit it; this repository never contains key values):

| Name | Required | Meaning |
|---|---|---|
| `GROQ_API_KEY` | yes | Your Groq API key |
| `GROQ_MODEL` | no | Model name (default `openai/gpt-oss-120b`) |
| `CODEPILOT_SANDBOX_IMAGE` | no | Sandbox image (default `autodebug-sandbox:2`) |
| `CODEPILOT_API_URL` | no | Backend URL used by the frontend (default `http://127.0.0.1:8000`) |

### Build the sandbox image (once, for test verification)

```bash
docker build -t autodebug-sandbox:2 .
```

## Run

Backend (terminal 1):

```bash
uv run uvicorn codepilot_ai.backend.main:app --host 127.0.0.1 --port 8000
# or: uv run codepilot-ai
```

Frontend (terminal 2):

```bash
uv run streamlit run frontend/app.py
```

Check the backend: `curl http://127.0.0.1:8000/health` (liveness) and
`curl http://127.0.0.1:8000/health/details` (is the Groq key set? is Docker /
the sandbox image ready?). Neither shows secret values.

> The backend has **no authentication** and can run code (in the sandbox) when
> `test_code` is supplied. Keep it bound to `127.0.0.1`; do not expose it.

### API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /health/details` | Liveness; provider and sandbox readiness |
| `POST /upload` | Store a `.py` file (never executed); returns `upload_id` |
| `POST /upload/{id}/analyze` | Static + AI analysis of an upload |
| `POST /analyze` | Static + AI analysis of posted code |
| `POST /run` | Run the workflow (`task`, `source_code`, optional `error_context`, `test_report`, `test_code`, `source_filename`) |
| `POST /sandbox/test` | Run pytest code against source in the sandbox |

`POST /run` answers HTTP 200 even when the workflow fails; check `status`.

### Using sandbox verification

In the UI, open *Advanced options* and paste pytest tests. Import your code by
its file name: for `inventory.py` write `from inventory import add_item`
(`solution` if the file name is not a valid module name). Uploads are deleted
from the backend after 24 hours.

## Tests

```bash
uv run pytest                      # everything; Docker tests skip if Docker/image is missing
uv run pytest -m "not docker"      # no Docker needed
uv run pytest -m docker            # only the real-container tests
uv run ruff check src frontend tests
```

No test needs an API key or network: LLM calls are mocked. `tests/sandbox_fixtures/`
holds sample test files that are executed *inside* the sandbox by the Docker
tests (some fail or sleep on purpose); they are not part of the normal run.

## Limitations

* The sandbox protects your machine, not the verdict: tests and the code under
  test run in the same container, so deliberately malicious code could forge a
  passing report.
* LLM output is non-deterministic. A syntactically valid fix can still be wrong;
  only your tests can show otherwise.
* One shot per agent: there is no automatic retry loop beyond the LangChain
  client's single retry on provider errors.
* Limits: 100,000 characters of code for Debug/Refactor, 5 MB per upload,
  50,000 characters of test code, 30 s sandbox timeout.
* `deepagents` is not used anywhere in the code and was removed from the
  dependencies; add it back if you plan to use it.

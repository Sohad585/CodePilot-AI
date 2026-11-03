def main() -> None:
    """Start the CodePilot AI backend (FastAPI) with uvicorn.

    Environment variables:
    - CODEPILOT_HOST (default 127.0.0.1)
    - CODEPILOT_PORT (default 8000)
    """

    import os

    import uvicorn

    uvicorn.run(
        "codepilot_ai.backend.main:app",
        host=os.getenv("CODEPILOT_HOST", "127.0.0.1"),
        port=int(os.getenv("CODEPILOT_PORT", "8000")),
    )

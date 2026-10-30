
from fastapi import FastAPI
from pydantic import BaseModel

from codepilot_ai.agents.analyzer import analyze_python_code


app = FastAPI(
    title="CodePilot AI",
    description="Autonomous Code Debugging & Refactoring Agent",
    version="0.1.0"
)


class CodeAnalysisRequest(BaseModel):
    source_code: str


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


@app.post("/analyze")
def analyze_code(request: CodeAnalysisRequest):
    result = analyze_python_code(request.source_code)

    return result

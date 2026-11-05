---
title: CodePilot AI Demo
emoji: 🧑‍💻
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# CodePilot AI Demo

Copy this `app.py` and `requirements.txt` into a Hugging Face Gradio Space, and also copy the existing project `src/` directory so the app can import the original analyzer.

In Space Settings → Variables and secrets, add secret `GROQ_API_KEY`. Optionally add variable `GROQ_MODEL`; default is `openai/gpt-oss-120b`.

This demo reuses `src/codepilot_ai/agents/analyzer.py`, so it requires the original project source folder. It supports static checks, Groq code review, and AI chat. It does not execute uploaded code and does not include Debug, Refactor, pytest, or Docker workflows.

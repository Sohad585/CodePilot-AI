import os

from dotenv import load_dotenv

from codepilot_ai.llm_utils import LLMConfigurationError

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_llm(provider: str = "groq"):
    """
    Create and return an LLM instance.

    Currently supported provider:
    - groq

    Environment variables:
    - GROQ_API_KEY (required)
    - GROQ_MODEL   (optional, defaults to DEFAULT_GROQ_MODEL)
    """

    if provider.lower() == "groq":

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL

        if not api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not set. Add it to the backend's .env file "
                "(see .env.example) and restart the backend."
            )

        # Imported here so the rest of the package can be imported (and
        # tested) without the provider SDK being installed.
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0,
            timeout=160,
            max_retries=1
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider: {provider}"
    )

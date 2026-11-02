
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm(provider: str = "groq"):
    """
    Create and return an LLM instance.

    Currently supported provider:
    - groq
    """

    if provider.lower() == "groq":

        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Please check your .env file."
            )

        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0,
            timeout=160,
            max_retries=1
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )

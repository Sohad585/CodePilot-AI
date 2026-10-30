
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables from .env
load_dotenv()


def get_llm(provider: str = "google"):
    """
    Create and return an LLM instance.

    Currently supported provider:
    - google (Gemini)
    """

    if provider.lower() == "google":

        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. "
                "Please check your .env file."
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )

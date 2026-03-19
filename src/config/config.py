"""Configuration module for Agentic RAG system"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# Load environment variables from .env
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()

    # LLM configuration (LangChain syntax: "provider:model_name")
    LLM_MODEL = "google_genai:gemini-2.5-flash"

    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Default URLs (if you ever want to use web docs)
    DEFAULT_URLS = [
        "https://en.wikipedia.org/wiki/SpaceX_Starship"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # LangChain Google integration reads this env var internally
        os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY

        # temperature=0 is recommended for RAG
        return init_chat_model(cls.LLM_MODEL, temperature=0)

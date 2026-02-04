from dataclasses import dataclass, field
from enum import Enum
import os
from threading import Lock

from openai import OpenAI


CLIENT_LOCK = Lock()
CLIENTS = {} # mapping of base_url -> client instance


def get_or_create_openai_client(
    base_url: str,
    api_key: str
) -> OpenAI:
    """
    Instantiates a client for the OpenAI API (or compatible APIs, e.g. OpenRouter)
    with the given base URL.
    """
    with CLIENT_LOCK:
        if base_url not in CLIENTS:
            # Create a new client instance (pseudo-code, replace with actual client creation)
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            CLIENTS[base_url] = client
        return CLIENTS[base_url]


class LLMProvider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    """
    Configuration for connecting to a Large Language Model (LLM) provider,
    including model name, temperature, provider type, and retry settings.
    """
    model_name: str
    temperature: int = field(default=0)
    llm_provider: str = field(
        default=LLMProvider.OPENAI.value,
        metadata=dict(
            choices=[e.value for e in LLMProvider]
        ))
    max_retries: int = field(default=5)

    def get_base_url(self):
        if self.llm_provider == LLMProvider.OPENAI.value:
            return None
        if self.llm_provider == LLMProvider.OPENROUTER.value:
            return "https://openrouter.ai/api/v1"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def get_client(self, api_key: str | None=None):
        return get_or_create_openai_client(
            self.get_base_url(),
            api_key or self.get_key_from_env()
        )

    def get_key_from_env(self):
        if self.llm_provider == LLMProvider.OPENAI.value:
            return os.environ.get("OPENAI_API_KEY")
        if self.llm_provider == LLMProvider.OPENROUTER.value:
            return os.environ.get("OPENROUTER_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
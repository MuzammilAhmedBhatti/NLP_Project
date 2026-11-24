"""Conversational model used for open-ended follow-up questions."""

from __future__ import annotations

from functools import lru_cache

from ..config import settings
from .ollama import OllamaClient
from .providers import configured_client


@lru_cache(maxsize=1)
def conversation_model():
    """Create the configured provider with local fallback for follow-up chat."""
    if settings.chat_backend == "ollama":
        return OllamaClient(settings.ollama_model, settings.ollama_url)
    return configured_client(settings.chat_backend, settings.chat_model)

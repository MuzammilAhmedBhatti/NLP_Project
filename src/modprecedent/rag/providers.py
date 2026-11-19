"""Configurable local and future hosted-model provider factory."""

from __future__ import annotations

import json

import requests

from ..config import settings
from .local_instruct import TransformersInstructClient


class OpenAICompatibleClient:
    """Minimal adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, model: str, api_key: str, base_url: str):
        """Initialize the OpenAICompatibleClient instance and its required dependencies."""
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        text = self.generate_text(prompt + "\nReturn only valid JSON.")
        return json.loads(text[text.index("{") : text.rindex("}") + 1])


class AnthropicClient:
    """Small direct Messages API adapter with conservative token limits."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 320):
        """Initialize the AnthropicClient instance and its required dependencies."""
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        response.raise_for_status()
        blocks = response.json().get("content", [])
        return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        instruction = (
            '\nReturn only compact JSON: {"outcome":"upheld|overturned",'
            '"confidence":0.0,"reasoning":"two sentences maximum",'
            '"policies":["policy"],"citation_ids":["valid evidence ID"]}. '
            "Use only citation IDs shown in EVIDENCE."
        )
        text = self.generate_text(prompt + instruction)
        return json.loads(text[text.index("{") : text.rindex("}") + 1])


class ResilientClient:
    """Use the low-cost hosted model first and local Qwen on API failure."""

    def __init__(self, primary, fallback_model: str):
        """Initialize the ResilientClient instance and its required dependencies."""
        self.primary = primary
        self.fallback_model = fallback_model
        self._fallback = None
        self.model = primary.model

    @property
    def fallback(self):
        """Return the lazily initialized local fallback client."""
        if self._fallback is None:
            self._fallback = TransformersInstructClient(self.fallback_model)
        return self._fallback

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        try:
            return self.primary.generate_text(prompt)
        except (requests.RequestException, KeyError, ValueError):
            return self.fallback.generate_text(prompt)

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        try:
            return self.primary.generate_json(prompt)
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            return self.fallback.generate_json(prompt)


def configured_client(backend: str, model: str):
    """Build the selected API or local model client with resilient fallback."""
    if backend == "transformers":
        try:
            return TransformersInstructClient(model)
        except Exception:
            if model == settings.fallback_model:
                raise
            return TransformersInstructClient(settings.fallback_model)
    if backend == "openai":
        if not settings.api_key:
            raise ValueError("MODPRECEDENT_API_KEY is required for the OpenAI backend")
        return OpenAICompatibleClient(model, settings.api_key, settings.api_base_url)
    if backend == "anthropic":
        api_key = settings.claude_api_key
        if not api_key:
            raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY is required")
        primary = AnthropicClient(
            model or settings.anthropic_model,
            api_key,
            settings.anthropic_max_tokens,
        )
        return ResilientClient(primary, settings.fallback_model)
    raise ValueError(f"Unsupported model backend: {backend}")

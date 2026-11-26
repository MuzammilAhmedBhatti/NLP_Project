from __future__ import annotations

import json
import logging
import re

import requests

LOG = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434"):
        """Initialize the OllamaClient instance and its required dependencies."""
        self.model, self.url = model, url.rstrip("/")

    def _generate(self, prompt: str, json_mode: bool = False) -> str:
        """Run model inference with the backend-specific generation settings."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        }
        if json_mode:
            payload["format"] = "json"
        response = requests.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        return self._generate(prompt)

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        raw = self._generate(prompt, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("LLM did not return JSON")
            return json.loads(match.group())

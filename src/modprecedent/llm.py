from __future__ import annotations

import json
import logging
import re

import requests

LOG = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def generate_json(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 42},
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("LLM did not return JSON")
            return json.loads(match.group())

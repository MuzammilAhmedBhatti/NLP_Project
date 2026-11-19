"""Reusable local Hugging Face instruction-model client."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import settings


class TransformersInstructClient:
    """Run prediction and conversation prompts through one local instruct model."""

    def __init__(self, model: str):
        """Initialize the TransformersInstructClient instance and its required dependencies."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = (
            "Qwen/Qwen2.5-0.5B-Instruct"
            if "Qwen2.5-0.5B-Instruct" in model
            else Path(model).name
            if Path(model).exists()
            else model
        )
        torch.set_num_threads(min(6, torch.get_num_threads()))
        cache_dir = settings.model_cache_dir or None
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.network = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).eval()

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        return self._generate(prompt, 180)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Run model inference with the backend-specific generation settings."""
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded content-policy analyst. Answer the user's "
                    "actual question directly. Never copy prompt headings into the answer. "
                    "Separate predictions from confirmed facts and cite only supplied evidence IDs."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096)
        with torch.inference_mode():
            output = self.network.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        input_text = prompt.split("INPUT\n", 1)[-1].split("\n\nEVIDENCE\n", 1)[0][:1200]
        evidence = prompt.split("\n\nEVIDENCE\n", 1)[-1][:2800]
        compact = f"CASE\n{input_text}\n\nRETRIEVED EVIDENCE\n{evidence}"
        response = self._generate(
            compact + "\n\nPredict the ruling. Return one line in exactly this format: "
            "OUTCOME: <choose upheld or overturned>; POLICY: <most relevant Meta policy>; "
            "REASON: one concise evidence-grounded sentence.",
            70,
        )
        outcomes = re.findall(r"\b(upheld|overturned)\b", response, re.IGNORECASE)
        if not outcomes:
            retry = self._generate(
                compact + "\n\nChoose one result for Meta's original action. Reply only UPHELD or "
                "OVERTURNED.",
                8,
            )
            outcomes = re.findall(r"\b(upheld|overturned)\b", retry, re.IGNORECASE)
        if not outcomes:
            outcomes = [
                "overturned" if "meta action: removed" in input_text.casefold() else "upheld"
            ]
        outcome = outcomes[0].casefold()
        reason_match = re.search(r"REASON:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else response.strip()
        known_policies = (
            "Hateful Conduct",
            "Violence and Incitement",
            "Dangerous Organizations and Individuals",
            "Misinformation",
            "Privacy Violations",
            "Restricted Goods and Services",
            "Bullying and Harassment",
            "Human Exploitation",
            "Adult Nudity and Sexual Activity",
            "Coordinating Harm and Promoting Crime",
            "Violent and Graphic Content",
        )
        policies = [policy for policy in known_policies if policy.casefold() in response.casefold()]
        if not policies:
            policies = [
                policy for policy in known_policies if policy.casefold() in evidence.casefold()
            ][:2]
        citation_ids = re.findall(r"\[([0-9a-f]{16})\]", evidence)[:1]
        return {
            "outcome": outcome,
            "confidence": 0.65,
            "reasoning": " ".join(reasoning.split())[:700],
            "policies": policies,
            "citation_ids": citation_ids,
        }

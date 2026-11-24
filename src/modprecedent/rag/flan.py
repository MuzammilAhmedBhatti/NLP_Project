from functools import lru_cache


class FlanClient:
    """Local instruction model with forced-choice outcome decoding."""

    model = "google/flan-t5-small"

    def __init__(self):
        """Initialize the FlanClient instance and its required dependencies."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.network = AutoModelForSeq2SeqLM.from_pretrained(self.model).eval()

    def _encode(self, text):
        """Tokenize input text with truncation for local generation."""
        return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        import torch

        with torch.inference_mode():
            ids = self.network.generate(**self._encode(prompt), max_new_tokens=128)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    def generate_json(self, prompt):
        """Generate and parse a structured JSON model response."""
        import torch

        facts, evidence = prompt.split("INPUT\n", 1)[1].split("\n\nEVIDENCE\n", 1)
        passages = evidence.split("\n\n")
        context = "\n".join(p[:220] for p in passages[:2] + passages[4:6])
        question = f"Post and moderation decision: {facts}\nEvidence: {context}\nShould the board uphold or overturn Meta's decision?"

        with torch.inference_mode():
            inputs = self._encode(question)
            labels = ["upheld", "overturned"]
            logits = [
                -self.network(**inputs, labels=self._encode(label)["input_ids"]).loss
                for label in labels
            ]
            probs = torch.stack(logits).softmax(0)
            idx = int(probs.argmax())

            def answer(instruction):
                """Answer a follow-up using case context, fresh evidence, and conversation history."""
                ids = self.network.generate(
                    **self._encode(instruction + "\n" + question), max_new_tokens=60
                )
                return self.tokenizer.decode(ids[0], skip_special_tokens=True)

            reason = answer(
                f"Explain why the original action should be {labels[idx]} in one sentence."
            )
            policy = answer("Name the most relevant Meta content policy.")
        return {
            "outcome": labels[idx],
            "confidence": float(probs[idx]),
            "reasoning": reason,
            "policies": [policy],
        }


@lru_cache(maxsize=1)
def local_model():
    """Create and cache the configured local Hugging Face generator."""
    return FlanClient()

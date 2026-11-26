from __future__ import annotations

from collections import Counter
from typing import Protocol

import requests

from ..config import settings
from ..data.models import Citation, Prediction, PredictionRequest
from .ollama import OllamaClient
from .retrieval import Hit, HybridIndex

SYSTEM_GUIDANCE = """You are predicting how Meta's independent Oversight Board would rule.
Return JSON with keys: outcome (upheld or overturned), confidence (0 to 1), reasoning,
and policies (array). Judge consistency with Meta policy, necessity/proportionality, context,
voice, intent, likelihood and severity of harm, and international human rights standards.
Do not treat retrieved precedent outcomes as certain. Cite evidence using [chunk_id]."""


class GenerativeClient(Protocol):
    model: str

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a structured JSON model response."""
        ...

    def generate_text(self, prompt: str) -> str:
        """Generate a natural-language response from the configured model backend."""
        ...


class Predictor:
    def __init__(self, index: HybridIndex | None = None, llm: GenerativeClient | None = None):
        """Initialize the Predictor instance and its required dependencies."""
        self.index = index
        if llm is None:
            if settings.llm_backend == "flan":
                from .flan import local_model

                llm = local_model()
            elif settings.llm_backend == "ollama":
                # Preserved as an inactive local-server option; select it through .env when needed.
                llm = OllamaClient(settings.ollama_model, settings.ollama_url)
            else:
                from .providers import configured_client

                llm = configured_client(settings.llm_backend, settings.llm_model)
        self.llm = llm

    @staticmethod
    def _query(request: PredictionRequest) -> str:
        """Build a retrieval query from the post, action, and optional question."""
        return f"Meta action: {request.meta_action}. Content: {request.post_description}"

    def _hits(self, request: PredictionRequest, system: str) -> list[Hit]:
        """Retrieve policy and precedent evidence for the selected RAG system."""
        if not self.index or system == "S0":
            return []
        query = self._query(request)
        hits = self.index.search(query, 4, "policy", request.as_of_date)
        if system == "S2":
            hits += self.index.search(query, 6, "precedent", request.as_of_date)
        return hits

    def follow_up_citations(
        self,
        request: PredictionRequest,
        question: str,
        history: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
        k: int = 6,
    ) -> list[Citation]:
        """Retrieve evidence for the current turn, not only the original prediction."""
        if not self.index:
            return []
        recent = " ".join(user for user, _ in history[-2:])
        query = f"{self._query(request)} Follow-up: {question}. Earlier questions: {recent}"
        hits = self.index.search(query, max(2, k // 2), "policy", request.as_of_date)
        hits += self.index.search(query, k, "precedent", request.as_of_date)
        best: dict[str, Hit] = {}
        for hit in hits:
            current = best.get(hit.chunk.chunk_id)
            if current is None or hit.score > current.score:
                best[hit.chunk.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)[:k]
        return [
            Citation(
                chunk_id=hit.chunk.chunk_id,
                case_id=hit.chunk.case_id,
                title=hit.chunk.title,
                url=hit.chunk.url,
                excerpt=hit.chunk.text[:360],
                score=round(hit.score, 4),
                source_type=hit.chunk.source_type,
            )
            for hit in ranked
        ]

    @staticmethod
    def _fallback(request: PredictionRequest, hits: list[Hit]) -> tuple[str, float, str, list[str]]:
        """Create a conservative prediction when structured generation fails."""
        outcomes = Counter(h.chunk.outcome.value for h in hits if h.chunk.outcome.value != "other")
        if outcomes:
            outcome, count = outcomes.most_common(1)[0]
            confidence = min(0.78, 0.5 + count / max(20, 2 * sum(outcomes.values())))
            reason = "The closest retrieved decisions lean toward this outcome; contextual and human-rights review could still change it."
        else:
            outcome = "overturned" if request.meta_action == "removed" else "upheld"
            confidence = 0.51
            reason = "No generative model or labelled precedents were available, so this is a low-confidence baseline based on the original action."
        policies = list(dict.fromkeys(p for hit in hits for p in hit.chunk.policies))[:5]
        return outcome, confidence, reason, policies

    def predict(self, request: PredictionRequest, system: str = "S2") -> Prediction:
        """Predict the Board outcome with constrained generation and validated citations."""
        if system not in {"S0", "S1", "S2"}:
            raise ValueError("system must be S0, S1, or S2")
        hits = self._hits(request, system)
        context = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.title} — {h.chunk.section}\n{h.chunk.text[:600]}"
            for h in hits[:8]
        )
        prompt = f"{SYSTEM_GUIDANCE}\n\nINPUT\n{self._query(request)}\n\nEVIDENCE\n{context or 'None (closed book)'}"
        fallback = False
        try:
            data = self.llm.generate_json(prompt)
            outcome = str(data["outcome"]).lower()
            if outcome not in {"upheld", "overturned"}:
                raise ValueError("invalid outcome")
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))
            policies = [str(p) for p in data.get("policies", [])]
            requested_citations = [str(item) for item in data.get("citation_ids", [])]
        except (requests.RequestException, KeyError, ValueError, TypeError):
            fallback = True
            outcome, confidence, reasoning, policies = self._fallback(request, hits)
            requested_citations = []
        citations = [
            Citation(
                chunk_id=h.chunk.chunk_id,
                case_id=h.chunk.case_id,
                title=h.chunk.title,
                url=h.chunk.url,
                excerpt=h.chunk.text[:360],
                score=round(h.score, 4),
                source_type=h.chunk.source_type,
            )
            for h in hits
        ]
        valid_ids = {citation.chunk_id for citation in citations}
        citation_ids = [item for item in requested_citations if item in valid_ids]
        if system != "S0" and citations and not citation_ids:
            citation_ids = [citations[0].chunk_id]
        if citation_ids and not any(f"[{item}]" in reasoning for item in citation_ids):
            reasoning = reasoning.rstrip() + f" [{citation_ids[0]}]"
        return Prediction(
            system=system,
            outcome=outcome,
            confidence=max(0, min(1, confidence)),
            reasoning=reasoning,
            policies=policies,
            citations=citations,
            citation_ids=citation_ids,
            model=self.llm.model,
            fallback_used=fallback,
        )

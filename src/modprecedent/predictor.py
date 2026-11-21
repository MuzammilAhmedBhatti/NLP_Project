from __future__ import annotations

from collections import Counter

import requests

from .llm import OllamaClient
from .models import Citation, Prediction, PredictionRequest
from .retrieval import Hit, HybridIndex
from .settings import settings

SYSTEM_GUIDANCE = """You are predicting how Meta's independent Oversight Board would rule.
Return JSON with keys: outcome (upheld or overturned), confidence (0 to 1), reasoning,
and policies (array). Judge consistency with Meta policy, necessity/proportionality, context,
voice, intent, likelihood and severity of harm, and international human rights standards.
Do not treat retrieved precedent outcomes as certain. Cite evidence using [chunk_id]."""


class Predictor:
    def __init__(self, index: HybridIndex | None = None, llm: OllamaClient | None = None):
        self.index = index
        self.llm = llm or OllamaClient(
            model=settings.ollama_model,
            url=settings.ollama_url,
        )

    @staticmethod
    def _query(request: PredictionRequest) -> str:
        return f"Meta action: {request.meta_action}. Content: {request.post_description}"

    def _hits(self, request: PredictionRequest, system: str) -> list[Hit]:
        if not self.index or system == "S0":
            return []
        query = self._query(request)
        hits = self.index.search(query, 4, "policy", request.as_of_date)
        if system == "S2":
            hits += self.index.search(query, 6, "precedent", request.as_of_date)
        return hits

    @staticmethod
    def _fallback(request: PredictionRequest, hits: list[Hit]) -> tuple[str, float, str, list[str]]:
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
        if system not in {"S0", "S1", "S2"}:
            raise ValueError("system must be S0, S1, or S2")
        hits = self._hits(request, system)
        context = "\n\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.title} — {h.chunk.section}\n{h.chunk.text}"
            for h in hits
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
        except (requests.RequestException, KeyError, ValueError, TypeError):
            fallback = True
            outcome, confidence, reasoning, policies = self._fallback(request, hits)
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
        return Prediction(
            system=system,
            outcome=outcome,
            confidence=max(0, min(1, confidence)),
            reasoning=reasoning,
            policies=policies,
            citations=citations,
            model=self.llm.model,
            fallback_used=fallback,
        )

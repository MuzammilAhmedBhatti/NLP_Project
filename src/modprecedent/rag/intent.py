"""Semantic intent routing for follow-up questions."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import settings

INTENT_EXAMPLES = {
    "poster_revision": [
        "What could I have done differently so my post would be accepted?",
        "How should I change my content to avoid removal?",
        "What can the person posting this do to make it compliant?",
        "What should I change so it stays up?",
        "Could I rewrite the caption to make it acceptable?",
    ],
    "explanation": [
        "Why was the decision upheld?",
        "Why did the model reach this ruling?",
        "Explain the reason for this outcome.",
    ],
    "policy": [
        "Which moderation policy applies and why?",
        "Explain the relevant Community Standard.",
        "What rule did this content violate?",
        "Is this hate speech under the policy?",
    ],
    "precedent": [
        "Which past case is most similar?",
        "What retrieved evidence supports this answer?",
        "Show me the strongest precedent.",
        "Where did you get this from?",
        "Which source supports that answer?",
        "What is the closest case?",
    ],
    "counterfactual": [
        "Would the outcome change if this were clearly satire?",
        "What different facts could reverse the ruling?",
        "What if the post included a warning label?",
    ],
    "comparison": [
        "How does this differ from the closest case?",
        "Can you compare this with the closest decision?",
        "Compare this post with the precedent.",
        "What are the similarities and differences?",
        "What do both cases have in common?",
    ],
    "outcome_patterns": [
        "What types of posts are usually upheld?",
        "Which content categories tend to be overturned?",
        "What patterns appear in past Board outcomes?",
        "Are certain kinds of posts more likely to win an appeal?",
        "Are removals often overturned?",
    ],
    "definition": [
        "What does upheld mean?",
        "What does overturned mean in this result?",
        "What is a citation?",
    ],
    "critique": [
        "Do you agree with the ruling?",
        "What is the strongest argument against this decision?",
        "Is this outcome fair?",
    ],
    "conversation_history": [
        "What did I ask before?",
        "Recap our conversation.",
        "What was your previous answer?",
    ],
}


class SemanticIntentRouter:
    """Zero-shot intent classifier using normalized sentence embeddings."""

    def __init__(self, encoder=None, threshold: float = 0.34):
        """Initialize the SemanticIntentRouter instance and its required dependencies."""
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(settings.embedding_model)
        self.encoder = encoder
        self.threshold = threshold
        self.labels = [label for label, examples in INTENT_EXAMPLES.items() for _ in examples]
        examples = [example for values in INTENT_EXAMPLES.values() for example in values]
        self.prototype_vectors = np.asarray(
            self.encoder.encode(examples, normalize_embeddings=True, show_progress_bar=False)
        )

    def route(self, question: str) -> str:
        """Classify a question into a bounded operational intent or general conversation."""
        query = np.asarray(
            self.encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        )[0]
        similarities = self.prototype_vectors @ query
        scores = {
            label: max(
                float(score)
                for candidate, score in zip(self.labels, similarities, strict=True)
                if candidate == label
            )
            for label in INTENT_EXAMPLES
        }
        intent, score = max(scores.items(), key=lambda item: item[1])
        return intent if score >= self.threshold else "general"


@lru_cache(maxsize=1)
def intent_router() -> SemanticIntentRouter:
    """Create and cache the MiniLM semantic intent router."""
    return SemanticIntentRouter()

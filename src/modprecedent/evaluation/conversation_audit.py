"""Reproducible semantic-routing audit for representative follow-up questions."""

from __future__ import annotations

import csv
from pathlib import Path

from ..rag.intent import intent_router

AUDIT_CASES = [
    ("what should I change so it stays up?", "poster_revision"),
    ("could I rewrite the caption?", "poster_revision"),
    ("what did I do wrong?", "poster_revision"),
    ("why was this upheld?", "explanation"),
    ("explain this verdict", "explanation"),
    ("is this hate speech?", "policy"),
    ("which rule applies?", "policy"),
    ("what is the closest case?", "precedent"),
    ("where did you get this from?", "precedent"),
    ("which source supports that?", "precedent"),
    ("what if it was clearly satire?", "counterfactual"),
    ("would a warning label change it?", "counterfactual"),
    ("what do both cases have in common?", "comparison"),
    ("compare this post with the precedent", "comparison"),
    ("what type of posts are usually upheld?", "outcome_patterns"),
    ("are removals often overturned?", "outcome_patterns"),
    ("what does upheld mean?", "definition"),
    ("what is a citation?", "definition"),
    ("do you agree with the ruling?", "critique"),
    ("is this outcome fair?", "critique"),
    ("what did I ask before?", "conversation_history"),
    ("recap our conversation", "conversation_history"),
]


def run_audit(output_dir: Path = Path("reports/generated/latest")) -> dict[str, float | int]:
    """Run the reproducible conversation-intent regression audit."""
    router = intent_router()
    rows = []
    for question, expected in AUDIT_CASES:
        predicted = router.route(question)
        rows.append(
            {
                "question": question,
                "expected_intent": expected,
                "predicted_intent": predicted,
                "correct": predicted == expected,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "conversation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    correct = sum(row["correct"] for row in rows)
    summary = {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}
    (output_dir / "conversation_audit.md").write_text(
        "# Conversation intent audit\n\n"
        f"- Cases: {summary['cases']}\n- Correct: {summary['correct']}\n"
        f"- Accuracy: {summary['accuracy']:.1%}\n\n"
        "This finite regression suite measures intent routing, not unrestricted answer quality.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(run_audit())

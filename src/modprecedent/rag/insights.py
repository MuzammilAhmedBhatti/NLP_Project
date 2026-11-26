"""Corpus-level answers for questions that require aggregate evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..data.models import Decision


def outcome_pattern_answer(decisions: Iterable[Decision], question: str) -> str:
    """Answer corpus-pattern questions with counts and explicit limitations."""
    rows = list(decisions)
    counts = Counter(decision.outcome.value for decision in rows)
    target = "overturned" if "overturn" in question.casefold() else "upheld"
    matching = [decision for decision in rows if decision.outcome.value == target]
    other = "upheld" if target == "overturned" else "overturned"
    titles = ", ".join(f"“{decision.title}”" for decision in matching[:3])
    label = "upholding Meta's action" if target == "upheld" else "overturning Meta's action"
    examples = f" The {target} records are {titles}." if titles else ""
    return (
        f"This dataset does not support a reliable claim about which posts are usually {target}. "
        f"Among {len(rows)} collected decisions, {counts[target]} are marked {target} and "
        f"{counts[other]} are marked {other}, and {counts['other']} are unclassified/other."
        f"{examples} Because the classes are highly imbalanced "
        f"and several records lack policy labels, the defensible conclusion is only that {label} "
        "depends on policy fit, context, harm, and proportionality. A larger balanced review set "
        "would be needed to identify a genuine content-category pattern."
    )

"""Leakage-aware evaluation and reproducible experiments."""

from .metrics import candidate_from_decision, citation_metrics, evaluate, policy_f1, save_report

__all__ = [
    "candidate_from_decision",
    "citation_metrics",
    "evaluate",
    "policy_f1",
    "save_report",
]

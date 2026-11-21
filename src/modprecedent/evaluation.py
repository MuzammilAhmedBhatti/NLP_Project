from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .models import Decision, PredictionRequest
from .predictor import Predictor


def policy_f1(gold: list[list[str]], predicted: list[list[str]]) -> float:
    scores = []
    for expected, actual in zip(gold, predicted, strict=True):
        e, a = {x.lower() for x in expected}, {x.lower() for x in actual}
        scores.append(
            1.0 if not e and not a else (2 * len(e & a) / (len(e) + len(a)) if e or a else 0)
        )
    return sum(scores) / len(scores) if scores else 0.0


def citation_metrics(predictions) -> dict[str, float]:
    if not predictions:
        return {"citation_coverage": 0.0, "citation_relevance": 0.0}
    return {
        "citation_coverage": sum(bool(p.citations) for p in predictions) / len(predictions),
        "citation_relevance": sum(
            sum(c.score for c in p.citations) / max(1, len(p.citations)) for p in predictions
        )
        / len(predictions),
    }


def evaluate(
    predictor: Predictor, cases: list[dict], systems=("S0", "S1", "S2")
) -> tuple[pd.DataFrame, list[dict]]:
    metrics, rows = [], []
    for system in systems:
        predictions = []
        for case in cases:
            request = PredictionRequest.model_validate(case["input"])
            prediction = predictor.predict(request, system)
            predictions.append(prediction)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "system": system,
                    "gold": case["outcome"],
                    **prediction.model_dump(mode="json"),
                }
            )
        gold = [c["outcome"] for c in cases]
        pred = [p.outcome for p in predictions]
        precision, recall, _, _ = precision_recall_fscore_support(
            gold, pred, average="macro", zero_division=0
        )
        citation = citation_metrics(predictions)
        metrics.append(
            {
                "system": system,
                "n": len(cases),
                "accuracy": accuracy_score(gold, pred),
                "macro_f1": f1_score(gold, pred, average="macro"),
                "macro_precision": precision,
                "macro_recall": recall,
                "policy_f1": policy_f1(
                    [c.get("policies", []) for c in cases], [p.policies for p in predictions]
                ),
                **citation,
            }
        )
    return pd.DataFrame(metrics), rows


def candidate_from_decision(decision: Decision) -> dict | None:
    if decision.decision_type.lower() != "standard" or decision.outcome.value == "other":
        return None
    # Only case facts/summary enter the prompt; ruling-specific sections stay excluded.
    description = decision.summary.split("The Oversight Board")[0].strip()
    if len(description) < 80:
        return None
    meta_action = "removed" if "remove" in decision.summary[:500].lower() else "left_up"
    return {
        "case_id": decision.case_id,
        "source_url": str(decision.url),
        "input": {
            "post_description": description[:1800],
            "meta_action": meta_action,
            "as_of_date": decision.published_date.isoformat() if decision.published_date else None,
        },
        "outcome": decision.outcome.value,
        "policies": decision.policies,
        "status": "pending_review",
    }


def save_report(metrics: pd.DataFrame, rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    (directory / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (directory / "report.md").write_text(
        "# Evaluation\n\n"
        + metrics.to_markdown(index=False)
        + "\n\nResults are generated, not claimed production metrics.\n",
        encoding="utf-8",
    )

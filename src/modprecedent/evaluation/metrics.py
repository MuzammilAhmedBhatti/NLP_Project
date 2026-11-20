from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ..data.models import Decision, PredictionRequest
from ..rag import Predictor


def policy_f1(gold: list[list[str]], predicted: list[list[str]]) -> float:
    """Compute multilabel policy F1 across evaluated cases."""
    scores = []
    for expected, actual in zip(gold, predicted, strict=True):
        e, a = {x.lower() for x in expected}, {x.lower() for x in actual}
        scores.append(
            1.0 if not e and not a else (2 * len(e & a) / (len(e) + len(a)) if e or a else 0)
        )
    return sum(scores) / len(scores) if scores else 0.0


def citation_metrics(predictions) -> dict[str, float]:
    """Measure whether generated citations reference retrieved evidence."""
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
    """Compute outcome, policy, and citation metrics for predictions."""
    metrics, rows = [], []
    for system in systems:
        predictions = []
        for case in cases:
            request = PredictionRequest.model_validate(case["input"])
            prediction = predictor.predict(request, system)
            if prediction.fallback_used:
                raise ValueError(
                    "Evaluation requires a working language model. Install the pretrained-model dependencies or configure Ollama."
                )
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
    """Create an evaluation candidate from a normalized Board decision."""
    if decision.decision_type.lower() != "standard" or decision.outcome.value == "other":
        return None
    # Only case facts/summary enter the prompt; ruling-specific sections stay excluded.
    curated = {
        "IG-XLA2OKM2": (
            "An Instagram user uses memes and film clips to describe side effects from prescription medications during pregnancy and confusion after clinically administered fentanyl during labor.",
            "removed",
            "overturned",
        ),
        "FB-46USUH23": (
            "During the Israel-Iran conflict, a Facebook page presents footage of damaged buildings as live reporting from Haifa. Users question its authenticity. A fact-checker identified a similar clip as AI-generated. The post carries no AI label.",
            "left_up",
            "overturned",
        ),
        "FB-LX3CUY5O": (
            "A German news agency shares a photograph from a political protest. An activist tore up his own passport at a press event, and the published image shows the identifying information on its biodata page.",
            "left_up",
            "upheld",
        ),
        "FB-92IOQ88S": (
            "A Facebook user posts an AI-generated video depicting a UK politician making derogatory statements about refugees. The video is presented as authentic political footage and the platform initially leaves it online.",
            "left_up",
            "overturned",
        ),
        "BUN-LJFRC0VQ": (
            "Two Instagram users share eating-disorder awareness posts during an awareness campaign. The posts discuss recovery and direct viewers toward support resources, but Meta removes them under its self-injury and eating-disorders rules.",
            "removed",
            "overturned",
        ),
        "FB-O7AI7UAX": (
            "A Facebook post uses an AI-manipulated video of a famous Brazilian footballer to promote an online game. The download link opens a different game, and the organic post remains online after a user reports it as a scam.",
            "left_up",
            "overturned",
        ),
        "TH-NC063KAD": (
            "A Threads user replies to a news post about the Japanese prime minister and a tax scandal. The reply uses a Japanese expression translated as 'drop dead' alongside criticism and receives no likes before Meta removes it as a threat.",
            "removed",
            "overturned",
        ),
        "BUN-ZHSLVJO2": (
            "Meta disables an account after repeated posts targeting public figures. The reviewed content includes hostile commentary, and the case concerns whether the accumulated violations justified an account-level penalty.",
            "account_action",
            "upheld",
        ),
        "FB-Q8WG6571": (
            "A Facebook post alleges corruption involving the adult children of a politician. Meta leaves the post online, and the case concerns political criticism, the targets' public relevance and possible harassment.",
            "left_up",
            "upheld",
        ),
        "BUN-86TJ0RK5": (
            "Three Facebook posts use the phrase 'From the River to the Sea' in content expressing solidarity with Palestinians. The posts contain no explicit call for violence or exclusion, and Meta leaves them online.",
            "left_up",
            "upheld",
        ),
        "BUN-IJ9KG9E9": (
            "Two Facebook posts reproduce voting information attributed to the Australian Electoral Commission shortly before a referendum. Meta removes the posts under rules addressing election-related harm and misinformation.",
            "removed",
            "upheld",
        ),
        "IG-3R8RQIAQ": (
            "An Instagram beautician posts a video showing make-up preparation for a 14-year-old girl's wedding. The caption promotes bridal services, and the conversation in the video presents marriage as more important than the child's education.",
            "removed",
            "upheld",
        ),
    }
    if decision.case_id not in curated:
        return None
    description, meta_action, outcome = curated[decision.case_id]
    if len(description) < 80:
        return None
    return {
        "case_id": decision.case_id,
        "source_url": str(decision.url),
        "input": {
            "post_description": description[:1800],
            "meta_action": meta_action,
            "as_of_date": decision.published_date.isoformat() if decision.published_date else None,
        },
        "outcome": outcome,
        "policies": decision.policies,
        "status": "approved",
    }


def save_report(metrics: pd.DataFrame, rows: list[dict], directory: Path) -> None:
    """Persist metrics, predictions, charts, and the Markdown evaluation report."""
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    (directory / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (directory / "report.md").write_text(
        "# Evaluation\n\n"
        + metrics.to_markdown(index=False)
        + "\n\nResults are generated, not claimed production metrics.\n",
        encoding="utf-8",
    )

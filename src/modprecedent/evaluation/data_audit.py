"""Generate a reproducible quality report for the processed decision corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ..data.io import read_jsonl
from ..data.models import Decision


def run_data_audit(
    source: Path = Path("data/processed/decisions.jsonl"),
    output_dir: Path = Path("reports/generated/latest"),
) -> pd.DataFrame:
    """Summarize corpus size, labels, decision types, and missing fields."""
    decisions = read_jsonl(source, Decision)
    rows = [
        {
            "case_id": item.case_id,
            "decision_type": item.decision_type,
            "outcome": item.outcome.value,
            "meta_action": item.meta_action,
            "published_date_present": item.published_date is not None,
            "policies_present": bool(item.policies),
            "full_text_characters": len(item.full_text),
        }
        for item in decisions
    ]
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "data_audit.csv", index=False)
    outcomes = Counter(frame["outcome"])
    types = Counter(frame["decision_type"])
    duplicate_ids = int(frame["case_id"].duplicated().sum())
    report = (
        "# Corpus quality audit\n\n"
        f"- Decisions: {len(frame)}\n"
        f"- Standard decisions: {types['Standard']}\n"
        f"- Summary decisions: {types['Summary']}\n"
        f"- Outcomes: {outcomes['upheld']} upheld, {outcomes['overturned']} overturned, "
        f"{outcomes['other']} mixed/advisory\n"
        f"- Records with policy labels: {int(frame['policies_present'].sum())}\n"
        f"- Records with known original action: "
        f"{int((frame['meta_action'] != 'unknown').sum())}\n"
        f"- Duplicate case IDs: {duplicate_ids}\n"
    )
    (output_dir / "data_audit.md").write_text(report, encoding="utf-8")
    return frame


if __name__ == "__main__":
    result = run_data_audit()
    print(f"Audited {len(result)} decisions")

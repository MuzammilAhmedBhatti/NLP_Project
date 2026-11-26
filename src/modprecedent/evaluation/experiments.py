"""Reconstructed retrieval sensitivity experiment."""

import json
from pathlib import Path

import pandas as pd

from ..data.io import read_jsonl
from ..data.models import Decision, PredictionRequest
from ..data.processing import decision_to_chunks
from ..rag.retrieval import HybridIndex


def run():
    """Execute the leakage-aware S0, S1, and S2 evaluation experiment."""
    cases = [
        json.loads(s)
        for s in Path("data/evaluation/approved_cases.jsonl").read_text().splitlines()
        if s
    ]
    if not cases or any(c["status"] != "approved" for c in cases):
        raise ValueError("An approved review set is required.")
    decisions = read_jsonl(Path("data/processed/decisions.jsonl"), Decision)
    heldout = {c["case_id"] for c in cases}
    comparisons = []
    requests_by_case = {
        case["case_id"]: PredictionRequest.model_validate(case["input"]) for case in cases
    }
    cutoff = min(request.as_of_date for request in requests_by_case.values())
    prior = [
        decision
        for decision in decisions
        if decision.case_id not in heldout
        and decision.published_date
        and decision.published_date < cutoff
    ]
    sensitivity_indexes = {}
    for size, dense in [(480, False), (900, False), (900, True)]:
        index = HybridIndex()
        index.build([part for d in prior for part in decision_to_chunks(d, size)], dense=dense)
        if dense and index.dense is None:
            raise RuntimeError("Install the rag extra before running the experiment.")
        sensitivity_indexes[(size, dense)] = index
    for c in cases:
        req = requests_by_case[c["case_id"]]
        for (size, dense), index in sensitivity_indexes.items():
            for diverse in [False, True]:
                index.diversify = diverse
                for k in [3, 6]:
                    hits = index.search(req.post_description, k, "precedent")
                    comparisons.append(
                        {
                            "case_id": c["case_id"],
                            "chunk_size": size,
                            "backend": "hybrid" if dense else "lexical",
                            "rerank": diverse,
                            "k": k,
                            "distinct_cases": len({h.chunk.case_id for h in hits}),
                            "policy_hit": int(
                                any(set(h.chunk.policies) & set(c["policies"]) for h in hits)
                            ),
                        }
                    )

    out = Path("reports/generated/latest")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(
        out / "retrieval.csv", index=False
    )


if __name__ == "__main__":
    run()

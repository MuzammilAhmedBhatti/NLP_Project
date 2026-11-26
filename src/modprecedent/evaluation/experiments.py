"""Reproduce the three-system study and retrieval sensitivity report."""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..data.io import read_jsonl
from ..data.models import Decision, PredictionRequest
from ..data.processing import decision_to_chunks
from ..rag import HybridIndex, Predictor
from .metrics import policy_f1, save_report


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
    rows, comparisons = [], []
    llm = Predictor().llm
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
    index = sensitivity_indexes[(900, True)]
    index.diversify = False
    for c in cases:
        req = requests_by_case[c["case_id"]]
        for system in ("S0", "S1", "S2"):
            p = Predictor(index, llm).predict(req, system)
            if p.fallback_used:
                raise RuntimeError("Generative inference failed")
            rows.append(
                dict(
                    case_id=c["case_id"],
                    gold=c["outcome"],
                    gold_policies=c["policies"],
                    **p.model_dump(mode="json"),
                )
            )
    metrics = []
    for system in ("S0", "S1", "S2"):
        subset = [r for r in rows if r["system"] == system]
        gold = [r["gold"] for r in subset]
        pred = [r["outcome"] for r in subset]
        cites = [
            (
                set(r.get("citation_ids", []))
                | set(re.findall(r"\[([0-9a-f]{16})\]", r["reasoning"])),
                {c["chunk_id"] for c in r["citations"]},
            )
            for r in subset
        ]
        metrics.append(
            {
                "system": system,
                "n": len(subset),
                "accuracy": accuracy_score(gold, pred),
                "macro_f1": f1_score(
                    gold, pred, labels=["upheld", "overturned"], average="macro", zero_division=0
                ),
                "policy_f1": policy_f1(
                    [r["gold_policies"] for r in subset], [r["policies"] for r in subset]
                ),
                "valid_inline_citation_coverage": sum(bool(a & b) for a, b in cites) / len(cites),
            }
        )
    out = Path("reports/generated/latest")
    frame = pd.DataFrame(metrics)
    save_report(frame, rows, out)
    pd.DataFrame(comparisons).to_csv(out / "retrieval.csv", index=False)
    case_count = len(cases)
    bars = "".join(
        f'<text x="12" y="{36 + i * 46}">{r["system"]}</text><rect x="58" y="{19 + i * 46}" width="{300 * r["accuracy"]}" height="22" fill="#456544"/><text x="370" y="{36 + i * 46}">{r["accuracy"]:.0%}</text>'
        for i, r in enumerate(metrics)
    )
    (out / "accuracy.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="440" height="155"><title>Outcome accuracy on {case_count} approved cases</title>{bars}</svg>'
    )
    with (out / "report.md").open("a", encoding="utf8") as f:
        f.write(
            f"\n![Accuracy on {case_count} approved decisions](accuracy.svg)\n\nGenerator: {llm.model}. The retrieval corpus excludes every reviewed case and uses only documents earlier than the earliest test case. Citation coverage checks identifiable passage references, not evidentiary support. Retrieval sensitivity measures policy overlap and source diversity.\n"
        )
    summarize(out)
    print(frame.to_string(index=False))


def summarize(out: Path):
    """Summarize one evaluated system into report-ready metrics."""
    rows = json.loads((out / "predictions.json").read_text(encoding="utf8"))
    scores = pd.read_csv(out / "metrics.csv")

    def policies(row, key):
        """Normalize a policy collection for evaluation."""
        return {value.strip().casefold() for value in row[key]}

    scores["policy_exact_accuracy"] = [
        sum(
            policies(r, "policies") == policies(r, "gold_policies")
            for r in rows
            if r["system"] == system
        )
        / sum(r["system"] == system for r in rows)
        for system in scores["system"]
    ]
    scores.to_csv(out / "metrics.csv", index=False)
    case_count = int(scores["n"].max())
    corpus_size = len(read_jsonl(Path("data/processed/decisions.jsonl"), Decision))
    summary = (
        pd.read_csv(out / "retrieval.csv")
        .groupby(["backend", "chunk_size", "k", "rerank"], as_index=False)[
            ["distinct_cases", "policy_hit"]
        ]
        .mean()
    )
    (out / "report.md").write_text(
        "# Approved Standard-decision study\n\n"
        + scores.to_markdown(index=False)
        + f"\n\n![Accuracy on {case_count} approved decisions](accuracy.svg)\n\n"
        + f"Generator: `{rows[0]['model'] if rows else 'unknown'}`. Dense retriever: `sentence-transformers/all-MiniLM-L6-v2`. "
        + f"The processed corpus contains {corpus_size} decisions and the human-approved review set contains {case_count} cases. "
        + "The configured pretrained instruction model generated the outcome, rationale, policy labels and citation IDs. "
        + f"The retrieval corpus excluded all {case_count} test cases and used only documents published before the earliest test case, including for TF-IDF fitting. "
        + f"With {case_count} examples, accuracy changes in steps of {100 / case_count:.1f} percentage points. The scores do not establish generalization. "
        + "Inline citation coverage counts rationales containing a valid retrieved passage ID. It does not measure entailment. "
        + "Policy exact accuracy requires matching the complete normalized label set.\n\n"
        + "## Retrieval sensitivity\n\n"
        + summary.to_markdown(index=False)
        + "\n\nDistinct cases measures source diversity. Policy hit measures whether any retrieved source shares an annotated policy. "
        + "Compare the two lexical sizes for chunk length, and the two backends at 900 for embedding effects. "
        + "The reranker promotes the first passage from each distinct decision.\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    run()

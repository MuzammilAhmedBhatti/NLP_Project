"""Reproduce the three-system study and retrieval sensitivity report."""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .evaluation import policy_f1, save_report
from .io import read_jsonl
from .local_model import local_model
from .models import Decision, PredictionRequest
from .predictor import Predictor
from .processing import decision_to_chunks
from .retrieval import HybridIndex


def run():
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
    llm = local_model()
    for c in cases:
        req = PredictionRequest.model_validate(c["input"])
        prior = [
            d
            for d in decisions
            if d.case_id not in heldout and d.published_date and d.published_date < req.as_of_date
        ]
        for size, dense in [(480, False), (900, False), (900, True)]:
            index = HybridIndex()
            index.build([part for d in prior for part in decision_to_chunks(d, size)], dense=dense)
            if dense and index.dense is None:
                raise RuntimeError("Install the rag extra before running the experiment.")
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
        index.diversify = False
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
                set(re.findall(r"\[([0-9a-f]{16})\]", r["reasoning"])),
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
    bars = "".join(
        f'<text x="12" y="{36 + i * 46}">{r["system"]}</text><rect x="58" y="{19 + i * 46}" width="{300 * r["accuracy"]}" height="22" fill="#456544"/><text x="370" y="{36 + i * 46}">{r["accuracy"]:.0%}</text>'
        for i, r in enumerate(metrics)
    )
    (out / "accuracy.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="440" height="155"><title>Outcome accuracy on three approved cases</title>{bars}</svg>'
    )
    with (out / "report.md").open("a", encoding="utf8") as f:
        f.write(
            "\n![Accuracy on three approved decisions](accuracy.svg)\n\nFLAN-T5-small, forced-choice decoding. Each fold excludes all reviewed cases and fits retrieval only on earlier dated documents. Three cases support a demonstration only. Citation coverage checks identifiable passage references, not evidentiary support. Retrieval sensitivity measures policy overlap and source diversity; the two chunk sizes also use different encoders.\n"
        )
    summarize(out)
    print(frame.to_string(index=False))


def summarize(out: Path):
    rows = json.loads((out / "predictions.json").read_text(encoding="utf8"))
    scores = pd.read_csv(out / "metrics.csv")

    def policies(row, key):
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
        + "\n\n![Accuracy on three approved decisions](accuracy.svg)\n\n"
        + "FLAN-T5-small scored two possible outcomes and generated a rationale and policy for each input. "
        + "Each fold excluded all three test cases and used only earlier dated documents, including for TF-IDF fitting. "
        + "Three examples give coarse accuracy steps of 33 percentage points. The scores do not establish generalization. "
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

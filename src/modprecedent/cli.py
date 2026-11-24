from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import settings
from .data.collector import DecisionCollector
from .data.io import read_jsonl, write_jsonl
from .data.models import Decision, PredictionRequest
from .data.processing import decision_to_chunks, split_sections
from .evaluation import candidate_from_decision, evaluate, save_report
from .rag import HybridIndex, Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def decisions_path() -> Path:
    """Resolve the configured path to the processed decision corpus."""
    real = Path("data/raw/decisions.jsonl")
    processed = Path("data/processed/decisions.jsonl")
    return (
        real
        if real.exists()
        else processed
        if processed.exists()
        else Path("data/processed/sample_decisions.jsonl")
    )


def load_index() -> HybridIndex:
    """Load a saved retrieval index or build one from the processed corpus."""
    path = Path("data/index")
    if (path / "metadata.json").exists():
        return HybridIndex.load(path)
    decisions = read_jsonl(decisions_path(), Decision)
    chunks = [c for d in decisions for c in decision_to_chunks(d)]
    index = HybridIndex(settings.embedding_model)
    index.build(chunks)
    return index


def main() -> None:
    """Run the command-line entry point for the selected workflow."""
    parser = argparse.ArgumentParser(prog="modprecedent")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--max-pages", type=int, default=5)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--refresh-cached", action="store_true")
    sub.add_parser("process")
    index = sub.add_parser("index")
    index.add_argument("--dense", action="store_true")
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--output", default="data/evaluation/candidate_cases.jsonl")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--output", default="reports/generated/latest")
    predict = sub.add_parser("predict")
    predict.add_argument("--text")
    predict.add_argument(
        "--action", choices=["removed", "left_up", "warning", "account_action"], default="removed"
    )
    predict.add_argument("--system", choices=["S0", "S1", "S2"], default="S2")
    args = parser.parse_args()
    if args.command == "collect":
        collector = DecisionCollector()
        decisions = (
            collector.refresh_cached()
            if args.refresh_cached
            else collector.collect(args.max_pages, args.limit)
        )
        print(f"Collected {len(decisions)} decisions")
    elif args.command == "process":
        decisions = read_jsonl(decisions_path(), Decision)
        for d in decisions:
            d.sections = split_sections(d.full_text)
        write_jsonl(Path("data/processed/decisions.jsonl"), decisions)
        print(f"Processed {len(decisions)} decisions")
    elif args.command == "index":
        source = Path("data/processed/decisions.jsonl")
        decisions = read_jsonl(source if source.exists() else decisions_path(), Decision)
        chunks = [c for d in decisions for c in decision_to_chunks(d)]
        idx = HybridIndex(settings.embedding_model)
        idx.build(chunks, args.dense)
        idx.save(Path("data/index"))
        print(f"Indexed {len(chunks)} chunks")
    elif args.command == "candidates":
        rows = [
            x for d in read_jsonl(decisions_path(), Decision) if (x := candidate_from_decision(d))
        ]
        print(f"Wrote {write_jsonl(Path(args.output), rows)} leakage-safe evaluation cases")
    elif args.command == "evaluate":
        cases = [
            json.loads(x)
            for x in Path(args.dataset).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        pending = [c["case_id"] for c in cases if c.get("status") != "approved"]
        if pending:
            raise SystemExit(f"Refusing final evaluation: unapproved cases: {', '.join(pending)}")
        metrics, rows = evaluate(Predictor(load_index()), cases)
        save_report(metrics, rows, Path(args.output))
        print(metrics.to_string(index=False))
    elif args.command == "predict":
        text = args.text or input("Describe the post: ")
        result = Predictor(load_index()).predict(
            PredictionRequest(post_description=text, meta_action=args.action), args.system
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .config.settings import settings
from .data.io import read_jsonl
from .data.models import Decision
from .data.processing import decision_to_chunks
from .rag.retrieval import HybridIndex


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
    parser = argparse.ArgumentParser(prog="modprecedent")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    if args.command == "index":
        source = Path("data/processed/decisions.jsonl")
        decisions = read_jsonl(source if source.exists() else decisions_path(), Decision)
        chunks = [c for d in decisions for c in decision_to_chunks(d)]
        idx = HybridIndex(settings.embedding_model)
        idx.build(chunks, args.dense)
        idx.save(Path("data/index"))
        print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class HybridIndex:
    """Portable TF-IDF index with optional dense Sentence Transformer retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.sparse = None
        self.dense: np.ndarray | None = None
        self._encoder = None
        self.diversify = False

    def build(self, chunks: list[Chunk], dense: bool = False) -> None:
        self.chunks = chunks
        corpus = [f"{c.title} {' '.join(c.policies)} {c.section} {c.text}" for c in chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True
        )
        self.sparse = self.vectorizer.fit_transform(corpus)
        if dense:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
                self.dense = self._encoder.encode(
                    corpus, normalize_embeddings=True, show_progress_bar=False
                )
            except ImportError:
                self.dense = None

    def search(
        self,
        query: str,
        k: int = 6,
        source_type: str | None = None,
        before: date | None = None,
        exclude_case_id: str | None = None,
    ) -> list[Hit]:
        if not self.chunks or self.vectorizer is None:
            return []
        sparse_scores = cosine_similarity(self.vectorizer.transform([query]), self.sparse)[0]
        scores = sparse_scores.copy()
        if self.dense is not None:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
            q = self._encoder.encode([query], normalize_embeddings=True)[0]
            scores = 0.35 * sparse_scores + 0.65 * np.dot(self.dense, q)
        valid = []
        for i, chunk in enumerate(self.chunks):
            if source_type and chunk.source_type != source_type:
                continue
            if exclude_case_id and chunk.case_id == exclude_case_id:
                continue
            if before and (chunk.published_date is None or chunk.published_date >= before):
                continue
            valid.append(i)
        ranked = sorted(valid, key=lambda i: float(scores[i]), reverse=True)
        if self.diversify:
            seen = set()
            distinct = []
            repeated = []
            for i in ranked:
                case = self.chunks[i].case_id
                (repeated if case in seen else distinct).append(i)
                seen.add(case)
            ranked = distinct + repeated
        ranked = ranked[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "sparse": self.sparse}, directory / "sparse.joblib"
        )
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks]), encoding="utf-8"
        )
        if self.dense is not None:
            np.save(directory / "dense.npy", self.dense)
        else:
            (directory / "dense.npy").unlink(missing_ok=True)
        (directory / "metadata.json").write_text(
            json.dumps({"model_name": self.model_name}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> HybridIndex:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(metadata["model_name"])
        stored = joblib.load(directory / "sparse.joblib")
        obj.vectorizer, obj.sparse = stored["vectorizer"], stored["sparse"]
        obj.chunks = [
            Chunk.model_validate(x)
            for x in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        ]
        dense_path = directory / "dense.npy"
        if dense_path.exists():
            obj.dense = np.load(dense_path)
        return obj

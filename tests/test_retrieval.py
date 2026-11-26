from datetime import date

from modprecedent.data.models import Chunk
from modprecedent.rag import HybridIndex


def chunk(cid, text, published="2020-01-01", source="precedent"):
    return Chunk(
        chunk_id=cid,
        case_id=cid,
        source_type=source,
        title=cid,
        text=text,
        url="https://example.com",
        published_date=published,
    )


def test_retrieval_and_temporal_filter():
    idx = HybridIndex()
    idx.build(
        [
            chunk("old", "rhetorical political threat context"),
            chunk("new", "rhetorical threat", "2025-01-01"),
        ]
    )
    hits = idx.search("rhetorical threat", before=date(2024, 1, 1))
    assert [h.chunk.case_id for h in hits] == ["old"]


def test_case_exclusion():
    idx = HybridIndex()
    idx.build([chunk("a", "hate speech slur"), chunk("b", "hate speech slur")])
    assert all(h.chunk.case_id != "a" for h in idx.search("hate speech", exclude_case_id="a"))

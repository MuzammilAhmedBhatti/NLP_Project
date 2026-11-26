from modprecedent.data.models import Citation
from modprecedent.ui.web import _render_chat_answer


def test_chat_citation_uses_readable_label_and_keeps_traceability():
    citation = Citation(
        chunk_id="18b49f1022986d50",
        case_id="case",
        title="AI-Generated Video in Israel-Iran Conflict",
        url="https://example.com/case",
        excerpt="Evidence",
        score=0.8,
        source_type="precedent",
    )
    rendered = _render_chat_answer(
        "The evidence supports this conclusion [18b49f1022986d50].",
        {citation.chunk_id: citation},
    )
    assert "[Source 1]" in rendered
    assert "AI-Generated Video in Israel-Iran Conflict" in rendered
    assert "internal ID retained for evaluation" in rendered
    assert ">[18b49f1022986d50]<" not in rendered


def test_multiple_prediction_citations_receive_stable_source_numbers():
    first = Citation(
        chunk_id="18b49f1022986d50",
        case_id="case-1",
        title="First decision",
        url="https://example.com/first",
        excerpt="Evidence",
        score=0.8,
        source_type="policy",
    )
    second = first.model_copy(
        update={
            "chunk_id": "7918f20084769ad8",
            "case_id": "case-2",
            "title": "Second decision",
            "url": "https://example.com/second",
        }
    )
    rendered = _render_chat_answer(
        "First claim [18b49f1022986d50]. Second claim [7918f20084769ad8].",
        {first.chunk_id: first, second.chunk_id: second},
    )
    assert "[Source 1]" in rendered and "[Source 2]" in rendered
    assert "First decision" in rendered and "Second decision" in rendered

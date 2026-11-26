from modprecedent.data.collector import DecisionCollector
from modprecedent.data.models import Decision
from modprecedent.data.processing import decision_to_chunks, sliding_chunks, split_sections


def test_sections_and_chunks():
    text = "Case description\n" + "context words " * 30 + "\nDecision\n" + "analysis words " * 30
    sections = split_sections(text)
    assert "Case description" in sections
    decision = Decision(case_id="X", title="Test", url="https://example.com", full_text=text)
    chunks = decision_to_chunks(decision, size=300, overlap=30)
    assert chunks and all(c.case_id == "X" for c in chunks)


def test_chunk_overlap_does_not_loop():
    assert len(list(sliding_chunks("word " * 500, size=300, overlap=60))) > 1


def test_page_label_wins_over_unrelated_outcome_mentions():
    text = "Upheld\nSummary\nThe Board later discussed a case where it overturned Meta."
    assert DecisionCollector._extract_outcome(text).value == "upheld"


def test_mixed_bundle_is_not_forced_into_one_outcome():
    assert DecisionCollector._extract_outcome("Upheld\nOverturned").value == "other"

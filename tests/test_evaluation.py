from modprecedent.data.models import Decision
from modprecedent.evaluation import candidate_from_decision, policy_f1


def test_policy_f1():
    assert policy_f1([["Hate speech"]], [["hate speech"]]) == 1


def test_candidate_excludes_ruling_phrase():
    decision = Decision(
        case_id="IG-XLA2OKM2",
        title="x",
        url="https://example.com",
        decision_type="Standard",
        outcome="overturned",
        published_date="2024-01-01",
        summary="A post described a contested political event in substantial contextual detail without an explicit outcome. The Oversight Board overturned Meta.",
    )
    decision.meta_action = "removed"
    row = candidate_from_decision(decision)
    assert row and "overturned" not in row["input"]["post_description"].lower()


def test_uncurated_case_is_excluded_from_leakage_safe_review_set():
    decision = Decision(
        case_id="unknown-case",
        title="Unknown",
        url="https://example.com",
        decision_type="Standard",
        outcome="upheld",
    )
    assert candidate_from_decision(decision) is None

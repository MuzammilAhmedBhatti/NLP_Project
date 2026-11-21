from modprecedent.evaluation import candidate_from_decision, policy_f1
from modprecedent.models import Decision


def test_policy_f1():
    assert policy_f1([["Hate speech"]], [["hate speech"]]) == 1


def test_candidate_excludes_ruling_phrase():
    decision = Decision(
        case_id="x",
        title="x",
        url="https://example.com",
        decision_type="Standard",
        outcome="overturned",
        published_date="2024-01-01",
        summary="A post described a contested political event in substantial contextual detail without an explicit outcome. The Oversight Board overturned Meta.",
    )
    row = candidate_from_decision(decision)
    assert row and "overturned" not in row["input"]["post_description"].lower()

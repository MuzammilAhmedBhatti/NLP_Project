import json
from pathlib import Path

from modprecedent.models import PredictionRequest


def test_reviewed_inputs_have_cutoffs_and_hide_rulings():
    cases = [
        json.loads(line)
        for line in Path("data/evaluation/approved_cases.jsonl").read_text().splitlines()
    ]
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["outcome"] for case in cases} == {"upheld", "overturned"}
    for case in cases:
        assert case["status"] == "approved"
        request = PredictionRequest.model_validate(case["input"])
        assert request.as_of_date
        assert all(
            word not in request.post_description.lower()
            for word in ("board", "overturned", "upheld")
        )

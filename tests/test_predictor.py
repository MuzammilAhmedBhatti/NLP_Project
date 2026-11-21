from modprecedent.models import Chunk, PredictionRequest
from modprecedent.predictor import Predictor
from modprecedent.retrieval import HybridIndex


class BrokenLLM:
    model = "offline"

    def generate_json(self, prompt):
        raise ValueError("offline")


def test_predictor_offline_fallback():
    chunk = Chunk(
        chunk_id="c",
        case_id="case",
        source_type="precedent",
        title="Case",
        text="neutral news report",
        url="https://example.com",
        outcome="overturned",
    )
    index = HybridIndex()
    index.build([chunk])
    request = PredictionRequest(
        post_description="A neutral newspaper report quoted a political group.",
        meta_action="removed",
    )
    result = Predictor(index, BrokenLLM()).predict(request, "S2")
    assert result.outcome == "overturned" and result.fallback_used and result.citations

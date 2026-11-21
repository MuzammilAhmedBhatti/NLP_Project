import numpy as np

from modprecedent.data.models import Chunk, Decision, Prediction, PredictionRequest
from modprecedent.rag import (
    ConversationChain,
    HybridIndex,
    Predictor,
    SemanticIntentRouter,
    outcome_pattern_answer,
)
from modprecedent.rag.intent import INTENT_EXAMPLES


class BrokenLLM:
    model = "offline"

    def generate_json(self, prompt):
        raise ValueError("offline")

    def generate_text(self, prompt):
        raise ValueError("offline")


class RecordingLLM:
    model = "test-model"

    def __init__(self):
        self.prompt = ""

    def generate_text(self, prompt):
        self.prompt = prompt
        return "The retrieved precedent supports this explanation [c]."


class WeakFlanLLM(RecordingLLM):
    model = "google/flan-t5-small"

    def generate_text(self, prompt):
        self.prompt = prompt
        return "Outcome: upheld"


class PosterIntentEncoder:
    def encode(self, texts, **kwargs):
        size = sum(len(examples) for examples in INTENT_EXAMPLES.values())
        if len(texts) == size:
            return np.eye(size)
        vector = np.zeros(size)
        vector[0] = 1
        return np.asarray([vector])


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
    assert result.citation_ids == ["c"]
    assert "[c]" in result.reasoning


def test_langchain_follow_up_contains_case_history_and_evidence():
    llm = RecordingLLM()
    request = PredictionRequest(
        post_description="A news report includes identifying information from a passport.",
        meta_action="left_up",
    )
    prediction = Prediction(
        system="S2",
        outcome="upheld",
        confidence=0.7,
        reasoning="Newsworthiness affects the proportionality assessment.",
        policies=["Privacy"],
        citations=[
            {
                "chunk_id": "c",
                "case_id": "case",
                "title": "Passport case",
                "url": "https://example.com",
                "excerpt": "The Board weighed privacy against public interest.",
                "score": 0.8,
                "source_type": "precedent",
            }
        ],
        model=llm.model,
    )
    answer = ConversationChain(llm).answer(
        request,
        prediction,
        "Why does privacy matter?",
        [("What was predicted?", "The action was likely upheld.")],
    )
    assert answer.endswith("[c].")
    assert "Why does privacy matter?" in llm.prompt
    assert "What was predicted?" in llm.prompt
    assert "The Board weighed privacy" in llm.prompt


def test_weak_flan_chat_response_becomes_a_grounded_explanation():
    llm = WeakFlanLLM()
    request = PredictionRequest(
        post_description="A news report includes identifying information from a passport.",
        meta_action="left_up",
    )
    prediction = Prediction(
        system="S2",
        outcome="upheld",
        confidence=0.7,
        reasoning="The public-interest context outweighed the privacy interference.",
        policies=["Privacy"],
        citations=[
            {
                "chunk_id": "privacy-1",
                "case_id": "case",
                "title": "Passport case",
                "url": "https://example.com",
                "excerpt": "The Board balanced privacy against public interest.",
                "score": 0.8,
                "source_type": "precedent",
            }
        ],
        model=llm.model,
    )
    answer = ConversationChain(llm).answer(request, prediction, "Why was it upheld?")
    assert "uphold Meta's decision to keep the post online" in answer
    assert "privacy-1" in answer
    assert "Meta action:" not in answer
    assert answer != "Outcome: upheld"


def test_repeated_chat_answer_uses_question_specific_counterfactual():
    llm = RecordingLLM()
    llm.generate_text = lambda prompt: "The post was misleading."
    request = PredictionRequest(
        post_description="An AI-generated conflict video was presented as live reporting.",
        meta_action="removed",
    )
    prediction = Prediction(
        system="S2",
        outcome="upheld",
        confidence=0.7,
        reasoning="Viewers could mistake synthetic footage for live reporting.",
        policies=["Misinformation"],
        model=llm.model,
    )
    answer = ConversationChain(llm).answer(
        request,
        prediction,
        "Could the answer change if it was clearly satire?",
        [("Why was it upheld?", "The post was misleading.")],
    )
    assert "satire label" in answer
    assert answer != "The post was misleading."


def test_first_person_question_advises_poster_instead_of_meta():
    llm = RecordingLLM()
    llm.generate_text = lambda prompt: "Meta should have added a warning to the post."
    request = PredictionRequest(
        post_description="An AI-generated conflict video was presented as live eyewitness footage.",
        meta_action="removed",
    )
    prediction = Prediction(
        system="S2",
        outcome="upheld",
        confidence=0.7,
        reasoning="The presentation could deceive viewers.",
        policies=["Misinformation"],
        model=llm.model,
    )
    answer = ConversationChain(llm).answer(
        request,
        prediction,
        "What could I have done differently for it to be accepted?",
    )
    assert "You could have clearly labeled" in answer
    assert "original source" in answer
    assert "Meta should" not in answer


def test_semantic_router_classifies_paraphrased_poster_request():
    router = SemanticIntentRouter(PosterIntentEncoder())
    assert router.route("How might the uploader revise this?") == "poster_revision"


def test_outcome_pattern_answer_reports_corpus_limits():
    decisions = [
        Decision(case_id="u", title="Upheld case", url="https://example.com/u", outcome="upheld"),
        Decision(
            case_id="o1",
            title="First overturned",
            url="https://example.com/o1",
            outcome="overturned",
        ),
        Decision(
            case_id="o2",
            title="Second overturned",
            url="https://example.com/o2",
            outcome="overturned",
        ),
    ]
    answer = outcome_pattern_answer(decisions, "What types of posts are usually upheld?")
    assert "1 are marked upheld" in answer
    assert "2 are marked overturned" in answer
    assert "does not support a reliable claim" in answer


def test_follow_up_retrieval_uses_current_question():
    chunks = [
        Chunk(
            chunk_id="privacy",
            case_id="p",
            source_type="precedent",
            title="Privacy case",
            text="passport biodata privacy public interest",
            url="https://example.com/p",
        ),
        Chunk(
            chunk_id="satire",
            case_id="s",
            source_type="precedent",
            title="Satire case",
            text="satire parody humorous context",
            url="https://example.com/s",
        ),
    ]
    index = HybridIndex()
    index.build(chunks)
    predictor = Predictor(index, BrokenLLM())
    request = PredictionRequest(
        post_description="A post includes a photograph and contextual caption.",
        meta_action="removed",
    )
    citations = predictor.follow_up_citations(request, "Would clear satire change the result?")
    assert citations[0].chunk_id == "satire"


def test_definition_and_history_do_not_depend_on_local_model():
    request = PredictionRequest(
        post_description="A sufficiently detailed post description.", meta_action="removed"
    )
    prediction = Prediction(
        system="S2", outcome="upheld", confidence=0.7, reasoning="Policy fit.", model="offline"
    )
    chain = ConversationChain(BrokenLLM())
    definition = chain.answer(request, prediction, "What does upheld mean?", intent="definition")
    history = chain.answer(
        request,
        prediction,
        "What did I ask before?",
        [("Why?", "Because policy applied.")],
        "conversation_history",
    )
    assert "removal" in definition and "stays in place" in definition
    assert 'you asked "Why?"' in history


def test_prompt_leak_is_rejected_for_grounded_fallback():
    llm = RecordingLLM()
    llm.generate_text = lambda prompt: "Meta action: removed Content: copied prompt text"
    request = PredictionRequest(
        post_description="An AI-generated conflict clip was shown as real.", meta_action="removed"
    )
    prediction = Prediction(
        system="S2",
        outcome="upheld",
        confidence=0.7,
        reasoning="It could mislead viewers.",
        policies=["Misinformation"],
        model=llm.model,
    )
    answer = ConversationChain(llm).answer(
        request, prediction, "Why was it upheld?", intent="explanation"
    )
    assert "Meta action:" not in answer
    assert "model's estimate" in answer


def test_general_conversation_is_handled_by_the_language_model():
    class ConversationalLLM(RecordingLLM):
        def generate_text(self, prompt):
            self.prompt = prompt
            return "I'm doing well and ready to help you today."

    request = PredictionRequest(
        post_description="A sufficiently detailed social media post description.",
        meta_action="removed",
    )
    prediction = Prediction(
        system="S2",
        outcome="overturned",
        confidence=0.7,
        reasoning="Context supports restoration.",
        model="test-model",
    )
    llm = ConversationalLLM()
    answer = ConversationChain(llm).answer(request, prediction, "how r u", intent="general")
    assert "doing well" in answer
    assert "reverse Meta" not in answer
    assert "Respond naturally to greetings" in llm.prompt

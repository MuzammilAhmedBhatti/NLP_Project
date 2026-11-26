"""LangChain-powered, evidence-grounded follow-up conversation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from ..data.models import Prediction, PredictionRequest
from .predictor import GenerativeClient

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are ModPrecedent, an evidence-grounded assistant explaining a predicted "
                "Meta Oversight Board ruling. Respond naturally to greetings and ordinary "
                "conversation. For unrelated general questions, answer briefly and offer to "
                "return to the analyzed case. For case-specific claims, rely only on the case, "
                "moderation policies, human-rights considerations, and retrieved precedents. Distinguish "
                "the model prediction from a confirmed Board ruling. Use [chunk_id] citations "
                "when the evidence supports the answer. If evidence is insufficient, say so. "
                "Track who the user is asking about. When they ask what 'I' or the poster could "
                "do differently, advise the content creator rather than Meta."
            ),
        ),
        (
            "human",
            (
                "FOLLOW-UP QUESTION\n{question}\n\nANALYZED CASE\n{case}\n\n"
                "MODEL PREDICTION\n{prediction}\n\n"
                "RETRIEVED EVIDENCE\n{evidence}\n\nCONVERSATION\n{history}\n\n"
                "Answer the follow-up question directly in two to four sentences."
            ),
        ),
    ]
)


class IntentRouter(Protocol):
    def route(self, question: str) -> str:
        """Classify a question into a bounded operational intent or general conversation."""
        ...


class ConversationChain:
    """Run case-grounded follow-up questions through a LangChain LCEL pipeline."""

    def __init__(self, client: GenerativeClient, router: IntentRouter | None = None):
        """Initialize the ConversationChain instance and its required dependencies."""
        self.client = client
        self.router = router
        self.chain = (
            FOLLOW_UP_PROMPT
            | RunnableLambda(lambda prompt: self.client.generate_text(prompt.to_string()))
            | StrOutputParser()
        )

    def answer(
        self,
        request: PredictionRequest,
        prediction: Prediction,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        intent: str | None = None,
    ) -> str:
        """Answer a follow-up using case context, fresh evidence, and conversation history."""
        evidence = "\n\n".join(
            f"[{citation.chunk_id}] {citation.title}\n{citation.excerpt[:240]}"
            for citation in prediction.citations[:4]
        )
        transcript = "\n".join(
            f"User: {user_message}\nAssistant: {assistant_message}"
            for user_message, assistant_message in history[-6:]
        )
        intent = intent or (
            self.router.route(question) if self.router else self._heuristic_intent(question)
        )
        if intent in {
            "poster_revision",
            "definition",
            "critique",
            "conversation_history",
        }:
            return self._grounded_fallback(request, prediction, question, history, intent)
        response = self.chain.invoke(
            {
                "case": (
                    f"Meta action: {request.meta_action}\n"
                    f"Post and context: {request.post_description}"
                ),
                "prediction": (
                    f"Outcome: {prediction.outcome}\n"
                    f"Reasoning: {prediction.reasoning}\n"
                    f"Policies: {', '.join(prediction.policies) or 'None returned'}"
                ),
                "evidence": evidence or "No retrieved evidence was available.",
                "history": transcript or "No earlier follow-up messages.",
                "question": question,
            }
        ).strip()
        if self._is_weak_response(response, question, history):
            return self._grounded_fallback(request, prediction, question, history, intent)
        return response

    def _is_weak_response(
        self,
        response: str,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> bool:
        """Detect unusable, repeated, leaked, or wrong-actor model responses."""
        normalized = response.casefold().strip()
        repeated = any(normalized == answer.casefold().strip() for _, answer in history)
        asks_poster = self._asks_poster(question)
        addresses_poster = any(word in normalized.split() for word in ("you", "your"))
        answers_meta_instead = asks_poster and "meta" in normalized and not addresses_poster
        return (
            not normalized
            or "flan-t5" in self.client.model.casefold()
            or normalized.startswith(("outcome:", "prediction:"))
            or normalized in {"upheld", "overturned"}
            or repeated
            or answers_meta_instead
            or len(normalized.split()) < 5
            or any(
                marker in normalized
                for marker in (
                    "follow-up question",
                    "analyzed case",
                    "meta action:",
                    "post and context:",
                )
            )
        )

    @staticmethod
    def _grounded_fallback(
        request: PredictionRequest,
        prediction: Prediction,
        question: str,
        history: Sequence[tuple[str, str]],
        intent: str = "general",
    ) -> str:
        """Produce a safe evidence-grounded answer when generation is unreliable."""
        question_text = question.casefold()
        citation = prediction.citations[0] if prediction.citations else None
        policy = ConversationChain._policy_label(prediction, citation.title if citation else "")
        evidence = (
            f" The closest retrieved case is “{citation.title}” [{citation.chunk_id}], which "
            "the retriever selected as the most similar prior decision."
            if citation
            else " The retrieved material does not provide enough evidence for a stronger claim."
        )

        action_effect = ConversationChain._action_effect(request.meta_action, prediction.outcome)
        if intent == "definition":
            if "overturn" in question_text:
                opposite = ConversationChain._action_effect(
                    request.meta_action,
                    "overturned" if prediction.outcome == "upheld" else "upheld",
                )
                return f"Overturned means the Board would {opposite}. It describes whether Meta's original action is reversed, not whether the post itself is morally good or bad."
            if "citation" in question_text:
                return "A citation is a retrieved policy or prior Board passage used as evidence. Its bracketed ID lets you match the answer to the evidence shown on this result page."
            return f"Upheld means the Board would {ConversationChain._action_effect(request.meta_action, 'upheld')}. In other words, Meta's original action stays in place."
        if intent == "conversation_history":
            if not history:
                return "This is the first follow-up question in this analysis, so there is no earlier conversation to recap."
            recap = "; ".join(
                f'you asked "{q}" and I answered "{a[:120]}"' for q, a in history[-3:]
            )
            return f"Here is the recent conversation: {recap}."
        if intent == "critique":
            support = f"The strongest case for the prediction is that {ConversationChain._clean_reasoning(prediction.reasoning)}"
            challenge = "The strongest challenge is that context, public interest, intent, and a less restrictive enforcement option may not have been weighed enough."
            return f"I do not have a personal opinion. {support} {challenge} This is a model-assisted assessment, not a confirmed Board judgment."
        if intent == "poster_revision" or ConversationChain._asks_poster(question):
            return ConversationChain._poster_changes(request, prediction, citation)
        if intent == "explanation" or "why" in question_text:
            return (
                f"The prediction is upheld because the model believes {action_effect}. "
                f"It identified {policy} as the main policy consideration."
                f"{evidence} This is the model's estimate, not a confirmed Board ruling."
            )

        if any(word in question_text for word in ("reject", "remove", "taken down")):
            return (
                f"The model predicts that the Board {action_effect}. It identified {policy} "
                f"as the main policy consideration.{evidence}"
            )
        if intent == "policy" or "policy" in question_text:
            return (
                f"The model identified {policy} because the described content and Meta's "
                f"action raise the concerns summarized in its reasoning: {ConversationChain._clean_reasoning(prediction.reasoning)}"
                f"{evidence}"
            )
        if intent == "precedent" or any(
            word in question_text for word in ("precedent", "case", "evidence", "source")
        ):
            if citation:
                return (
                    f"The strongest retrieved precedent is “{citation.title}.” "
                    f"{ConversationChain._complete_excerpt(citation.excerpt, 260)} "
                    f"[{citation.chunk_id}]"
                )
            return "No precedent was retrieved for this prediction, so I cannot identify one."
        if intent == "counterfactual" or any(
            word in question_text for word in ("change", "opposite", "different fact")
        ):
            if "satire" in question_text or "satirical" in question_text:
                return (
                    "Yes. A clear satire label and an obviously humorous presentation could "
                    "reduce the risk that viewers mistake the video for genuine reporting. The "
                    f"Board would then need to reassess whether {policy} still justifies "
                    f"{ConversationChain._action_description(request.meta_action)}."
                )
            return (
                "The outcome could change if the post's context, intent, authenticity, likely "
                "harm, newsworthiness, or available enforcement option differs materially from "
                f"the description. The current prediction relies on {policy} and this reasoning: "
                f"{ConversationChain._clean_reasoning(prediction.reasoning)}"
            )
        if intent == "comparison":
            if citation:
                return (
                    f"The current post concerns {request.post_description[:180].rstrip()}. "
                    f"The closest retrieved decision is “{citation.title},” where "
                    f"{ConversationChain._complete_excerpt(citation.excerpt, 220)} "
                    f"[{citation.chunk_id}] The shared policy issue makes them comparable, while "
                    "differences in context, intent, audience, and likely harm may change the ruling."
                )
            return (
                "No precedent was retrieved, so there is not enough evidence for a case comparison."
            )
        return (
            "I couldn't produce a reliable answer to that question with the available local "
            "fallback. Please try rephrasing it, or ask about the ruling, policy, or evidence."
        )

    @staticmethod
    def _action_effect(meta_action: str, outcome: str) -> str:
        """Translate an outcome into its practical effect on Meta's action."""
        actions = {
            "removed": ("uphold Meta's removal", "reverse Meta's removal and restore the post"),
            "left_up": (
                "uphold Meta's decision to keep the post online",
                "reverse Meta's decision to keep the post online",
            ),
            "warning": ("uphold Meta's warning", "reverse Meta's warning"),
            "account_action": (
                "uphold Meta's account restriction",
                "reverse Meta's account restriction",
            ),
        }
        upheld, overturned = actions[meta_action]
        return upheld if outcome == "upheld" else overturned

    @staticmethod
    def _action_description(meta_action: str) -> str:
        """Return a readable description of Meta's original action."""
        return {
            "removed": "removing the post",
            "left_up": "leaving the post online",
            "warning": "adding the warning",
            "account_action": "restricting the account",
        }[meta_action]

    @staticmethod
    def _complete_excerpt(text: str, limit: int = 190) -> str:
        """Trim an evidence excerpt at a readable sentence or word boundary."""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        shortened = compact[:limit]
        sentence_end = max(shortened.rfind("."), shortened.rfind("!"), shortened.rfind("?"))
        if sentence_end >= 70:
            return shortened[: sentence_end + 1]
        return shortened.rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _policy_label(prediction: Prediction, citation_title: str) -> str:
        """Select a clean policy label from prediction or citation context."""
        valid = [
            policy.strip()
            for policy in prediction.policies
            if 2 < len(policy.strip()) <= 80
            and "meta action:" not in policy.casefold()
            and "content:" not in policy.casefold()
        ]
        if valid:
            return ", ".join(valid)
        title = citation_title.casefold()
        if any(word in title for word in ("ai", "fake", "misinformation")):
            return "the misinformation and manipulated-media rules"
        if any(word in title for word in ("privacy", "passport", "identity")):
            return "the privacy rules"
        if any(word in title for word in ("drug", "medicine", "fentanyl")):
            return "the Restricted Goods and Services policy"
        return "the relevant moderation policy"

    @staticmethod
    def _clean_reasoning(reasoning: str) -> str:
        """Remove prompt leakage and truncate model reasoning safely."""
        compact = " ".join(reasoning.split())
        lowered = compact.casefold()
        if any(marker in lowered for marker in ("meta action:", "content:", "follow-up question")):
            return "the retrieved policy and precedent evidence favors the predicted outcome."
        if not compact:
            return "the retrieved policy and precedent evidence favors the predicted outcome."
        return ConversationChain._complete_excerpt(compact, 320)

    @staticmethod
    def _asks_poster(question: str) -> bool:
        """Detect whether the user wants advice for the content creator."""
        question_text = question.casefold()
        return any(
            phrase in question_text
            for phrase in (
                "could i",
                "should i",
                "my post",
                "i have done",
                "for it to be accepted",
                "for it to be allowed",
            )
        )

    @staticmethod
    def _heuristic_intent(question: str) -> str:
        """Provide a lightweight intent fallback when no semantic router is present."""
        return "poster_revision" if ConversationChain._asks_poster(question) else "general"

    @staticmethod
    def _poster_changes(
        request: PredictionRequest,
        prediction: Prediction,
        citation,
    ) -> str:
        """Suggest case-specific content changes without guaranteeing acceptance."""
        context = " ".join(
            [
                request.post_description,
                " ".join(prediction.policies),
                citation.title if citation else "",
            ]
        ).casefold()
        if any(word in context for word in ("ai-generated", "synthetic", "misinformation")):
            return (
                "You could have clearly labeled the video as AI-generated, avoided presenting it "
                "as live eyewitness footage, and included its original source and creation context. "
                "If it was satire, that should have been unmistakable in the caption and presentation. "
                "Those changes would reduce the risk of deception, although they would not guarantee "
                "that Meta would leave the post online."
            )
        if any(word in context for word in ("privacy", "passport", "identity", "biodata")):
            return (
                "You could have blurred the name, document number, date of birth, photograph, and "
                "other identifying details while preserving the newsworthy part of the image. Adding "
                "context about why the photograph was in the public interest would also help."
            )
        if any(word in context for word in ("drug", "medicine", "fentanyl", "prescription")):
            return (
                "You could have made the medical context explicit, stated that the medication was "
                "professionally administered, and avoided wording that could look like promotion, "
                "sale, or encouragement of non-medical drug use."
            )
        return (
            "You could have removed or clarified the element that triggered the policy concern, "
            "provided the original source, and made the post's intent and context explicit. The exact "
            "change depends on the cited policy, and acceptance would still not be guaranteed."
        )

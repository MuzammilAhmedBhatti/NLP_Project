# ModPrecedent presentation content

Maximum length: 10 slides. Replace the placeholders on Slide 1 before submission.

## Slide 1 — ModPrecedent

Evidence-grounded prediction of Meta Oversight Board decisions

- Project category: niche-domain RAG bot
- Team members: add names and registration numbers

## Slide 2 — Problem and objectives

- Moderation appeals depend on policy, context, likely harm, public interest and precedent.
- Users rarely receive an evidence-linked explanation of a likely ruling.
- ModPrecedent predicts upheld or overturned, retrieves supporting sources and answers follow-up questions.

## Slide 3 — System architecture

Image or text input → OCR and optional BLIP caption → preprocessing → TF-IDF and MiniLM retrieval → policy and precedent context → outcome prediction → LangChain follow-up conversation

Outputs: outcome, confidence, policy labels, explanation and validated citation IDs.

The saved experiment uses local Qwen. The live application uses Claude Haiku when an API key is configured and falls back to local Qwen if the API is unavailable.

## Slide 4 — Tools and NLP techniques

- Requests and BeautifulSoup for resumable official-data collection
- TF-IDF bigrams for lexical retrieval
- MiniLM sentence embeddings for semantic retrieval and bounded zero-shot intent routing
- Qwen2.5 Instruct for the reproducible local evaluation and offline fallback
- Claude Haiku for live open-ended generation through the Anthropic API
- LangChain LCEL for prompt, context and output orchestration
- RapidOCR plus optional BLIP captioning for image input
- FastAPI, pandas, scikit-learn and pytest

## Slide 5 — Dataset and preprocessing

- 94 public Oversight Board decisions
- 44 Standard, 46 Summary and 4 advisory or unknown records
- 18 upheld, 65 overturned and 11 mixed or advisory outcomes
- 88 records have normalized policy labels and 81 have an extracted original Meta action
- Processing extracts sections, removes page noise, creates overlapping chunks and attaches metadata
- The final evaluation uses 12 leakage-safe Standard cases balanced 6 upheld and 6 overturned

## Slide 6 — RAG methodology

- Hybrid relevance = 0.35 × TF-IDF similarity + 0.65 × MiniLM similarity
- S0 receives only the case and original Meta action
- S1 adds policy passages
- S2 adds policy passages and analogous prior decisions
- The retrieval corpus excludes all test cases and documents published after the earliest test case
- The local model returns a constrained outcome, policy, rationale and citation ID

## Slide 7 — Conversational and multimodal application

- Every follow-up expands the query with the original case and recent conversation
- Fresh hybrid retrieval runs for each question
- Claude handles greetings, general language and unexpected follow-up questions
- MiniLM routes ten bounded case operations, including policy, precedent and counterfactual questions
- Deterministic handlers cover definitions, corpus statistics, history and poster-revision guidance
- Response validation catches prompt leaks, repetition, weak output and wrong-actor advice
- The interface supports OCR, basic visual context, loading feedback and multipage precedent exploration

## Slide 8 — Evaluation design

- Outcome accuracy and macro-F1
- Policy-label F1 and exact policy accuracy
- Valid citation-ID coverage
- Retrieval experiments compare 480 and 900 character chunks, lexical and hybrid search, top 3 and top 6 results, and source diversification
- Conversation audit: 22 paraphrased questions
- Software verification: 25 automated tests plus Ruff static checks

## Slide 9 — Results

| System | Accuracy | Macro-F1 | Policy F1 | Citation coverage |
|---|---:|---:|---:|---:|
| S0 | 58.3% | 55.6% | 5.6% | 0% |
| S1 | 75.0% | 73.3% | 8.9% | 100% |
| S2 | 50.0% | 48.6% | 8.9% | 100% |

- Policy retrieval produced the strongest outcome performance.
- Precedent retrieval introduced noise for the small local model.
- Policy-label extraction remains weak and needs a stronger model or dedicated classifier.
- Intent routing scored 22 of 22 on the bounded audit. This measures routing, not unrestricted answer quality.

## Slide 10 — Conclusion and next steps

- ModPrecedent demonstrates a complete niche-domain conversational RAG methodology.
- S1 improved accuracy by 16.7 percentage points over the closed-book system.
- More retrieval did not automatically improve prediction, which is an important experimental finding.
- Remaining limits: policy multilabel prediction, CPU fallback latency, moderate corpus size and API cost.
- Future work: cross-encoder reranking, human citation-relevance judgments and a larger held-out review set.

# ModPrecedent

## Reproduce the approved study

Install the application and research dependencies with `python -m pip install -e ".[all]"`. Run `python -m modprecedent.evaluation.experiments` from the repository root. The first run downloads the configured Qwen instruction model and MiniLM for dense retrieval. The study writes per-case predictions, outcome and policy metrics, citation coverage, a chart, and retrieval sensitivity results to `reports/generated/latest`.

ModPrecedent is a leakage-aware retrieval-augmented assistant for predicting whether Meta's Oversight Board would **uphold** or **overturn** a moderation action. It retrieves policy passages and analogous published decisions, produces structured explanations with citations, and compares three experimental systems.

After each prediction, a LangChain LCEL conversation keeps the analyzed post, ruling, retrieved evidence, and recent message history in context. Every turn expands the query with the original case and recent questions, then runs fresh TF-IDF/MiniLM hybrid retrieval. Claude Haiku handles natural conversation when the Anthropic backend is configured. The default local model is `Qwen/Qwen2.5-1.5B-Instruct`, with automatic fallback to the 0.5B variant on constrained machines. A zero-shot MiniLM semantic intent router distinguishes ten bounded case operations; it does not attempt to enumerate every possible user question. High-precision answers such as definitions, corpus counts, poster corrections, and history recap are deterministic and grounded. General language goes to the LLM. Response validation rejects prompt leaks, empty or very short output, repeated answers, and wrong-actor advice.

The application includes the approved **Casebook** interface, a warm editorial design built around source reading and evidence traceability. Its five views cover prediction input, results, the precedent library, case details, and evaluation.

## Systems

- **S0 — Closed Book:** LLM receives only the case description and Meta action.
- **S1 — Policy RAG:** adds retrieved policy passages.
- **S2 — Policy + Precedent RAG:** adds policies and similar decisions.

The live demo can use `claude-haiku-4-5-20251001` through the Anthropic Messages API, with automatic fallback to local Qwen if the API is unavailable. The reproducible evaluation artifacts remain the local-Qwen baseline. FLAN-T5-small and Ollama remain optional local backends. Retrieval always has a portable TF-IDF implementation; installing the `rag` extra enables Sentence Transformer dense embeddings and hybrid scoring.

## Quick start (Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
python -m pytest
python -m modprecedent.cli index
python -m modprecedent.cli predict --text "A news page neutrally quotes a designated group announcing a policy change." --action removed --system S2
modprecedent-app
```

For dense retrieval: `python -m pip install -e ".[rag,dev]"` then `python -m modprecedent.cli index --dense`.

For Claude Haiku, set both backends to `anthropic`, use `claude-haiku-4-5-20251001`, and provide `CLAUDE_API_KEY`. To use Ollama, set the backends to `ollama`; the Ollama implementation remains preserved but inactive by default. OpenAI-compatible endpoints also require only backend, model, key and base-URL configuration. Retrieval and LangChain logic remain unchanged.

The web app is available through the installed `modprecedent-app` command. If the configured generator is unavailable, it clearly marks its retrieval-based fallback result.

## Real-data pipeline

```powershell
python -m modprecedent.cli collect --max-pages 10
python -m modprecedent.cli process
python -m modprecedent.cli index --dense
```

Collection is cached under `data/raw/pages`, rate-limited, retried, and checkpointed after each decision. The processed corpus contains 94 decisions. Run `python -m modprecedent.evaluation.data_audit` to reproduce the corpus-quality report.

## Human-approved, leakage-aware evaluation

Generate candidates from real Standard decisions:

```powershell
python -m modprecedent.cli candidates
```

Review `data/evaluation/candidate_cases.jsonl`, edit descriptions if needed, and change `status` to `approved`. Final evaluation refuses to run while any case is pending:

```powershell
python -m modprecedent.evaluation.experiments
```

Run `python -m modprecedent.evaluation.conversation_audit` to reproduce the 22-question semantic-routing regression audit. It writes CSV and Markdown results beside the main evaluation artifacts. This finite suite verifies routing behavior, not unlimited question-answer quality.

For each held-out case, `as_of_date` excludes contemporaneous and future documents. The case itself can also be excluded by ID at retrieval time. Reports contain accuracy, macro precision/recall/F1, set-based policy F1, citation coverage, and mean retrieval relevance. No production metrics are hard-coded.

## Layout

```text
src/modprecedent/data/        schemas, collection, persistence, preprocessing
src/modprecedent/rag/         retrieval, pretrained generators, prediction
src/modprecedent/evaluation/  metrics and reproducible experiments
src/modprecedent/ui/          FastAPI/Gradio interfaces, theme, OCR
src/modprecedent/config/      environment-backed settings
src/modprecedent/cli.py       command-line entry point
config/            reproducible defaults
data/              cached raw data, processed corpus, indexes, review sets
tests/             unit tests for processing, leakage controls, retrieval, fallback
reports/generated/ generated metrics and predictions
```

## Responsible use

This is a research and teaching tool, not legal advice or an official Oversight Board system. Predictions are uncertain, may reproduce source/model biases, and should be interpreted alongside the cited primary decisions. The collector uses only public pages and identifies itself with a conservative request rate.


# ModPrecedent

## Reproduce the approved study

Install the application and research dependencies with `python -m pip install -e ".[frontend,rag,dev]"`. Run `python -m modprecedent.experiments` from the repository root. The first run downloads FLAN-T5-small for local generation and MiniLM for dense retrieval. The study writes per-case predictions, outcome and policy metrics, citation coverage, a chart, and retrieval sensitivity results to `reports/generated/latest`. Set `MODPRECEDENT_LLM_BACKEND=flan` to use the same local model in the web application. The three approved examples form a small course demonstration.

ModPrecedent is a leakage-aware retrieval-augmented assistant for predicting whether Meta's Oversight Board would **uphold** or **overturn** a moderation action. It retrieves policy passages and analogous published decisions, produces structured explanations with citations, and compares three experimental systems.

The application includes the approved **Casebook** interface, a warm editorial design built around source reading and evidence traceability. Its five views cover prediction input, results, the precedent library, case details, and evaluation.

## Systems

- **S0 — Closed Book:** LLM receives only the case description and Meta action.
- **S1 — Policy RAG:** adds retrieved policy passages.
- **S2 — Policy + Precedent RAG:** adds policies and similar decisions.

The default LLM is local Ollama. If Ollama is unavailable, the application returns an explicitly marked, low-confidence retrieval baseline so demos remain runnable. Retrieval always has a portable TF-IDF implementation; installing the `rag` extra enables Sentence Transformer dense embeddings and hybrid scoring.

## Quick start (Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,frontend]"
python -m pytest
python -m modprecedent.cli index
python -m modprecedent.cli predict --text "A news page neutrally quotes a designated group announcing a policy change." --action removed --system S2
python -m modprecedent.app
```

For dense retrieval: `python -m pip install -e ".[rag,dev]"` then `python -m modprecedent.cli index --dense`.

For generative predictions, install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, and keep the service running. Copy `.env.example` to `.env` to change the Ollama model, endpoint, or embedding model.

The web app is also available through the installed `modprecedent-app` command. It remains usable when Ollama is offline by clearly marking its retrieval-based fallback result.

## Real-data pipeline

```powershell
python -m modprecedent.cli collect --max-pages 10
python -m modprecedent.cli process
python -m modprecedent.cli index --dense
```

Collection is cached under `data/raw/pages`, rate-limited, retried, and checkpointed after each decision. The included three-case sample is only a smoke-test corpus; it is not presented as a complete dataset or as evaluation evidence.

## Human-approved, leakage-aware evaluation

Generate candidates from real Standard decisions:

```powershell
python -m modprecedent.cli candidates
```

Review `data/evaluation/candidate_cases.jsonl`, edit descriptions if needed, and change `status` to `approved`. Final evaluation refuses to run while any case is pending:

```powershell
python -m modprecedent.experiments
```

For each held-out case, `as_of_date` excludes contemporaneous and future documents. The case itself can also be excluded by ID at retrieval time. Reports contain accuracy, macro precision/recall/F1, set-based policy F1, citation coverage, and mean retrieval relevance. No production metrics are hard-coded.

## Layout

```text
src/modprecedent/  collector, processing, retrieval, LLM, prediction, evaluation, CLI
config/            reproducible defaults
data/              cached raw data, processed corpus, indexes, review sets
tests/             unit tests for processing, leakage controls, retrieval, fallback
reports/generated/ generated metrics and predictions
```

## Responsible use

This is a research and teaching tool, not legal advice or an official Oversight Board system. Predictions are uncertain, may reproduce source/model biases, and should be interpreted alongside the cited primary decisions. The collector uses only public pages and identifies itself with a conservative request rate.


from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import pandas as pd

from .cli import decisions_path, load_index
from .io import read_jsonl
from .models import Decision, PredictionRequest
from .predictor import Predictor

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Spectral:ital,wght@0,500;0,600;1,500&display=swap');

:root {
  --paper: #f7f3ea;
  --paper-deep: #eee8dc;
  --ink: #182321;
  --forest: #164a3d;
  --forest-soft: #dbe7e0;
  --rust: #a34329;
  --rust-soft: #f2dfd7;
  --rule: #c9c0b2;
  --muted: #6d716c;
}

body, .gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'Source Sans 3', sans-serif !important;
}
.gradio-container { max-width: none !important; padding: 0 !important; }
.app-shell { min-height: 100vh; gap: 0 !important; }
.case-sidebar {
  min-height: 100vh;
  background: var(--forest);
  color: #fffdf7;
  padding: 2rem 1.35rem 1.5rem;
  border: 0 !important;
  position: sticky;
  top: 0;
}
.brand-word { font-family: 'Spectral', serif; font-size: 2rem; font-weight: 600; letter-spacing: -.05em; }
.brand-kicker { font-size: .73rem; letter-spacing: .15em; text-transform: uppercase; opacity: .72; margin-top: .3rem; line-height: 1.4; }
.sidebar-rule { border-top: 1px solid rgba(255,255,255,.3); margin: 1.6rem 0; }
.nav-button { margin: .32rem 0 !important; background: transparent !important; color: #fffdf7 !important; border: 1px solid transparent !important; box-shadow: none !important; text-align: left !important; justify-content: flex-start !important; font-size: 1rem !important; }
.nav-button:hover { background: rgba(255,255,255,.09) !important; border-color: rgba(255,255,255,.14) !important; }
.sidebar-note { margin-top: 2.2rem; color: rgba(255,255,255,.72); font-family: 'Spectral', serif; font-style: italic; line-height: 1.55; font-size: .92rem; }
.content-column { padding: 2.4rem clamp(1.4rem, 4vw, 4.7rem) 4rem; max-width: 1180px; margin: 0 auto; }
.page-kicker { color: var(--rust); text-transform: uppercase; letter-spacing: .14em; font-size: .76rem; font-weight: 700; margin-bottom: .45rem; }
.page-title { font-family: 'Spectral', serif; font-size: clamp(2.6rem, 5vw, 5rem); line-height: .95; letter-spacing: -.055em; margin: 0 0 .85rem; color: var(--ink); }
.page-deck { max-width: 690px; font-family: 'Spectral', serif; font-size: 1.15rem; line-height: 1.55; color: #4b534f; margin-bottom: 2.2rem; }
.paper-panel { background: rgba(255,255,255,.31) !important; border: 1px solid var(--rule) !important; border-radius: 2px !important; box-shadow: 0 8px 30px rgba(46,42,34,.045) !important; padding: 1.25rem !important; }
.paper-panel label span { text-transform: uppercase; letter-spacing: .11em; font-size: .76rem !important; font-weight: 700; color: var(--forest) !important; }
textarea, input, .wrap, select { background: #fffdf7 !important; border-color: #bdb4a7 !important; color: var(--ink) !important; border-radius: 2px !important; }
.primary-action { background: var(--rust) !important; color: white !important; border: 1px solid var(--rust) !important; border-radius: 2px !important; font-weight: 700 !important; min-height: 3.1rem; }
.secondary-action { background: transparent !important; color: var(--forest) !important; border: 1px solid var(--forest) !important; border-radius: 2px !important; }
.result-banner { border-top: 4px solid var(--rust); padding: 1.3rem 1.5rem; background: var(--rust-soft); margin: 1rem 0 1.5rem; }
.result-label { text-transform: uppercase; letter-spacing: .14em; color: var(--rust); font-size: .76rem; font-weight: 700; }
.result-outcome { font-family: 'Spectral', serif; font-style: italic; font-size: clamp(2.1rem, 5vw, 4rem); line-height: 1; color: var(--rust); margin: .35rem 0; }
.result-meta { color: #544a43; }
.evidence-card { border-top: 1px solid var(--rule); padding: 1rem 0; }
.evidence-card:first-child { border-top: 0; }
.evidence-type { color: var(--rust); text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 700; }
.evidence-title { font-family: 'Spectral', serif; font-size: 1.25rem; font-weight: 600; color: var(--forest); margin: .2rem 0; }
.evidence-excerpt { color: #4e5652; line-height: 1.55; }
.policy-chip { display: inline-block; padding: .34rem .65rem; margin: .2rem; background: var(--forest-soft); color: var(--forest); font-size: .85rem; border-radius: 2px; }
.comparison { display: grid; grid-template-columns: 1fr 1fr; border-block: 1px solid var(--rule); margin: 1rem 0 1.4rem; }
.comparison > div { padding: 1rem; }
.comparison > div + div { border-left: 1px solid var(--rule); }
.comparison small { text-transform: uppercase; letter-spacing: .11em; color: var(--muted); }
.comparison strong { display: block; font-family: 'Spectral', serif; font-size: 1.2rem; margin-top: .25rem; }
.case-reader { max-width: 820px; }
.case-reader h2 { font-family: 'Spectral', serif; font-size: 1.55rem; color: var(--forest); border-top: 1px solid var(--rule); padding-top: 1.1rem; margin-top: 1.6rem; }
.case-reader p { font-family: 'Spectral', serif; font-size: 1.08rem; line-height: 1.72; }
.case-meta { color: var(--rust); text-transform: uppercase; letter-spacing: .1em; font-size: .76rem; font-weight: 700; }
.library-table table, .metrics-table table { background: transparent !important; }
.library-table th, .metrics-table th { background: var(--forest) !important; color: white !important; }
.library-table td, .metrics-table td { background: #fffdf7 !important; border-color: var(--rule) !important; }
.section-heading { font-family: 'Spectral', serif; font-size: 1.65rem; margin: 1.5rem 0 .4rem; color: var(--ink); }
.loading-copy { font-family: 'Spectral', serif; font-style: italic; color: var(--muted); }
.footer-note { border-top: 1px solid var(--rule); margin-top: 3rem; padding-top: 1rem; color: var(--muted); font-size: .85rem; }
@media (max-width: 800px) {
  .app-shell { display: block !important; }
  .case-sidebar { min-height: auto; position: static; padding: 1rem; }
  .brand-kicker, .sidebar-note, .sidebar-rule { display: none; }
  .case-sidebar .form { display: grid !important; grid-template-columns: repeat(4, 1fr); gap: .3rem; }
  .nav-button { font-size: .84rem !important; padding: .55rem .35rem !important; justify-content: center !important; }
  .content-column { padding: 1.6rem 1rem 3rem; }
  .page-title { font-size: 2.7rem; }
}
"""


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _decisions() -> list[Decision]:
    processed = Path("data/processed/decisions.jsonl")
    return read_jsonl(processed if processed.exists() else decisions_path(), Decision)


def _library_frame(decisions: list[Decision]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Case": decision.title,
                "Outcome": decision.outcome.value.title(),
                "Policy": ", ".join(decision.policies) or "—",
                "Published": decision.published_date.isoformat()
                if decision.published_date
                else "—",
                "ID": decision.case_id,
            }
            for decision in decisions
        ]
    )


def _page_updates(page: str):
    names = ("predict", "result", "library", "case", "evaluation")
    return tuple(gr.update(visible=name == page) for name in names)


def _prediction_html(prediction, request: PredictionRequest) -> str:
    outcome = prediction.outcome.title()
    guess = (request.user_guess or "No guess recorded").replace("_", " ").title()
    policies = "".join(
        f'<span class="policy-chip">{_esc(policy)}</span>' for policy in prediction.policies
    )
    if not policies:
        policies = '<span class="policy-chip">No policy label returned</span>'
    evidence = (
        "".join(
            f"""<div class="evidence-card">
          <div class="evidence-type">{_esc(c.source_type)} · relevance {c.score:.2f}</div>
          <div class="evidence-title">{_esc(c.title)}</div>
          <div class="evidence-excerpt">{_esc(c.excerpt)}</div>
        </div>"""
            for c in prediction.citations
        )
        or '<p class="loading-copy">No retrieved evidence was available for this run.</p>'
    )
    fallback = " · retrieval baseline used" if prediction.fallback_used else ""
    return f"""
      <div class="result-banner">
        <div class="result-label">Board prediction</div>
        <div class="result-outcome">Likely {outcome}</div>
        <div class="result-meta">{prediction.confidence:.0%} confidence{fallback}</div>
      </div>
      <div class="comparison">
        <div><small>Your guess</small><strong>{_esc(guess)}</strong></div>
        <div><small>Model prediction</small><strong>{_esc(outcome)}</strong></div>
      </div>
      <h3 class="section-heading">Reasoning</h3>
      <p>{_esc(prediction.reasoning)}</p>
      <h3 class="section-heading">Relevant policy</h3>
      <div>{policies}</div>
      <h3 class="section-heading">Evidence</h3>
      <div>{evidence}</div>
    """


def _run_prediction(description: str, action: str, guess: str, system: str):
    request = PredictionRequest(
        post_description=description,
        meta_action=action,
        user_guess=guess,
    )
    prediction = Predictor(load_index()).predict(request, system)
    cited_cases = list(dict.fromkeys(c.case_id for c in prediction.citations))
    choices = [
        (next((d.title for d in _decisions() if d.case_id == cid), cid), cid) for cid in cited_cases
    ]
    return (
        *_page_updates("result"),
        _prediction_html(prediction, request),
        gr.update(choices=choices, value=choices[0][1] if choices else None),
    )


def _filter_library(query: str, policy: str, outcome: str):
    decisions = _decisions()
    query_lower = query.strip().lower()
    filtered = []
    for decision in decisions:
        searchable = " ".join(
            [decision.title, decision.summary, *decision.policies, *decision.topics]
        ).lower()
        if query_lower and query_lower not in searchable:
            continue
        if policy != "All policies" and policy not in decision.policies:
            continue
        if outcome != "All outcomes" and decision.outcome.value != outcome.lower():
            continue
        filtered.append(decision)
    return _library_frame(filtered), gr.update(
        choices=[(d.title, d.case_id) for d in filtered],
        value=filtered[0].case_id if filtered else None,
    )


def _case_html(case_id: str) -> str:
    decision = next((item for item in _decisions() if item.case_id == case_id), None)
    if decision is None:
        return '<div class="case-reader"><p>Select a precedent from the library.</p></div>'
    majority = (
        decision.majority
        or "The published decision does not include a separately extracted majority passage in the local dataset."
    )
    minority = decision.minority or "No separate minority argument was extracted for this decision."
    policies = ", ".join(decision.policies) or "Policy details in full decision"
    date = (
        decision.published_date.strftime("%B %d, %Y")
        if decision.published_date
        else "Publication date unavailable"
    )
    return f'''
      <article class="case-reader">
        <div class="case-meta">{_esc(decision.decision_type)} · {_esc(date)} · {_esc(decision.outcome.value.title())}</div>
        <h1 class="page-title">{_esc(decision.title)}</h1>
        <p class="page-deck">{_esc(decision.summary)}</p>
        <h2>Policy considered</h2><p>{_esc(policies)}</p>
        <h2>Majority</h2><p>{_esc(majority)}</p>
        <h2>Minority</h2><p>{_esc(minority)}</p>
        <h2>Source</h2><p><a href="{_esc(decision.url)}" target="_blank" rel="noopener">Read the published Oversight Board decision</a></p>
      </article>
    '''


def _open_case(case_id: str):
    return (*_page_updates("case"), _case_html(case_id))


def build_app() -> gr.Blocks:
    decisions = _decisions()
    policies = sorted({policy for decision in decisions for policy in decision.policies})
    case_choices = [(decision.title, decision.case_id) for decision in decisions]

    with gr.Blocks(title="ModPrecedent · Oversight Board research") as demo:
        with gr.Row(elem_classes="app-shell"):
            with gr.Column(scale=1, min_width=220, elem_classes="case-sidebar"):
                gr.HTML(
                    '<div class="brand-word">ModPrecedent</div><div class="brand-kicker">Oversight Board<br>moderation research</div><div class="sidebar-rule"></div>'
                )
                nav_predict = gr.Button("Predict", elem_classes="nav-button")
                nav_library = gr.Button("Precedents", elem_classes="nav-button")
                nav_evaluation = gr.Button("Evaluation", elem_classes="nav-button")
                gr.HTML(
                    '<div class="sidebar-note">Evidence before prediction.<br>Policy, context, and precedent in view.</div>'
                )

            with gr.Column(scale=5, elem_classes="content-column"):
                with gr.Group(visible=True) as predict_page:
                    gr.HTML(
                        '<div class="page-kicker">New analysis</div><h1 class="page-title">Predict</h1><p class="page-deck">Describe the post as a reviewer would encounter it. Add Meta’s original action, then record your own judgment before seeing the model’s evidence.</p>'
                    )
                    with gr.Group(elem_classes="paper-panel"):
                        post_description = gr.Textbox(
                            label="Post description",
                            placeholder="Describe the content, speaker, audience, context, language, and any plausible risk of harm…",
                            lines=9,
                        )
                        with gr.Row():
                            meta_action = gr.Dropdown(
                                [
                                    ("Removed the content", "removed"),
                                    ("Left the content up", "left_up"),
                                    ("Added a warning screen", "warning"),
                                    ("Took action on the account", "account_action"),
                                ],
                                value="removed",
                                label="Meta’s original action",
                            )
                            user_guess = gr.Radio(
                                [("Uphold Meta", "upheld"), ("Overturn Meta", "overturned")],
                                value="overturned",
                                label="Your guess",
                            )
                        with gr.Row():
                            system = gr.Radio(
                                [
                                    ("Closed book", "S0"),
                                    ("Policy RAG", "S1"),
                                    ("Policy + precedent", "S2"),
                                ],
                                value="S2",
                                label="Research method",
                            )
                        analyze = gr.Button(
                            "Analyze precedent", variant="primary", elem_classes="primary-action"
                        )
                    gr.HTML(
                        '<div class="footer-note">Predictions are generated from the configured language model and retrieved public decisions. Source links remain available for verification.</div>'
                    )

                with gr.Group(visible=False) as result_page:
                    gr.HTML(
                        '<div class="page-kicker">Analysis</div><h1 class="page-title">Prediction</h1><p class="page-deck">The ruling estimate is presented alongside your guess, model reasoning, policy references, and retrieved precedent.</p>'
                    )
                    result_html = gr.HTML(
                        '<p class="loading-copy">Run an analysis to see a prediction.</p>'
                    )
                    with gr.Group(elem_classes="paper-panel"):
                        cited_case = gr.Dropdown(label="Open a cited precedent", choices=[])
                        open_cited = gr.Button(
                            "Read selected precedent", elem_classes="secondary-action"
                        )
                    new_analysis = gr.Button(
                        "Start another analysis", elem_classes="secondary-action"
                    )

                with gr.Group(visible=False) as library_page:
                    gr.HTML(
                        '<div class="page-kicker">Published decisions</div><h1 class="page-title">Precedents</h1><p class="page-deck">Search the local collection by subject, policy, or outcome. Each entry opens into a source-oriented case reader.</p>'
                    )
                    with gr.Group(elem_classes="paper-panel"):
                        with gr.Row():
                            library_query = gr.Textbox(
                                label="Search", placeholder="Journalism, political speech, threats…"
                            )
                            policy_filter = gr.Dropdown(
                                ["All policies", *policies], value="All policies", label="Policy"
                            )
                            outcome_filter = gr.Dropdown(
                                ["All outcomes", "Upheld", "Overturned"],
                                value="All outcomes",
                                label="Outcome",
                            )
                        search_library = gr.Button(
                            "Search collection", elem_classes="secondary-action"
                        )
                    library_table = gr.Dataframe(
                        _library_frame(decisions), interactive=False, elem_classes="library-table"
                    )
                    with gr.Group(elem_classes="paper-panel"):
                        library_case = gr.Dropdown(
                            case_choices,
                            value=case_choices[0][1] if case_choices else None,
                            label="Selected precedent",
                        )
                        open_library_case = gr.Button(
                            "Read selected precedent", elem_classes="primary-action"
                        )

                with gr.Group(visible=False) as case_page:
                    case_html = gr.HTML(
                        _case_html(case_choices[0][1]) if case_choices else _case_html("")
                    )
                    back_library = gr.Button(
                        "Return to precedents", elem_classes="secondary-action"
                    )

                with gr.Group(visible=False) as evaluation_page:
                    gr.HTML(
                        '<div class="page-kicker">Methods</div><h1 class="page-title">Evaluation</h1><p class="page-deck">Compare the three prediction systems after approved Standard-decision test cases have been evaluated.</p>'
                    )
                    metrics_file = Path("reports/generated/latest/metrics.csv")
                    if metrics_file.exists():
                        metrics = pd.read_csv(metrics_file)
                        gr.Dataframe(metrics, interactive=False, elem_classes="metrics-table")
                    else:
                        gr.HTML(
                            """<div class="paper-panel"><h3 class="section-heading">Awaiting approved cases</h3><p>Generate candidate cases, review descriptions for ruling leakage, mark the selected cases approved, and run the evaluation command. Results will appear here automatically.</p><p><code>python -m modprecedent.cli candidates</code><br><code>python -m modprecedent.cli evaluate --dataset data/evaluation/approved_cases.jsonl</code></p></div>"""
                        )
                    gr.HTML(
                        """<h3 class="section-heading">Compared systems</h3><div class="evidence-card"><div class="evidence-type">S0</div><div class="evidence-title">Closed book</div><div class="evidence-excerpt">The language model receives the case description and original action.</div></div><div class="evidence-card"><div class="evidence-type">S1</div><div class="evidence-title">Policy RAG</div><div class="evidence-excerpt">Adds retrieved Community Standards and policy passages.</div></div><div class="evidence-card"><div class="evidence-type">S2</div><div class="evidence-title">Policy + precedent RAG</div><div class="evidence-excerpt">Adds policy evidence and analogous prior decisions with temporal safeguards.</div></div>"""
                    )

        page_outputs = [predict_page, result_page, library_page, case_page, evaluation_page]
        nav_predict.click(lambda: _page_updates("predict"), outputs=page_outputs)
        nav_library.click(lambda: _page_updates("library"), outputs=page_outputs)
        nav_evaluation.click(lambda: _page_updates("evaluation"), outputs=page_outputs)
        new_analysis.click(lambda: _page_updates("predict"), outputs=page_outputs)
        back_library.click(lambda: _page_updates("library"), outputs=page_outputs)

        analyze.click(
            _run_prediction,
            inputs=[post_description, meta_action, user_guess, system],
            outputs=[*page_outputs, result_html, cited_case],
        )
        search_library.click(
            _filter_library,
            inputs=[library_query, policy_filter, outcome_filter],
            outputs=[library_table, library_case],
        )
        open_cited.click(_open_case, inputs=[cited_case], outputs=[*page_outputs, case_html])
        open_library_case.click(
            _open_case, inputs=[library_case], outputs=[*page_outputs, case_html]
        )

    return demo


def main() -> None:
    import uvicorn

    uvicorn.run("modprecedent.web:site", host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()

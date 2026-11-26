"""Local, server-rendered Casebook interface."""

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from ..cli import load_index
from ..data.models import Citation, Prediction, PredictionRequest
from ..rag import (
    ConversationChain,
    Predictor,
    conversation_model,
    intent_router,
    outcome_pattern_answer,
)
from .gradio_app import APP_CSS, _case_html, _decisions, _esc
from .ocr import OCRFailure, extract_content
from .theme import STYLE

site = FastAPI()
app_index = lru_cache(maxsize=1)(load_index)


@dataclass
class AnalysisSession:
    request: PredictionRequest
    prediction: Prediction
    predictor: Predictor
    messages: list[tuple[str, str]] = field(default_factory=list)
    chat_sources: dict[str, Citation] = field(default_factory=dict)
    chat_error: str = ""


results: OrderedDict[str, AnalysisSession] = OrderedDict()
css = """
*{box-sizing:border-box}body{margin:0;color:#27331f;background:#faf4e8;font:17px/1.55 'Source Sans 3',system-ui,sans-serif}a{color:#36472b}a:focus-visible,button:focus-visible{outline:3px solid #8456c7;outline-offset:4px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
header,nav{display:flex;align-items:center;gap:1.5rem}header{padding:18px 5vw;border-bottom:1px solid #b9baa6;justify-content:space-between}.brand,h1,h2,h3{font-family:Spectral,Georgia,serif}.brand{font-size:1.6rem;text-decoration:none;font-weight:bold}.hero{max-width:1400px;margin:auto;display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:5vw;padding:8vh 6vw}.hero h1{font-size:clamp(3rem,6vw,6rem);line-height:1;letter-spacing:-.055em;margin:0}.hero p{max-width:36em;margin:1.5em 0}.cta,button{display:inline-block;border:0;border-radius:8px;background:#ff8874;color:#351910;padding:.8em 1.3em;text-decoration:none;font:inherit;font-weight:bold;cursor:pointer}.art{position:relative;min-height:330px;display:grid;place-items:center}.folder{width:78%;padding:38px 28px;background:#efe7cf;border:1px solid #a3a38b;border-radius:12px;box-shadow:12px -14px #c8b6ed,22px -31px #495b37,8px 15px 32px #323b2922;transform:rotate(4deg)}.folder:before{content:'';position:absolute;right:15%;top:0;width:28px;height:75px;background:coral;clip-path:polygon(0 0,100% 0,100% 100%,50% 80%,0 100%)}.folder h2{font-size:3rem;margin:.4em 0}.folder span{display:block;border-top:1px solid #b6b39c;padding:5px}
.shell{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:90vh}.rail{background:#39472c;padding:30px 16px;display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;gap:7px}.rail a{color:#fff9ed;text-decoration:none;padding:10px;border-radius:6px}.rail a:hover,.rail a[aria-current]{background:#596943}.work{padding:clamp(20px,4vw,60px);min-width:0}.split{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(210px,.85fr);gap:3vw}.work h1{font-size:clamp(2.6rem,4vw,4rem);line-height:1.05;margin:.1em 0 .4em}.well{background:#e5edd9;border-radius:14px;padding:22px;align-self:start}.card{display:block;padding:14px;background:#fffaf1;border:1px solid #cdcabc;border-radius:8px;margin:10px 0;color:#26351e;text-decoration:none}.card:hover{border-color:#8060ab}.tag{background:#e4d7fa;padding:2px 8px;border-radius:12px;font-size:.8rem}.work label{display:block;font-weight:bold;margin:15px 0 6px}textarea,input,select{width:100%;padding:12px;border:1px solid #a7aa99;border-radius:7px;background:#fffaf1;font:inherit}textarea{min-height:160px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}form button{margin-top:20px}article{max-width:800px}.tablewrap{overflow:auto}td,th{padding:.6em;border-bottom:1px solid #cac9bb}
.result-layout{display:grid;grid-template-columns:minmax(300px,.85fr) minmax(430px,1.15fr);gap:clamp(18px,3vw,42px);align-items:start}.decision-panel{min-width:0}.decision-panel details{margin-top:1.5rem}.chat{position:sticky;top:18px;height:calc(100vh - 120px);min-height:570px;padding:clamp(18px,2.4vw,28px);border:1px solid #bcb4c7;border-radius:22px;background:#fffaf1;box-shadow:0 18px 50px #36294118;display:flex;flex-direction:column;overflow:hidden}.chat[aria-busy=true]{border-color:#8d6db4}.chat-head{display:flex;align-items:center;gap:12px}.bot-orb{width:42px;height:42px;border-radius:50%;background:conic-gradient(from 40deg,#ff8874,#c4a1ff,#cadf9f,#ff8874);box-shadow:inset 0 0 0 6px #fffaf1}.chat[aria-busy=true] .bot-orb{animation:botThink 1.1s linear infinite}.chat h2{font-size:clamp(1.8rem,2.5vw,2.5rem);margin:0}.chat-log{display:flex;flex:1;flex-direction:column;gap:12px;margin:1rem 0;overflow-y:auto;padding:4px 6px 12px;scroll-behavior:smooth}.message{max-width:84%;padding:11px 14px;border-radius:16px;white-space:pre-wrap}.message.user{align-self:flex-end;background:#d7c0fa;border-bottom-right-radius:5px}.message.assistant{align-self:flex-start;background:#e5edd9;border-bottom-left-radius:5px}.message.pending{display:flex;align-items:center;gap:5px}.typing-dot{width:7px;height:7px;border-radius:50%;background:#526348;animation:typingDot .9s ease-in-out infinite}.typing-dot:nth-child(3){animation-delay:.15s}.typing-dot:nth-child(4){animation-delay:.3s}.message strong{display:block;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:3px}.chat form{display:grid;grid-template-columns:1fr auto;gap:9px;align-items:end}.chat label{grid-column:1/-1;margin:0}.chat textarea{min-height:54px;max-height:130px;resize:none}.chat textarea[readonly]{opacity:.7}.chat form button{margin:0;height:54px;min-width:148px}.suggestions{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.suggestions button{margin:0;padding:.4rem .65rem;border:1px solid #9d91a8;background:#f2eafb;color:#402d4e;font-weight:600;font-size:.85rem}.chat-error{color:#7e3025;font-weight:600;margin:.25rem 0}@keyframes typingDot{0%,100%{translate:0 0;opacity:.35}50%{translate:0 -4px;opacity:1}}@keyframes botThink{to{rotate:1turn}}
.source-ref{font-weight:700;color:#584078;text-decoration:none}.source-ref:hover{text-decoration:underline}.message-sources{margin:.65rem 0 0;padding-top:.55rem;border-top:1px solid #becbb4;white-space:normal;font-size:.82rem}.message-sources summary{cursor:pointer;font-weight:700}.message-source{margin:.4rem 0}.message-source a{font-weight:700}.message-source small{display:block;color:#596153}
@media(max-width:980px){.result-layout{grid-template-columns:1fr}.chat{position:relative;top:auto;height:72vh;min-height:540px}.decision-panel{max-width:800px}}
@media(max-width:790px){.hero,.split,.shell{grid-template-columns:1fr}.hero{padding-top:45px}.art{min-height:290px}.rail{flex-direction:row;flex-wrap:wrap;padding:6px 12px}.well{margin-top:20px}header{flex-wrap:wrap;gap:8px}nav{gap:12px}.chat form{grid-template-columns:1fr}.chat form button{width:100%}}
"""

UPLOAD_SCRIPT = r"""
const form=document.querySelector('form[method="post"]'), post=document.querySelector('#post');
if(form&&post){
 form.enctype='multipart/form-data';post.required=false;
 const field=document.createElement('div');field.innerHTML='<label for="image">Post screenshot <span class="optional">optional</span></label><div class="upload"><input id="image" name="image" type="file" accept="image/png,image/jpeg,image/webp,image/bmp"><span id="file-label">PNG, JPEG, WebP or BMP</span></div><p id="ocr-status" class="status" role="status" aria-live="polite"></p>';
 const postLabel=document.querySelector('label[for="post"]');
 (postLabel||post).before(field);
 const image=field.querySelector('#image'),status=field.querySelector('#ocr-status'),label=field.querySelector('#file-label');
 image.addEventListener('change',async()=>{if(!image.files.length)return;label.textContent=image.files[0].name;status.className='status loading';status.textContent='Extracting text…';const data=new FormData();data.append('image',image.files[0]);const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),25000);try{const response=await fetch('/ocr',{method:'POST',body:data,signal:controller.signal});const payload=await response.json();if(!response.ok)throw Error(payload.error||'OCR failed.');post.value=payload.text;status.className='status success';status.textContent='Text extracted. Review or edit it before analysis.';post.focus()}catch(error){status.className='status error';status.textContent=error.name==='AbortError'?'Text extraction timed out. Try a smaller or sharper image.':error.message}finally{clearTimeout(timeout)}});
}
document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',()=>{const question=document.querySelector('#question');if(question){question.value=button.dataset.question;question.focus()}}));
const question=document.querySelector('#question');
if(question){const chat=question.closest('.chat'),chatForm=question.form,log=document.querySelector('.chat-log'),submit=chatForm.querySelector('button[type=submit]');question.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();if(question.value.trim().length>=3)chatForm.requestSubmit()}});chatForm.addEventListener('submit',()=>{const text=question.value.trim();if(text.length<3)return;chat.setAttribute('aria-busy','true');const user=document.createElement('div');user.className='message user';const userLabel=document.createElement('strong');userLabel.textContent='You';user.append(userLabel,document.createTextNode(text));const pending=document.createElement('div');pending.className='message assistant pending';pending.innerHTML='<strong>ModPrecedent</strong><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span><span class="sr-only">ModPrecedent is researching the answer</span>';log.append(user,pending);question.readOnly=true;submit.disabled=true;submit.textContent='Researching...';document.querySelector('.suggestions')?.setAttribute('hidden','');log.scrollTop=log.scrollHeight});if(log)log.scrollTop=log.scrollHeight}
"""


def page(title, body, active=None):
    """Render a complete page inside the shared application shell."""
    links = [
        ("/predict", "New analysis"),
        ("/precedents", "Precedents"),
        ("/evaluation", "Evaluation"),
    ]
    rail = "".join(
        f'<a href="{url}" ' + ('aria-current="page"' if active == url else "") + f">{label}</a>"
        for url, label in links
    )
    if active:
        body = f'<div class="shell"><nav class="rail">{rail}</nav><main class="work">{body}</main></div>'
    return HTMLResponse(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)} · ModPrecedent</title><style>{APP_CSS.splitlines()[1]}{css}{STYLE}</style><header><a class="brand" href="/">ModPrecedent</a><nav><a href="/precedents">Precedents</a><a href="/predict">Open workspace</a></nav></header>{body}<script>{UPLOAD_SCRIPT}</script></html>'
    )


@site.get("/")
def home():
    """Render the animated landing page."""
    return page(
        "Home",
        """<main class="hero"><section><h1 aria-label="Would you uphold or overturn the decision?"><span aria-hidden="true">Would you</span> <span class="verdict-stage" aria-hidden="true"><span class="verdict-face uphold">uphold</span><span class="verdict-face overturn">overturn</span></span> <span aria-hidden="true">the decision?</span></h1><p>Explore how the Oversight Board weighs a social media post, its context, and past decisions.</p><a class="cta" href="/predict">Analyze a post</a></section><div class="art"><div class="folder" aria-hidden="true"><small>OVERSIGHT BOARD</small><h2>Case file</h2><span>Post and context</span><span>Meta’s action</span><span>Related precedents</span></div></div></main>""",
    )


def card(case_id, title, tag):
    """Render a compact informational card used by application pages."""
    return f'<a class="card" href="/precedents/{quote(case_id, safe="")}"><span class="tag">{_esc(tag)}</span><h3>{_esc(title)}</h3></a>'


def _render_chat_answer(answer: str, sources: dict[str, Citation]) -> str:
    """Replace internal chunk IDs visually while preserving the stored raw answer."""
    ids = list(dict.fromkeys(re.findall(r"\[([0-9a-f]{16})\]", answer)))
    valid = [chunk_id for chunk_id in ids if chunk_id in sources]
    labels = {chunk_id: index + 1 for index, chunk_id in enumerate(valid)}
    escaped = _esc(answer)
    for chunk_id, number in labels.items():
        escaped = escaped.replace(
            f"[{chunk_id}]",
            f'<a class="source-ref" href="#source-{chunk_id}">[Source {number}]</a>',
        )
    if not valid:
        return escaped
    items = "".join(
        f'<div class="message-source" id="source-{chunk_id}"><a href="{_esc(sources[chunk_id].url)}" target="_blank" rel="noopener">Source {labels[chunk_id]} — {_esc(sources[chunk_id].title)}</a><small>{_esc(sources[chunk_id].source_type.title())} evidence · internal ID retained for evaluation</small></div>'
        for chunk_id in valid
    )
    return (
        escaped
        + f'<details class="message-sources"><summary>View {len(valid)} source{"s" if len(valid) != 1 else ""}</summary>{items}</details>'
    )


@site.get("/predict")
def workspace():
    """Render the text and image analysis workspace."""
    examples = "".join(card(d.case_id, d.title, d.decision_type) for d in _decisions()[:3])
    return page(
        "New analysis",
        """<div class="split"><section><h1>Analyze a post</h1><p>Describe what someone scrolling their feed would see, including the caption and context.</p><form method="post" onsubmit="this.querySelector('button').textContent='Analyzing…';this.querySelector('button').disabled=true"><label for="post">Post and context</label><textarea id="post" name="post" minlength="15" required placeholder="What was shared, by whom, and with what context?"></textarea><div class="pair"><div><label for="action">Meta’s action</label><select id="action" name="action"><option value="removed">Removed the post</option><option value="left_up">Kept it up</option><option value="warning">Added a warning</option><option value="account_action">Restricted the account</option></select></div><div><label for="guess">Your guess</label><select name="guess" id="guess"><option value="">Skip guess</option><option value="upheld">Uphold</option><option value="overturned">Overturn</option></select></div></div><label for="method">Research method</label><select name="method" id="method"><option value="S2">Policy + precedent</option><option value="S1">Policy only</option><option value="S0">Closed book</option></select><button>Analyze post</button></form></section><aside class="well"><h2>From the library</h2>"""
        + examples
        + "</aside></div>",
        "/predict",
    )


@site.post("/predict")
def analyze(
    post: str = Form(""),
    image: UploadFile | None = File(None),  # noqa: B008 - FastAPI dependency marker
    action: str = Form(...),
    guess: str = Form(""),
    method: str = Form("S2"),
):
    """Validate a request, run prediction, and redirect to its result page."""
    post = post.strip()
    if len(post) < 15 and image and image.filename:
        try:
            post = extract_content(image.file.read())
        except OCRFailure as error:
            return page(
                "Image text unavailable",
                f'<div class="error-panel"><h1>We could not read that image</h1><p>{_esc(error)}</p><a class="cta" href="/predict">Try another image</a></div>',
                "/predict",
            )
    if len(post) < 15:
        return page(
            "Post text required",
            '<div class="error-panel"><h1>Add some post text</h1><p>Enter at least 15 characters or upload an image containing readable text.</p><a class="cta" href="/predict">Return to analysis</a></div>',
            "/predict",
        )
    req = PredictionRequest(post_description=post, meta_action=action, user_guess=guess or None)
    predictor = Predictor(app_index())
    prediction = predictor.predict(req, method)
    key = uuid4().hex
    results[key] = AnalysisSession(
        req,
        prediction,
        predictor,
        chat_sources={citation.chunk_id: citation for citation in prediction.citations},
    )
    if len(results) > 128:
        results.popitem(last=False)
    return RedirectResponse("/results/" + key, 303)


@site.post("/ocr")
def ocr_preview(image: UploadFile = File(...)):  # noqa: B008 - FastAPI dependency marker
    """Extract OCR and optional visual context for an uploaded image."""
    if image.content_type not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        return JSONResponse({"error": "Upload a PNG, JPEG, WebP, or BMP image."}, status_code=415)
    try:
        return {"text": extract_content(image.file.read())}
    except OCRFailure as error:
        return JSONResponse({"error": str(error)}, status_code=422)


@site.get("/results/{key}")
def result(key: str):
    """Render a stored prediction and its evidence-backed conversation."""
    if key not in results:
        raise HTTPException(404, "Result expired. Start a new analysis.")
    session = results[key]
    req, p = session.request, session.prediction
    evidence = "".join(
        card(c.case_id, c.title, c.source_type) + f"<p>{_esc(c.excerpt)}</p>" for c in p.citations
    )
    prediction_sources = {citation.chunk_id: citation for citation in p.citations}
    rendered_reasoning = _render_chat_answer(p.reasoning, prediction_sources)
    messages = "".join(
        f'<div class="message user"><strong>You</strong>{_esc(question)}</div>'
        f'<div class="message assistant"><strong>ModPrecedent</strong>{_render_chat_answer(answer, session.chat_sources)}</div>'
        for question, answer in session.messages
    )
    starter = (
        '<p class="status">Ask why the policy applies, which precedent mattered, or what facts could change the ruling.</p>'
        if not messages
        else ""
    )
    chat = (
        '<section class="chat"><div class="chat-head"><span class="bot-orb" aria-hidden="true"></span>'
        "<div><small>Evidence-grounded case assistant</small><h2>Ask ModPrecedent</h2></div></div>"
        f"{starter}"
        f'<div class="chat-log" aria-live="polite">{messages}</div>'
        '<div class="suggestions"><button type="button" data-question="Why does this policy apply?">Why this policy?</button>'
        '<button type="button" data-question="Which retrieved precedent mattered most?">Strongest precedent</button>'
        '<button type="button" data-question="What facts could change this outcome?">What could change?</button></div>'
        f'<p class="chat-error">{_esc(session.chat_error)}</p>'
        f'<form method="post" action="/results/{key}/chat">'
        '<label for="question">Follow-up question</label><textarea id="question" name="question" minlength="3" maxlength="1000" required placeholder="Ask about the reasoning, policy, evidence, or a counterfactual..."></textarea>'
        '<button type="submit">Ask ModPrecedent</button></form></section>'
    )
    return page(
        "Prediction",
        f'<div class="result-layout"><section class="decision-panel"><small>{"Retrieval baseline" if p.fallback_used else _esc(p.model)}</small><h1>Likely {_esc(p.outcome)}</h1><p>Your guess · {_esc(req.user_guess or "Skipped")}</p><h2>Reasoning</h2><div>{rendered_reasoning}</div><h2>Policy</h2><p>{_esc(", ".join(p.policies) or "No policy returned")}</p><details><summary>Original post</summary><p>{_esc(req.post_description)}</p></details><details><summary>Retrieved evidence ({len(p.citations)})</summary>{evidence or "No passages retrieved."}</details></section>'
        + chat
        + "</div>",
        "/predict",
    )


@site.post("/results/{key}/chat")
def follow_up(key: str, question: str = Form(...)):
    """Retrieve fresh evidence and answer one conversational follow-up turn."""
    if key not in results:
        raise HTTPException(404, "Result expired. Start a new analysis.")
    session = results[key]
    question = question.strip()
    if not 3 <= len(question) <= 1000:
        session.chat_error = "Enter a question between 3 and 1,000 characters."
        return RedirectResponse(f"/results/{key}#question", 303)
    session.chat_error = ""
    try:
        intent = intent_router().route(question)
        if intent == "outcome_patterns":
            answer = outcome_pattern_answer(_decisions(), question)
        else:
            fresh = session.predictor.follow_up_citations(
                session.request, question, session.messages
            )
            merged = {citation.chunk_id: citation for citation in fresh}
            for citation in session.prediction.citations:
                merged.setdefault(citation.chunk_id, citation)
            turn_prediction = session.prediction.model_copy(
                update={"citations": list(merged.values())[:8]}
            )
            session.chat_sources.update(merged)
            client = session.predictor.llm if intent == "poster_revision" else conversation_model()
            answer = ConversationChain(client).answer(
                session.request,
                turn_prediction,
                question,
                session.messages,
                intent,
            )
        if not answer:
            raise ValueError("The model returned an empty answer.")
    except Exception:  # noqa: BLE001 - isolate optional local/API model failures
        answer = (
            "I could not generate a new response. The current prediction is based on "
            f"{', '.join(session.prediction.policies) or 'the retrieved precedents'}: "
            f"{session.prediction.reasoning}"
        )
    session.messages.append((question, answer))
    session.messages = session.messages[-12:]
    return RedirectResponse(f"/results/{key}#question", 303)


@site.get("/precedents")
def library(q: str = ""):
    """Render the searchable precedent library."""
    cards = "".join(
        card(d.case_id, d.title, d.decision_type)
        for d in _decisions()
        if q.casefold() in " ".join([d.title, d.summary, *d.policies]).casefold()
    )
    return page(
        "Precedents",
        f'<h1>Precedent library</h1><form><label for="q">Search by case or policy</label><input id="q" name="q" value="{_esc(q)}"><button>Search</button></form>{cards or "<p>No matching decisions.</p>"}',
        "/precedents",
    )


@site.get("/precedents/{case_id}")
def reader(case_id: str):
    """Render one normalized Board decision for close reading."""
    if not any(d.case_id == case_id for d in _decisions()):
        raise HTTPException(404, "Case unavailable")
    return page("Case detail", _case_html(case_id), "/precedents")


@site.get("/evaluation")
def evaluation():
    """Render the saved experimental metrics and limitations."""
    path = Path("reports/generated/latest/metrics.csv")
    body = (
        pd.read_csv(path).to_html(index=False)
        if path.exists()
        else "<p>Results will appear here after approved cases have been evaluated.</p>"
    )
    if path.with_name("accuracy.svg").exists():
        body += '<figure><img style="max-width:100%" src="/evaluation/accuracy.svg" alt="Outcome accuracy by system"><figcaption>Accuracy on the approved Standard-decision review set</figcaption></figure><a href="/evaluation/retrieval.csv">Download retrieval experiment results</a>'
    return page(
        "Evaluation",
        '<h1>Evaluation</h1><p>Compare closed-book, policy-only, and policy-plus-precedent predictions on reviewed cases.</p><div class="tablewrap">'
        + body
        + "</div>",
        "/evaluation",
    )


@site.get("/evaluation/{artifact}")
def evaluation_artifact(artifact: str):
    """Serve an approved generated evaluation artifact safely."""
    if artifact not in {"accuracy.svg", "retrieval.csv"}:
        raise HTTPException(404)
    path = Path("reports/generated/latest") / artifact
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)

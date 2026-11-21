"""Local, server-rendered Casebook interface."""
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote
from collections import OrderedDict

import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from .app import _case_html, _decisions, _esc, APP_CSS
from .cli import load_index
from .models import PredictionRequest
from .predictor import Predictor

site = FastAPI()
results = OrderedDict()
css = """
*{box-sizing:border-box}body{margin:0;color:#27331f;background:#faf4e8;font:17px/1.55 'Source Sans 3',system-ui,sans-serif}a{color:#36472b}a:focus-visible,button:focus-visible{outline:3px solid #8456c7;outline-offset:4px}
header,nav{display:flex;align-items:center;gap:1.5rem}header{padding:18px 5vw;border-bottom:1px solid #b9baa6;justify-content:space-between}.brand,h1,h2,h3{font-family:Spectral,Georgia,serif}.brand{font-size:1.6rem;text-decoration:none;font-weight:bold}.hero{max-width:1400px;margin:auto;display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:5vw;padding:8vh 6vw}.hero h1{font-size:clamp(3rem,6vw,6rem);line-height:1;letter-spacing:-.055em;margin:0}.hero p{max-width:36em;margin:1.5em 0}.cta,button{display:inline-block;border:0;border-radius:8px;background:#ff8874;color:#351910;padding:.8em 1.3em;text-decoration:none;font:inherit;font-weight:bold;cursor:pointer}.art{position:relative;min-height:330px;display:grid;place-items:center}.folder{width:78%;padding:38px 28px;background:#efe7cf;border:1px solid #a3a38b;border-radius:12px;box-shadow:12px -14px #c8b6ed,22px -31px #495b37,8px 15px 32px #323b2922;transform:rotate(4deg)}.folder:before{content:'';position:absolute;right:15%;top:0;width:28px;height:75px;background:coral;clip-path:polygon(0 0,100% 0,100% 100%,50% 80%,0 100%)}.folder h2{font-size:3rem;margin:.4em 0}.folder span{display:block;border-top:1px solid #b6b39c;padding:5px}
.shell{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:90vh}.rail{background:#39472c;padding:30px 16px;display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;gap:7px}.rail a{color:#fff9ed;text-decoration:none;padding:10px;border-radius:6px}.rail a:hover,.rail a[aria-current]{background:#596943}.work{padding:clamp(20px,4vw,60px);min-width:0}.split{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(210px,.85fr);gap:3vw}.work h1{font-size:clamp(2.6rem,4vw,4rem);line-height:1.05;margin:.1em 0 .4em}.well{background:#e5edd9;border-radius:14px;padding:22px;align-self:start}.card{display:block;padding:14px;background:#fffaf1;border:1px solid #cdcabc;border-radius:8px;margin:10px 0;color:#26351e;text-decoration:none}.card:hover{border-color:#8060ab}.tag{background:#e4d7fa;padding:2px 8px;border-radius:12px;font-size:.8rem}.work label{display:block;font-weight:bold;margin:15px 0 6px}textarea,input,select{width:100%;padding:12px;border:1px solid #a7aa99;border-radius:7px;background:#fffaf1;font:inherit}textarea{min-height:160px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}form button{margin-top:20px}article{max-width:800px}.tablewrap{overflow:auto}td,th{padding:.6em;border-bottom:1px solid #cac9bb}
@media(max-width:790px){.hero,.split,.shell{grid-template-columns:1fr}.hero{padding-top:45px}.art{min-height:290px}.rail{flex-direction:row;flex-wrap:wrap;padding:6px 12px}.well{margin-top:20px}header{flex-wrap:wrap;gap:8px}nav{gap:12px}}
"""

def page(title, body, active=None):
    links = [("/predict","New analysis"),("/precedents","Precedents"),("/evaluation","Evaluation")]
    rail = "".join(f'<a href="{url}" '+('aria-current="page"' if active == url else '')+f'>{label}</a>' for url,label in links)
    if active:
        body = f'<div class="shell"><nav class="rail">{rail}</nav><main class="work">{body}</main></div>'
    return HTMLResponse(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)} · ModPrecedent</title><style>{APP_CSS.splitlines()[1]}{css}</style><header><a class="brand" href="/">ModPrecedent</a><nav><a href="/precedents">Precedents</a><a href="/predict">Open workspace</a></nav></header>{body}</html>')

@site.get("/")
def home():
    return page("Home", """<main class="hero"><section><h1>Would you uphold the decision?</h1><p>Explore how the Oversight Board weighs a social media post, its context, and past decisions.</p><a class="cta" href="/predict">Analyze a post</a></section><div class="art" aria-hidden="true"><div class="folder"><small>OVERSIGHT BOARD</small><h2>Case file</h2><span>Post and context</span><span>Meta’s action</span><span>Related precedents</span></div></div></main>""")

def card(case_id,title,tag):
    return f'<a class="card" href="/precedents/{quote(case_id,safe="")}"><span class="tag">{_esc(tag)}</span><h3>{_esc(title)}</h3></a>'

@site.get("/predict")
def workspace():
    examples = "".join(card(d.case_id,d.title,d.decision_type) for d in _decisions()[:3])
    return page("New analysis",'''<div class="split"><section><h1>Analyze a post</h1><p>Describe what someone scrolling their feed would see, including the caption and context.</p><form method="post" onsubmit="this.querySelector('button').textContent='Analyzing…';this.querySelector('button').disabled=true"><label for="post">Post and context</label><textarea id="post" name="post" minlength="15" required placeholder="What was shared, by whom, and with what context?"></textarea><div class="pair"><div><label for="action">Meta’s action</label><select id="action" name="action"><option value="removed">Removed the post</option><option value="left_up">Kept it up</option><option value="warning">Added a warning</option><option value="account_action">Restricted the account</option></select></div><div><label for="guess">Your guess</label><select name="guess" id="guess"><option value="">Skip guess</option><option value="upheld">Uphold</option><option value="overturned">Overturn</option></select></div></div><label for="method">Research method</label><select name="method" id="method"><option value="S2">Policy + precedent</option><option value="S1">Policy only</option><option value="S0">Closed book</option></select><button>Analyze post</button></form></section><aside class="well"><h2>From the library</h2>'''+examples+'</aside></div>',"/predict")

@site.post("/predict")
def analyze(post:str=Form(...,min_length=15),action:str=Form(...),guess:str=Form(""),method:str=Form("S2")):
    req = PredictionRequest(post_description=post,meta_action=action,user_guess=guess or None)
    prediction = Predictor(load_index()).predict(req,method)
    key = uuid4().hex
    results[key] = (req,prediction)
    if len(results)>128:
        results.popitem(last=False)
    return RedirectResponse("/results/"+key,303)

@site.get("/results/{key}")
def result(key:str):
    if key not in results:
        raise HTTPException(404,"Result expired. Start a new analysis.")
    req,p=results[key]
    evidence="".join(card(c.case_id,c.title,c.source_type)+f'<p>{_esc(c.excerpt)}</p>' for c in p.citations)
    return page("Prediction",f'<div class="split"><article><small>{"Retrieval baseline" if p.fallback_used else _esc(p.model)}</small><h1>Likely {_esc(p.outcome)}</h1><p>Your guess · {_esc(req.user_guess or "Skipped")}</p><h2>Reasoning</h2><p>{_esc(p.reasoning)}</p><h2>Policy</h2><p>{_esc(", ".join(p.policies) or "No policy returned")}</p><details><summary>Original post</summary><p>{_esc(req.post_description)}</p></details></article><aside class="well"><h2>Retrieved evidence</h2>{evidence or "No passages retrieved."}</aside></div>',"/predict")

@site.get("/precedents")
def library(q:str=""):
    cards="".join(card(d.case_id,d.title,d.decision_type) for d in _decisions() if q.casefold() in " ".join([d.title,d.summary,*d.policies]).casefold())
    return page("Precedents",f'<h1>Precedent library</h1><form><label for="q">Search by case or policy</label><input id="q" name="q" value="{_esc(q)}"><button>Search</button></form>{cards or "<p>No matching decisions.</p>"}',"/precedents")

@site.get("/precedents/{case_id}")
def reader(case_id:str):
    if not any(d.case_id==case_id for d in _decisions()):
        raise HTTPException(404,"Case unavailable")
    return page("Case detail",_case_html(case_id),"/precedents")

@site.get("/evaluation")
def evaluation():
    path=Path("reports/generated/latest/metrics.csv")
    body=pd.read_csv(path).to_html(index=False) if path.exists() else "<p>Results will appear here after approved cases have been evaluated.</p>"
    return page("Evaluation",'<h1>Evaluation</h1><p>Compare closed-book, policy-only, and policy-plus-precedent predictions on reviewed cases.</p><div class="tablewrap">'+body+'</div>',"/evaluation")

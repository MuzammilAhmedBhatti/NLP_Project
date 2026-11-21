"""Shared Casebook controls, reading surfaces and motion."""

STYLE = """
@media(prefers-reduced-motion:no-preference){
 .folder{animation:fileFloat 4.6s ease-in-out infinite alternate}
 .folder span{animation:paperMark 6s ease-in-out infinite;background:linear-gradient(90deg,#c6dba6,transparent) left center/0% 90% no-repeat}
 .folder span:nth-of-type(2){animation-delay:1.4s}.folder span:nth-of-type(3){animation-delay:2.8s}
 @keyframes fileFloat{to{translate:0 -9px;rotate:-2deg}}
 @keyframes paperMark{0%,55%,100%{background-size:0% 90%}25%,40%{background-size:94% 90%}}
 .card,.tablewrap,.case-reader,.split>article{animation:unfile .38s cubic-bezier(.2,.8,.25,1) backwards}
 .card:nth-of-type(2){animation-delay:.06s}.card:nth-of-type(3){animation-delay:.12s}
 @keyframes unfile{from{opacity:0;translate:0 9px}}
 form button:disabled,.upload:has(+.loading){background-image:repeating-linear-gradient(115deg,transparent 0 12px,#fff5 12px 20px);background-size:44px 100%;animation:travel 1s linear infinite}
 @keyframes travel{to{background-position:44px 0}}
 .rail a:hover,.rail a:focus-visible{translate:3px 0}.rail a{transition:translate .2s ease,background-color .2s}
 .tablewrap img{animation:plotIn .65s ease-out}
 @keyframes plotIn{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0)}}
}
.motion-toggle{position:absolute;bottom:-2rem;right:8%;font:600 .85rem system-ui;padding:.35rem .85rem;border:1px solid #bbc3aa;background:#fffaf1;color:#34462f}
.art:has(.motion-toggle[aria-pressed=true]) .folder,.art:has(.motion-toggle[aria-pressed=true]) span{animation-play-state:paused}
@media(prefers-reduced-motion:reduce){.motion-toggle{display:none}}
:root{--olive:#34462f;--ink:#243322;--peach:#ff947c;--iris:#ddd1ff;--line:#d8d7c8;--spring:cubic-bezier(.22,1.32,.36,1)}
body{color:var(--ink);background:#faf7ee}header{background:#fffdf5;border-color:var(--line)}header nav a{text-decoration:none;font-weight:600;padding:.45rem .8rem;border-radius:2rem}header nav a:last-child{background:var(--olive);color:white}
.brand{letter-spacing:-.055em}.brand:before{content:"";display:inline-block;width:.7em;height:.7em;margin-right:.4em;background:var(--peach);border-radius:3px;box-shadow:3px -3px var(--olive);transform:rotate(-9deg)}
.hero{min-height:73vh}.folder{transition:transform .5s var(--spring)}.art:hover .folder{transform:rotate(-2deg) translateY(-5px)}
.rail{background:var(--olive);padding-top:2rem}.rail a{border-radius:10px}.rail a[aria-current]{background:#dcebaf;color:#26341f;font-weight:700}
.work{max-width:1550px;width:100%;margin:auto}.work h1,.hero h1,.well h2{letter-spacing:-.05em}.well{background:#e8eedb;border:1px solid #d3ddc2}.well h2{margin-top:0}.card{border-radius:12px;background:#fffdf7;text-decoration:none;transition:transform .23s var(--spring),box-shadow .23s,border-color .23s}.card:hover{transform:translateY(-3px);box-shadow:0 7px 0 #35482d12}.card h3{margin:.45em 0 .2em}.tag{background:var(--iris);color:#49376b;font-weight:650}
.cta,button{background:var(--peach);border-radius:100px;transition:transform .23s var(--spring),box-shadow .23s}.cta:hover,button:hover{transform:translateY(-2px);box-shadow:0 4px 0 #763c2d}button:active,.cta:active{transform:scale(.97)}button:disabled{opacity:.75;cursor:progress}
textarea,input,select{border:1px solid #acb49f;border-radius:12px;background:#fffdf8;transition:border-color .18s,box-shadow .18s}textarea:focus,input:focus,select:focus-visible{outline:0;border-color:#72599a;box-shadow:0 0 0 4px #bfa7eb55}textarea{resize:vertical}.work label{font-size:.95rem}select{cursor:pointer}
.optional{font-weight:400;color:#697064}.upload{position:relative;display:flex;align-items:center;gap:.75rem;border:1.5px dashed #87947d;border-radius:14px;padding:.75rem;background:#f4f7e9;transition:border-color .2s,background .2s}.upload:hover{border-color:#705295;background:#f2edff}.upload input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}.upload:before{content:"＋";display:grid;place-items:center;width:2rem;height:2rem;border-radius:50%;background:var(--iris);font-weight:700}.status{min-height:1.5em;margin:.45rem 0}.status.loading{color:#635477}.status.success{color:#36582c}.status.error,.error-panel{color:#7e3025}.status.loading:before{content:"";display:inline-block;width:.7em;height:.7em;margin-right:.45em;border:2px solid #a18bbd;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite}.error-panel{max-width:650px;padding:clamp(1.5rem,5vw,4rem);background:#fff0e9;border:1px solid #e7b7a9;border-radius:18px}@keyframes spin{to{transform:rotate(1turn)}}
@supports (appearance:base-select){select,::picker(select){appearance:base-select}select::picker-icon{transition:rotate .25s var(--spring)}select:open::picker-icon{rotate:180deg}::picker(select){border:1px solid var(--line);border-radius:12px;padding:5px;background:#fffdf8;box-shadow:0 12px 32px #20302022;opacity:0;transform:translateY(-6px) scale(.98);transition:opacity .18s,transform .23s var(--spring),display .23s allow-discrete,overlay .23s allow-discrete}::picker(select):popover-open{opacity:1;transform:none}@starting-style{::picker(select):popover-open{opacity:0;transform:translateY(-6px) scale(.98)}}option{padding:.6em .75em;border-radius:7px}option:checked{background:#ece5ff}option:hover,option:focus-visible{background:#e4edcc}}
.case-reader{max-width:76ch}.case-reader p{line-height:1.75}.case-meta{color:#5a6355}.case-reader h2{padding-top:.8em;border-top:1px solid var(--line)}details,.tablewrap{border:1px solid var(--line);background:#fffdf7;border-radius:12px;padding:1rem;margin:1rem 0}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%}th{background:#e6edda}
@view-transition{navigation:auto}::view-transition-old(root){animation:depart .14s ease-in both}::view-transition-new(root){animation:arrive .32s cubic-bezier(.2,.85,.3,1) both}@keyframes depart{to{opacity:0;transform:translateY(-4px)}}@keyframes arrive{from{opacity:0;transform:translateY(9px)}}@media(max-width:790px){.rail{padding:.5rem}.hero{min-height:auto}.work{margin:0}header nav{gap:0}.pair>*{min-width:0}}@media(prefers-reduced-motion:reduce){*,*::before,*::after,::view-transition-old(root),::view-transition-new(root){animation:none!important;transition:none!important}}
"""

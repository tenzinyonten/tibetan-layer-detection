#!/usr/bin/env python3
"""Human review tool for quotation annotations. Book triage first, then spans.

Runs a tiny local server and opens a keyboard-driven review UI. Every decision is
written to disk immediately, so you can close the tab and pick up where you left
off. At the end it exports the set of spans that are safe to train and evaluate on.

    python3 quotation/src/review_quotation_spans.py                 # start reviewing
    python3 quotation/src/review_quotation_spans.py --export        # just write the usable set
    python3 quotation/src/review_quotation_spans.py --reset-spans   # keep book verdicts, redo spans

Two stages:

  BOOKS   ~420 decisions. Each screen shows the diagnostics that matter
          (closed-by rate, median length, class mix, boundary health), a
          recommendation with its reason, and eight sampled spans in context.
          Most are one keystroke.

  SPANS   Only the flagged spans inside accepted books. Accept, reject, or mark
          the boundary for snapping.

Keys:  A accept · R reject · S needs span review (books) / snap boundary (spans)
       ← → move · U undo last · E export

Ledger:  Data/audit/review_ledger.json      (every decision, resumable)
Export:  Data/audit/quotation_usable.json   (the clean set + per-book summary)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import statistics
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / "Data" / "repos"
AUDIT = ROOT / "Data" / "audit"
LEDGER = AUDIT / "review_ledger.json"
EXPORT = AUDIT / "quotation_gold.json"

BLOCK = re.compile(
    r"^  (?P<ann>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)", re.M)
DELIM = "་༌ \t\n\r།༎༏༑༈༔༴༵༶྅"
PUN = set(DELIM)
SHAD = set("།༎༏༑༈༔")
SPLIT = re.compile("[" + re.escape(DELIM) + "]+")

SPEECH = {"གསུངས", "སྩལ", "སྨྲས", "བཤད", "བརྗོད", "ཞུས", "གསོལ", "ཟེར"}
NOMZ = {"པ", "པར", "བ", "བར", "པོ", "པའི", "བའི"}
ERG = {"གིས", "ཀྱིས", "གྱིས", "ཡིས"}
ERG_NAME = re.compile(r"(?:པས|པོས|བས|ཆེས|ལྷས|ཧས|བོས|མས|རས|ཏིས|བུས|རྗེས|མཚོས|གྲགས|གེས|ཤས)$")
TERM = {"དུ", "ཏུ", "རུ", "སུ"}
MARKERS = {"ལས", "ནས"} | TERM | ERG
CLOSERS = ("ཞེས", "ཅེས", "གསུངས")
COLOUR = {"TEXT": "#2563eb", "PERSON": "#059669", "FORMULA": "#7c3aed",
          "CHAINED": "#d97706", "MISC": "#e11d48"}


def syl(x): return [p for p in SPLIT.split(x) if p]


def prev2(t, s):
    i, sh = s - 1, False
    while i >= 0 and t[i] in PUN:
        if t[i] in SHAD: sh = True
        i -= 1
    out = []
    for _ in range(2):
        if i < 0: break
        j = i
        while j >= 0 and t[j] not in PUN: j -= 1
        out.append(t[j + 1:i + 1]); i = j - 1
        while i >= 0 and t[i] in PUN: i -= 1
    return (out + ["", ""])[:2], sh


def classify(a, b):
    if a == "རེ" and b == "ན": return "PERSON"
    if a == "དང" and b in ("ཞེས", "ཅེས"): return "CHAINED"
    if a in NOMZ and b in SPEECH: return "PERSON"
    if a in ("ཀྱང", "ཡང") and b: a = b
    if a in ("ལས", "ནས"): return "PERSON" if b in ("ཞལ", "གསུང") else "TEXT"
    if a in SPEECH or a in ERG or ERG_NAME.search(a): return "PERSON"
    if a in TERM: return "FORMULA"
    if a in ("ཞེས", "ཅེས"): return "CHAINED"
    return "MISC"


def meta_title(book: str) -> str:
    p = REPOS / book / f"{book}.opf" / "meta.yml"
    if not p.is_file(): return ""
    txt = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s+en:\s*(.+)$", txt, re.M)
    if m and len(m.group(1).strip()) > 4: return m.group(1).strip().strip("'\"")[:120]
    m = re.search(r"^\s+title:\s*(.+)$", txt, re.M)
    return m.group(1).strip()[:120] if m else ""


def scan_book(book: str) -> dict | None:
    opf = REPOS / book / f"{book}.opf"
    base = opf / "base" / "v001.txt"
    layer = path = None
    for name in ("Quotation", "Citation"):
        cand = opf / "layers" / "v001" / f"{name}.yml"
        if cand.is_file(): layer, path = name, cand; break
    if not (base.is_file() and path): return None
    t = base.read_text(encoding="utf-8", errors="replace")
    spans = []
    for m in BLOCK.finditer(path.read_text(encoding="utf-8", errors="replace")):
        s, e = int(m["s"]), int(m["e"])
        if e < s: continue
        (a, b), sh = prev2(t, s)
        body = t[s:e + 1]
        tail = syl(body[-40:])
        after = syl(t[e + 1:e + 40])
        flags = []
        bad_s = s > 0 and t[s - 1] not in PUN
        bad_e = e + 1 < len(t) and t[e + 1] not in PUN
        if bad_s: flags.append("start cuts a syllable")
        if bad_e: flags.append("end cuts a syllable")
        if len(body) <= 30 and tail and tail[-1] in MARKERS:
            flags.append("looks like the introducer, not the quote")
        if not sh: flags.append("starts mid-sentence")
        end_mid = True
        for k in range(1, 7):
            q = e + k
            if q >= len(t) or t[q] in SHAD: end_mid = False; break
            if t[q] not in PUN: break
        if end_mid: flags.append("ends mid-sentence")
        spans.append({"both_mid": (not sh) and end_mid,
                      "ann": m["ann"], "start": s, "end": e, "len": e - s + 1,
                      "cls": classify(a, b), "prev": a, "new_sent": sh,
                      "closed": after[0] if after and after[0] in CLOSERS else "",
                      "flags": flags, "bad_s": bad_s, "bad_e": bad_e})
    if not spans: return None
    spans.sort(key=lambda x: x["start"])
    # Some books annotate each VERSE LINE as its own span. Consecutive spans with a
    # tiny gap are one logical quotation, so only the LAST of a run can carry a
    # closer -- scoring every span for a closer would wrongly condemn those books.
    for i, sp in enumerate(spans):
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        sp["group_end"] = not (nxt and nxt["start"] - sp["end"] <= 5)
    n = len(spans)
    counts = {}
    for sp in spans: counts[sp["cls"]] = counts.get(sp["cls"], 0) + 1
    e_delim = sum(1 for sp in spans if t[sp["end"]] in PUN)
    stats = {
        "n": n, "layer": layer, "batch": "old" if book.startswith("P0") else "new",
        "median_len": statistics.median(sp["len"] for sp in spans),
        "closed_pct": (100 * sum(1 for sp in spans if sp["group_end"] and sp["closed"])
                       / max(sum(1 for sp in spans if sp["group_end"]), 1)),
        "groups": sum(1 for sp in spans if sp["group_end"]),
        "line_split_pct": 100 * (1 - sum(1 for sp in spans if sp["group_end"]) / n),
        "misaligned_pct": 100 * sum(1 for sp in spans if sp["bad_s"] or sp["bad_e"]) / n,
        "intro_pct": 100 * sum(1 for sp in spans
                               if "looks like the introducer, not the quote" in sp["flags"]) / n,
        "midsent_pct": 100 * sum(1 for sp in spans if not sp["new_sent"]) / n,
        "both_mid_pct": 100 * sum(1 for sp in spans if sp["both_mid"]) / n,
        "exclusive_end_pct": 100 * e_delim / n,
        "counts": counts, "title": meta_title(book), "doc_chars": len(t),
    }
    stats["flagged"] = sum(1 for sp in spans if sp["flags"])
    return {"book": book, "text": t, "spans": spans, "stats": stats}


def recommend(st: dict) -> tuple[str, str]:
    """(verdict, reason) — pre-computed so most books are one keystroke."""
    if st["intro_pct"] > 50:
        return "reject", (f"{st['intro_pct']:.0f}% of spans are just the citation introducer — "
                          "this book annotates source references, not quotations (cf. P000115)")
    if st["n"] < 10:
        return "reject", f"only {st['n']} spans — too few to be worth including"
    # Boundary health is checked BEFORE content, because a span that ends mid-syllable
    # makes the closed-by rate unmeasurable: what follows the span is a fragment, not
    # a word. Snap first, then judge the content on the repaired offsets.
    if st["both_mid_pct"] >= 50:
        return "reject", (f"{st['both_mid_pct']:.0f}% of spans both start AND end mid-sentence — "
                          "the boundaries do not line up with sentence structure at all")
    if st["exclusive_end_pct"] > 90 and st["misaligned_pct"] > 80:
        return "snap", ("this book uses exclusive-end offsets — every span needs end−1 "
                        "(cf. P000128). Content looks fine.")
    if st["misaligned_pct"] > 55:
        return "snap", (f"{st['misaligned_pct']:.0f}% of boundaries land mid-syllable, about "
                        f"chance level — offsets were never snapped, so the closed-by rate "
                        f"({st['closed_pct']:.0f}%) can't be trusted yet. Snap, then re-judge.")
    if st["closed_pct"] < 20 and st["groups"] >= 20:
        return "reject", (f"only {st['closed_pct']:.0f}% of quotations are closed by ཞེས/ཅེས/གསུངས "
                          f"against a corpus norm of ~69% (over {st['groups']} grouped quotations, "
                          "so line-splitting is accounted for) — likely root or instruction text "
                          "rather than citations (cf. P000250)")
    return "accept", (f"{st['closed_pct']:.0f}% closed, {st['misaligned_pct']:.0f}% misaligned, "
                      f"median {st['median_len']:.0f} chars — looks healthy")


def load_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"version": 1, "books": {}, "spans": {}, "updated": None}


def save_ledger(led: dict) -> None:
    led["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    AUDIT.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(LEDGER)


def snap(t: str, pos: int, is_start: bool, rng: int = 3) -> int | None:
    for k in sorted(range(-rng, rng + 1), key=abs):
        p = pos + k
        if not (0 <= p < len(t)): continue
        ok = (p == 0 or t[p - 1] in PUN) if is_start else (p + 1 >= len(t) or t[p + 1] in PUN)
        if ok: return p
    return None


def do_export(cache: dict, led: dict) -> dict:
    """Book-level gold set.

    A book is kept whole or dropped whole. That is deliberate: when a book's
    boundaries are placed at chance (P000101 and friends sit at ~29% aligned,
    exactly the delimiter density), a span that happens to land on clean edges
    only got lucky -- it is not evidence the annotator meant those offsets. So a
    corrupted book contributes nothing, and an accepted book contributes all of
    its spans.

    Individual spans you rejected by hand are still honoured, and `snap` books
    get their offsets repaired on the way out.
    """
    books_out, kept, dropped, snapped = {}, 0, 0, 0
    for book, rec in led["books"].items():
        data = cache.get(book) or scan_book(book)
        if not data: continue
        if rec["verdict"] == "reject":
            dropped += data["stats"]["n"]; continue
        t = data["text"]; out = []
        for sp in data["spans"]:
            if led["spans"].get(f"{book}:{sp['ann']}", {}).get("verdict") == "reject":
                dropped += 1; continue
            s0, e0 = sp["start"], sp["end"]
            if rec["verdict"] == "snap":
                if data["stats"]["exclusive_end_pct"] > 90: e0 -= 1
                ns, ne = snap(t, s0, True), snap(t, e0, False)
                if ns is None or ne is None or ne < ns:
                    dropped += 1; continue
                if (ns, ne) != (s0, e0): snapped += 1
                s0, e0 = ns, ne
            out.append({"ann": sp["ann"], "start": s0, "end": e0,
                        "cls": sp["cls"], "closed": bool(sp["closed"])})
            kept += 1
        if not out: continue
        st = data["stats"]
        books_out[book] = {"layer": st["layer"], "batch": st["batch"],
                           "verdict": rec["verdict"], "title": st["title"],
                           "doc_chars": st["doc_chars"],
                           "closed_pct": round(st["closed_pct"], 1),
                           "misaligned_pct": round(st["misaligned_pct"], 1),
                           "both_mid_pct": round(st["both_mid_pct"], 1),
                           "line_split_pct": round(st["line_split_pct"], 1),
                           "n_original": st["n"], "spans": out}
    cls_counts = {}
    for bo in books_out.values():
        for sp in bo["spans"]: cls_counts[sp["cls"]] = cls_counts.get(sp["cls"], 0) + 1
    undecided = sum(d["stats"]["n"] for b, d in cache.items()
                    if d and b not in led["books"])
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "policy": "book-level: an accepted book contributes every span; "
                         "a rejected book contributes none",
               "kept_books": len(books_out), "kept_spans": kept,
               "dropped_spans": dropped, "snapped_spans": snapped,
               "spans_in_unreviewed_books": undecided,
               "class_counts": cls_counts, "books": books_out}
    EXPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>gold set review</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1c1917;overflow:hidden;
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.app{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:100vh}
header{padding:12px 24px;border-bottom:1px solid #e7e5e4;background:#fff}
h2{margin:0;font-size:17px;font-weight:600}
.mut{color:#78716c;font-size:12.5px}
.bar{height:3px;background:#e7e5e4;border-radius:9px;overflow:hidden;margin-top:9px}
.bar>div{height:100%;background:#059669;transition:width .25s}
main{overflow:auto;padding:14px 22px 80px;background:#fafaf9}
.row{background:#fff;border:1px solid #e7e5e4;border-left:4px solid;border-radius:8px;
 padding:9px 13px;margin-bottom:7px}
.row.f{border-color:#fca5a5;background:#fffbfb}
.tib{font-family:"Noto Serif Tibetan","Microsoft Himalaya","Kailasa",serif;
 font-size:18px;line-height:2.1}
.ctx{color:#a8a29e}
.sp{padding:2px 0;border-radius:3px;border-bottom:2px solid}
.gap{color:#a8a29e;font-size:13px;padding:0 5px}
.meta{color:#78716c;font-size:11.5px;margin-top:4px}
.flag{display:inline-block;background:#fee2e2;color:#991b1b;border-radius:4px;
 padding:1px 6px;margin:3px 4px 0 0;font-size:10.5px}
aside{border-left:1px solid #e7e5e4;background:#fff;overflow:auto;padding:18px}
.rec{padding:10px 12px;border-radius:8px;font-size:12.5px;border:1px solid;margin:0 0 14px}
.rec.accept{background:#ecfdf5;border-color:#6ee7b7;color:#065f46}
.rec.reject{background:#fef2f2;border-color:#fca5a5;color:#991b1b}
.rec.snap{background:#fffbeb;border-color:#fcd34d;color:#92400e}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:14px}
.t{background:#fafaf9;border:1px solid #e7e5e4;border-radius:7px;padding:8px 10px}
.t b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.t span{font-size:10.5px;color:#78716c}
.t.bad b{color:#dc2626}
button{font:inherit;font-size:13px;padding:9px 13px;border-radius:7px;border:1px solid #e7e5e4;
 background:#fafaf9;color:#1c1917;cursor:pointer;font-weight:600;width:100%;
 margin-bottom:7px;display:flex;justify-content:space-between;align-items:center}
button.a{background:#059669;color:#fff;border-color:#059669}
button.r{background:#dc2626;color:#fff;border-color:#dc2626}
button.s{background:#d97706;color:#fff;border-color:#d97706}
kbd{font:11px ui-monospace,monospace;background:rgba(0,0,0,.18);border-radius:4px;padding:1px 6px}
hr{border:0;border-top:1px solid #e7e5e4;margin:14px 0}
.leg{display:flex;flex-wrap:wrap;gap:9px;font-size:11px;margin-bottom:12px}
.leg i{display:inline-block;width:13px;height:11px;margin-right:5px;vertical-align:-1px;
 border-radius:3px;border-bottom:2px solid}
.toast{position:fixed;top:14px;left:50%;transform:translateX(-50%);padding:6px 16px;
 border-radius:99px;font-weight:700;font-size:12px;color:#fff;opacity:0;transition:opacity .25s;
 pointer-events:none;z-index:20}
.filt{display:flex;gap:6px;margin-bottom:10px}
.filt button{width:auto;padding:5px 11px;font-size:12px;margin:0}
.filt button.on{background:#1c1917;color:#fff;border-color:#1c1917}
</style></head><body>
<div class="toast" id="toast"></div>
<div class="app">
 <div style="display:flex;flex-direction:column;min-height:0">
  <header><h2 id="bk">loading…</h2><div class="mut" id="meta"></div>
   <div class="bar"><div id="pb" style="width:0"></div></div></header>
  <main id="doc"></main></div>
 <aside id="side"></aside></div>
<script>
let books=[],bi=0,B=null,only=false;
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const COL={TEXT:'#2563eb',PERSON:'#059669',FORMULA:'#7c3aed',CHAINED:'#d97706',MISC:'#e11d48'};

async function boot(){books=await (await fetch('/api/books')).json();
  bi=books.findIndex(b=>!b.verdict); if(bi<0)bi=0; open_();}

async function open_(){
  while(bi<books.length && books[bi].verdict) bi++;
  if(bi>=books.length){done();return;}
  if(bi<0)bi=0;
  B=await (await fetch('/api/book?id='+books[bi].book)).json();
  draw(); paint();
  document.getElementById('doc').scrollTop=0;
}
function draw(){
  const pick=only==='m'?B.rows.filter(r=>r.both_mid)
                 :only==='f'?B.rows.filter(r=>r.flags.length):B.rows;
  const rows=pick.map(r=>{
    const c=COL[r.cls];
    const body=r.trunc
      ? `<span class="sp" style="background:${c}30;border-color:${c}">${esc(r.head)}</span>`+
        `<span class="gap">⟨${r.len-r.head.length-r.tail.length} more⟩</span>`+
        `<span class="sp" style="background:${c}30;border-color:${c}">${esc(r.tail)}</span>`
      : `<span class="sp" style="background:${c}30;border-color:${c}">${esc(r.head)}</span>`;
    return `<div class="row ${r.flags.length?'f':''}" style="border-left-color:${c}">
      <div class="tib"><span class="ctx">${esc(r.before)}</span>${body}<span class="ctx">${esc(r.after)}</span></div>
      <div class="meta">${r.cls} · ${r.len} chars · ${r.start}–${r.end} ·
        prev <b>${esc(r.prev)||'—'}</b> · ${r.closed?'closed by '+r.closed:'not closed'}</div>
      ${r.flags.map(f=>`<span class="flag">${esc(f)}</span>`).join('')}</div>`;
  }).join('');
  document.getElementById('doc').innerHTML=
    `<div class="filt">
      <button class="${only?'':'on'}" onclick="only=false;draw()">all ${B.rows.length}</button>
      <button class="${only==='f'?'on':''}" onclick="only='f';draw()">flagged ${B.stats.flagged}</button>
      <button class="${only==='m'?'on':''}" onclick="only='m';draw()">starts+ends mid-sentence ${B.rows.filter(r=>r.both_mid).length}</button>
     </div>`+ (rows||'<div class="mut">no spans match</div>');
}
function paint(){
  const st=B.stats,c=st.counts||{};
  document.getElementById('bk').textContent=B.book+'  ·  '+st.layer+'.yml  ·  '+st.batch;
  const left=books.filter(b=>!b.verdict).length;
  const pos=books.slice(0,bi).filter(b=>!b.verdict).length+1;
  const done=books.length-left;
  document.getElementById('meta').innerHTML=
    (B.title?esc(B.title)+' &nbsp;·&nbsp; ':'')+
    `<b>${Math.min(pos,left)} of ${left} left to review</b>`+
    ` &nbsp;·&nbsp; ${done} of ${books.length} books already decided`;
  document.getElementById('pb').style.width=(100*done/books.length)+'%';
  const tile=(v,l,bad)=>`<div class="t ${bad?'bad':''}"><b>${v}</b><span>${l}</span></div>`;
  const cur=books[bi].verdict;
  document.getElementById('side').innerHTML=`
   <div class="rec ${B.rec}"><b>${B.rec.toUpperCase()}</b> — ${esc(B.reason)}</div>
   ${cur?`<div class="mut" style="margin:-8px 0 12px">already marked <b>${cur}</b></div>`:''}
   <div class="tiles">
     ${tile(st.n,'spans')}${tile(st.groups,'quotations')}
     ${tile(st.median_len.toFixed(0),'median chars',st.median_len<40)}
     ${tile(st.closed_pct.toFixed(0)+'%','closed by ཞེས',st.closed_pct<25)}
     ${tile(st.misaligned_pct.toFixed(0)+'%','misaligned',st.misaligned_pct>55)}
     ${tile(st.both_mid_pct.toFixed(0)+'%','starts+ends mid-sent',st.both_mid_pct>=50)}
   </div>
   <div class="leg">${Object.entries(COL).map(([k,v])=>
     `<span><i style="background:${v}30;border-color:${v}"></i>${k} ${c[k]||0}</span>`).join('')}</div>
   <button class="a" onclick="mark('accept')">Accept — goes in gold<kbd>A</kbd></button>
   <button class="s" onclick="mark('snap')">Accept + snap boundaries<kbd>S</kbd></button>
   <button class="r" onclick="mark('reject')">Reject whole book<kbd>R</kbd></button>
   <hr>
   <button onclick="go(-1)">← previous book<kbd>←</kbd></button>
   <button onclick="go(1)">skip / next book<kbd>→</kbd></button>
   <button onclick="exp()">Export gold set<kbd>E</kbd></button>
   <div class="mut" style="margin-top:10px">Each row is one span with 10 characters of
    context either side. Long spans show head and tail.</div>`;
}
function toast(t,c){const e=document.getElementById('toast');
  e.textContent=t;e.style.background=c;e.style.opacity='1';setTimeout(()=>e.style.opacity='0',700);}
async function mark(v){
  await fetch('/api/book',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({book:B.book,verdict:v})});
  books[bi].verdict=v;
  toast(v==='reject'?'rejected':(v==='snap'?'accepted + snap':'accepted'),
        v==='reject'?'#dc2626':(v==='snap'?'#d97706':'#059669'));
  while(bi<books.length && books[bi].verdict) bi++;
  open_();
}
function go(d){
  let j=bi+d;
  while(j>=0 && j<books.length && books[j].verdict) j+=d;   // step over decided books
  if(j<0)j=0; if(j>=books.length){done();return;}
  bi=j; open_();
}
async function exp(){
  const r=await (await fetch('/api/export',{method:'POST'})).json();
  alert(`Gold set exported\n\n${r.kept_spans} spans from ${r.kept_books} books\n`+
        `${r.dropped_spans} dropped · ${r.snapped_spans} snapped\n\nData/audit/quotation_gold.json`);
}
function done(){document.getElementById('doc').innerHTML=
  '<div style="text-align:center;padding:90px"><h2>All books reviewed</h2></div>';
  document.getElementById('side').innerHTML=
  '<button class="a" onclick="exp()">Export gold set<kbd>E</kbd></button>';}
addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||!B)return;
  const k=e.key.toLowerCase();
  if(k==='a')mark('accept'); else if(k==='r')mark('reject'); else if(k==='s')mark('snap');
  else if(k==='f'){only=only==='f'?false:'f';draw();}
  else if(k==='m'){only=only==='m'?false:'m';draw();}
  else if(k==='x')mark('reject_all'); else if(e.key==='ArrowLeft')go(-1);
  else if(e.key==='ArrowRight')go(1); else if(k==='e')exp(); else return;
  e.preventDefault();
});
boot();
</script></body></html>"""

ORDER = "broken"  # queue order: "broken" (worst first) or "size"
CTX = 10        # characters of context either side
HEAD = 55       # chars of a long span shown at each end
class State:
    def __init__(self):
        print("scanning corpus …")
        self.cache = {}
        names = sorted(d.name for d in REPOS.iterdir() if d.is_dir())
        for n, b in enumerate(names, 1):
            d = scan_book(b)
            if d: self.cache[b] = d
            if n % 100 == 0: print(f"  {n}/{len(names)}")
        self.led = load_ledger()
        pinned = self.led.get("pinned", [])
        base = ((lambda b: -self.cache[b]["stats"]["n"]) if ORDER == "size"
                else (lambda b: (-self.cache[b]["stats"]["both_mid_pct"],
                                 -self.cache[b]["stats"]["n"])))
        # books in led["pinned"] jump the queue, in the order listed
        self.order = sorted(self.cache,
                            key=lambda b: (pinned.index(b) if b in pinned else len(pinned),
                                           base(b)))
        print(f"ready: {len(self.cache)} books, "
              f"{sum(d['stats']['n'] for d in self.cache.values()):,} spans")

    def books(self):
        return [{"book": b, "n": self.cache[b]["stats"]["n"],
                 "verdict": self.led["books"].get(b, {}).get("verdict")} for b in self.order]

    def book_payload(self, book):
        d = self.cache[book]; t = d["text"]
        flat = lambda x: x.replace("\n", " ").replace("\r", " ")
        rows = []
        for sp in d["spans"]:
            s, e = sp["start"], sp["end"]
            body = t[s:e + 1]
            trunc = len(body) > HEAD * 2 + 20
            rows.append({
                "ann": sp["ann"], "start": s, "end": e, "len": sp["len"], "cls": sp["cls"],
                "both_mid": sp["both_mid"],
                "prev": sp["prev"], "closed": sp["closed"], "flags": sp["flags"],
                "before": flat(t[max(0, s - CTX):s]),
                "after": flat(t[e + 1:e + 1 + CTX]),
                "head": flat(body[:HEAD] if trunc else body),
                "tail": flat(body[-HEAD:]) if trunc else "",
                "trunc": trunc})
        rec, why = recommend(d["stats"])
        return {"book": book, "title": d["stats"]["title"], "stats": d["stats"],
                "rec": rec, "reason": why, "rows": rows}

    def book(self, book, verdict):
        self.led["books"][book] = {
            "verdict": verdict, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_ledger(self.led)


def serve(state: State, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _s(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def _json(self, o):
            self._s(json.dumps(o, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/books": self._json(state.books())
            elif u.path == "/api/book":
                self._json(state.book_payload(dict(p.split("=") for p in u.query.split("&"))["id"]))
            else: self._s(PAGE.encode(), "text/html; charset=utf-8")
        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            b = json.loads(self.rfile.read(n)) if n else {}
            if u.path == "/api/book": state.book(b["book"], b["verdict"]); self._json({"ok": 1})
            elif u.path == "/api/export":
                r = do_export(state.cache, state.led)
                self._json({k: v for k, v in r.items() if k != "books"})
            else: self._json({"ok": 0})
    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"\n  review UI  →  http://127.0.0.1:{port}/\n  ledger     →  {LEDGER}\n  ctrl-c to stop\n")
    threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nstopped — progress saved")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--order", choices=("broken", "size"), default="broken",
                    help="broken = most obviously-broken books first (default)")
    a = ap.parse_args()
    globals()["ORDER"] = a.order
    if a.export:
        led = load_ledger()
        cache = {b: scan_book(b) for b in led["books"]}
        r = do_export({k: v for k, v in cache.items() if v}, led)
        print(f"gold: {r['kept_spans']:,} spans from {r['kept_books']} books · "
              f"{r['dropped_spans']:,} dropped · {r['snapped_spans']:,} snapped → {EXPORT}")
        return 0
    serve(State(), a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

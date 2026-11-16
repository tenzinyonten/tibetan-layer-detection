#!/usr/bin/env python3
"""Trial: Gemini with Google Search grounding checks the AUTHOR of a flagged work (validation split only, never test).
Same works, same order and same baseline as ta_agent_trial.py (flagged validation works of the saved gepa_v2 flash-lite answers, seed 0), so the
two trials are comparable. PAID (grounding is not on the free tier): the key must have billing. The key comes only from GEMINI_API_KEY.

    GEMINI_API_KEY=... python3 title_author/src/ta_search_trial.py [N=40] [MODEL=gemini-3.5-flash]      DRY=1 prints the plan and makes no call
Safety: stops at MAX_WORKS (default = N), stops after 3 errors in a row, and caches every reply in data/gemini_runs/search_trial_<model>/results.json,
so a rerun repeats nothing. Prints token use and the number of search queries Gemini made per work, so the cost can be checked."""
import os, sys, json, random, time, urllib.request, urllib.error
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skills/tibetan-title-author/scripts"))
import ta_rule_extract as T, title_author_skill as S, title_author_fix as F
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, parse, score, answers, answered = _g["load"], _g["parse"], _g["score"], _g["answers"], _g["answered"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash"
DRY = os.environ.get("DRY") == "1"; MAX_WORKS = int(os.environ.get("MAX_WORKS", N))
KEY = os.environ.get("GEMINI_API_KEY")
if not KEY and not DRY: sys.exit("set GEMINI_API_KEY (or DRY=1)")
FORCE = os.environ.get("FORCE_SEARCH") == "1"        # tell the model to search at least once (without it, gemini-3.5-flash made 0 searches on 10 works)
OUT = ROOT / "title_author/data/gemini_runs" / f"search_trial_{MODEL}{'_forced' if FORCE else ''}"; OUT.mkdir(parents=True, exist_ok=True)
RF = OUT / "results.json"; RES = json.loads(RF.read_text(encoding="utf-8")) if RF.exists() else {}
SYSTEM = """You check the AUTHOR (composer) of one Tibetan work. You get the first answer from another model and the work text.
You may use Google Search if you doubt the answer (search the Tibetan title, or the author's name with the title). A search result describes a whole book or a person, so it is only a hint; the text of the work decides.
The author is the person who composed THIS work. Not the translator, stanza writer, compiler, copyist, proofreader or the person who requested it.
@FORCE@Finish with ONLY this JSON: {"author": "<exact text copied from the work>" or null, "reason": "<short>"}. Keep the first answer if you find no clear reason to change it."""


if FORCE: SYSTEM = SYSTEM.replace("@FORCE@", "BEFORE you answer, you MUST run at least one Google Search: search the Tibetan title together with the author's name, or the title alone.\n")
else: SYSTEM = SYSTEM.replace("@FORCE@", "")


def build_user(a, head, tail, whole):
    body = "[HEAD]\n" + head + ("" if whole or not tail else "\n[TAIL]\n" + tail)
    ask = ("\n\nBefore you answer, run a Google Search for the Tibetan title of this work (or the title with the author's name) and use what you find as a hint." if FORCE else "")
    return f"First answer: author = {a.get('author')!r} (in the {a.get('author_region')}).\n\n{body}{ask}"


def parse_json(raw):
    dec, i = json.JSONDecoder(), 0
    while raw and (i := raw.find("{", i)) >= 0:
        try:
            d, _ = dec.raw_decode(raw[i:])
            if isinstance(d, dict): return d
        except Exception: pass
        i += 1
    return {}


def call(user):
    body = json.dumps({"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "tools": [{"google_search": {}}], "generationConfig": {"temperature": 0}}).encode()
    for wait in (5, 20, 60):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent", data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300].replace("\n", " ")
            if e.code in (429, 500, 503): print(f"  HTTP {e.code}, wait {wait}s", flush=True); time.sleep(wait); continue
            return {"error": f"HTTP {e.code}: {msg}"}
        except Exception as e: print(f"  {type(e).__name__}, wait {wait}s", flush=True); time.sleep(wait)
    return {"error": "gave up"}


rows = {r.work_id: r for r in load("validation", 0, 0).itertuples()}
items = []
for w, raw in answers("gepa_v2_validation_heldout").items():
    if w not in rows or not answered(raw): continue
    r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
    a, notes = S.postprocess(S.parse_answer(raw), head, tail)
    if F.flags(a, notes, head, tail): items.append((w, r, a, head, tail))
random.seed(0); random.shuffle(items); items = items[:N]
todo = [it for it in items if it[0] not in RES][:MAX_WORKS]
if DRY:
    ex = items[0]; u = build_user(ex[2], ex[3], ex[4], bool(ex[1].whole_work))
    print(f"DRY: {len(items)} flagged works selected, {len(todo)} not yet cached, model {MODEL}; first prompt {len(SYSTEM) + len(u)} characters; no call made"); sys.exit()
tin = tout = queries = errs = 0
for k, (w, r, a, head, tail) in enumerate(todo, 1):
    out = call(build_user(a, head, tail, bool(r.whole_work)))
    if "error" in out:
        errs += 1; print(f"{k}/{len(todo)} ERROR {out['error'][:150]}", flush=True)
        if errs >= 3 or "HTTP 4" in out["error"] and "429" not in out["error"]: print("stopping: repeated or non-retryable errors"); break
        continue
    errs = 0; c = out["candidates"][0]; txt = "".join(p.get("text", "") for p in c["content"]["parts"] if not p.get("thought"))
    q = (c.get("groundingMetadata") or {}).get("webSearchQueries") or []; u = out.get("usageMetadata") or {}
    RES[w] = {"raw": txt, "queries": q, "tokens": [u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), u.get("thoughtsTokenCount", 0)]}
    RF.write_text(json.dumps(RES, ensure_ascii=False), encoding="utf-8")
    tin += u.get("promptTokenCount", 0); tout += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0); queries += len(q)
    print(f"{k}/{len(todo)} search queries={len(q)}  tokens in {tin} out {tout}  total queries {queries}", flush=True)
    if FORCE and k == 3 and queries == 0: print("stopping: the model still makes no searches after 3 works (money saved)"); break
before = after = fixed = broke = changed = searched = have = 0
for w, r, a, head, tail in items:
    if w not in RES: continue
    have += 1; d = parse_json(RES[w]["raw"]); cand = d.get("author"); o = dict(a)
    if cand and cand != a.get("author"):
        f = T.repair_span(cand, head, tail, "tail", last=True)
        if f: o["author"], o["author_region"] = f; o, _ = S.apply_rules(o, head, tail)
    b = score(r, a, "author")[0] == "ok"; n = score(r, o, "author")[0] == "ok"
    before += b; after += n; fixed += (n and not b); broke += (b and not n); changed += (o.get("author") != a.get("author")); searched += bool(RES[w]["queries"])
if have:
    print(f"\nSEARCH ({MODEL}) on {have} flagged works: author correct before {before/have:.3f} -> after {after/have:.3f}")
    print(f"  searched on {searched} works; changed the answer on {changed}: fixed {fixed}, broke {broke}, net {fixed-broke:+d}")

#!/usr/bin/env python3
"""Trial C: the LLM calls tools by itself to check the author of a flagged work (validation split only, never test).
Tools the model may call: list_candidates (rule-script name candidates with context), find_in_text (search the work text),
bdrc_author (author of the book in the BDRC catalog, Unicode). The final answer must be an exact substring of the text.

    GEMINI_API_KEY=... python3 title_author/src/ta_agent_trial.py [N] [MODEL]      (free-tier key works; function calling is not grounding)
Replies are cached per work in data/gemini_runs/agent_trial_<model>/results.json so a rerun resumes."""
import os, sys, json, re, time, random, urllib.request, urllib.error
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skill"))
import ta_rule_extract as T, title_author_skill as S, title_author_fix as F
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, parse, score, answers, answered = _g["load"], _g["parse"], _g["score"], _g["answers"], _g["answered"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash-lite"
BASE = "gepa_v2_validation_heldout"
KEY = os.environ.get("GEMINI_API_KEY") or sys.exit("set GEMINI_API_KEY")
OUT = ROOT / "title_author/data/gemini_runs" / f"agent_trial_{MODEL}"; OUT.mkdir(parents=True, exist_ok=True)
RF = OUT / "results.json"; RES = json.loads(RF.read_text(encoding="utf-8")) if RF.exists() else {}
PAUSE = float(os.environ.get("PAUSE", "4"))

SYSTEM = """You check the AUTHOR (composer) of one Tibetan work. You get the first answer from another model and the work text.
Use the tools if you doubt the answer: list_candidates, find_in_text, bdrc_author (catalog author of the whole book; a book holds many works, so it is only a hint).
The author is the person who composed THIS work. Not the translator, stanza writer, compiler, copyist, proofreader or the person who requested it.
Finish with ONLY this JSON: {"author": "<exact text copied from the work>" or null, "reason": "<short>"}. Keep the first answer if you find no clear reason to change it."""
TOOLS = [{"functionDeclarations": [
    {"name": "list_candidates", "description": "Rule-script author name candidates in this work, each with the surrounding text.", "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "find_in_text", "description": "Find a Tibetan string in the work text and return the text around each hit.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}},
    {"name": "bdrc_author", "description": "Author of the whole book from the BDRC catalog (Tibetan Unicode), or none.", "parameters": {"type": "OBJECT", "properties": {}}}]}]

def post(contents, force=False):
    body = json.dumps({"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents, "tools": TOOLS, "toolConfig": {"functionCallingConfig": {"mode": "ANY" if force else "AUTO"}},
                       "generationConfig": {"temperature": 0}}).encode()
    for wait in (5, 15, 40, 90, 180):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent", data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                print(f'  HTTP {e.code}, waiting {wait}s', flush=True); time.sleep(wait); continue
            raise SystemExit(f"HTTP {e.code} (stopping; results are cached): {e.read()[:200]!r}")
        except Exception as e:
            print("  retry after", type(e).__name__, str(e)[:120], flush=True); time.sleep(wait)
    raise SystemExit("too many retries; rerun later")

def parse_json(raw):
    """First complete JSON object in the text (not a greedy match); falls back to pulling the "author" string out of broken JSON."""
    raw = raw or ""; dec = json.JSONDecoder()
    for m in re.finditer(r"\{", raw):
        try: d, _ = dec.raw_decode(raw[m.start():]); return d if isinstance(d, dict) else {}
        except Exception: continue
    m = re.search(r'"author"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    return {"author": m.group(1)} if m else {}

# BDRC lookup (same chain as the earlier test: instance -> work -> creator -> person label, Wylie -> Unicode)
sys.path.insert(0, os.environ.get("PYLIB", ""))
try: import pyewts; conv = pyewts.pyewts()
except Exception: conv = None
_bc = {}
def _get(rid):
    if rid in _bc: return _bc[rid]
    time.sleep(0.25)
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://purl.bdrc.io/resource/{rid}", headers={"Accept": "application/json", "User-Agent": "title-author-feasibility-test"}), timeout=20))
    except Exception: d = None
    _bc[rid] = d; return d
def _val(d, key, rid):
    v = (d or {}).get("http://purl.bdrc.io/resource/" + rid, {})
    return [x.get("value") for k, l in v.items() if k.endswith("/" + key) or k.endswith("#" + key) for x in l]
def _creators(wid, depth=0):
    d = _get(wid)
    if not d: return []
    out = []
    for c in _val(d, "creator", wid):
        for a in _val(d, "agent", c.rsplit("/", 1)[-1]):
            pr = a.rsplit("/", 1)[-1]; lab = _val(_get(pr), "prefLabel", pr)
            out.append(lab[0] if lab else pr)
    if out or depth >= 3: return out
    for key in ("instanceOf", "instanceReproductionOf"):
        for u in _val(d, key, wid):
            r = _creators(u.rsplit("/", 1)[-1], depth + 1)
            if r: return r
    return []

def run_tool(name, args, ctx):
    head, tail, whole, work = ctx
    if name == "list_candidates":
        return {"candidates": F.candidates(head, tail, whole)}
    if name == "find_in_text":
        q = (args or {}).get("query") or ""
        if not q.strip(): return {"hits": [], "error": "empty query"}
        text = head + "\n" + (tail or ""); out = []
        for m in re.finditer(re.escape(q), text):
            out.append(text[max(0, m.start() - 110): m.end() + 110].replace("\n", " "))
            if len(out) >= 4: break
        return {"hits": out}
    if name == "bdrc_author":
        names = _creators(work.split("_")[0].split("#")[0])
        return {"authors": [conv.toUnicode(n) if conv else n for n in names][:3]}
    return {"error": "unknown tool"}

def agent(a, head, tail, whole, work):
    user = S.build_input(head, tail, whole) + f'\n\nFirst answer: author = {json.dumps(a.get("author"), ensure_ascii=False)}'
    contents = [{"role": "user", "parts": [{"text": user}]}]; calls = []
    for _ in range(5):
        out = post(contents, force=(len(calls) == 0)); time.sleep(PAUSE)
        try: parts = out["candidates"][0]["content"]["parts"]
        except Exception: return None, calls
        contents.append({"role": "model", "parts": parts})
        fcs = [p["functionCall"] for p in parts if "functionCall" in p]
        if not fcs:
            return "".join(p.get("text", "") for p in parts if not p.get("thought")), calls
        resp = []
        for fc in fcs:
            calls.append(fc["name"])
            resp.append({"functionResponse": {**({"id": fc["id"]} if fc.get("id") else {}), "name": fc["name"], "response": run_tool(fc["name"], fc.get("args"), (head, tail, whole, work))}})
        contents.append({"role": "user", "parts": resp})
    return None, calls

rows = {r.work_id: r for r in load("validation", 0, 0).itertuples()}
items = []
for w, raw in answers(BASE).items():
    if w not in rows or not answered(raw): continue
    r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
    a = dict(parse(raw) or {})
    if not S.usable(a, "title", head, tail):
        f = T.repair_span(a.get("title"), head, tail, a.get("title_region"), last=False) if a.get("title") else None
        a["title"], a["title_region"] = (f if f else (None, None))
    a, notes = S.apply_rules(a, head, tail)
    if F.flags(a, notes, head, tail): items.append((w, r, a, head, tail))
random.seed(0); random.shuffle(items); items = items[:N]          # same order as the earlier fix-pass trial
before = after = fixed = broke = changed = used = failed = 0
for k, (w, r, a, head, tail) in enumerate(items, 1):
    if w not in RES:
        raw, calls = agent(a, head, tail, bool(r.whole_work), w)
        RES[w] = {"raw": raw, "calls": calls}; RF.write_text(json.dumps(RES, ensure_ascii=False), encoding="utf-8")
    raw, calls = RES[w]["raw"], RES[w]["calls"]; o = dict(a)
    d = parse_json(raw); cand = d.get("author")
    if raw is None: failed += 1
    if cand and cand != a.get("author"):
        f = T.repair_span(cand, head, tail, "tail", last=True)             # the colophon is usually in the tail: look there first
        if f:
            o["author"], o["author_region"] = f
            o, _ = S.apply_rules(o, head, tail)                             # same role-word step the baseline already went through
    b = score(r, a, "author")[0] == "ok"; n = score(r, o, "author")[0] == "ok"
    before += b; after += n; fixed += (n and not b); broke += (b and not n); changed += (o.get("author") != a.get("author")); used += bool(calls)
    print(f"{k}/{len(items)} calls={calls}", flush=True)
n = len(items)
print(f"\nAGENT ({MODEL}) on {n} flagged works: author correct before {before/n:.3f} -> after {after/n:.3f}")
print(f"  works with no usable answer (stored nothing): {failed}")
print(f"  model used tools on {used} works; changed the answer on {changed}: fixed {fixed}, broke {broke}, net {fixed-broke:+d}")

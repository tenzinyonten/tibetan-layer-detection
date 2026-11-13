#!/usr/bin/env python3
"""Trial I: retrieval of similar labelled works as examples for the author check (validation split only, never test).
For a flagged work, take the text window around the predicted author, find the K most similar labelled windows in TRAIN (character
n-gram TF-IDF) and show them to the model as "how the author is marked in similar works". The model keeps or changes its answer.
Neighbours from the same book (same bdrc_work_id) are excluded, so this does not use the same-book shortcut.

    GEMINI_API_KEY=... python3 title_author/src/ta_retrieval_trial.py [N] [MODEL] [K]
Replies are cached in data/gemini_runs/retrieval_trial_<model>/results.json (resumes on rerun). Same N-work order as ta_agent_trial.py."""
import os, sys, json, re, time, random, urllib.request, urllib.error
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skills/tibetan-title-author/scripts"))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ta_rule_extract as T, title_author_skill as S, title_author_fix as F
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, parse, score, answers, answered = _g["load"], _g["parse"], _g["score"], _g["answers"], _g["answered"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash-lite"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
BASE = "gepa_v2_validation_heldout"; W = 150
KEY = os.environ.get("GEMINI_API_KEY") or sys.exit("set GEMINI_API_KEY")
OUT = ROOT / "title_author/data/gemini_runs" / f"retrieval_trial_{MODEL}"; OUT.mkdir(parents=True, exist_ok=True)
RF = OUT / "results.json"; RES = json.loads(RF.read_text(encoding="utf-8")) if RF.exists() else {}
PAUSE = float(os.environ.get("PAUSE", "4"))

SYSTEM = """You check the AUTHOR (composer) of one Tibetan work. You get a first answer, the work text, and K similar labelled examples:
for each example, a short passage with the correct author marked as [[name]]. Use them only to see how authors are marked, not to copy a name.
The author is the person who composed THIS work. Not the translator, stanza writer, compiler, copyist, proofreader or the person who requested it.
Reply with ONLY this JSON: {"author": "<exact text copied from the work>" or null, "reason": "<short>"}. Keep the first answer unless you see a clear reason to change it."""

# index: windows around the gold author in TRAIN
tr = pd.read_parquet(ROOT / "title_author/data/processed/parquet_export_strat_both_notrans_nous/train.parquet")
tr = tr[tr.author.notna() & tr.author_start.notna()].reset_index(drop=True)
def window(row):
    text = row.text_head if row.author_region == "head" else (row.text_tail or "")
    s, e = int(row.author_start), int(row.author_end)
    return (text[max(0, s - W): s] + "[[" + text[s:e] + "]]" + text[e: e + W]).replace("\n", " ")
tr["win"] = [window(r) for r in tr.itertuples()]
vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=2, max_features=300000, sublinear_tf=True)
X = vec.fit_transform(tr.win.str.replace(r"\[\[.*?\]\]", " ", regex=True))     # match on the phrasing around the author, not the name itself
print(f"index: {len(tr)} train windows", flush=True)

def neighbours(win_query, book):
    sims = (X @ vec.transform([win_query]).T).toarray().ravel()
    out = []
    for i in sims.argsort()[::-1]:
        if tr.bdrc_work_id.iat[i] == book: continue                         # no same-book neighbours
        out.append(tr.win.iat[i])
        if len(out) == K: break
    return out

def post(user):
    body = json.dumps({"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    for wait in (5, 15, 40, 90, 180):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent", data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
            out = json.load(urllib.request.urlopen(req, timeout=180))
            return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503): print(f"  HTTP {e.code}, waiting {wait}s", flush=True); time.sleep(wait); continue
            raise SystemExit(f"HTTP {e.code} (stopping; results are cached): {e.read()[:200]!r}")
        except Exception as e:
            print("  retry after", type(e).__name__, str(e)[:100], flush=True); time.sleep(wait)
    raise SystemExit("too many retries; rerun later")

rows = {r.work_id: r for r in load("validation", 0, 0).itertuples()}
va_book = pd.read_parquet(ROOT / "title_author/data/processed/parquet_export_strat_both_notrans_nous/validation.parquet").set_index("work_id").bdrc_work_id.to_dict()
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
random.seed(0); random.shuffle(items); items = items[:N]                   # same works, same order as the agent trial
before = after = fixed = broke = changed = 0
for k, (w, r, a, head, tail) in enumerate(items, 1):
    if w not in RES:
        text = head + "\n" + tail; p = a.get("author") or ""; i = text.find(p) if p else -1
        q = text[max(0, i - W): i + len(p) + W] if i >= 0 else (tail or head)[-2 * W:]
        ex = neighbours(q.replace("\n", " "), va_book.get(w))
        user = (S.build_input(head, tail, bool(r.whole_work)) + f'\n\nFirst answer: author = {json.dumps(a.get("author"), ensure_ascii=False)}\n\nSimilar labelled examples:\n'
                + "\n".join(f"{n}. {e}" for n, e in enumerate(ex, 1)))
        RES[w] = {"raw": post(user)}; RF.write_text(json.dumps(RES, ensure_ascii=False), encoding="utf-8"); time.sleep(PAUSE)
    m = re.search(r"\{.*\}", RES[w]["raw"] or "", re.S)
    try: d = json.loads(m.group(0)) if m else {}
    except Exception: d = {}
    o = dict(a); cand = d.get("author")
    if cand and cand != a.get("author"):
        f = T.repair_span(cand, head, tail, None, last=True)
        if f: o["author"], o["author_region"] = f
    b = score(r, a, "author")[0] == "ok"; n = score(r, o, "author")[0] == "ok"
    before += b; after += n; fixed += (n and not b); broke += (b and not n); changed += (o.get("author") != a.get("author"))
    print(f"{k}/{len(items)}", flush=True)
n = len(items)
print(f"\nRETRIEVAL K={K} ({MODEL}) on {n} flagged works: author correct before {before/n:.3f} -> after {after/n:.3f}")
print(f"  changed the answer on {changed}: fixed {fixed}, broke {broke}, net {fixed-broke:+d}")

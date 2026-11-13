#!/usr/bin/env python3
"""Trial of the fix pass (skill/title_author_fix.py) on saved validation answers. The LLM calls are live (works on a free-tier key,
slowly), cached per work so a rerun resumes. Key from the environment only (GEMINI_API_KEY).

    GEMINI_API_KEY=... TA_PQ=<parquet folder> python3 title_author/src/ta_fix_trial.py [N] [MODEL] [BASE_RUN]
    N works (default 300, the first N flagged works in a fixed random order), MODEL default gemini-3.5-flash-lite,
    BASE_RUN a folder in data/gemini_runs with the first-pass answers (default gepa_v2_validation_heldout).
Never opens the test split."""
import os, sys, json, time, random, urllib.error
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skill"))
import ta_rule_extract as T, title_author_skill as S, title_author_fix as F
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, parse, score, answers, answered = _g["load"], _g["parse"], _g["score"], _g["answers"], _g["answered"]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash-lite"
BASE = sys.argv[3] if len(sys.argv) > 3 else "gepa_v2_validation_heldout"
OUT = ROOT / "title_author/data/gemini_runs" / f"fix_trial_{MODEL}_{BASE}"; OUT.mkdir(parents=True, exist_ok=True)
CACHE_F = OUT / "replies.json"; CACHE = json.loads(CACHE_F.read_text(encoding="utf-8")) if CACHE_F.exists() else {}
base_call = S.gemini_call(MODEL, thinking=None)

def call_with_retry(system, user):
    for wait in (5, 15, 40, 90, 180, 300):
        try:
            return base_call(system, user)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503): time.sleep(wait); continue
            raise SystemExit(f"HTTP {e.code} from Gemini (stopping, nothing is lost: replies are cached): {e.read()[:200]!r}")
        except Exception:
            time.sleep(wait)
    raise SystemExit("too many retries; rerun later, cached replies are kept")

rows = {r.work_id: r for r in load("validation", 0, 0).itertuples()}
A = answers(BASE); items = []
for w, raw in A.items():
    if w not in rows or not answered(raw): continue
    r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
    a = dict(parse(raw) or {})
    if not S.usable(a, "title", head, tail):
        f = T.repair_span(a.get("title"), head, tail, a.get("title_region"), last=False) if a.get("title") else None
        a["title"], a["title_region"] = (f if f else (None, None))
    a, notes = S.apply_rules(a, head, tail)
    items.append((w, r, a, notes, head, tail))
print(f"validation answers: {len(items)}; author correct after the rule layer: {sum(score(r,a,'author')[0]=='ok' for _,r,a,*_ in items)/len(items):.3f}")
flagged = [x for x in items if F.flags(x[2], x[3], x[4], x[5])]
random.seed(0); random.shuffle(flagged); flagged = flagged[:N]
print(f"flagged works in this trial: {len(flagged)} (of {sum(1 for x in items if F.flags(x[2], x[3], x[4], x[5]))} flagged, {len(items)} works)")
before = after = fixed = broke = changed = 0
for k, (w, r, a, notes, head, tail) in enumerate(flagged, 1):
    def llm(system, user, w=w):
        if w not in CACHE:
            CACHE[w] = call_with_retry(system, user); CACHE_F.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8"); time.sleep(float(os.environ.get("PAUSE", "4")))
        return CACHE[w]
    o, _ = F.fix(a, head, tail, bool(r.whole_work), llm, notes=notes)
    b = score(r, a, "author")[0] == "ok"; n = score(r, o, "author")[0] == "ok"
    before += b; after += n; fixed += (n and not b); broke += (b and not n); changed += (o.get("author") != a.get("author"))
    if k % 25 == 0: print(f"  {k}/{len(flagged)} done", flush=True)
n = len(flagged)
print(f"\nFIX PASS ({MODEL}) on {n} flagged works: author correct before {before/n:.3f} -> after {after/n:.3f}")
print(f"  changed the answer on {changed} works: fixed {fixed}, broke {broke}, net {fixed-broke:+d}")
tot = len(items); nfl = sum(1 for x in items if F.flags(x[2], x[3], x[4], x[5]))
print(f"  if applied to all {nfl} flagged works of {tot}: overall author about {(sum(score(r,a,'author')[0]=='ok' for _,r,a,*_ in items) + (after-before)/n*nfl)/tot:.3f} (from {sum(score(r,a,'author')[0]=='ok' for _,r,a,*_ in items)/tot:.3f})")

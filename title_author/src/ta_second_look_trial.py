#!/usr/bin/env python3
"""Trial of the skill's NO-TOOLS second look (title_author_skill.second_look) on flagged validation works (validation only, never test).
Same works, same order and same baseline as ta_agent_trial.py (flagged works of the saved gepa_v2 flash-lite answers, seed 0), so the two trials compare directly.
One model call per work. Free-tier key works for gemini-3.5-flash-lite. The key comes only from GEMINI_API_KEY.

    GEMINI_API_KEY=... python3 title_author/src/ta_second_look_trial.py [N=139] [MODEL=gemini-3.5-flash-lite]
Replies are cached per work (data/gemini_runs/second_look_trial_<model>/), so a rerun resumes. Stops after 3 failed works in a row."""
import os, sys, json, random, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skills/tibetan-title-author/scripts"))
import ta_rule_extract as T, title_author_skill as S, title_author_fix as F, efficiency as E
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, score, answers, answered = _g["load"], _g["score"], _g["answers"], _g["answered"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 139
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash-lite"
PAUSE = float(os.environ.get("PAUSE", "4"))
OUT = ROOT / "title_author/data/gemini_runs" / f"second_look_trial_{MODEL}"; OUT.mkdir(parents=True, exist_ok=True)
llm = E.metered(E.cached(E.retrying(S.gemini_call(MODEL), tries=4, base=10, cap=90), OUT))
rows = {r.work_id: r for r in load("validation", 0, 0).itertuples()}
items = []
for w, raw in answers("gepa_v2_validation_heldout").items():
    if w not in rows or not answered(raw): continue
    r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
    a, notes = S.postprocess(S.parse_answer(raw), head, tail)
    if F.flags(a, notes, head, tail): items.append((w, r, a, head, tail))
random.seed(0); random.shuffle(items); items = items[:N]
before = after = fixed = broke = changed = noreply = done = fails = 0
for k, (w, r, a, head, tail) in enumerate(items, 1):
    hit0 = llm.inner.hits
    try: new, outcome = S.second_look(a, head, tail, bool(r.whole_work), llm)
    except Exception as e:
        fails += 1; print(f"{k}/{len(items)} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
        if fails >= 3: print("stopping: 3 failures in a row (quota?). Rerun later; replies are cached."); break
        continue
    fails = 0; done += 1
    b = score(r, a, "author")[0] == "ok"; n = score(r, new, "author")[0] == "ok"
    before += b; after += n; fixed += (n and not b); broke += (b and not n); changed += (outcome == "changed"); noreply += (outcome == "no reply")
    print(f"{k}/{len(items)} {outcome}", flush=True)
    if llm.inner.hits == hit0: time.sleep(PAUSE)
if done:
    print(f"\nSECOND LOOK, NO TOOLS ({MODEL}) on {done} flagged works: author correct before {before/done:.3f} -> after {after/done:.3f}")
    print(f"  changed the answer on {changed}: fixed {fixed}, broke {broke}, net {fixed-broke:+d}; no usable reply on {noreply}")
    print("  usage:", llm.meter.report())

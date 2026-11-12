#!/usr/bin/env python3
"""Benchmark the skill on a split with the SAME scorer as every other result (IoU >= 0.5, same block). Free in rules mode.

    python3 bench.py validation [N=300] [rules]          # rule-based only: no key, no cost
    GEMINI_API_KEY=... python3 bench.py validation 100 gemini-3.8-flash    # PAID: state the cost first; answers are cached in ~/.cache/ta_skill
    ALLOW_TEST=1 python3 bench.py test ...               # test split only for a final comparison, never for tuning
Prints title / author / mean share-correct, seconds per work, tokens, and how well `needs_review` separates right from wrong."""
import os, sys, time, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import title_author_skill as S, efficiency as E
_src = (HERE.parent / "src/ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE.parent / "src/ta_compare_runs.py")}; exec(_src, _g)
load, score = _g["load"], _g["score"]


def main(split, n, mode):
    if split == "test" and os.environ.get("ALLOW_TEST") != "1": sys.exit("test needs ALLOW_TEST=1 (final comparison only)")
    rows = list(load(split, 0, 0).itertuples()); random.Random(0).shuffle(rows); rows = rows[:n]
    llm = None if mode == "rules" else E.metered(E.cached(E.retrying(S.gemini_call(mode)), "~/.cache/ta_skill", tag=mode))
    ok = {"title": 0, "author": 0}; rev = {True: [0, 0], False: [0, 0]}; t0 = time.time()
    for r in rows:
        whole = bool(r.whole_work); tail = "" if whole else (r.text_tail or "")
        a = S.extract(r.text_head, tail, whole, call_llm=llm)
        s = {w: score(r, {k: a.get(k) for k in S.FIELDS}, w)[0] == "ok" for w in ok}
        for w in ok: ok[w] += s[w]
        rev[bool(a["needs_review"])][0] += s["author"]; rev[bool(a["needs_review"])][1] += 1
    N = len(rows); t, au = ok["title"] / N, ok["author"] / N
    print(f"{split} {N} works, mode={mode}: title {t:.3f}  author {au:.3f}  mean {(t + au) / 2:.3f}  | {(time.time() - t0) / N:.3f} s/work")
    for k, name in ((True, "flagged needs_review"), (False, "not flagged")):
        c, m = rev[k]; print(f"  author right, {name}: {c / m:.3f} of {m}" if m else f"  {name}: 0 works")
    if llm: print("  usage:", llm.meter.report())


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    main(a[0], int(a[1]) if len(a) > 1 else 300, a[2] if len(a) > 2 else "rules")

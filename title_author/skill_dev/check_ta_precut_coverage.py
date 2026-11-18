#!/usr/bin/env python3
"""Free measurement (no model call): if the text sent to the model is cut shorter, does it still CONTAIN the gold title and the gold author?
Coverage is an upper limit on accuracy: a span that is not in the text sent cannot be found. Validation split only.

    python3 skill_dev/check_ta_precut_coverage.py [validation]
Policies: first H characters of the head + last T characters of the tail, or a 'smart' cut: first H characters + windows around colophon cues in the tail.
Tokens are estimated at 1.5 Tibetan characters per token (calibrated on one trial: about 4,200 input tokens for 6,700 characters)."""
import sys, re, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skills/tibetan-title-author/scripts")); sys.path.insert(0, str(HERE.parent / "src"))
import ta_rule_extract as T, title_author_fix as F
import pandas as pd
CPT = 1.5
CUE = re.compile("|".join([F.SIGNATURE.pattern, F.REQUESTER.pattern] + [re.escape(r) for r in T.ROLES]))


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out


def covered(iv, a, b): return any(s <= a and b <= e for s, e in iv)


def policy(name, H, T_, smart, before=200, after=120, last=300):
    def f(r):
        head, tail = r.text_head, ("" if r.whole_work else (r.text_tail or ""))
        hiv = [[0, min(H, len(head))]]
        if r.whole_work:                                               # whole work: cut the same way from its own start and end
            n = len(head); tiv = [[max(0, n - T_), n]] if not smart else None
            if smart: tiv = merge([[max(0, m.start() - before), min(n, m.end() + after)] for m in CUE.finditer(head[hiv[0][1]:])] and [[hiv[0][1] + max(0, m.start() - before), min(n, hiv[0][1] + m.end() + after)] for m in CUE.finditer(head[hiv[0][1]:])] + [[max(0, n - last), n]])
            return {"head": hiv, "tail": tiv, "whole": True, "n": len(head)}
        n = len(tail)
        tiv = [[max(0, n - T_), n]] if not smart else merge([[max(0, m.start() - before), min(n, m.end() + after)] for m in CUE.finditer(tail)] + [[max(0, n - last), n]])
        return {"head": hiv, "tail": tiv, "whole": False, "n": n}
    f.name = name; return f


def run(df, pol):
    ct = ca = 0; chars = []
    for r in df.itertuples():
        p = pol(r)
        def ok(which):
            reg, a, b = getattr(r, which + "_region"), int(getattr(r, which + "_start")), int(getattr(r, which + "_end"))
            if p["whole"]: return covered(p["head"], a, b) or covered(p["tail"], a, b)
            return covered(p["head"] if reg == "head" else p["tail"], a, b)
        ct += ok("title"); ca += ok("author")
        chars.append(sum(e - s for s, e in p["head"]) + (0 if p["whole"] else sum(e - s for s, e in p["tail"])) if not p["whole"] else
                     len({i for s, e in p["head"] + p["tail"] for i in range(s, e)}))
    n = len(df); return ct / n, ca / n, statistics.mean(chars)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    df = pd.read_parquet(HERE.parent / f"data/processed/parquet_export_strat_both_notrans_nous/{split}.parquet")
    df = df[df.source != "v3_auto"].dropna(subset=["title", "author"]); print(f"{split}: {len(df)} works")
    print(f"{'policy':34} {'title in cut':>13} {'author in cut':>14} {'chars sent':>11} {'~tokens':>8} {'vs 3000/3000':>13}")
    base = None
    for name, H, T_, sm in [("3000 + 3000 (current)", 3000, 3000, False), ("2000 + 2000", 2000, 2000, False), ("1000 + 1000", 1000, 1000, False), ("800 + 600", 800, 600, False),
                            ("smart: 1000 + cue windows", 1000, 0, True), ("smart: 600 + cue windows", 600, 0, True)]:
        t, a, c = run(df, policy(name, H, T_, sm)); base = base or c
        print(f"{name:34} {t:13.3f} {a:14.3f} {c:11.0f} {c / CPT:8.0f} {c / base:12.0%}")

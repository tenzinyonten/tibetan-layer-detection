#!/usr/bin/env python3
"""
count_trailing_particles.py — how often do tsawa spans END with a citation
particle (ཅེས་/ཞེས་...) that arguably shouldn't be inside the span?

If common in clean books -> worth a trim rule (strip trailing particle from
span end, in BOTH train labels and eval, consistently).
If mostly in junk books (I058DD999 etc.) -> just more reason to exclude them.

    python count_trailing_particles.py
"""
import glob
import pandas as pd

SPANS = "tsawa/data/processed/tsawa_spans_merged.csv"
# closing particles that mark quote-end (longest first)
CLOSERS = ["ཞེས་གསུངས", "ཅེས་གསུངས", "ཞེས་དང", "ཅེས་དང", "ཞེས་པ", "ཅེས་པ",
           "ཞེས་སོགས", "ཅེས་སོགས", "ཞེས", "ཅེས"]
# books already known to be mislabeled / excluded
BAD = {"I058DD999", "I100E7DAD"}

d = pd.read_csv(SPANS)
d = d[~d.dropped.astype(bool)] if "dropped" in d.columns else d

def base(pid):
    for p in glob.glob(f"data/raw_opf/{pid}.opf/*/base/v001.txt"):
        return open(p, encoding="utf-8").read()
    return ""

texts = {}
def span_text(pid, s, e):
    if pid not in texts:
        texts[pid] = base(pid)
    return texts[pid][s:e]

stats = {}  # (batch, is_bad) -> [total, ends_with_particle]
for r in d.itertuples():
    pid = r.pecha_id
    batch = "old" if str(pid).startswith("P") else "new"
    bad = pid in BAD
    key = (batch, bad)
    stats.setdefault(key, [0, 0])
    txt = span_text(pid, int(r.start), int(r.end)).rstrip("།༎ \n\t")
    stats[key][0] += 1
    if any(txt.endswith(c) for c in CLOSERS):
        stats[key][1] += 1

print(f"{'group':<22}{'spans':>8}{'end w/ particle':>18}{'%':>7}")
for (batch, bad), (tot, hit) in sorted(stats.items()):
    label = f"{batch}{' (BAD books)' if bad else ''}"
    print(f"{label:<22}{tot:>8,}{hit:>18,}{100*hit/max(tot,1):>6.1f}%")

print("""
READING:
- If 'new' (excluding BAD) has a high %, tsawa spans systematically swallow
  the trailing citation particle -> a trim rule would clean boundaries.
- If it's mostly the BAD-books rows, it's isolated -> just exclude those.
""")
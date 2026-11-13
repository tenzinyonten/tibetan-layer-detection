#!/usr/bin/env python3
"""
show_new_gaps.py — visualize what sits BETWEEN consecutive new-batch tsawa
spans, to decide if a looser merge rule would help.

The merge rule joined spans separated only by punctuation/whitespace. Median
new-batch span is still 42 chars vs 143 old, so pieces remain split. This
shows the actual gap text between consecutive new-batch spans, colored:
  - GREEN gap  = only punctuation/whitespace (SHOULD have merged — bug?)
  - YELLOW gap = short text (<20 chars) — maybe should merge
  - RED gap    = real commentary between them (correctly separate)

Writes scratch/tsawa/analysis/new_gaps.html — open in a browser.

    python show_new_gaps.py --n 40
"""
import argparse
import glob
import html
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_merged.csv")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="scratch/tsawa/analysis/new_gaps.html")
args = ap.parse_args()

SHAD = set("།༎༏༐༑༔ \t\n\r་")

d = pd.read_csv(args.spans)
d = d[~d.dropped.astype(bool)] if "dropped" in d.columns else d
d = d[~d.pecha_id.str.startswith("P")].copy()  # new batch only

def base(pid):
    for p in glob.glob(f"data/raw_opf/{pid}.opf/*/base/v001.txt"):
        return open(p, encoding="utf-8").read()
    return ""

rows = []
shown = 0
for pid, g in d.groupby("pecha_id"):
    t = base(pid)
    if not t:
        continue
    g = g.sort_values("start")
    spans = list(g.itertuples())
    for a, b in zip(spans, spans[1:]):
        gap = t[int(a.end):int(b.start)]
        gap_clean = gap.strip("".join(SHAD))
        if len(gap) > 200:
            continue  # far apart, not a merge candidate
        if all(c in SHAD for c in gap):
            kind = "punct"        # green — should have merged
        elif len(gap_clean) < 20:
            kind = "short"        # yellow — maybe
        else:
            kind = "text"         # red — real commentary
        rows.append((pid, t, int(a.start), int(a.end), int(b.start), int(b.end), gap, kind))
        shown += 1
    if shown >= args.n:
        break

# order so the interesting ones (punct/short gaps) show first
order = {"punct": 0, "short": 1, "text": 2}
rows.sort(key=lambda r: order[r[7]])
rows = rows[:args.n]

colors = {"punct": "#c8f7c5", "short": "#fff3b0", "text": "#f7c5c5"}
labels = {"punct": "GAP = punctuation only (should have merged?)",
          "short": "GAP = short text (<20 chars)",
          "text": "GAP = real commentary between"}

parts = ["""<html><head><meta charset='utf-8'><style>
body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em;background:#111;color:#eee}
.span{background:#4a90d9;color:#fff;padding:1px 3px;border-radius:3px}
.gap-punct{background:#2d7a2d;padding:1px 3px;border-radius:3px}
.gap-short{background:#8a7a2d;padding:1px 3px;border-radius:3px}
.gap-text{background:#7a2d2d;padding:1px 3px;border-radius:3px}
.item{margin:1.2em 0;padding:1em;background:#1c1c1c;border-radius:6px;font-size:1.1em;line-height:1.9}
.pid{font-size:.75em;color:#888;font-family:monospace}
.tag{font-size:.75em;color:#aaa;margin-bottom:.4em}
h2{color:#4a90d9}
</style></head><body>
<h2>What sits between consecutive new-batch tsawa spans</h2>
<p>Blue = tsawa spans. Gap between them colored:
<span class='gap-punct'>punctuation only</span>
<span class='gap-short'>short text</span>
<span class='gap-text'>real commentary</span>.
If most gaps are green/yellow, a looser merge would help.</p>"""]

for pid, t, a_s, a_e, b_s, b_e, gap, kind in rows:
    ctx0 = max(0, a_s - 30)
    ctx1 = min(len(t), b_e + 30)
    seg = (html.escape(t[ctx0:a_s])
           + f"<span class='span'>{html.escape(t[a_s:a_e])}</span>"
           + f"<span class='gap-{kind}'>{html.escape(gap)}</span>"
           + f"<span class='span'>{html.escape(t[b_e-(b_e-b_s):b_e])}</span>"
           + html.escape(t[b_e:ctx1]))
    seg = seg.replace("\n", " ")
    parts.append(f"<div class='item'><div class='pid'>{pid}</div>"
                 f"<div class='tag'>{labels[kind]}</div>{seg}</div>")

parts.append("</body></html>")
import os
os.makedirs("scratch", exist_ok=True)
open(args.out, "w", encoding="utf-8").write("".join(parts))

from collections import Counter
c = Counter(r[7] for r in rows)
print(f"punct-only gaps: {c['punct']}   short-text: {c['short']}   real-commentary: {c['text']}")
print(f"written to {args.out}")

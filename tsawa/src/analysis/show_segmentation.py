#!/usr/bin/env python3
"""
show_segmentation.py — see how the two batches mark tsawa.

Builds an HTML page with real passages from old-batch and new-batch books,
tsawa spans highlighted in alternating colours so the boundary between two
adjacent spans is visible. Old batch should show a stanza as one block;
new batch as several pieces separated by a shad.

Measured behind it (raw offsets, interlinear books excluded):
    old batch: 0.8% of gaps between spans are shad/whitespace only, median span 140 chars
    new batch: 27.4%,                                           median span  78 chars

Usage
-----
    python show_segmentation.py --examples 3
    open scratch/tsawa/analysis/tsawa_segmentation.html
"""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path

import pandas as pd

INTERLINEAR = {"I1637B774", "I881A57E8", "I9FF2B59B", "IDAD44BA2",
               "IF3ACC3E1", "I895C519A"}
STRIP = " \n\t\u0f0d\u0f0e"
CONTEXT = 90
COLOURS = ["#f6c945", "#7cc4f5"]


def load(raw: Path, pid: str) -> str | None:
    for c in (raw / f"{pid}.opf" / f"{pid}.opf" / "base" / "v001.txt",
              raw / f"{pid}.opf" / "base" / "v001.txt"):
        if c.is_file():
            return c.read_text(encoding="utf-8")
    return None


def render(text: str, spans: list[tuple[int, int]], lo: int, hi: int) -> str:
    out, cur = [], lo
    for n, (s, e) in enumerate(spans):
        s, e = max(s, lo), min(e, hi)
        if s >= hi or e <= lo:
            continue
        out.append(f'<span class="ctx">{html.escape(text[cur:s])}</span>')
        c = COLOURS[n % 2]
        out.append(f'<span class="ts" style="background:{c}" title="span {n+1}">'
                   f'{html.escape(text[s:e])}</span>')
        cur = e
    out.append(f'<span class="ctx">{html.escape(text[cur:hi])}</span>')
    return "".join(out).replace("\n", "<br>")


def find_new_runs(d, raw, want, rng):
    """Places where 3+ tsawa spans sit back to back, separated only by a shad."""
    found = []
    pids = [p for p in d.pecha_id.unique()
            if not p.startswith("P") and p not in INTERLINEAR]
    rng.shuffle(pids)
    for pid in pids:
        t = load(raw, pid)
        if not t:
            continue
        rows = [(int(r.start_orig), int(r.end_orig))
                for r in d[d.pecha_id == pid].sort_values("start_orig").itertuples()]
        run = [rows[0]] if rows else []
        for a, b in zip(rows, rows[1:]):
            if t[a[1]:b[0]].strip(STRIP) == "":
                run.append(b)
            else:
                if len(run) >= 3:
                    found.append((pid, t, run[:6]))
                    break
                run = [b]
        if len(found) >= want:
            break
    return found


def find_old_blocks(d, raw, want, rng):
    """Single spans of stanza length surrounded by commentary."""
    found = []
    old = d[d.pecha_id.str.startswith("P")]
    cand = old[(old.end_orig - old.start_orig).between(110, 260)]
    idx = list(range(len(cand)))
    rng.shuffle(idx)
    seen = set()
    for i in idx:
        r = cand.iloc[i]
        if r.pecha_id in seen:
            continue
        t = load(raw, r.pecha_id)
        if not t:
            continue
        s, e = int(r.start_orig), int(r.end_orig)
        found.append((r.pecha_id, t, [(s, e)]))
        seen.add(r.pecha_id)
        if len(found) >= want:
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="scratch/tsawa/analysis/tsawa_segmentation.html")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = Path(args.raw_opf)
    d = pd.read_csv(args.spans)
    d = d[~d.dropped.astype(bool)]

    old = find_old_blocks(d, raw, args.examples, rng)
    new = find_new_runs(d, raw, args.examples, rng)

    def block(title, items, note):
        parts = [f"<h2>{title}</h2><p class='note'>{note}</p>"]
        for pid, t, sp in items:
            lo = max(0, sp[0][0] - CONTEXT)
            hi = min(len(t), sp[-1][1] + CONTEXT)
            parts.append(f"<div class='ex'><div class='pid'>{pid} — "
                         f"{len(sp)} span{'s' if len(sp) > 1 else ''}</div>"
                         f"<div class='txt'>{render(t, sp, lo, hi)}</div></div>")
        return "".join(parts)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tsawa segmentation: old vs new batch</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.4rem}} h2{{margin-top:2rem;font-size:1.15rem}}
.stats{{background:#f4f4f4;padding:.8rem 1rem;border-radius:6px}}
.note{{color:#555}}
.ex{{border:1px solid #ddd;border-radius:6px;padding:.8rem;margin:1rem 0}}
.pid{{font-size:.8rem;color:#777;margin-bottom:.4rem}}
.txt{{font-family:"Noto Serif Tibetan","Jomolhari","Microsoft Himalaya",serif;
      font-size:1.35rem;line-height:2.4}}
.ctx{{color:#999}}
.ts{{color:#111;border-radius:3px;padding:2px 0}}
</style></head><body>
<h1>How the two batches mark tsawa</h1>
<div class="stats">
Measured on raw offsets, interlinear books excluded:<br>
<b>Old batch</b> — 0.8% of gaps between spans are only a shad or line break; median span 140 chars.<br>
<b>New batch</b> — 27.4%; median span 78 chars.<br>
Grey is commentary. Tsawa alternates yellow and blue, so a colour change with
no grey in between is a boundary between two separate spans.
</div>
{block("Old batch", old, "A stanza is marked as one span — one colour.")}
{block("New batch", new, "The same kind of passage is cut into several spans at each shad — colours alternate with no commentary between them.")}
</body></html>"""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"old-batch examples: {len(old)}   new-batch examples: {len(new)}")
    print(f"written to {args.out}")
    print(f"open it with:  open {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
audit_before_v6.py — four data checks before building v6. CPU only.

  1. UNDER-LABELLED BOOKS  lots of verse outside tsawa, very little tsawa.
                           Unmarked root verse teaches "verse = not tsawa".
  2. PROSE TSAWA           books whose tsawa is mostly irregular prose.
                           Prose root texts exist, but so do mislabels.
  3. CROSS-LAYER OVERLAP   tsawa also marked BookTitle / Author / Chapter / Sabche.
  4. LEAKAGE               val/test tsawa text that appears verbatim in a TRAIN
                           book. Inflates scores through memorisation.

Usage:  python audit_before_v6.py > scratch/tsawa/analysis/audit_v6.txt
"""

import glob
import os
import re
import statistics
from collections import defaultdict

import pandas as pd
import yaml

SPANS = "tsawa/data/processed/tsawa_spans_merged.csv"
SPLIT = "tsawa/data/processed/split_v2_frozen.csv"
RAW = "data/raw_opf"
SHADS = "།༎༏༐༑༔"
ISO = {7, 9, 11, 13}

d = pd.read_csv(SPANS)
if "dropped" in d.columns:
    d = d[~d.dropped.astype(bool)]
split = pd.read_csv(SPLIT, comment="#").set_index("pecha_id")["split"].to_dict()


def base(pid):
    for p in glob.glob(f"{RAW}/{pid}.opf/*/base/v001.txt"):
        return open(p, encoding="utf-8").read()
    return ""


def layer(pid, name):
    for p in glob.glob(f"{RAW}/{pid}.opf/*/layers/v001/{name}.yml"):
        try:
            y = yaml.safe_load(open(p, encoding="utf-8")) or {}
        except Exception:
            return []
        anns = y.get("annotations") or {}
        items = anns.values() if isinstance(anns, dict) else anns
        return [(int(a["span"]["start"]), int(a["span"]["end"]))
                for a in items
                if isinstance(a, dict) and a.get("span") and "start" in a["span"]]
    return []


def clauses(text):
    """(start, end, syllables) for each shad-delimited clause."""
    out, s = [], 0
    for m in re.finditer(f"[{SHADS}]", text):
        seg = text[s:m.start()]
        n = len([x for x in seg.split("་") if x.strip()])
        if n:
            out.append((s, m.start(), n))
        s = m.end()
    return out


texts = {pid: base(pid) for pid in d.pecha_id.unique()}

# ---------------------------------------------------------------- 1 & 2
rows = []
for pid, g in d.groupby("pecha_id"):
    t = texts[pid]
    if not t:
        continue
    mask = bytearray(len(t))
    for r in g.itertuples():
        mask[int(r.start):int(r.end)] = b"\x01" * (int(r.end) - int(r.start))
    iso_in = n_in = iso_out = n_out = 0
    for s, e, n in clauses(t):
        inside = sum(mask[s:e]) > (e - s) / 2
        if inside:
            n_in += 1; iso_in += n in ISO
        else:
            n_out += 1; iso_out += n in ISO
    rows.append(dict(pid=pid, split=split.get(pid, "?"), spans=len(g),
                     density=100 * sum(mask) / len(t),
                     tsawa_iso=100 * iso_in / n_in if n_in else float("nan"),
                     outside_iso_clauses=iso_out, outside_clauses=n_out))
bk = pd.DataFrame(rows)

print("=" * 72)
print("1. POSSIBLY UNDER-LABELLED — little tsawa, lots of verse outside it")
print("=" * 72)
u = bk[(bk.density < 2) & (bk.outside_iso_clauses > 200)]
u = u.sort_values("outside_iso_clauses", ascending=False)
print(u[["pid", "split", "spans", "density", "outside_iso_clauses"]]
      .to_string(index=False) if len(u) else "  none")
print("  Verse outside tsawa is normal (citations, the author's own verse),")
print("  so read a few before concluding. High counts + tiny density is the flag.")

print("\n" + "=" * 72)
print("2. PROSE TSAWA — tsawa clauses rarely metrical (corpus average ~88%)")
print("=" * 72)
p = bk[(bk.spans >= 20) & (bk.tsawa_iso < 40)].sort_values("tsawa_iso")
print(p[["pid", "split", "spans", "density", "tsawa_iso"]]
      .to_string(index=False) if len(p) else "  none")
print("  Could be a prose root text (fine) or commentary labelled as tsawa.")

# ---------------------------------------------------------------- 3
print("\n" + "=" * 72)
print("3. TSAWA OVERLAPPING OTHER LAYERS (>= half the tsawa span covered)")
print("=" * 72)
ov = defaultdict(int)
ov_books = defaultdict(set)
for pid, g in d.groupby("pecha_id"):
    lay = {L: layer(pid, L) for L in ("BookTitle", "Author", "Chapter", "Sabche")}
    for r in g.itertuples():
        s, e = int(r.start), int(r.end)
        for L, sp in lay.items():
            cov = sum(max(0, min(e, b) - max(s, a)) for a, b in sp)
            if cov >= (e - s) / 2:
                ov[L] += 1; ov_books[L].add(pid)
for L in ("BookTitle", "Author", "Chapter", "Sabche"):
    ex = sorted(ov_books[L])[:6]
    print(f"  {L:<10} {ov[L]:>5} spans in {len(ov_books[L]):>3} books   e.g. {', '.join(ex)}")

# ---------------------------------------------------------------- 4
print("\n" + "=" * 72)
print("4. LEAKAGE — val/test tsawa text found verbatim in a TRAIN book")
print("=" * 72)
norm = lambda s: "".join(s.split())
train_text = "\n".join(norm(texts[p]) for p in texts if split.get(p) == "train")
for sp in ("validation", "val", "test"):
    sub = d[d.pecha_id.map(lambda p: split.get(p)) == sp]
    if sub.empty:
        continue
    n = hit = chars = hit_chars = 0
    books = defaultdict(lambda: [0, 0])
    for r in sub.itertuples():
        s = norm(texts[r.pecha_id][int(r.start):int(r.end)])
        if len(s) < 40:
            continue
        n += 1; chars += len(s)
        books[r.pecha_id][0] += 1
        if s in train_text:
            hit += 1; hit_chars += len(s); books[r.pecha_id][1] += 1
    print(f"  {sp}: {hit} of {n} spans (>=40 chars) appear in train "
          f"({100*hit/max(n,1):.1f}% of spans, {100*hit_chars/max(chars,1):.1f}% of text)")
    worst = sorted(books.items(), key=lambda kv: -kv[1][1])[:5]
    for pid, (tot, h) in worst:
        if h:
            print(f"      {pid}: {h}/{tot} spans leaked")
print("  Some overlap is natural — commentaries on the same root text quote the")
print("  same verses. A high share means the test score partly measures memory.")
#!/usr/bin/env python3
"""Export the title/author dataset as Parquet (raw text, not tokens) for review / sharing.

Works kept: only works that have BOTH a title span and an author span.
  gold      : the works of gold_dataset_all (annotator spans), same train/validation/test split.
  v3_auto   : the BDRC v3 works with both spans found by string matching (train only).
Text kept: first and last ~3,000 characters (same crop rule as the training data; whole work if <= 6,000 chars).
Spans are character offsets inside text_head / text_tail (see title_region / author_region).

    python title_author/src/ta_export_parquet.py V3_CLEAN_DIR [OUT_DIR]
"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common")); sys.path.insert(0, str(ROOT / "title_author/src"))
from ta_build_gold_dataset import OUT, SPLIT_NAMES, regions, resolve_overlap, strip_title_marks, strip_title_formula  # noqa: E402
from ta_load_spans import GOLD, TRAIN_SOURCES, load, parse_name  # noqa: E402
from datasets import load_from_disk  # noqa: E402

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("v3_dir", type=Path); ap.add_argument("--base", default="gold_dataset_all")
ap.add_argument("--v3-works", default="gold_dataset_all_v3_works.csv"); ap.add_argument("--out-dir", type=Path, default=None)
args = ap.parse_args()
v3_dir = args.v3_dir; out_dir = args.out_dir or OUT / "parquet_export"
out_dir.mkdir(parents=True, exist_ok=True)
K = 3000
bdrc = {r["work_id"]: r for r in csv.DictReader(open(OUT / "bdrc_metadata.csv", encoding="utf-8"))}
dd = load_from_disk(str(OUT / args.base))
keep = {s: set(dd[s].unique("work_id")) for s in dd}
keep_split = {w: s for s, ws in keep.items() for w in ws}
TYPE = {"tei": "clean", "google_books": "clean", "google_vision": "clean"}   # clean-text sources; the rest are OCR


def meta(bid):
    m = bdrc.get(bid or "", {})
    return dict(bdrc_work_id=bid or "", book_type=m.get("print_method") or "unknown")


def row(work_id, split, text, t, a, origin, source, src_type, bid, typed_t, typed_a):
    """t, a: (start, end) in text. Returns None if a region edge would cut a span."""
    a2, _ = resolve_overlap(t, a)
    if a2 is None: return None
    regs = regions(text, K, [t, a2])
    parts = {n: (s, e) for n, s, e in regs}
    head = parts.get("head") or parts.get("full"); tail = parts.get("tail")
    def where(sp):
        for n, (s, e) in parts.items():
            if s <= sp[0] and sp[1] <= e: return ("head" if n == "full" else n), sp[0] - s, sp[1] - s
        return None
    wt, wa = where(t), where(a2)
    if not wt or not wa: return None   # span fell into the dropped middle
    r = dict(work_id=work_id, split=split, source=source, source_type=src_type, span_origin=origin,
             text_head=text[head[0]:head[1]], text_tail=text[tail[0]:tail[1]] if tail else "", whole_work=tail is None,
             work_chars=len(text), title=text[t[0]:t[1]], title_region=wt[0], title_start=wt[1], title_end=wt[2],
             author=text[a2[0]:a2[1]], author_region=wa[0], author_start=wa[1], author_end=wa[2],
             typed_title=typed_t, typed_author=typed_a)
    r.update(meta(bid)); return r


rows, drop = [], Counter()
byf = defaultdict(list)
for r in load()[0]: byf[r["file"]].append(r)
for f, rs in sorted(byf.items()):
    c = None
    for r in rs:
        wid = f"{f[:-5]}#{r['seg']}"
        if wid not in keep_split: continue
        if c is None: c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        s0 = r["span_start"]; text = c[s0:r["span_end"]]
        t = strip_title_formula(text, strip_title_marks(text, (r["title_span"][0] - s0, r["title_span"][1] - s0))); a = (r["author_span"][0] - s0, r["author_span"][1] - s0)
        bid = r["cid"]   # W... work id or IE... e-text id (both are in bdrc_metadata.csv)
        x = row(wid, keep_split[wid], text, t, a, "annotator", r["source"],
                TYPE.get(r["source"], "ocr"), bid, r["title_text"], r["author_text"])
        if x is None: drop["gold_no_author_left_or_span_lost"] += 1
        else: rows.append(x)
n_gold = len(rows)
# v3: both-span works only, train only
for l in csv.DictReader(open(OUT / args.v3_works, encoding="utf-8")):
    if l["kind"] != "both": continue
    d = l["v3_id"]
    text = open(v3_dir / d / f"{d}.txt", encoding="utf-8").read(); m = json.load(open(v3_dir / d / "meta.json", encoding="utf-8"))
    x = row(f"v3:{d}#0", "train", text, (int(l["title_start"]), int(l["title_end"])), (int(l["author_start"]), int(l["author_end"])),
            "auto", "v3_auto", "ocr", m["work_id"], l["typed_title"], l["typed_author"])
    if x is None: drop["v3_no_author_left_or_span_lost"] += 1
    else: rows.append(x)
df = pd.DataFrame(rows)
for s in ("train", "validation", "test"):
    df[df.split == s].drop(columns="split").to_parquet(out_dir / f"{s}.parquet", index=False)
print("works exported:", len(df), "(gold", n_gold, ", v3", len(df) - n_gold, ")", dict(drop))
print(df.groupby(["split", "span_origin"]).size().unstack(fill_value=0))
print(df.groupby(["split", "source_type"]).size().unstack(fill_value=0))
print(df.groupby(["split", "book_type"]).size().unstack(fill_value=0))
print(df.groupby("source").size())
df.drop(columns=["text_head", "text_tail"]).to_csv(out_dir / "works_index.csv", index=False)

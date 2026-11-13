#!/usr/bin/env python3
"""
check_double_labels.py — how often is a quotation also labelled tsawa?

Found in I27E347E9, a commentary on Chakrasamvara: eight lines introduced as
"from the Hevajra" (དགྱེས་རྡོར་ལས།) are annotated BOTH as one Citation span and
as eight Tsawa spans. A quotation from a different tantra is a citation, not
this book's root text, so the Tsawa label there looks wrong.

This counts how widespread that is. For every tsawa span it asks:
  - is it (mostly) covered by a Citation/Quotation span?  -> double-labelled
  - is it introduced by a source name ending in ལས། / ནས། etc.? -> quote-shaped

Double-labelled AND introduced by a named source is the strong pattern: the
likeliest cases of a citation mislabelled as tsawa. Not proof — a commentary
on the Hevajra would legitimately name its own root text — so read the
examples.

Nothing is modified.

Usage
-----
    python check_double_labels.py
    python check_double_labels.py --examples 12
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict

import pandas as pd
import yaml

# source attribution just before a span: "from X", "X says", "as stated in X"
SOURCE_MARKS = ["ལས།", "ལས་", "ནས།", "གསུངས་པ།", "ཞེས་གསུངས་པ", "ལུང་ལས",
                "མདོ་ལས", "རྒྱུད་ལས", "གཞུང་ལས", "དུ།"]
STRONG = ["ལས།", "ལས་", "ནས།", "ལུང་ལས", "མདོ་ལས", "རྒྱུད་ལས", "གཞུང་ལས"]


def load_spans(path: str):
    try:
        y = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception:
        return []
    out = []
    for a in (y.get("annotations") or {}).values():
        sp = a.get("span") or {}
        if "start" in sp and "end" in sp:
            out.append((int(sp["start"]), int(sp["end"])))
    return sorted(out)


def covered(span, cits) -> float:
    s, e = span
    tot = 0
    for cs, ce in cits:
        tot += max(0, min(e, ce) - max(s, cs))
    return tot / max(e - s, 1)


def title_of(meta_path: str) -> str:
    try:
        m = yaml.safe_load(open(meta_path, encoding="utf-8")) or {}
        return ((m.get("source_metadata") or {}).get("title") or "").strip()[:60]
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    d = pd.read_csv(args.spans)
    d = d[~d.dropped.astype(bool)]

    stats = defaultdict(lambda: Counter())
    per_book = Counter()
    per_book_total = Counter()
    examples = []

    for layer_dir in sorted(glob.glob(f"{args.raw_opf}/*/*/layers/v001")):
        pid = os.path.basename(os.path.dirname(os.path.dirname(layer_dir))).replace(".opf", "")
        cits = []
        for f in ("Citation.yml", "Quotation.yml"):
            p = os.path.join(layer_dir, f)
            if os.path.isfile(p):
                cits += load_spans(p)
        tsawa = d[d.pecha_id == pid]
        if tsawa.empty:
            continue
        batch = "old" if pid.startswith("P") else "new"
        root = os.path.dirname(os.path.dirname(layer_dir))
        base = os.path.join(root, "base", "v001.txt")
        if not os.path.isfile(base):
            continue
        text = open(base, encoding="utf-8").read()

        for r in tsawa.itertuples():
            s, e = int(r.start_orig), int(r.end_orig)
            stats[batch]["tsawa"] += 1
            per_book_total[pid] += 1
            before = text[max(0, s - 40):s].rstrip(" \n")
            sourced = any(before.endswith(m) or before.endswith(m + "།")
                          for m in STRONG)
            if sourced:
                stats[batch]["sourced"] += 1
            if cits and covered((s, e), cits) >= 0.5:
                stats[batch]["double"] += 1
                per_book[pid] += 1
                if sourced:
                    stats[batch]["double_sourced"] += 1
                    if len(examples) < args.examples:
                        examples.append((pid, text[max(0, s - 80):s],
                                         text[s:e], os.path.join(root, "meta.yml")))

    print(f"{'='*66}\nTSAWA SPANS ALSO LABELLED CITATION/QUOTATION\n{'='*66}")
    print(f"  {'batch':<6}{'tsawa':>8}{'double':>9}{'%':>7}"
          f"{'  sourced':>10}{'double+src':>12}")
    for b in ("old", "new"):
        c = stats[b]
        if not c["tsawa"]:
            continue
        print(f"  {b:<6}{c['tsawa']:>8,}{c['double']:>9,}"
              f"{100*c['double']/c['tsawa']:>6.1f}%"
              f"{c['sourced']:>10,}{c['double_sourced']:>12,}")
    print("\n  double      = at least half the span is also a citation span")
    print("  sourced     = span is introduced by a source name (…ལས། etc.)")
    print("  double+src  = both — the strongest candidates for a quotation")
    print("                wrongly labelled tsawa")

    print(f"\n{'='*66}\nBOOKS WITH THE MOST DOUBLE-LABELLED TSAWA\n{'='*66}")
    for pid, n in per_book.most_common(10):
        tot = per_book_total[pid]
        root = glob.glob(f"{args.raw_opf}/{pid}.opf/*")
        t = title_of(os.path.join(root[0], "meta.yml")) if root else ""
        print(f"  {pid}: {n:>4} of {tot:>4} tsawa spans ({100*n/tot:5.1f}%)  {t}")

    print(f"\n{'='*66}\nEXAMPLES — double-labelled and introduced by a source\n{'='*66}")
    print("  Read the line before each span: does it name ANOTHER text? And is")
    print("  that text the one this book comments on (then tsawa is right) or a")
    print("  different one (then it is a citation)?\n")
    for pid, before, span, meta in examples:
        print(f"  {pid}  —  {title_of(meta)}")
        print(f"    before: …{before}".replace("\n", " "))
        print(f"    span  : {span[:120]}".replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
    
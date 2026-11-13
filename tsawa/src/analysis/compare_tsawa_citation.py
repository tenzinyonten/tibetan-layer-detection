#!/usr/bin/env python3
"""
compare_tsawa_citation.py — are citations distinguishable from tsawa by surface form?

The model's biggest error class is predicting spans that overlap no gold tsawa
(53.7% of predictions). A plausible explanation: those are CITATIONS —
quotations from other works, carrying the same framing particles and the same
verse structure as root text, but not annotated as tsawa. The model has never
been told that category exists.

If citations and tsawa look identical on the surface — same closers, same
metre, same lengths — then no text-only model can separate them from local
evidence, and the fix is either an extra label or an extra signal (position in
the document, source attribution before the span, monotonicity).

If they differ measurably, a three-label model (O / TSAWA / CITATION) should
pick that up and stop paying for the confusion.

This compares the two layers directly, reading Quotation.yml / Citation.yml
alongside the tsawa sidecar. Nothing is modified.

Usage
-----
    python compare_tsawa_citation.py
    python compare_tsawa_citation.py --books 60 --examples 4
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

TSHEG = "\u0f0b"
SHADS = "\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f14"
CLOSERS = ["ཞེས་པ་སྟེ", "ཞེས་པ་ནི", "ཞེས་གསུངས", "ཅེས་གསུངས", "ཞེས་བྱ་བ",
           "ཞེས་པ", "ཅེས་པ", "ཞེས་", "ཅེས་", "གསུངས་"]
# attribution markers: a citation often names its source first
SOURCE_MARKS = ["མདོ་ལས", "རྒྱུད་ལས", "ལས།", "གསུངས་པ་ལྟར", "ཞེས་བཤད",
                "སློབ་དཔོན", "ལུང་ལས", "གཞུང་ལས"]


def syllables(s: str) -> int:
    return len([p for p in s.split(TSHEG) if p.strip()])


def clause_profile(texts: list[str]) -> tuple[float, float, int]:
    counts = []
    for t in texts:
        for seg in "".join(c if c not in SHADS else "\n" for c in t).split("\n"):
            n = syllables(seg)
            if 1 <= n <= 30:
                counts.append(n)
    if not counts:
        return 0.0, 0.0, 0
    iso = sum(1 for n in counts if n in (7, 9, 11, 13))
    return 100 * iso / len(counts), statistics.pstdev(counts), len(counts)


def load_layer(layer_dir: str, names: list[str]):
    for nm in names:
        p = os.path.join(layer_dir, nm)
        if os.path.isfile(p):
            try:
                y = yaml.safe_load(open(p, encoding="utf-8")) or {}
            except Exception:
                return []
            out = []
            for _, a in (y.get("annotations") or {}).items():
                sp = a.get("span") or {}
                if "start" in sp and "end" in sp:
                    out.append((int(sp["start"]), int(sp["end"])))
            return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--books", type=int, default=60)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]

    ts_texts, ct_texts = [], []
    ts_lens, ct_lens = [], []
    ts_closer = ct_closer = 0
    ts_source = ct_source = 0
    overlap_pairs = 0
    books = 0
    examples = []

    for layer_dir in sorted(glob.glob(f"{args.raw_opf}/*/*/layers/v001")):
        pid = os.path.basename(os.path.dirname(os.path.dirname(layer_dir))).replace(".opf", "")
        fs = set(os.listdir(layer_dir))
        if "Tsawa.yml" not in fs:
            continue
        cits = load_layer(layer_dir, ["Quotation.yml", "Citation.yml"])
        if not cits:
            continue
        base = os.path.join(os.path.dirname(os.path.dirname(layer_dir)),
                            "base", "v001.txt")
        if not os.path.isfile(base):
            continue
        text = open(base, encoding="utf-8").read()
        tsawa = [(int(r.start), int(r.end)) for _, r in d[d.pecha_id == pid].iterrows()]
        if not tsawa:
            continue
        books += 1

        for s, e in tsawa:
            seg = text[s:e]
            ts_texts.append(seg); ts_lens.append(e - s)
            after = text[e:e + 16].lstrip(" \n།")
            before = text[max(0, s - 40):s]
            if any(after.startswith(c) for c in CLOSERS):
                ts_closer += 1
            if any(m in before for m in SOURCE_MARKS):
                ts_source += 1

        for s, e in cits:
            if e <= s or e > len(text):
                continue
            seg = text[s:e]
            ct_texts.append(seg); ct_lens.append(e - s)
            after = text[e:e + 16].lstrip(" \n།")
            before = text[max(0, s - 40):s]
            if any(after.startswith(c) for c in CLOSERS):
                ct_closer += 1
            if any(m in before for m in SOURCE_MARKS):
                ct_source += 1
            if any(min(e, te) - max(s, ts) > 0 for ts, te in tsawa):
                overlap_pairs += 1

        if len(examples) < args.examples and cits:
            s, e = cits[len(cits) // 2]
            examples.append((pid, text[max(0, s-60):s], text[s:e][:90],
                             text[e:e+50]))
        if books >= args.books:
            break

    if not books:
        print("no books found with both layers")
        return

    print(f"{books} books with both a Tsawa and a Quotation/Citation layer")
    print(f"  tsawa spans:    {len(ts_lens):,}")
    print(f"  citation spans: {len(ct_lens):,}")
    print(f"  citation spans overlapping a tsawa span: {overlap_pairs:,} "
          f"({100*overlap_pairs/max(len(ct_lens),1):.2f}%)")

    print(f"\n{'='*64}")
    print("SURFACE COMPARISON")
    print("=" * 64)
    print(f"{'':>26} {'tsawa':>10} {'citation':>10}")
    print(f"{'median length (chars)':>26} "
          f"{statistics.median(ts_lens):>10.0f} {statistics.median(ct_lens):>10.0f}")
    ts_iso, ts_sd, ts_n = clause_profile(ts_texts)
    ct_iso, ct_sd, ct_n = clause_profile(ct_texts)
    print(f"{'isometric clauses %':>26} {ts_iso:>10.1f} {ct_iso:>10.1f}")
    print(f"{'clause length stdev':>26} {ts_sd:>10.1f} {ct_sd:>10.1f}")
    print(f"{'followed by a closer %':>26} "
          f"{100*ts_closer/max(len(ts_lens),1):>10.1f} "
          f"{100*ct_closer/max(len(ct_lens),1):>10.1f}")
    print(f"{'preceded by attribution %':>26} "
          f"{100*ts_source/max(len(ts_lens),1):>10.1f} "
          f"{100*ct_source/max(len(ct_lens),1):>10.1f}")

    print(f"\n{'='*64}")
    print("READING")
    print("=" * 64)
    close = (abs(ts_iso - ct_iso) < 15 and
             abs(100*ts_closer/max(len(ts_lens),1) -
                 100*ct_closer/max(len(ct_lens),1)) < 15)
    if close:
        print("  The two look ALIKE on the surface. A model seeing only local")
        print("  text cannot separate them, which would explain a large share")
        print("  of the orphan predictions: they may be correctly-identified")
        print("  quotations that simply are not root text.")
        print("  -> an explicit CITATION label, or a document-level signal")
        print("     (attribution before the span, monotonic position), is")
        print("     needed rather than more capacity.")
    else:
        print("  The two DIFFER measurably. A three-label model should be able")
        print("  to learn the distinction from surface form alone.")
    print("\n  Attribution rate is the one to watch: if citations are far more")
    print("  often preceded by a source name, that is a usable feature.")

    print(f"\n{'='*64}")
    print("CITATION EXAMPLES (for reading)")
    print("=" * 64)
    for pid, before, span, after in examples:
        print(f"\n  {pid}")
        print(f"    …{before}«{span}»{after}…".replace("\n", " "))


if __name__ == "__main__":
    main()
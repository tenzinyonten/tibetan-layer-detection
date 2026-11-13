#!/usr/bin/env python3
"""
count_incipits.py — how much of the tsawa layer is quote-pointers rather than
quoted text?

An incipit pointer names the opening words of a root-text passage and then says
how long it is, e.g.

    <<མདོར་ན་རྣལ་འབྱོར་པ་>> སོགས་རྐང་པ་ལྔའོ།།
    "'In short, the yogi...' etc., five lines."

Only the opening words are annotated as tsawa; the five lines they point to are
labelled O. Elsewhere in the same corpus a full block is annotated. Both
patterns are present, and nothing in the text distinguishes them, so a token
classifier cannot satisfy both.

This counts them, and while it is there, counts two other categories Gemini
flagged in a 9-span sample: section headings ending in NI (ནི།) and spans whose
edges cut a multi-syllable word.

Nothing is modified. This is a measurement.

Usage
-----
    python count_incipits.py
    python count_incipits.py --dump scratch/tsawa/analysis/incipits.csv --examples 20
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# "and so on" — the marker that turns a quotation into a pointer
SOGS = "སོགས་"
# line/verse counters that usually follow it
COUNTERS = ["རྐང་པ་", "ཤོ་ལོ་ཀ་", "ཚིགས་བཅད་", "ཚིག་རྐང་"]
# Tibetan numerals, plus the spelled-out numbers that actually appear
TIB_DIGITS = "༠༡༢༣༤༥༦༧༨༩"
NUM_WORDS = ["གཅིག", "གཉིས", "གསུམ", "བཞི", "ལྔ", "དྲུག", "བདུན", "བརྒྱད",
             "དགུ", "བཅུ", "བཅུག", "ཉི་ཤུ"]

TSHEG, SHAD = "་", "།"
BOUNDARY = set(TSHEG + SHAD + "༎༏༐༑༔ \t\n\r\xa0༼༽༺༻[]()")


def looks_like_incipit(after: str) -> tuple[bool, str]:
    """Does the text right after the span turn it into a pointer?"""
    head = after[:40]
    if SOGS not in head:
        return False, ""
    tail = head[head.index(SOGS) + len(SOGS):][:25]
    for c in COUNTERS:
        if c in tail:
            return True, "sogs+counter"
    if any(d in tail for d in TIB_DIGITS):
        return True, "sogs+numeral"
    if any(w in tail for w in NUM_WORDS):
        return True, "sogs+number-word"
    # sogs present but no length marker — weaker evidence
    return True, "sogs-only"


def ends_in_ni(span: str) -> bool:
    """Section headings look like '<topic> ནི།' — commentary, not root text."""
    s = span.rstrip()
    return s.endswith("ནི།") or s.endswith("ནི་།") or s.endswith("ནི")


def cuts_word(text: str, start: int, end: int) -> bool:
    """
    Edge sits mid-word: the character just outside the span is a letter, and the
    boundary is not a tsheg/shad. The snapper fixed mid-SYLLABLE breaks; this
    catches breaks at a valid syllable boundary that still split a compound —
    only detectable as 'no tsheg immediately outside'.
    """
    before_ok = start == 0 or text[start - 1] in BOUNDARY
    after_ok = end >= len(text) or text[end - 1] in BOUNDARY or text[end] in BOUNDARY
    return not (before_ok and after_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--max-len", type=int, default=60,
                    help="only classify spans up to this many characters")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--dump", default="scratch/tsawa/analysis/incipit_analysis.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.spans)
    if "dropped" in d.columns:
        d = d[~d.dropped.astype(bool)]
    d = d.copy()
    d["len"] = d.end - d.start
    print(f"{len(d):,} active spans across {d.pecha_id.nunique()} pechas")

    raw = Path(args.raw_opf)
    texts: dict[str, str] = {}

    def base(p: str) -> str:
        if p not in texts:
            for c in [raw / f"{p}.opf" / f"{p}.opf" / "base" / "v001.txt",
                      raw / f"{p}.opf" / "base" / "v001.txt"]:
                if c.is_file():
                    texts[p] = c.read_text(encoding="utf-8")
                    break
            else:
                texts[p] = ""
        return texts[p]

    rows = []
    for _, r in d.iterrows():
        t = base(r.pecha_id)
        if not t:
            continue
        s, e = int(r.start), int(r.end)
        span_txt = t[s:e]
        after = t[e:e + 60]
        short = r["len"] <= args.max_len

        inc, kind = looks_like_incipit(after) if short else (False, "")
        rows.append({
            "pecha_id": r.pecha_id, "batch": r.batch,
            "start": s, "end": e, "len": int(r["len"]),
            "is_incipit": inc, "incipit_kind": kind,
            "is_ni_heading": ends_in_ni(span_txt) if short else False,
            "cuts_word": cuts_word(t, s, e),
            "span_text": span_txt[:50],
            "after_text": after[:40],
        })

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"{n:,} spans classified\n")

    # ---- incipits --------------------------------------------------------
    inc = df[df.is_incipit]
    print("=" * 62)
    print("INCIPIT POINTERS  (span followed by སོགས་ + a length marker)")
    print("=" * 62)
    print(f"  total: {len(inc):,} ({100*len(inc)/n:.2f}% of all spans)")
    if len(inc):
        print("\n  by evidence strength:")
        for k, c in inc.incipit_kind.value_counts().items():
            note = "  <- weak, sogs with no length marker" if k == "sogs-only" else ""
            print(f"    {k:>18}: {c:>6,}{note}")
        strong = inc[inc.incipit_kind != "sogs-only"]
        print(f"\n  strong evidence only: {len(strong):,} "
              f"({100*len(strong)/n:.2f}% of all spans)")
        print(f"  median length: {inc['len'].median():.0f} chars "
              f"(corpus median {df['len'].median():.0f})")
        print("\n  by batch:")
        print(inc.batch.value_counts().to_string())
        print("\n  top books:")
        for p, c in inc.pecha_id.value_counts().head(8).items():
            tot = (df.pecha_id == p).sum()
            print(f"    {p}: {c:,} of {tot:,} ({100*c/tot:.0f}%)")

    # ---- other categories ------------------------------------------------
    print("\n" + "=" * 62)
    print("OTHER FLAGGED CATEGORIES")
    print("=" * 62)
    ni = df[df.is_ni_heading]
    cw = df[df.cuts_word]
    print(f"  spans ending in ནི (section headings?): {len(ni):,} "
          f"({100*len(ni)/n:.2f}%)")
    print(f"  spans with an edge not on a boundary:   {len(cw):,} "
          f"({100*len(cw)/n:.2f}%)")
    print("    (the snapper should have left none — anything here is a bug)")

    # ---- overlap with the short-span problem -----------------------------
    print("\n" + "=" * 62)
    print("HOW MUCH OF THE SHORT-SPAN PROBLEM IS INCIPITS?")
    print("=" * 62)
    for lo, hi in [(0, 15), (16, 30), (31, 60), (61, 10**9)]:
        sub = df[(df["len"] > lo) & (df["len"] <= hi)]
        if not len(sub):
            continue
        tag = f"{lo+1}-{hi}" if hi < 10**9 else f"{lo+1}+"
        share = 100 * sub.is_incipit.sum() / len(sub)
        print(f"  {tag:>8} chars: {len(sub):>6,} spans, "
              f"{sub.is_incipit.sum():>5,} incipits ({share:5.1f}%)")

    print("\n  If incipits dominate the short buckets, the 0.000 recall is")
    print("  explained: the model marks the whole quoted block while gold")
    print("  marks only the pointer. That is a target definition problem,")
    print("  not a detection failure.")

    # ---- examples --------------------------------------------------------
    if len(inc):
        print("\n" + "=" * 62)
        print("EXAMPLES")
        print("=" * 62)
        for _, r in inc.sample(min(args.examples, len(inc)), random_state=3).iterrows():
            print(f"\n  {r.pecha_id} [{r.start}:{r.end}] len={r['len']} ({r.incipit_kind})")
            print(f"    《《{r.span_text}》》{r.after_text}")

    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dump, index=False)
    print(f"\nwritten to {args.dump}")
    print("\nNOTE: these patterns were derived from a 9-span sample. Read the")
    print("examples above before trusting the counts — if the regex is catching")
    print("things that are not incipits, the numbers mean nothing.")


if __name__ == "__main__":
    main()
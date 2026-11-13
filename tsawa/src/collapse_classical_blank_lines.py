#!/usr/bin/env python3
"""Collapse blank-line gaps in the classical (interleaved_v3) docs and remap every span.

A blank-line run (a newline followed by one or more whitespace-only lines) becomes a single
newline. Single newlines (the line breaks inside a verse) are untouched. In this data every
run is exactly one blank line ("\\n\\n"), so the effect is to remove the blank separator
between blocks.

Reads:   ~/tsawa_commentary_data/interleaved_v3/docs/*.txt (source, never modified) and
         tsawa/data/processed/classical_v3/{tsawa_spans,ignore_spans,tsawa_audit}_classical_v3.csv
Writes:  ~/tsawa_commentary_data/interleaved_v3_collapsed/docs/*.txt
         tsawa/data/processed/classical_v3_collapsed/ (same file names, remapped offsets,
         classical_split.csv copied, collapse_report.json)

Offsets: a position p maps to p minus the characters removed before it. A position inside a
removed run maps to just after the one kept newline. Every span is then checked: the collapsed
old text must equal the new text at the new offsets, the start must still be at a line start,
the end must still be on the shad, and no spans may overlap.
"""
from __future__ import annotations

import bisect
import csv
import importlib.util
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DOCS = Path.home() / "tsawa_commentary_data/interleaved_v3/docs"
OUT_DOCS = Path.home() / "tsawa_commentary_data/interleaved_v3_collapsed/docs"
CL = ROOT / "tsawa/data/processed/classical_v3"
OUT = ROOT / "tsawa/data/processed/classical_v3_collapsed"
RUN = re.compile(r"\n(?:[ \t\xa0​]*\n)+")

spec = importlib.util.spec_from_file_location("ebc", ROOT / "scratch/tsawa/analysis/end_boundary_check.py")
ebc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebc)


class Mapper:
    def __init__(self, text: str):
        self.runs = [(m.start(), m.end()) for m in RUN.finditer(text)]
        self.starts = [a for a, _ in self.runs]
        self.cum = [0]
        for a, b in self.runs:
            self.cum.append(self.cum[-1] + (b - a - 1))

    def __call__(self, p: int) -> int:
        j = bisect.bisect_left(self.starts, p) - 1  # last run with a < p
        if j < 0:
            return p
        a, b = self.runs[j]
        if p <= a + 1:
            return p - self.cum[j]
        if p <= b:
            return a + 1 - self.cum[j]
        return p - self.cum[j + 1]


def main() -> int:
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    old, new, maps = {}, {}, {}
    stats = Counter()
    for f in sorted(SRC_DOCS.glob("*.txt")):
        d = f.stem
        t = f.read_text(encoding="utf-8")
        nt = RUN.sub("\n", t)
        (OUT_DOCS / f.name).write_text(nt, encoding="utf-8")
        old[d], new[d], maps[d] = t, nt, Mapper(t)
        stats["docs"] += 1
        stats["runs_collapsed"] += len(maps[d].runs)
        stats["chars_removed"] += len(t) - len(nt)
        stats["chars_before"] += len(t)
        stats["chars_after"] += len(nt)
        assert "\n\n" not in nt
        assert len(nt) == maps[d](len(t))

    def remap_csv(name, cols_start="start", cols_end="end"):
        rows = list(csv.DictReader((CL / name).open(encoding="utf-8")))
        out = []
        for r in rows:
            m = maps[r["pecha_id"]]
            r["start"], r["end"] = m(int(r["start"])), m(int(r["end"]))
            out.append(r)
        return rows, out

    problems = Counter()
    # --- labelled spans
    src_rows = list(csv.DictReader((CL / "tsawa_spans_classical_v3.csv").open(encoding="utf-8")))
    new_rows = []
    by_doc = defaultdict(list)
    for r in src_rows:
        d = r["pecha_id"]
        s0, e0 = int(r["start"]), int(r["end"])
        s1, e1 = maps[d](s0), maps[d](e0)
        r2 = dict(r)
        r2["start"], r2["end"] = s1, e1
        r2["end_before_norm"] = maps[d](int(r["end_before_norm"]))
        # content preserved
        if RUN.sub("\n", old[d][s0:e0]) != new[d][s1:e1]:
            problems["span_text_changed"] += 1
        if not (s1 == 0 or new[d][s1 - 1] == "\n"):
            problems["start_not_at_line_start"] += 1
        if not ebc.is_valid_end(new[d], e1) or ebc.end_class(new[d], e1) != "shad":
            problems["end_not_on_shad"] += 1
        if new[d][e1 - 1] not in "།༔༎":
            problems["last_char_not_shad"] += 1
        if e1 <= s1:
            problems["empty_after_collapse"] += 1
        stats["spans_with_blank_inside"] += (e0 - s0) != (e1 - s1)
        new_rows.append(r2)
        by_doc[d].append((s1, e1))
    for d, S in by_doc.items():
        S.sort()
        problems["overlaps"] += sum(1 for (a, b), (c, e) in zip(S, S[1:]) if c < b)
    with (OUT / "tsawa_spans_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new_rows[0]))
        w.writeheader()
        w.writerows(new_rows)

    # --- ignore spans (repeats)
    ig_old = list(csv.DictReader((CL / "ignore_spans_classical_v3.csv").open(encoding="utf-8")))
    ig_new = []
    for r in ig_old:
        d = r["pecha_id"]
        s0, e0 = int(r["start"]), int(r["end"])
        s1, e1 = maps[d](s0), maps[d](e0)
        if RUN.sub("\n", old[d][s0:e0]) != new[d][s1:e1]:
            problems["ignore_text_changed"] += 1
        if e1 <= s1:
            problems["ignore_empty"] += 1
        ig_new.append({**r, "start": s1, "end": e1})
    with (OUT / "ignore_spans_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ig_new[0]))
        w.writeheader()
        w.writerows(ig_new)
    # ignore spans must still not overlap a labelled span
    for d in by_doc:
        S = sorted(by_doc[d])
        starts = [x for x, _ in S]
        for r in ig_new:
            if r["pecha_id"] != d:
                continue
            a, e = r["start"], r["end"]
            k = bisect.bisect_left(starts, e)
            if any(S[j][1] > a and S[j][0] < e for j in range(max(0, k - 3), k)):
                problems["ignore_overlaps_label"] += 1

    # --- audit rows
    audit = list(csv.DictReader((CL / "tsawa_audit_classical_v3.csv").open(encoding="utf-8")))
    for r in audit:
        d = r["pecha_id"]
        n = len(new[d])
        union = sum(e - s for s, e in by_doc[d])
        r["base_chars"] = n
        r["tsawa_chars_sum"] = r["tsawa_chars_union"] = union
        r["coverage_pct"] = 100 * union / n
        r["base_path"] = str(OUT_DOCS / f"{d}.txt")
    with (OUT / "tsawa_audit_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    shutil.copy(CL / "classical_split.csv", OUT / "classical_split.csv")

    problems = Counter({k: v for k, v in problems.items() if v})
    rep = {**stats, "n_spans": len(new_rows), "n_ignore": len(ig_new), "problems": dict(problems)}
    (OUT / "collapse_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

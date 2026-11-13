#!/usr/bin/env python3
"""Convert ~/tsawa_commentary_data/interleaved_v3 (25 docs, 6 classical root texts)
to the tsawa training-pipeline format. Reads the source folder, writes only to
tsawa/data/processed/classical_v3/.

Outputs
  tsawa_spans_classical_v3.csv     pecha_id,batch,ann_id,start,end,dropped,end_before_norm,normalized
  ignore_spans_classical_v3.csv    pecha_id,start,end,reason (tokens inside are labelled -100)
  tsawa_audit_classical_v3.csv     audit rows (same columns as tsawa_audit.csv), base_path -> the doc txt
  convert_report.json              counts and boundary-check numbers

Steps
  1. spans.csv (pecha_id,start,end_exclusive) -> add ann_id, rename end_exclusive -> end
  2. normalise span ends to the shad-INSIDE convention (a span that stops just before the
     shad is extended by the shad), same rule as tsawa_spans_merged_reannotated_normalized.csv
  3. boundary check on the result (end class, start at a line boundary)
  4. audit rows
"""
from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path.home() / "tsawa_commentary_data" / "interleaved_v3"
OUT = ROOT / "tsawa/data/processed/classical_v3"

spec = importlib.util.spec_from_file_location(
    "end_boundary_check", ROOT / "scratch/tsawa/analysis/end_boundary_check.py")
ebc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebc)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {r["doc_id"]: r for r in csv.DictReader((SRC / "manifest.csv").open(encoding="utf-8"))}
    texts = {d: (SRC / "docs" / f"{d}.txt").read_text(encoding="utf-8") for d in manifest}

    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in csv.DictReader((SRC / "spans.csv").open(encoding="utf-8")):
        spans[r["pecha_id"]].append((int(r["start"]), int(r["end_exclusive"])))
    unknown = sorted(set(spans) - set(manifest))
    if unknown:
        raise SystemExit(f"spans for docs not in manifest: {unknown}")

    rows, rep = [], {"before": Counter(), "after": Counter(), "per_doc": {}}
    n_blocked = 0
    for doc in sorted(manifest):
        t = texts[doc]
        S = sorted(spans[doc])
        for (a, b), (c, d) in zip(S, S[1:]):
            if c < b:
                raise SystemExit(f"{doc}: overlapping spans {(a, b)} {(c, d)}")
        for k, (s, e) in enumerate(S, 1):
            if not (0 <= s < e <= len(t)):
                raise SystemExit(f"{doc}: span out of bounds {(s, e)} len {len(t)}")
            rep["before"][ebc.end_class(t, e)] += 1
            ne, normed = e, False
            # shad + trailing NBSP/zero-width space: trim so the shad is the span's last char
            q = e
            while q > s and t[q - 1] in ebc.WS_INLINE:
                q -= 1
            if q < e and t[q - 1] in "།༔༎":
                ne, normed = q, True
            if not normed and ebc.end_class(t, e) == "before_shad":
                j = e
                while t[j] in ebc.WS_INLINE:
                    j += 1
                cand = j + 1  # include the shad
                nxt = S[k][0] if k < len(S) else len(t) + 1
                if cand <= nxt:
                    ne, normed = cand, True
                else:
                    n_blocked += 1
            rep["after"][ebc.end_class(t, ne)] += 1
            rows.append({"pecha_id": doc, "batch": "classical", "ann_id": f"{doc}-{k:05d}",
                         "start": s, "end": ne, "dropped": False,
                         "end_before_norm": e, "normalized": normed})

    with (OUT / "tsawa_spans_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ign = [(r["pecha_id"], r["start"], r["end"], r["reason"])
           for r in csv.DictReader((SRC / "ignore_spans.csv").open(encoding="utf-8"))]
    with (OUT / "ignore_spans_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pecha_id", "start", "end", "reason"])
        w.writerows(ign)

    # post-normalisation overlap check, per doc (spans and against ignore spans)
    by = defaultdict(list)
    for r in rows:
        by[r["pecha_id"]].append((r["start"], r["end"]))
    n_ov = sum(1 for d in by for (a, b), (c, e) in zip(sorted(by[d]), sorted(by[d])[1:]) if c < b)
    ig_by = defaultdict(list)
    for d, s, e, _ in ign:
        ig_by[d].append((int(s), int(e)))
    n_ov_ign = 0
    for d, S in by.items():
        S = sorted(S)
        starts = [x for x, _ in S]
        import bisect
        for a, e in ig_by[d]:
            k = bisect.bisect_left(starts, int(e))
            if any(S[j][1] > a and S[j][0] < e for j in range(max(0, k - 3), k)):
                n_ov_ign += 1

    audit_cols = next(csv.reader((ROOT / "tsawa/data/processed/tsawa_audit.csv").open(encoding="utf-8")))
    audit_rows = []
    for doc in sorted(manifest):
        t = texts[doc]
        S = sorted(by[doc])
        lens = [e - s for s, e in S]
        union = sum(lens)  # non-overlapping
        ls = sum(1 for s, e in S if s == 0 or t[s - 1] == "\n")
        ve = sum(1 for s, e in S if ebc.is_valid_end(t, e))
        rep["per_doc"][doc] = {
            "root_text": manifest[doc]["repo"], "chars": len(t), "spans": len(S),
            "ignore_spans": len(ig_by[doc]), "start_at_line_start_pct": round(100 * ls / len(S), 1),
            "end_valid_pct": round(100 * ve / len(S), 1), "median_span_chars": statistics.median(lens),
            "pct_under_10": round(100 * sum(1 for x in lens if x < 10) / len(S), 1),
            "tsawa_pct_of_text": round(100 * union / len(t), 1)}
        audit_rows.append({
            "pecha_id": doc, "source_batch": "classical", "has_tsawa_layer": True, "n_spans": len(S),
            "n_isverse_true": 0, "n_isverse_false": 0, "n_isverse_missing": len(S),
            "base_chars": len(t), "tsawa_chars_sum": union, "tsawa_chars_union": union,
            "coverage_pct": 100 * union / len(t), "avg_span_len": sum(lens) / len(S),
            "median_span_len": statistics.median(lens), "n_zero_length": 0, "n_inverted": 0,
            "n_out_of_bounds": 0, "n_overlap_pairs": 0, "n_spans_overlapping": 0,
            "parse_status": "ok", "notes": manifest[doc]["repo"], "opf_root": "", "tsawa_path": "",
            "base_path": str(SRC / "docs" / f"{doc}.txt")})
    with (OUT / "tsawa_audit_classical_v3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_cols)
        w.writeheader()
        for r in audit_rows:
            w.writerow({c: r.get(c, "") for c in audit_cols})

    rep.update(n_docs=len(manifest), n_spans=len(rows), n_ignore=len(ign), n_normalized=sum(r["normalized"] for r in rows),
               n_blocked=n_blocked, overlaps_after=n_ov, spans_overlapping_ignore=n_ov_ign,
               before=dict(rep["before"]), after=dict(rep["after"]))
    (OUT / "convert_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_doc"}, ensure_ascii=False, indent=1))
    print(f"{'doc':12}{'root text':38}{'spans':>7}{'start%':>8}{'end ok%':>9}{'median':>8}{'<10%':>6}{'tsawa%':>8}")
    for d, v in rep["per_doc"].items():
        print(f"{d:12}{v['root_text'][:36]:38}{v['spans']:>7}{v['start_at_line_start_pct']:>8}{v['end_valid_pct']:>9}"
              f"{v['median_span_chars']:>8}{v['pct_under_10']:>6}{v['tsawa_pct_of_text']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

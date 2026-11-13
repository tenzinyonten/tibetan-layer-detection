#!/usr/bin/env python3
"""Merge adjacent tsawa spans whose gap is only punctuation / whitespace.

Does **not** rewrite ``tsawa_spans_resolved.csv`` or any cloned YAML.
Writes ``tsawa/data/processed/tsawa_spans_merged.csv``.

Rule (both batches): within one pecha, sort by start and merge two consecutive
spans when ``base[end_i:start_{i+1}]`` contains only shad (། ༎), tsheg (་),
spaces, tabs or newlines. Chains (A+B+C) if every gap qualifies. A zero-width
gap (already adjacent) also qualifies.

Usage:
    python tsawa/src/merge_tsawa_spans.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

GAP_CHARS = set("།༎་ \t\n\r")
TOKENIZER = "jhu-clsp/mmBERT-base"
# User-facing window: max_length 8192, stride 5120 → 3072-token overlap.
# Content overlap after CLS/SEP is 8190 - 5120 = 3070; report against 3072
# as specified, and also flag content-fit separately if they differ.
WINDOW_OVERLAP = 3072
CONTENT_OVERLAP = 8190 - 5120


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge punctuation-separated tsawa spans.")
    p.add_argument(
        "--resolved",
        type=Path,
        default=Path("tsawa/data/processed/tsawa_spans_resolved.csv"),
    )
    p.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/shared/tsawa_audit.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tsawa/data/processed/tsawa_spans_merged.csv"),
    )
    p.add_argument("--tokenizer", default=TOKENIZER)
    return p.parse_args(argv)


def load_audit_bases(path: Path) -> dict[str, Path]:
    bases: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row["pecha_id"]
            bp = Path(row["base_path"]) if row.get("base_path") else Path()
            if bp.is_file():
                bases[pid] = bp
    return bases


def gap_is_mergeable(text: str, left_end: int, right_start: int) -> bool:
    if right_start < left_end:
        return True  # residual overlap: treat as already joined
    gap = text[left_end:right_start]
    return all(ch in GAP_CHARS for ch in gap)


def percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    return float(statistics.quantiles(vals, n=100)[min(99, max(0, int(p) - 1))])


def length_row(vals: list[int]) -> str:
    if not vals:
        return "n=0"
    return (
        f"n={len(vals):,}  median={statistics.median(vals):.0f}  "
        f"p90={percentile(vals, 90):.0f}  max={max(vals)}"
    )


def merge_pecha(
    spans: list[dict], text: str
) -> tuple[list[dict], int]:
    """Return merged rows and the number of qualifying gaps consumed."""
    ordered = sorted(spans, key=lambda r: (int(r["start"]), int(r["end"])))
    if not ordered:
        return [], 0
    groups: list[list[dict]] = [[ordered[0]]]
    n_merges = 0
    for rec in ordered[1:]:
        prev = groups[-1][-1]
        if gap_is_mergeable(text, int(prev["end"]), int(rec["start"])):
            groups[-1].append(rec)
            n_merges += 1
        else:
            groups.append([rec])

    out: list[dict] = []
    for g in groups:
        members = "+".join(r["ann_id"] for r in g)
        out.append(
            {
                "pecha_id": g[0]["pecha_id"],
                "batch": g[0]["batch"],
                "ann_id": g[0]["ann_id"] if len(g) == 1 else f"merge:{members}",
                "start": int(g[0]["start"]),
                "end": int(g[-1]["end"]),
                "n_members": len(g),
                "member_ann_ids": members,
                "dropped": False,
            }
        )
    return out, n_merges


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolved = args.resolved.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"resolved sidecar not found: {resolved}")

    bases = load_audit_bases(args.audit_csv.expanduser().resolve())
    by_repo: dict[str, list[dict]] = defaultdict(list)
    n_dropped = 0
    with resolved.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("dropped", "")).lower() == "true":
                n_dropped += 1
                continue
            s, e = int(row["start"]), int(row["end"])
            if e <= s:
                n_dropped += 1
                continue
            by_repo[row["pecha_id"]].append(row)

    merged_rows: list[dict] = []
    before_len: dict[str, list[int]] = {"old": [], "new": [], "all": []}
    after_len: dict[str, list[int]] = {"old": [], "new": [], "all": []}
    n_before = {"old": 0, "new": 0}
    n_after = {"old": 0, "new": 0}
    n_merges = {"old": 0, "new": 0}
    n_gaps = {"old": 0, "new": 0}

    for pid, spans in by_repo.items():
        batch = spans[0].get("batch") or ("old" if pid.startswith("P") else "new")
        if batch not in ("old", "new"):
            batch = "old" if pid.startswith("P") else "new"
        base = bases.get(pid)
        if base is None or not base.is_file():
            raise SystemExit(f"no base text for {pid}")
        text = base.read_text(encoding="utf-8")
        n_before[batch] += len(spans)
        n_gaps[batch] += max(0, len(spans) - 1)
        for r in spans:
            ln = int(r["end"]) - int(r["start"])
            before_len[batch].append(ln)
            before_len["all"].append(ln)
        out, merges = merge_pecha(spans, text)
        n_merges[batch] += merges
        n_after[batch] += len(out)
        for r in out:
            r["batch"] = batch
            ln = r["end"] - r["start"]
            after_len[batch].append(ln)
            after_len["all"].append(ln)
        merged_rows.extend(out)

    merged_rows.sort(key=lambda r: (r["pecha_id"], r["start"], r["end"]))
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pecha_id",
        "batch",
        "ann_id",
        "start",
        "end",
        "n_members",
        "member_ann_ids",
        "dropped",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(merged_rows)

    print(f"Source: {resolved}  (skipped dropped={n_dropped})")
    print(f"Wrote {len(merged_rows):,} merged spans to {out_path}")
    print("Cloned YAML and tsawa_spans_resolved.csv were not modified.\n")

    print(f"{'batch':8} {'before':>8} {'after':>8} {'merges':>8} "
          f"{'gaps':>8} {'gap% qualify':>13}")
    for b in ("old", "new"):
        gaps = n_gaps[b] or 1
        print(
            f"{b:8} {n_before[b]:8,} {n_after[b]:8,} {n_merges[b]:8,} "
            f"{n_gaps[b]:8,} {100.0 * n_merges[b] / gaps:12.1f}%"
        )
    print(
        f"{'total':8} {sum(n_before.values()):8,} {sum(n_after.values()):8,} "
        f"{sum(n_merges.values()):8,}"
    )

    print("\n=== span length (chars) ===")
    print(f"{'':8} {'before':>44}    {'after'}")
    for b in ("old", "new", "all"):
        print(f"{b:8} {length_row(before_len[b]):<44}    {length_row(after_len[b])}")

    old_rate = 100.0 * n_merges["old"] / (n_gaps["old"] or 1)
    new_rate = 100.0 * n_merges["new"] / (n_gaps["new"] or 1)
    if old_rate > 5.0:
        print(f"\nFLAG: old-batch qualifying-gap rate is {old_rate:.1f}%, "
              "expected ~0.8%.")
    if new_rate < 15.0:
        print(f"\nFLAG: new-batch qualifying-gap rate is {new_rate:.1f}%, "
              "expected ~27%.")

    print(f"\nTokenizing {len(merged_rows):,} merged spans with {args.tokenizer} …")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.model_max_length = int(1e12)
    texts_cache: dict[str, str] = {}
    over_3072: list[tuple[str, int, int, int]] = []
    over_content: list[tuple[str, int, int, int]] = []
    n_before_over = 0

    # before-merge token lengths for the "only 1 span did" check
    for pid, spans in by_repo.items():
        if pid not in texts_cache:
            texts_cache[pid] = bases[pid].read_text(encoding="utf-8")
        text = texts_cache[pid]
        for r in spans:
            piece = text[int(r["start"]):int(r["end"])]
            n = len(tok(piece, add_special_tokens=False)["input_ids"])
            if n > WINDOW_OVERLAP:
                n_before_over += 1

    for r in merged_rows:
        pid = r["pecha_id"]
        if pid not in texts_cache:
            texts_cache[pid] = bases[pid].read_text(encoding="utf-8")
        piece = texts_cache[pid][r["start"]:r["end"]]
        n = len(tok(piece, add_special_tokens=False)["input_ids"])
        r["_n_tok"] = n
        if n > WINDOW_OVERLAP:
            over_3072.append((pid, r["start"], r["end"], n))
        if n > CONTENT_OVERLAP:
            over_content.append((pid, r["start"], r["end"], n))

    print(f"Spans > {WINDOW_OVERLAP} tokens before merge: {n_before_over}")
    print(f"Spans > {WINDOW_OVERLAP} tokens after merge:  {len(over_3072)}")
    if over_3072:
        longest = max(over_3072, key=lambda t: t[3])
        print("  longest:", longest[0], f"[{longest[1]}:{longest[2]}]",
              f"{longest[3]} tokens")
        for pid, s, e, n in sorted(over_3072, key=lambda t: -t[3])[:15]:
            print(f"    {pid} [{s}:{e}] {n} tokens")
        L = longest[3]
        # Team convention (quotation_w8192_s4751.yaml): hf stride (overlap)
        # must be >= longest span; step = max_length - overlap.
        proposed_overlap = L
        proposed_stride = 8192 - proposed_overlap
        print(
            f"\nWINDOW FIT: {len(over_3072)} merged span(s) exceed the "
            f"{WINDOW_OVERLAP}-token overlap of 8192/5120, so they can never "
            "appear whole in one window."
        )
        print(
            f"  Propose max_length=8192 stride={proposed_stride} "
            f"(overlap={proposed_overlap} = longest span). "
            "That is the same rule as "
            "layer_detection_model_train/script/quotation_w8192_s4751.yaml "
            f"(they raised overlap to the measured longest span)."
        )
        if proposed_stride < 1:
            print("  Longest span exceeds 8192 tokens — no 8192-window stride "
                  "can contain it whole.")
    else:
        print("All merged spans fit inside a 3072-token overlap.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

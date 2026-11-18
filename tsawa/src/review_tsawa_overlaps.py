#!/usr/bin/env python3
"""Print post-snap overlapping tsawa pairs for manual review (read-only).

Reads the sidecar from ``snap_tsawa_boundaries.py`` and finds pairs in the
same pecha whose snapped half-open ``[start, end)`` ranges overlap. Does
not merge, trim, or write back to YAML.

Expected count is the 18 adjacent pairs from the last snap run. If this
script finds a different number, it stops so we do not review the wrong set.

Usage:
    python tsawa/src/review_tsawa_overlaps.py \\
        --sidecar tsawa/data/processed/tsawa_spans_snapped.csv \\
        --raw-opf-dir data/raw_opf \\
        --out tsawa/data/processed/overlap_review.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # check_boundary_snapping
from check_boundary_snapping import resolve_opf_root  # noqa: E402

EXPECTED_OVERLAP_PAIRS = 17
CONTEXT_CHARS = 50


def load_sidecar(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Sidecar not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"pecha_id", "ann_id", "start_orig", "end_orig", "start", "end"}
    missing = required - set(rows[0].keys() if rows else set())
    if missing:
        raise SystemExit(f"Sidecar missing columns {sorted(missing)}")
    return rows


def find_overlap_pairs(spans: list[dict]) -> list[tuple[dict, dict]]:
    """All unordered pairs in this repo whose snapped ranges overlap."""
    pairs: list[tuple[dict, dict]] = []
    n = len(spans)
    for i in range(n):
        a = spans[i]
        a_s, a_e = int(a["start"]), int(a["end"])
        for j in range(i + 1, n):
            b = spans[j]
            b_s, b_e = int(b["start"]), int(b["end"])
            if a_s < b_e and b_s < a_e:
                # Earlier-starting span first, for stable context.
                if (a_s, a_e, a["ann_id"]) <= (b_s, b_e, b["ann_id"]):
                    pairs.append((a, b))
                else:
                    pairs.append((b, a))
    return pairs


def overlap_len(a: dict, b: dict) -> int:
    lo = max(int(a["start"]), int(b["start"]))
    hi = min(int(a["end"]), int(b["end"]))
    return max(0, hi - lo)


def load_base(raw_opf_dir: Path, pecha_id: str) -> str:
    root = resolve_opf_root(raw_opf_dir, pecha_id)
    base = root / "base" / "v001.txt"
    if not base.is_file():
        raise SystemExit(f"Missing base text for {pecha_id}: {base}")
    return base.read_text(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report post-snap overlapping tsawa pairs (read-only)."
    )
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--raw-opf-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--expect",
        type=int,
        default=EXPECTED_OVERLAP_PAIRS,
        help=f"Abort if pair count != this (default: {EXPECTED_OVERLAP_PAIRS}; -1 = skip).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_sidecar(args.sidecar.expanduser().resolve())
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_repo[row["pecha_id"]].append(row)

    pairs: list[tuple[str, dict, dict]] = []
    for pecha_id, spans in sorted(by_repo.items()):
        for a, b in find_overlap_pairs(spans):
            pairs.append((pecha_id, a, b))

    if args.expect is not None and args.expect >= 0 and len(pairs) != args.expect:
        raise SystemExit(
            f"Found {len(pairs)} overlapping pairs, expected {args.expect}. "
            "Refusing to write a review file until this is reconciled "
            "(snap run vs all-pairs definition)."
        )

    out_rows: list[dict] = []
    raw_dir = args.raw_opf_dir.expanduser().resolve()
    cache: dict[str, str] = {}

    print(f"=== {len(pairs)} overlapping pairs (snapped [start, end)) ===\n")
    for i, (pecha_id, a, b) in enumerate(pairs, start=1):
        if pecha_id not in cache:
            cache[pecha_id] = load_base(raw_dir, pecha_id)
        text = cache[pecha_id]
        a_s, a_e = int(a["start"]), int(a["end"])
        b_s, b_e = int(b["start"]), int(b["end"])
        ctx_lo = min(a_s, b_s)
        ctx_hi = max(a_e, b_e)
        before = text[max(0, ctx_lo - CONTEXT_CHARS) : ctx_lo]
        after = text[ctx_hi : ctx_hi + CONTEXT_CHARS]
        text_a = text[a_s:a_e]
        text_b = text[b_s:b_e]
        ov = overlap_len(a, b)
        batch = a.get("batch") or b.get("batch") or ""

        print(f"---------- [{i}/{len(pairs)}] {pecha_id} ({batch}) ----------")
        print(f"  A  ann={a['ann_id']}")
        print(f"     orig [{a['start_orig']}:{a['end_orig']}]  "
              f"snapped [{a_s}:{a_e}] ({a_e - a_s} chars)")
        print(f"  B  ann={b['ann_id']}")
        print(f"     orig [{b['start_orig']}:{b['end_orig']}]  "
              f"snapped [{b_s}:{b_e}] ({b_e - b_s} chars)")
        print(f"  overlap: {ov} chars")
        print(f"--- CONTEXT BEFORE ---\n{before}")
        print(f"--- SPAN A ---\n{text_a}")
        print(f"--- SPAN B ---\n{text_b}")
        print(f"--- CONTEXT AFTER ---\n{after}\n")

        out_rows.append(
            {
                "repo_id": pecha_id,
                "batch": batch,
                "ann_id_a": a["ann_id"],
                "ann_id_b": b["ann_id"],
                "start_orig_a": a["start_orig"],
                "end_orig_a": a["end_orig"],
                "start_a": a_s,
                "end_a": a_e,
                "start_orig_b": b["start_orig"],
                "end_orig_b": b["end_orig"],
                "start_b": b_s,
                "end_b": b_e,
                "overlap_chars": ov,
                "text_a": text_a.replace("\n", " "),
                "text_b": text_b.replace("\n", " "),
                "context_before": before.replace("\n", " "),
                "context_after": after.replace("\n", " "),
            }
        )

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} pairs to {out}")
    print("No spans were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

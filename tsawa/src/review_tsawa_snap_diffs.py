#!/usr/bin/env python3
"""Sample before/after text for spans that the snapper actually moved.

Read-only. Uses nested ``.opf`` resolution from ``check_boundary_snapping``.

A start snap walks left (``start_delta < 0``): characters ADDED at the
head. An end snap walks right (``end_delta > 0``): characters ADDED at
the tail. Both sides are measured separately.

Usage:
    python tsawa/src/review_tsawa_snap_diffs.py \\
        --sidecar tsawa/data/processed/tsawa_spans_snapped.csv \\
        --raw-opf-dir data/raw_opf \\
        --n 15 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # check_boundary_snapping
from review_tsawa_overlaps import load_base, load_sidecar  # noqa: E402

LARGE_ADD_CHARS = 5


def _visible(s: str) -> str:
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print a random sample of snapper before/after diffs."
    )
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--raw-opf-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=15, help="sample size (default: 15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--large-add",
        type=int,
        default=LARGE_ADD_CHARS,
        help="Flag snaps that added more than this many chars on one side (default: 5).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_sidecar(args.sidecar.expanduser().resolve())
    changed: list[dict] = []
    start_adds: list[int] = []
    end_adds: list[int] = []
    large: list[dict] = []

    for row in rows:
        s0, e0 = int(row["start_orig"]), int(row["end_orig"])
        s1, e1 = int(row["start"]), int(row["end"])
        if s0 == s1 and e0 == e1:
            continue
        start_add = s0 - s1  # leftward start → positive chars added
        end_add = e1 - e0  # rightward end → positive chars added
        row = dict(row)
        row["_s0"] = s0
        row["_e0"] = e0
        row["_s1"] = s1
        row["_e1"] = e1
        row["_start_add"] = start_add
        row["_end_add"] = end_add
        changed.append(row)
        if start_add:
            start_adds.append(start_add)
        if end_add:
            end_adds.append(end_add)
        if start_add > args.large_add or end_add > args.large_add:
            large.append(row)

    print(
        f"Spans with any snap: {len(changed)} / {len(rows)} "
        f"(start-moved {len(start_adds)}, end-moved {len(end_adds)})"
    )
    if not changed:
        print("No snapped spans found.")
        return 0

    rng = random.Random(args.seed)
    sample = changed[:]
    rng.shuffle(sample)
    sample = sample[: min(args.n, len(sample))]

    raw_dir = args.raw_opf_dir.expanduser().resolve()
    cache: dict[str, str] = {}

    for i, row in enumerate(sample, start=1):
        pecha_id = row["pecha_id"]
        if pecha_id not in cache:
            cache[pecha_id] = load_base(raw_dir, pecha_id)
        text = cache[pecha_id]
        s0, e0, s1, e1 = row["_s0"], row["_e0"], row["_s1"], row["_e1"]
        orig = text[s0:e0]
        snapped = text[s1:e1]
        added_head = text[s1:s0] if s1 < s0 else ""
        added_tail = text[e0:e1] if e1 > e0 else ""
        print(f"\n========== [{i}/{len(sample)}] {pecha_id}  ann={row['ann_id']} ==========")
        print(f"  orig    [{s0}:{e0}] ({e0 - s0} chars)")
        print(f"  snapped [{s1}:{e1}] ({e1 - s1} chars)")
        print(f"  start-side added: {row['_start_add']} chars  "
              f"end-side added: {row['_end_add']} chars  "
              f"net length Δ: {(e1 - s1) - (e0 - s0)}")
        print(f"  ADDED HEAD: {_visible(added_head)!r}")
        print(f"  ADDED TAIL: {_visible(added_tail)!r}")
        print(f"--- ORIGINAL ---\n{orig}")
        print(f"--- SNAPPED ---\n{snapped}")

    def summarize(name: str, vals: list[int]) -> None:
        if not vals:
            print(f"  {name}: (none)")
            return
        print(
            f"  {name}: n={len(vals)}  min={min(vals)}  "
            f"median={statistics.median(vals)}  mean={statistics.mean(vals):.2f}  "
            f"max={max(vals)}"
        )

    print("\n=== SNAP SIZE DISTRIBUTION (chars added per side) ===")
    summarize("start-side (walk left)", start_adds)
    summarize("end-side (walk right)", end_adds)
    both = start_adds + end_adds
    summarize("either side", both)

    print(f"\n=== LARGE ADDS (>{args.large_add} chars on one side): {len(large)} ===")
    if not large:
        print("  (none)")
    else:
        for row in large[:50]:
            print(
                f"  {row['pecha_id']} {row['ann_id']}  "
                f"start+{row['_start_add']} end+{row['_end_add']}  "
                f"[{row['_s0']}:{row['_e0']}] → [{row['_s1']}:{row['_e1']}]"
            )
        if len(large) > 50:
            print(f"  … {len(large) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

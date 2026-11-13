#!/usr/bin/env python3
"""Snap dirty tsawa span edges to the nearest syllable boundary.

Does **not** rewrite cloned ``Tsawa.yml`` files. Writes a sidecar CSV of
proposed ``(start, end)`` so the originals stay pristine.

Snap rules (capped walk; abort and leave original if no marker in range):

* dirty start → walk **left** until a boundary char (or document start);
  new ``start`` is the first character after that marker.
* dirty end → walk **right** until a boundary char; new ``end`` is just
  after that marker (span last-char is the tsheg/shad).

Uses the same audit / nested-``.opf`` helpers and ``BOUNDARY_CHARS`` /
``check_span`` as ``check_boundary_snapping.py``.

Usage:
    python tsawa/src/snap_tsawa_boundaries.py \\
        --audit-csv data/shared/tsawa_audit.csv \\
        --raw-opf-dir data/raw_opf \\
        --out tsawa/data/processed/tsawa_spans_snapped.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # check_boundary_snapping
from check_boundary_snapping import (  # noqa: E402
    BOUNDARY_CHARS,
    check_span,
    infer_batch,
    load_repos_with_tsawa,
    load_spans,
    resolve_opf_root,
)


def snap_start(text: str, start: int, max_walk: int) -> tuple[int, str]:
    """Return (new_start, status). status is ok / unchanged / unsnapped_start."""
    flags = check_span(text, start, min(len(text), start + 1) if start < len(text) else start)
    # Only start cleanliness matters here.
    if start == 0 or (start > 0 and text[start - 1] in BOUNDARY_CHARS):
        return start, "unchanged"
    walked = 0
    i = start - 1
    while i >= 0 and walked < max_walk:
        if text[i] in BOUNDARY_CHARS:
            return i + 1, "ok"
        i -= 1
        walked += 1
    if i < 0:
        return 0, "ok"
    return start, "unsnapped_start"


def snap_end(text: str, end: int, max_walk: int) -> tuple[int, str]:
    """Return (new_end, status)."""
    n = len(text)
    if end <= 0 or end > n:
        return end, "unsnapped_end"
    if text[end - 1] in BOUNDARY_CHARS:
        return end, "unchanged"
    walked = 0
    # Look at characters from ``end`` onward; first boundary becomes last char.
    j = end
    while j < n and walked < max_walk:
        if text[j] in BOUNDARY_CHARS:
            return j + 1, "ok"
        j += 1
        walked += 1
    return end, "unsnapped_end"


def overlaps_in_repo(spans: list[tuple[str, int, int]]) -> list[tuple[str, str]]:
    """Return pairs of ann_ids that overlap after snapping (half-open)."""
    ordered = sorted(spans, key=lambda t: (t[1], t[2], t[0]))
    hits: list[tuple[str, str]] = []
    for i in range(len(ordered) - 1):
        a_id, a_s, a_e = ordered[i]
        b_id, b_s, b_e = ordered[i + 1]
        if a_s < b_e and b_s < a_e:
            hits.append((a_id, b_id))
    return hits


def empty_stats() -> dict[str, int]:
    return {
        "total": 0,
        "dirty_start": 0,
        "dirty_end": 0,
        "dirty_either": 0,
        "start_snapped": 0,
        "end_snapped": 0,
        "unsnapped_start": 0,
        "unsnapped_end": 0,
    }


def add_dirty(stats: dict[str, int], flags: dict) -> None:
    stats["total"] += 1
    if not flags["start_is_clean"]:
        stats["dirty_start"] += 1
    if not flags["end_is_clean"]:
        stats["dirty_end"] += 1
    if not flags["start_is_clean"] or not flags["end_is_clean"]:
        stats["dirty_either"] += 1


def print_table(title: str, by_batch: dict[str, dict[str, int]]) -> None:
    def cell(batch: str, key: str) -> str:
        s = by_batch[batch]
        n = s[key]
        tot = s["total"]
        if key == "total":
            return str(n)
        if tot == 0:
            return f"{n} (n/a)"
        return f"{n} ({100.0 * n / tot:.1f}%)"

    print(title)
    print(f"{'':22} {'old':>16} {'new':>16} {'combined':>16}")
    for label, key in (
        ("Total spans", "total"),
        ("Dirty start", "dirty_start"),
        ("Dirty end", "dirty_end"),
        ("Dirty either", "dirty_either"),
    ):
        print(
            f"{label:22} {cell('old', key):>16} {cell('new', key):>16} "
            f"{cell('combined', key):>16}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propose snapped tsawa offsets (does not edit cloned YAML)."
    )
    p.add_argument("--audit-csv", type=Path, required=True)
    p.add_argument("--raw-opf-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tsawa/data/processed/tsawa_spans_snapped.csv"),
        help="Sidecar of all spans with original + snapped offsets.",
    )
    p.add_argument(
        "--max-walk",
        type=int,
        default=12,
        help="Max characters to walk left/right when snapping (default: 12).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_walk < 1:
        raise SystemExit("--max-walk must be >= 1")

    repos = load_repos_with_tsawa(args.audit_csv)
    if not repos:
        raise SystemExit("No repos with tsawa spans found — check audit CSV.")

    before = {b: empty_stats() for b in ("old", "new", "combined")}
    after = {b: empty_stats() for b in ("old", "new", "combined")}
    rows: list[dict] = []
    overlap_count = 0
    example_p000207 = None

    for repo in repos:
        pecha_id = repo["pecha_id"]
        batch = infer_batch(pecha_id, repo)
        if batch not in ("old", "new"):
            batch = "new" if pecha_id.startswith("I") else "old"
        opf_dir = resolve_opf_root(args.raw_opf_dir, pecha_id, repo)
        base_path = (
            Path(repo["base_path"]) if repo.get("base_path") else opf_dir / "base" / "v001.txt"
        )
        tsawa_path = Path(repo["tsawa_path"]) if repo.get("tsawa_path") else None
        if not base_path.is_file():
            print(f"warning: missing base for {pecha_id}: {base_path}")
            continue
        text = base_path.read_text(encoding="utf-8")
        snapped_for_overlap: list[tuple[str, int, int]] = []

        for span in load_spans(opf_dir, tsawa_path):
            s0, e0 = span["start"], span["end"]
            flags0 = check_span(text, s0, e0)
            add_dirty(before[batch], flags0)
            add_dirty(before["combined"], flags0)

            new_s, st_status = snap_start(text, s0, args.max_walk)
            new_e, en_status = snap_end(text, e0, args.max_walk)
            if new_s >= new_e:
                new_s, new_e = s0, e0
                reason = "aborted_inverted"
            else:
                reasons = []
                if st_status == "unsnapped_start":
                    reasons.append(st_status)
                    after[batch]["unsnapped_start"] += 1
                    after["combined"]["unsnapped_start"] += 1
                if en_status == "unsnapped_end":
                    reasons.append(en_status)
                    after[batch]["unsnapped_end"] += 1
                    after["combined"]["unsnapped_end"] += 1
                reason = ",".join(reasons)

            if new_s != s0:
                after[batch]["start_snapped"] += 1
                after["combined"]["start_snapped"] += 1
            if new_e != e0:
                after[batch]["end_snapped"] += 1
                after["combined"]["end_snapped"] += 1

            flags1 = check_span(text, new_s, new_e)
            add_dirty(after[batch], flags1)
            add_dirty(after["combined"], flags1)
            snapped_for_overlap.append((span["ann_id"], new_s, new_e))

            rec = {
                "pecha_id": pecha_id,
                "batch": batch,
                "ann_id": span["ann_id"],
                "start_orig": s0,
                "end_orig": e0,
                "start": new_s,
                "end": new_e,
                "start_delta": new_s - s0,
                "end_delta": new_e - e0,
                "start_snapped": new_s != s0,
                "end_snapped": new_e != e0,
                "unsnapped_reason": reason,
                "start_is_clean_after": flags1["start_is_clean"],
                "end_is_clean_after": flags1["end_is_clean"],
            }
            rows.append(rec)
            if pecha_id == "P000207" and s0 == 92705 and e0 == 92820:
                example_p000207 = (text, rec)

        for a, b in overlaps_in_repo(snapped_for_overlap):
            overlap_count += 1

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "pecha_id",
        "batch",
        "ann_id",
        "start_orig",
        "end_orig",
        "start",
        "end",
        "start_delta",
        "end_delta",
        "start_snapped",
        "end_snapped",
        "unsnapped_reason",
        "start_is_clean_after",
        "end_is_clean_after",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} spans (all, not only dirty) to {out}")
    print(f"Cloned YAML was not modified. max_walk={args.max_walk}")
    print(
        f"Starts snapped: {after['combined']['start_snapped']}  "
        f"ends snapped: {after['combined']['end_snapped']}  "
        f"unsnapped start/end: {after['combined']['unsnapped_start']}/"
        f"{after['combined']['unsnapped_end']}  "
        f"adjacent overlaps after snap: {overlap_count}"
    )
    print()
    print_table("=== BEFORE (same check as boundary_check) ===", before)
    print()
    print_table("=== AFTER snap ===", after)

    if example_p000207:
        text, rec = example_p000207
        s, e = rec["start"], rec["end"]
        snippet = text[s:e][:20].replace("\n", " ")
        print()
        print(
            f"P000207 [92705:92820] → [{s}:{e}] "
            f"start_delta={rec['start_delta']} snippet={snippet!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resolve post-snap overlapping tsawa pairs (sidecar only).

Applies the three manual-review rules; does **not** edit cloned YAML.

1. Trim trailing citation particles (ཞེས་ / ཅེས་ / གསུངས་…) from the
   earlier span (``text_a``). Those belong to commentary, not the verse.
2. Strip bracketed folio/verse tags (``[༢༣]``) and Tibetan digit prefixes
   (``༡༤``, ``༢༤``) from A's tail and B's head.
3. If the remaining overlap is ≤ 10 characters, cut A back to its last
   shad (། / ༎ / ༔) and start B at the next non-boundary character.
   If A has no shad, clip A at the overlap start (overlap goes to B).

Empty / suffix-stub B spans are dropped. Writes a resolved sidecar and a
per-pair log. Re-counts overlaps and aborts if any remain (unless
``--allow-remaining``).

Usage:
    python tsawa/src/resolve_tsawa_overlaps.py \\
        --sidecar tsawa/data/processed/tsawa_spans_snapped.csv \\
        --raw-opf-dir data/raw_opf \\
        --out tsawa/data/processed/tsawa_spans_resolved.csv \\
        --log tsawa/data/processed/overlap_resolve_log.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # check_boundary_snapping
from check_boundary_snapping import BOUNDARY_CHARS, resolve_opf_root  # noqa: E402
from review_tsawa_overlaps import (  # noqa: E402
    EXPECTED_OVERLAP_PAIRS,
    find_overlap_pairs,
    load_base,
    load_sidecar,
    overlap_len,
)

# Longest-first. Written as Unicode escapes so the source stays ASCII-safe.
CITATION_SUFFIXES = [
    "\u0f5e\u0f7a\u0f66\u0f0b\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b\u0f4f\u0f7a",  # zhes gsungs te
    "\u0f45\u0f7a\u0f66\u0f0b\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b\u0f4f\u0f7a",  # ces gsungs te
    "\u0f5e\u0f7a\u0f66\u0f0b\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b",  # zhes gsungs
    "\u0f45\u0f7a\u0f66\u0f0b\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b",  # ces gsungs
    "\u0f5e\u0f7a\u0f66\u0f0b\u0f56\u0f66\u0f0b",  # zhes pas
    "\u0f45\u0f7a\u0f66\u0f0b\u0f56\u0f66\u0f0b",  # ces pas
    "\u0f45\u0f7a\u0f66\u0f0b\u0f56\u0f66",  # ces pas (no tsheg)
    "\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b\u0f4f\u0f7a",  # gsungs te
    "\u0f5e\u0f7a\u0f66\u0f0b",  # zhes
    "\u0f45\u0f7a\u0f66\u0f0b",  # ces
    "\u0f42\u0f66\u0f74\u0f44\u0f66\u0f0b",  # gsungs
]

SHAD_CHARS = frozenset("།༎༔")
TIBETAN_DIGITS = "༠༡༢༣༤༥༦༧༨༩"
BRACKET_TAG = re.compile(r"\[[^\[\]]+\]")
LEADING_DIGITS = re.compile(rf"^[{TIBETAN_DIGITS}]+")
TRAILING_DIGITS = re.compile(rf"[{TIBETAN_DIGITS}]+\s*$")
MAX_RULE3_OVERLAP = 10


def _rstrip_ws(text: str, end: int) -> int:
    while end > 0 and text[end - 1] in " \t\n\r\xa0":
        end -= 1
    return end


def trim_citation_end(text: str, start: int, end: int) -> tuple[int, str]:
    end = _rstrip_ws(text, end)
    chunk = text[start:end]
    for suf in CITATION_SUFFIXES:
        if chunk.endswith(suf):
            return end - len(suf), f"trim_citation:{suf}"
    return end, ""


def trim_metadata_end(text: str, start: int, end: int) -> tuple[int, list[str]]:
    notes: list[str] = []
    while end > start:
        end = _rstrip_ws(text, end)
        chunk = text[start:end]
        m = BRACKET_TAG.search(chunk)
        if m and m.end() == len(chunk):
            end = start + m.start()
            notes.append(f"trim_bracket_tail:{m.group()}")
            continue
        m = TRAILING_DIGITS.search(chunk)
        if m:
            end = start + m.start()
            notes.append(f"trim_digits_tail:{m.group().strip()}")
            continue
        break
    return end, notes


def trim_metadata_start(text: str, start: int, end: int) -> tuple[int, list[str]]:
    notes: list[str] = []
    while start < end:
        while start < end and text[start] in " \t\n\r\xa0":
            start += 1
        rest = text[start:end]
        m = BRACKET_TAG.match(rest)
        if m:
            start += m.end()
            notes.append(f"trim_bracket_head:{m.group()}")
            continue
        m = LEADING_DIGITS.match(rest)
        if m:
            start += m.end()
            notes.append(f"trim_digits_head:{m.group()}")
            continue
        break
    return start, notes


def last_shad_end(text: str, start: int, end: int) -> int | None:
    last = None
    for i in range(start, end):
        if text[i] in SHAD_CHARS:
            last = i + 1
    return last


def skip_boundaries(text: str, pos: int, limit: int) -> int:
    while pos < limit and text[pos] in BOUNDARY_CHARS | SHAD_CHARS:
        pos += 1
    return pos


def resolve_pair(text: str, a: dict, b: dict) -> tuple[dict, dict, list[str]]:
    """Return updated copies of a, b plus action notes."""
    a = dict(a)
    b = dict(b)
    notes: list[str] = []
    a_s, a_e = int(a["start"]), int(a["end"])
    b_s, b_e = int(b["start"]), int(b["end"])

    new_e, note = trim_citation_end(text, a_s, a_e)
    if note:
        a_e = new_e
        notes.append("A:" + note)

    a_e, meta_a = trim_metadata_end(text, a_s, a_e)
    notes.extend("A:" + n for n in meta_a)
    b_s, meta_b = trim_metadata_start(text, b_s, b_e)
    notes.extend("B:" + n for n in meta_b)

    ov = max(0, min(a_e, b_e) - max(a_s, b_s))
    if 0 < ov <= MAX_RULE3_OVERLAP:
        shad = last_shad_end(text, a_s, a_e)
        if shad is not None and shad > a_s:
            a_e = shad
            notes.append(f"A:cut_to_last_shad->{a_e}")
            b_s = skip_boundaries(text, max(b_s, a_e), b_e)
            notes.append(f"B:start_after_punct->{b_s}")
        else:
            a_e = min(a_e, max(a_s, min(a_e, b_s)))
            notes.append(f"A:clip_to_overlap_start->{a_e}")

    # Re-clip if they still overlap (tiny residual).
    if a_s < b_e and b_s < a_e and a_e > a_s:
        a_e = min(a_e, b_s)
        notes.append(f"A:final_clip->{a_e}")

    a["start"], a["end"] = a_s, a_e
    b["start"], b["end"] = b_s, b_e
    a["dropped"] = a_e <= a_s
    b["dropped"] = b_e <= b_s
    if a["dropped"]:
        notes.append("A:dropped_empty")
    if b["dropped"]:
        notes.append("B:dropped_empty")
    # Suffix stub: B is contained in A or leftover particle-only.
    if not b["dropped"] and not a["dropped"]:
        bt = text[b_s:b_e].strip()
        at = text[a_s:a_e]
        if bt and (bt in at or at.endswith(bt)) and (b_e - b_s) <= 20:
            b["dropped"] = True
            notes.append("B:dropped_suffix_stub")
        elif bt in CITATION_SUFFIXES or bt in {"[༢༣]", "[༢༢]", "[༥]"}:
            b["dropped"] = True
            notes.append("B:dropped_marker_only")
    return a, b, notes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resolve post-snap overlapping tsawa pairs (sidecar only)."
    )
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--raw-opf-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tsawa/data/processed/tsawa_spans_resolved.csv"),
    )
    p.add_argument(
        "--log",
        type=Path,
        default=Path("tsawa/data/processed/overlap_resolve_log.csv"),
    )
    p.add_argument(
        "--expect",
        type=int,
        default=EXPECTED_OVERLAP_PAIRS,
        help=f"Input pair count must match (default: {EXPECTED_OVERLAP_PAIRS}).",
    )
    p.add_argument(
        "--allow-remaining",
        action="store_true",
        help="Do not abort if overlaps remain after resolution.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_sidecar(args.sidecar.expanduser().resolve())
    by_key: dict[tuple[str, str], dict] = {}
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rec = dict(row)
        rec["start"] = int(rec["start"])
        rec["end"] = int(rec["end"])
        rec["dropped"] = False
        rec["resolve_notes"] = ""
        key = (rec["pecha_id"], rec["ann_id"])
        by_key[key] = rec
        by_repo[rec["pecha_id"]].append(rec)

    pairs: list[tuple[str, dict, dict]] = []
    for pecha_id, spans in sorted(by_repo.items()):
        for a, b in find_overlap_pairs(spans):
            pairs.append((pecha_id, a, b))
    if args.expect is not None and len(pairs) != args.expect:
        raise SystemExit(
            f"Found {len(pairs)} overlapping pairs, expected {args.expect}."
        )

    raw_dir = args.raw_opf_dir.expanduser().resolve()
    cache: dict[str, str] = {}
    logs: list[dict] = []

    for pecha_id, a0, b0 in pairs:
        if pecha_id not in cache:
            cache[pecha_id] = load_base(raw_dir, pecha_id)
        text = cache[pecha_id]
        # Use current (possibly already-updated) records.
        a = by_key[(pecha_id, a0["ann_id"])]
        b = by_key[(pecha_id, b0["ann_id"])]
        ov_before = overlap_len(a, b)
        a2, b2, notes = resolve_pair(text, a, b)
        by_key[(pecha_id, a0["ann_id"])].update(a2)
        by_key[(pecha_id, b0["ann_id"])].update(b2)
        a = by_key[(pecha_id, a0["ann_id"])]
        b = by_key[(pecha_id, b0["ann_id"])]
        ov_after = 0 if a["dropped"] or b["dropped"] else overlap_len(a, b)
        logs.append(
            {
                "repo_id": pecha_id,
                "ann_id_a": a0["ann_id"],
                "ann_id_b": b0["ann_id"],
                "overlap_before": ov_before,
                "overlap_after": ov_after,
                "a_start": a["start"],
                "a_end": a["end"],
                "b_start": b["start"],
                "b_end": b["end"],
                "a_dropped": a["dropped"],
                "b_dropped": b["dropped"],
                "actions": " | ".join(notes),
            }
        )
        a["resolve_notes"] = (a.get("resolve_notes") or "") + ";A:" + "|".join(notes)
        b["resolve_notes"] = (b.get("resolve_notes") or "") + ";B:" + "|".join(notes)

    # Remaining overlaps among non-dropped spans.
    remaining: list[tuple[str, str, str]] = []
    by_repo_after: dict[str, list[dict]] = defaultdict(list)
    for rec in by_key.values():
        if rec["dropped"]:
            continue
        by_repo_after[rec["pecha_id"]].append(rec)
    for pecha_id, spans in by_repo_after.items():
        for a, b in find_overlap_pairs(spans):
            remaining.append((pecha_id, a["ann_id"], b["ann_id"]))

    n_dropped = sum(1 for r in by_key.values() if r["dropped"])
    print(f"Resolved {len(pairs)} pairs. Dropped {n_dropped} stub spans.")
    print(f"Overlaps remaining: {len(remaining)}")
    for rec in remaining:
        print(f"  still overlapping: {rec[0]} {rec[1]} × {rec[2]}")
    if remaining and not args.allow_remaining:
        raise SystemExit("Overlaps remain; not writing resolved sidecar.")

    out_rows = []
    for rec in rows:
        cur = by_key[(rec["pecha_id"], rec["ann_id"])]
        out_rows.append(
            {
                **{k: rec[k] for k in rec},
                "start": cur["start"],
                "end": cur["end"],
                "dropped": cur["dropped"],
                "resolve_notes": cur.get("resolve_notes", ""),
            }
        )

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    log_path = args.log.expanduser().resolve()
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)

    print(f"Wrote {out} ({len(out_rows)} rows, {n_dropped} dropped)")
    print(f"Wrote {log_path}")
    print("Cloned YAML was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

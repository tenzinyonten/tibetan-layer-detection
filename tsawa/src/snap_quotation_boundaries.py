#!/usr/bin/env python3
"""Snap Quotation/Citation span edges to syllable boundaries, like Tsawa.

The old batch names the layer ``Quotation.yml`` and the new batch
``Citation.yml``; they are the same annotation and are merged here. One book
(``I8994AAB2``) ships both files with disjoint offsets, so the union is taken.

The boundary bug is not Tsawa-specific. Raw quotation offsets are 64.6% dirty
in the old batch (34.2% dirty start, 53.9% dirty end) against 3.9% in the new
batch, so these spans need the same treatment before they can be labeled.

Snapping is reused verbatim from ``snap_tsawa_boundaries`` so both layers move
by identical rules. Cloned YAML is never rewritten; this writes a sidecar.

After snapping, quote-quote overlaps within a document are resolved by
trimming the later span's start to the earlier span's end (they are the same
class, so a merge-like trim loses nothing); spans left empty are marked
``dropped``. Tsawa precedence is NOT applied here — that happens at label
time in ``build_tsawa_dataset.py``, which counts the suppressed tokens.

Usage:
    python tsawa/src/snap_quotation_boundaries.py \\
        --audit-csv data/shared/tsawa_audit.csv \\
        --raw-opf-dir data/raw_opf \\
        --out tsawa/data/processed/quotation_spans_snapped.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # check_boundary_snapping
from check_boundary_snapping import (  # noqa: E402
    check_span,
    infer_batch,
    load_repos_with_tsawa,
    resolve_opf_root,
)
from snap_tsawa_boundaries import snap_end, snap_start  # noqa: E402

LAYER_FILES = ("Quotation.yml", "Citation.yml")


def load_quote_spans(opf_dir: Path) -> list[dict]:
    """Merged, de-duplicated spans from whichever layer file(s) exist."""
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for name in LAYER_FILES:
        path = opf_dir / "layers" / "v001" / name
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for ann_id, ann in (data.get("annotations") or {}).items():
            sp = ann.get("span") or {}
            s, e = sp.get("start"), sp.get("end")
            if s is None or e is None:
                continue
            s, e = int(s), int(e)
            if s >= e or (s, e) in seen:
                continue
            seen.add((s, e))
            out.append({"ann_id": ann_id, "start": s, "end": e, "layer": name})
    out.sort(key=lambda r: (r["start"], r["end"]))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propose snapped quotation offsets (does not edit cloned YAML)."
    )
    p.add_argument("--audit-csv", type=Path, default=Path("data/shared/tsawa_audit.csv"))
    p.add_argument("--raw-opf-dir", type=Path, default=Path("data/raw_opf"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tsawa/data/processed/quotation_spans_snapped.csv"),
        help="Sidecar of all quotation spans with original + snapped offsets.",
    )
    p.add_argument("--max-walk", type=int, default=12, help="Snap walk cap (default: 12).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_walk < 1:
        raise SystemExit("--max-walk must be >= 1")

    repos = load_repos_with_tsawa(args.audit_csv.expanduser().resolve())
    if not repos:
        raise SystemExit("No tsawa repos found — check the audit CSV.")
    raw_dir = args.raw_opf_dir.expanduser().resolve()

    before = {b: Counter() for b in ("old", "new", "combined")}
    after = {b: Counter() for b in ("old", "new", "combined")}
    rows: list[dict] = []
    n_books = 0
    books_without = 0
    trimmed = 0
    dropped = 0

    for repo in repos:
        pid = repo["pecha_id"]
        batch = infer_batch(pid, repo)
        if batch not in ("old", "new"):
            batch = "new" if pid.startswith("I") else "old"
        opf = resolve_opf_root(raw_dir, pid, repo)
        base = Path(repo["base_path"]) if repo.get("base_path") else opf / "base" / "v001.txt"
        if not base.is_file():
            print(f"warning: missing base for {pid}: {base}", file=sys.stderr)
            continue
        spans = load_quote_spans(opf)
        if not spans:
            books_without += 1
            continue
        n_books += 1
        text = base.read_text(encoding="utf-8")

        snapped: list[dict] = []
        for sp in spans:
            s0, e0 = sp["start"], sp["end"]
            f0 = check_span(text, s0, e0)
            for k, c in (("total", True), ("dirty_start", not f0["start_is_clean"]),
                         ("dirty_end", not f0["end_is_clean"]),
                         ("dirty_either", not (f0["start_is_clean"] and f0["end_is_clean"]))):
                if c:
                    before[batch][k] += 1
                    before["combined"][k] += 1

            new_s, st = snap_start(text, s0, args.max_walk)
            new_e, en = snap_end(text, e0, args.max_walk)
            reasons = []
            if new_s >= new_e:
                new_s, new_e, reasons = s0, e0, ["aborted_inverted"]
            else:
                if st == "unsnapped_start":
                    reasons.append(st)
                if en == "unsnapped_end":
                    reasons.append(en)
            f1 = check_span(text, new_s, new_e)
            for k, c in (("total", True), ("dirty_start", not f1["start_is_clean"]),
                         ("dirty_end", not f1["end_is_clean"]),
                         ("dirty_either", not (f1["start_is_clean"] and f1["end_is_clean"]))):
                if c:
                    after[batch][k] += 1
                    after["combined"][k] += 1
            if new_s != s0:
                after[batch]["start_snapped"] += 1
                after["combined"]["start_snapped"] += 1
            if new_e != e0:
                after[batch]["end_snapped"] += 1
                after["combined"]["end_snapped"] += 1

            snapped.append({
                "pecha_id": pid, "batch": batch, "layer": sp["layer"],
                "ann_id": sp["ann_id"], "start_orig": s0, "end_orig": e0,
                "start": new_s, "end": new_e,
                "start_delta": new_s - s0, "end_delta": new_e - e0,
                "unsnapped_reason": ",".join(reasons),
                "start_is_clean_after": f1["start_is_clean"],
                "end_is_clean_after": f1["end_is_clean"],
                "overlap_action": "", "dropped": False,
            })

        # Same-class overlap: trim the later span's start up to the earlier end.
        snapped.sort(key=lambda r: (r["start"], r["end"]))
        prev_end = -1
        for rec in snapped:
            if rec["start"] < prev_end:
                rec["overlap_action"] = f"start_trimmed_to_{prev_end}"
                rec["start"] = prev_end
                trimmed += 1
                if rec["start"] >= rec["end"]:
                    rec["overlap_action"] = "dropped_contained"
                    rec["dropped"] = True
                    dropped += 1
                    continue
            prev_end = max(prev_end, rec["end"])
        rows.extend(snapped)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def table(title: str, st: dict) -> None:
        print(title)
        print(f"{'':16} {'old':>18} {'new':>18} {'combined':>18}")
        for label, k in (("Total spans", "total"), ("Dirty start", "dirty_start"),
                         ("Dirty end", "dirty_end"), ("Dirty either", "dirty_either")):
            cells = []
            for b in ("old", "new", "combined"):
                n, t = st[b][k], st[b]["total"]
                cells.append(str(n) if k == "total"
                             else f"{n} ({100.0*n/t:.1f}%)" if t else f"{n} (n/a)")
            print(f"{label:16} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")

    active = sum(1 for r in rows if not r["dropped"])
    print(f"Books with a quotation layer: {n_books} (of {len(repos)} tsawa books; "
          f"{books_without} without)")
    print(f"Wrote {len(rows):,} spans ({active:,} active, {dropped} dropped) to {out}")
    print(f"Cloned YAML was not modified. max_walk={args.max_walk}")
    print(f"Starts snapped: {after['combined']['start_snapped']:,}  "
          f"ends snapped: {after['combined']['end_snapped']:,}  "
          f"quote-quote overlaps trimmed: {trimmed}  dropped: {dropped}")
    print()
    table("=== BEFORE snap ===", before)
    print()
    table("=== AFTER snap ===", after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

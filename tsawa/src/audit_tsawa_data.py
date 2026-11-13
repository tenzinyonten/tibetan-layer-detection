#!/usr/bin/env python3
"""Audit Tsawa (root-text) annotation layers in cloned OpenPecha .opf repos.

OpenPecha / Pecha Format assumptions
------------------------------------
Each cloned repo lives at ``data/raw_opf/<PECHA_ID>.opf/``. That path is the
*git* root. The Pecha Format tree is usually nested one level down:

    <PECHA_ID>.opf/<PECHA_ID>.opf/base/v001.txt
    <PECHA_ID>.opf/<PECHA_ID>.opf/layers/v001/Tsawa.yml

Some checkouts may have ``base/`` at the git root; both layouts are accepted.

``base/v001.txt`` is the full book as one Unicode string. Layer YAML files
store character offsets into that string. OpenPecha spans are treated as
**start inclusive, end exclusive** (length = ``end - start``), matching the
OpenPecha Toolkit ``Span`` convention.

``Tsawa.yml`` (annotation_type: Tsawa / རྩ་བ) lists root-text spans, each
optionally carrying ``isverse``. This script does not train anything; it
only measures how much tsawa is present and flags data-quality problems.

This is still a rights-agnostic audit: a repo appearing here is not assumed
cleared for model training.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


AUDIT_COLUMNS = [
    "pecha_id",
    "source_batch",
    "has_tsawa_layer",
    "n_spans",
    "n_isverse_true",
    "n_isverse_false",
    "n_isverse_missing",
    "base_chars",
    "tsawa_chars_sum",
    "tsawa_chars_union",
    "coverage_pct",
    "avg_span_len",
    "median_span_len",
    "n_zero_length",
    "n_inverted",
    "n_out_of_bounds",
    "n_overlap_pairs",
    "n_spans_overlapping",
    "parse_status",
    "notes",
    "opf_root",
    "tsawa_path",
    "base_path",
]


def resolve_opf_root(clone_dir: Path) -> Path | None:
    """Return the directory that contains ``base/`` (and usually ``layers/``).

    Prefers ``<clone>/<pecha_id>.opf/`` when several ``*.opf`` dirs exist.
    """
    if (clone_dir / "base").is_dir():
        return clone_dir
    named = clone_dir / f"{clone_dir.name}"
    if (named / "base").is_dir():
        return named
    candidates = [
        p
        for p in clone_dir.glob("*.opf")
        if p.is_dir() and (p / "base").is_dir()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Stable pick; the audit row will still record the path used.
        return sorted(candidates)[0]
    return None


def load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    """Map pecha_id -> last manifest row (source_batch, dest, …)."""
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = (row.get("pecha_id") or "").strip()
        if pid:
            by_id[pid] = row
    return by_id


def infer_batch(pecha_id: str, manifest_row: dict[str, str] | None) -> str:
    if manifest_row and manifest_row.get("source_batch"):
        return manifest_row["source_batch"]
    if pecha_id.startswith("P"):
        return "old"
    if pecha_id.startswith("I"):
        return "new"
    return "unknown"


def iter_clone_dirs(raw_dir: Path) -> list[Path]:
    """Every ``*.opf`` checkout under raw_dir, skipping the raw dir itself."""
    dirs = [
        p
        for p in raw_dir.iterdir()
        if p.is_dir() and p.name.endswith(".opf") and not p.name.startswith(".")
    ]
    return sorted(dirs)


def parse_tsawa_yaml(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load annotations from Tsawa.yml.

    Returns ``(annotation_dicts, status)``. Status is ``ok`` or an error tag.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"unreadable:{exc}"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return [], f"yaml_error:{exc}"
    if not isinstance(data, dict):
        return [], "yaml_not_mapping"
    annotations = data.get("annotations")
    if annotations is None:
        return [], "ok_empty"
    if not isinstance(annotations, dict):
        return [], "annotations_not_mapping"
    out: list[dict[str, Any]] = []
    for ann_id, payload in annotations.items():
        if not isinstance(payload, dict):
            payload = {"_raw": payload}
        item = dict(payload)
        item["_id"] = ann_id
        out.append(item)
    return out, "ok"


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def union_coverage(spans: list[tuple[int, int]]) -> int:
    """Total characters covered by half-open [start, end) intervals."""
    if not spans:
        return 0
    ordered = sorted(spans)
    total = 0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def count_overlaps(spans: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (n_overlapping_pairs, n_unique_spans_in_at_least_one_pair)."""
    n_pairs = 0
    involved: set[int] = set()
    n = len(spans)
    for i in range(n):
        s1, e1 = spans[i]
        for j in range(i + 1, n):
            s2, e2 = spans[j]
            if s1 < e2 and s2 < e1:
                n_pairs += 1
                involved.add(i)
                involved.add(j)
    return n_pairs, len(involved)


@dataclass
class RepoAudit:
    pecha_id: str
    source_batch: str
    has_tsawa_layer: bool = False
    n_spans: int = 0
    n_isverse_true: int = 0
    n_isverse_false: int = 0
    n_isverse_missing: int = 0
    base_chars: int = 0
    tsawa_chars_sum: int = 0
    tsawa_chars_union: int = 0
    coverage_pct: float = 0.0
    avg_span_len: float | None = None
    median_span_len: float | None = None
    n_zero_length: int = 0
    n_inverted: int = 0
    n_out_of_bounds: int = 0
    n_overlap_pairs: int = 0
    n_spans_overlapping: int = 0
    parse_status: str = "no_opf_root"
    notes: str = ""
    opf_root: str = ""
    tsawa_path: str = ""
    base_path: str = ""
    valid_lengths: list[int] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "pecha_id": self.pecha_id,
            "source_batch": self.source_batch,
            "has_tsawa_layer": self.has_tsawa_layer,
            "n_spans": self.n_spans,
            "n_isverse_true": self.n_isverse_true,
            "n_isverse_false": self.n_isverse_false,
            "n_isverse_missing": self.n_isverse_missing,
            "base_chars": self.base_chars,
            "tsawa_chars_sum": self.tsawa_chars_sum,
            "tsawa_chars_union": self.tsawa_chars_union,
            "coverage_pct": round(self.coverage_pct, 6),
            "avg_span_len": (
                None if self.avg_span_len is None else round(self.avg_span_len, 3)
            ),
            "median_span_len": (
                None
                if self.median_span_len is None
                else round(self.median_span_len, 3)
            ),
            "n_zero_length": self.n_zero_length,
            "n_inverted": self.n_inverted,
            "n_out_of_bounds": self.n_out_of_bounds,
            "n_overlap_pairs": self.n_overlap_pairs,
            "n_spans_overlapping": self.n_spans_overlapping,
            "parse_status": self.parse_status,
            "notes": self.notes,
            "opf_root": self.opf_root,
            "tsawa_path": self.tsawa_path,
            "base_path": self.base_path,
        }


def audit_repo(clone_dir: Path, source_batch: str) -> RepoAudit:
    pecha_id = clone_dir.name.removesuffix(".opf")
    row = RepoAudit(pecha_id=pecha_id, source_batch=source_batch)
    root = resolve_opf_root(clone_dir)
    if root is None:
        row.parse_status = "no_opf_root"
        row.notes = "no directory with base/ found"
        return row
    row.opf_root = str(root)

    # Prefer the documented v001 pair; fall back to any single volume.
    base_path = root / "base" / "v001.txt"
    if not base_path.is_file():
        txts = sorted((root / "base").glob("*.txt")) if (root / "base").is_dir() else []
        if len(txts) == 1:
            base_path = txts[0]
            row.notes = f"base not v001.txt; used {base_path.name}"
        elif not txts:
            row.parse_status = "missing_base"
            row.notes = "no base/*.txt"
            return row
        else:
            row.parse_status = "multiple_base"
            row.notes = "multiple base texts and no v001.txt; skipped"
            return row
    row.base_path = str(base_path)
    try:
        base_text = base_path.read_text(encoding="utf-8")
    except OSError as exc:
        row.parse_status = f"base_unreadable:{exc}"
        return row
    row.base_chars = len(base_text)

    tsawa_path = root / "layers" / "v001" / "Tsawa.yml"
    if not tsawa_path.is_file():
        # Case-insensitive / other volume names.
        found = sorted(root.joinpath("layers").rglob("Tsawa.yml")) if (root / "layers").is_dir() else []
        if found:
            tsawa_path = found[0]
            extra = f"Tsawa.yml not at layers/v001; used {tsawa_path.relative_to(root)}"
            row.notes = f"{row.notes}; {extra}".strip("; ")
        else:
            row.has_tsawa_layer = False
            row.parse_status = "no_tsawa_layer"
            return row

    row.has_tsawa_layer = True
    row.tsawa_path = str(tsawa_path)
    annotations, status = parse_tsawa_yaml(tsawa_path)
    row.parse_status = status
    if status not in ("ok", "ok_empty"):
        return row

    valid_half_open: list[tuple[int, int]] = []
    lengths: list[int] = []
    for ann in annotations:
        span = ann.get("span") or {}
        if not isinstance(span, dict):
            row.n_inverted += 1  # treat as malformed span
            continue
        start = _as_int(span.get("start"))
        end = _as_int(span.get("end"))
        row.n_spans += 1

        isverse = ann.get("isverse")
        if isverse is True:
            row.n_isverse_true += 1
        elif isverse is False:
            row.n_isverse_false += 1
        else:
            row.n_isverse_missing += 1

        if start is None or end is None:
            row.n_inverted += 1
            continue
        if start > end:
            row.n_inverted += 1
            continue
        if start == end:
            row.n_zero_length += 1
            # Still OOB-check the empty point.
            if start < 0 or start > row.base_chars:
                row.n_out_of_bounds += 1
            continue
        if start < 0 or end > row.base_chars:
            row.n_out_of_bounds += 1
            # Clip for coverage so one bad span does not drop the repo.
            clipped_s = max(0, start)
            clipped_e = min(row.base_chars, end)
            if clipped_e > clipped_s:
                valid_half_open.append((clipped_s, clipped_e))
                lengths.append(end - start)  # report the *annotated* length
            continue
        valid_half_open.append((start, end))
        lengths.append(end - start)

    row.valid_lengths = lengths
    row.tsawa_chars_sum = sum(lengths)
    row.tsawa_chars_union = union_coverage(valid_half_open)
    if row.base_chars:
        row.coverage_pct = 100.0 * row.tsawa_chars_union / row.base_chars
    if lengths:
        row.avg_span_len = statistics.mean(lengths)
        row.median_span_len = statistics.median(lengths)
    row.n_overlap_pairs, row.n_spans_overlapping = count_overlaps(valid_half_open)
    return row


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100.0 * num / den:.1f}%"


def summarize_batch(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"=== {name} batch ({len(df)} repos) ==="]
    if df.empty:
        lines.append("  (no repos)")
        return lines

    has = df["has_tsawa_layer"].astype(bool)
    n_has = int(has.sum())
    n_no = int((~has).sum())
    lines.append(f"  Tsawa.yml present : {n_has}  ({_pct(n_has, len(df))})")
    lines.append(f"  Tsawa.yml absent  : {n_no}  ({_pct(n_no, len(df))})")

    with_layer = df[has]
    spans = with_layer["n_spans"].sum() if len(with_layer) else 0
    lines.append(f"  Total tsawa spans : {int(spans)}")

    all_base = int(df["base_chars"].sum())
    all_union = int(df["tsawa_chars_union"].sum())
    all_sum = int(df["tsawa_chars_sum"].sum())
    lines.append(f"  Base characters (all repos)          : {all_base:,}")
    lines.append(f"  Tsawa chars, union (all repos)       : {all_union:,}  ({_pct(all_union, all_base)} of all base)")
    lines.append(f"  Tsawa chars, raw span sum            : {all_sum:,}")

    if len(with_layer):
        layer_base = int(with_layer["base_chars"].sum())
        layer_union = int(with_layer["tsawa_chars_union"].sum())
        nonempty = with_layer[with_layer["n_spans"] > 0]
        lines.append(f"  Base characters (repos with Tsawa)   : {layer_base:,}")
        lines.append(
            f"  Tsawa coverage among those repos     : {layer_union:,}  "
            f"({_pct(layer_union, layer_base)} of their base)"
        )
        lines.append(f"  Repos with Tsawa.yml but 0 spans     : {int((with_layer['n_spans'] == 0).sum())}")
        if len(nonempty):
            # Recompute global avg/median from per-repo averages is wrong;
            # use span-weighted mean of repo averages only as a hint, plus
            # median of per-repo median_span_len.
            lines.append(
                f"  Mean of per-repo avg span length     : "
                f"{nonempty['avg_span_len'].mean():.1f} chars"
            )
            lines.append(
                f"  Median of per-repo median span len   : "
                f"{nonempty['median_span_len'].median():.1f} chars"
            )
            lines.append(
                f"  Mean / median spans per repo (w/ ≥1) : "
                f"{nonempty['n_spans'].mean():.1f} / {nonempty['n_spans'].median():.1f}"
            )
        verse_t = int(with_layer["n_isverse_true"].sum())
        verse_f = int(with_layer["n_isverse_false"].sum())
        verse_m = int(with_layer["n_isverse_missing"].sum())
        lines.append(
            f"  isverse  true / false / missing      : {verse_t} / {verse_f} / {verse_m}"
        )

    flags = {
        "zero-length spans (start==end)": int(df["n_zero_length"].sum()),
        "inverted / unparseable spans": int(df["n_inverted"].sum()),
        "out-of-bounds spans": int(df["n_out_of_bounds"].sum()),
        "overlap pairs": int(df["n_overlap_pairs"].sum()),
        "repos with any overlap": int((df["n_overlap_pairs"] > 0).sum()),
        "repos with any quality flag": int(
            (
                (df["n_zero_length"] > 0)
                | (df["n_inverted"] > 0)
                | (df["n_out_of_bounds"] > 0)
                | (df["n_overlap_pairs"] > 0)
            ).sum()
        ),
    }
    lines.append("  Quality flags:")
    for label, val in flags.items():
        lines.append(f"    {label:32s} {val}")

    status_counts = df["parse_status"].value_counts().to_dict()
    lines.append(f"  parse_status: {status_counts}")
    return lines


def print_comparison(df: pd.DataFrame) -> None:
    batches = ["new", "old"]
    present = [b for b in batches if (df["source_batch"] == b).any()]
    extra = sorted(set(df["source_batch"]) - set(batches))
    order = present + extra

    blocks: list[str] = []
    for b in order:
        blocks.extend(summarize_batch(b, df[df["source_batch"] == b]))
        blocks.append("")
    blocks.extend(summarize_batch("ALL", df))
    blocks.append("")

    # Side-by-side headline comparison
    if "new" in df["source_batch"].values and "old" in df["source_batch"].values:
        blocks.append("=== old vs new (headline) ===")
        header = f"{'metric':<42} {'new':>14} {'old':>14}"
        blocks.append(header)
        blocks.append("-" * len(header))

        def stats(batch: str) -> dict[str, float]:
            sub = df[df["source_batch"] == batch]
            has = sub["has_tsawa_layer"].astype(bool)
            base = int(sub["base_chars"].sum())
            union = int(sub["tsawa_chars_union"].sum())
            return {
                "repos": len(sub),
                "with_tsawa": int(has.sum()),
                "spans": int(sub["n_spans"].sum()),
                "coverage_all_pct": (100.0 * union / base) if base else 0.0,
                "flagged_repos": int(
                    (
                        (sub["n_zero_length"] > 0)
                        | (sub["n_inverted"] > 0)
                        | (sub["n_out_of_bounds"] > 0)
                        | (sub["n_overlap_pairs"] > 0)
                    ).sum()
                ),
            }

        a, b = stats("new"), stats("old")
        rows = [
            ("repos", "repos", ".0f"),
            ("with Tsawa.yml", "with_tsawa", ".0f"),
            ("total spans", "spans", ".0f"),
            ("tsawa ∪ / all base (%)", "coverage_all_pct", ".3f"),
            ("repos with any quality flag", "flagged_repos", ".0f"),
        ]
        for label, key, fmt in rows:
            blocks.append(f"{label:<42} {a[key]:>14{fmt}} {b[key]:>14{fmt}}")

    text = "\n".join(blocks)
    print(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Tsawa layers in cloned OpenPecha .opf repos (no training)."
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_opf"),
        help="Directory of cloned <ID>.opf checkouts (default: data/raw_opf).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw_opf/_manifest.csv"),
        help="Fetch manifest used to tag source_batch (default: data/raw_opf/_manifest.csv).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/shared/tsawa_audit.csv"),
        help="Per-repo audit CSV (default: data/shared/tsawa_audit.csv).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit at most N repos (smoke test).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    manifest = load_manifest(args.manifest.expanduser().resolve())
    clones = iter_clone_dirs(raw_dir)
    if args.limit is not None:
        clones = clones[: args.limit]
    if not clones:
        raise SystemExit(f"no *.opf checkouts under {raw_dir}")

    print(f"Auditing {len(clones)} repos under {raw_dir}")
    rows: list[dict[str, Any]] = []
    for i, clone in enumerate(clones, start=1):
        pecha_id = clone.name.removesuffix(".opf")
        batch = infer_batch(pecha_id, manifest.get(pecha_id))
        if i % 50 == 0 or i == 1 or i == len(clones):
            print(f"  [{i}/{len(clones)}] {pecha_id} ({batch})", file=sys.stderr)
        rows.append(audit_repo(clone, batch).to_row())

    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)\n")
    print_comparison(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

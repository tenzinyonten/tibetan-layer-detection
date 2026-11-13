"""
spot_check_tsawa.py

Pulls a random sample of tsawa spans from the audited .opf repos, prints the
actual span text with context, and scores each span against structural
heuristics for Tibetan root text (rtsa ba / རྩ་བ):

  1. Metrical regularity — lines within the span should have roughly equal
     syllable counts (classic meters 7/9/11/13). A multi-line stanza with
     low syllable stdev counts, and so does a *single* clean line whose
     syllable count is one of those meters (a lone verse line is
     structurally valid; requiring ≥2 lines was wrongly penalizing them).
  2. Trailing quotation marker — the text immediately AFTER the span often
     opens with a citation/gloss particle like ཞེས་, ཞེས་པས, ཞེས་གསུངས་,
     ཞེས་བྱ་བ, ཞེས་པ་ནི — commentary explicitly marking "that was the root
     text, here's what it means."
  3. Valid terminal punctuation — the span should end on a clean shad
     boundary (།, ། །, ༎, ༎ ༎) or gter-shad (༔) for treasure texts, not
     mid-clause.

Verdict v2: only ``is_regular_meter`` and ``ends_cleanly`` count toward
the verdict (``primary_signals_met``, 0–2). The trailing quote-marker
signal is still computed and reported as a *bonus / corroborating*
flag — it is not part of the verdict. v1 treated all three as equal
weight (``signals_met >= 2`` → LIKELY_OK), which had poor recall:
commentary often continues in plain prose without ཞེས་པས་ / ཞེས་
(confirmed in manual review, e.g. P000207 [92705:92820], a clean
3-line verse block that v1 scored AMBIGUOUS only because the
quote-marker was absent).

This is a heuristic triage tool, NOT a ground-truth verifier — it cannot
replace checking against source page images or a fluent reader's judgment.
Use its flags to prioritize which spans deserve a closer manual look, and
to give reviewers concrete, checkable reasons rather than a vague "this
looks off." A LOW score does not prove a span is mislabeled; a HIGH score
does not prove it's correct — both are just triage signals.

Usage:
    python spot_check_tsawa.py \
        --audit-csv data/shared/tsawa_audit.csv \
        --raw-opf-dir data/raw_opf \
        --n 20 \
        --seed 42 \
        --out tsawa/data/processed/spot_check_results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

import yaml

# Tibetan quotation/citation particles that classically follow a root-text
# quotation in commentarial literature (rtsa 'grel style).
TRAILING_QUOTE_MARKERS = [
    "ཞེས་པས་བསྟན་ཏེ",
    "ཞེས་གསུངས་སོ",
    "ཞེས་གསུངས་པ",
    "ཞེས་བྱ་བ",
    "ཞེས་པ་ནི",
    "ཞེས་",
    "ཅེས་པས",
    "ཅེས་",
]

# Valid terminal punctuation for a clean verse/stanza boundary.
# Ordered longest-first so multi-shad endings are matched before single ones.
VALID_ENDINGS = ["༎ ༎", "། །", "༎", "།", "༔"]

# Shad characters used to split a span into "lines" for the meter check.
SHAD_CHARS = ["།", "༔", "༎"]

# Classic Tibetan verse syllable counts (tsheg-based approximation).
CLASSIC_METERS = {7, 9, 11, 13}


def load_repos_with_tsawa(audit_csv: Path) -> list[dict]:
    """Read Phase 2 ``tsawa_audit.csv`` rows that have a non-empty Tsawa layer.

    Schema from ``audit_tsawa_data.py``: ``pecha_id``, ``has_tsawa_layer``,
    ``n_spans`` (not ``tsawa_span_count``), ``opf_root``, ``tsawa_path``,
    ``base_path``.
    """
    repos = []
    with open(audit_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_tsawa = row.get("has_tsawa_layer", "").strip().lower() == "true"
            span_count = int(row.get("n_spans") or 0)
            if has_tsawa and span_count > 0:
                repos.append(row)
    return repos


def resolve_opf_root(raw_opf_dir: Path, repo_id: str, audit_row: dict | None = None) -> Path:
    """Git clones nest the Pecha tree as ``<ID>.opf/<ID>.opf/{base,layers}``."""
    if audit_row and audit_row.get("opf_root"):
        root = Path(audit_row["opf_root"])
        if (root / "base").is_dir():
            return root
    clone = raw_opf_dir / f"{repo_id}.opf"
    nested = clone / f"{repo_id}.opf"
    if (nested / "base").is_dir():
        return nested
    return clone


def load_spans(opf_dir: Path, tsawa_path: Path | None = None) -> list[dict]:
    """Load all tsawa spans for a single .opf repo, dropping zero-length ones."""
    if tsawa_path is None or not tsawa_path.is_file():
        tsawa_path = opf_dir / "layers" / "v001" / "Tsawa.yml"
    if not tsawa_path.exists():
        return []

    with open(tsawa_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    spans = []
    for ann_id, ann in (data.get("annotations") or {}).items():
        span = ann.get("span", {})
        start, end = span.get("start"), span.get("end")
        if start is None or end is None or start == end:
            continue
        spans.append({"ann_id": ann_id, "start": start, "end": end})
    return spans


def split_into_lines(span_text: str) -> list[str]:
    """Split a span into pseudo-lines at shad boundaries, dropping empties."""
    lines = []
    current = ""
    for ch in span_text:
        current += ch
        if ch in SHAD_CHARS:
            stripped = current.strip()
            if stripped:
                lines.append(stripped)
            current = ""
    if current.strip():
        lines.append(current.strip())
    return lines


def syllable_count(line: str) -> int:
    """Rough syllable count: Tibetan syllables are separated by tsheg (་).
    This is an approximation, not a linguistically exact count — good enough
    for a regularity check, not for precise meter classification."""
    cleaned = line
    for ch in SHAD_CHARS + ["༄", "྅", " "]:
        cleaned = cleaned.replace(ch, "")
    if not cleaned:
        return 0
    return cleaned.count("་") + 1


def span_length_bucket(n_chars: int) -> str:
    """Character-length bucket for fragment reporting (not used as a filter)."""
    if n_chars < 15:
        return "very_short"
    if n_chars < 30:
        return "short"
    if n_chars < 100:
        return "typical"
    return "long"


def score_span(span_text: str, after_text: str) -> dict:
    """Score a span against the three structural heuristics.

    Verdict uses only the two primary signals (meter + clean ending).
    The trailing quote marker is returned as a bonus flag and does not
    affect ``verdict``.
    """
    lines = split_into_lines(span_text)
    syllable_counts = [syllable_count(l) for l in lines if syllable_count(l) > 0]

    has_multiple_lines = len(syllable_counts) >= 2
    meter_stdev = (
        statistics.pstdev(syllable_counts) if len(syllable_counts) >= 2 else None
    )
    is_regular_multiline = (
        has_multiple_lines and meter_stdev is not None and meter_stdev <= 1.5
    )
    is_regular_single_line = (
        len(syllable_counts) == 1 and syllable_counts[0] in CLASSIC_METERS
    )
    is_regular_meter = is_regular_multiline or is_regular_single_line

    after_stripped = after_text.strip()
    after_stripped_for_match = after_text.lstrip(" \n\t།༔༎")
    has_trailing_quote_marker = any(
        after_stripped_for_match.startswith(marker) for marker in TRAILING_QUOTE_MARKERS
    )

    span_stripped = span_text.rstrip()
    ends_cleanly = any(span_stripped.endswith(ending) for ending in VALID_ENDINGS)

    primary_signals_met = sum([is_regular_meter, ends_cleanly])

    if primary_signals_met == 2:
        verdict = "LIKELY_OK"
    elif primary_signals_met == 1:
        verdict = "AMBIGUOUS"
    else:
        verdict = "FLAG_FOR_REVIEW"

    return {
        "num_lines": len(lines),
        "syllable_counts": syllable_counts,
        "meter_stdev": round(meter_stdev, 2) if meter_stdev is not None else None,
        "is_regular_meter": is_regular_meter,
        "has_trailing_quote_marker": has_trailing_quote_marker,
        "trailing_text_preview": after_stripped[:20],
        "ends_cleanly": ends_cleanly,
        "primary_signals_met": primary_signals_met,
        "verdict": verdict,
    }


def show_span(
    opf_dir: Path,
    start: int,
    end: int,
    context: int = 100,
    base_path: Path | None = None,
) -> tuple[str, str, str]:
    """Return (before, span_text, after) strings for a span with context."""
    if base_path is None or not base_path.is_file():
        base_path = opf_dir / "base" / "v001.txt"
    with open(base_path, encoding="utf-8") as f:
        text = f.read()

    before = text[max(0, start - context):start]
    span_text = text[start:end]
    after = text[end:end + context]
    return before, span_text, after


def main():
    parser = argparse.ArgumentParser(description="Spot-check random tsawa spans with heuristic scoring.")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--raw-opf-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20, help="number of spans to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                         help="optional path to write results as CSV for later review")
    args = parser.parse_args()

    random.seed(args.seed)

    repos = load_repos_with_tsawa(args.audit_csv)
    if not repos:
        print("No repos with tsawa spans found — check audit CSV column names.")
        return

    pool = []
    for row in repos:
        repo_id = row["pecha_id"]
        opf_dir = resolve_opf_root(args.raw_opf_dir, repo_id, row)
        tsawa_path = Path(row["tsawa_path"]) if row.get("tsawa_path") else None
        spans = load_spans(opf_dir, tsawa_path)
        for span in spans:
            pool.append((row, span))

    if not pool:
        print("No non-zero-length spans found across included repos.")
        return

    sample = random.sample(pool, min(args.n, len(pool)))
    results = []

    for i, (row, span) in enumerate(sample, 1):
        repo_id = row["pecha_id"]
        opf_dir = resolve_opf_root(args.raw_opf_dir, repo_id, row)
        base_path = Path(row["base_path"]) if row.get("base_path") else None
        before, span_text, after = show_span(
            opf_dir, span["start"], span["end"], base_path=base_path
        )
        scored = score_span(span_text, after)

        print(f"\n========== [{i}/{len(sample)}] repo={repo_id} ann={span['ann_id']} "
              f"verdict={scored['verdict']} ==========")
        print(f"--- BEFORE ---\n{before}")
        print(f"\n--- TSAWA SPAN [{span['start']}:{span['end']}] ({span['end'] - span['start']} chars) ---\n{span_text}")
        print(f"\n--- AFTER ---\n{after}")
        print(f"\n[heuristics] lines={scored['num_lines']} "
              f"syllables/line={scored['syllable_counts']} "
              f"meter_stdev={scored['meter_stdev']} "
              f"regular_meter={scored['is_regular_meter']} "
              f"ends_cleanly={scored['ends_cleanly']} "
              f"primary_signals_met={scored['primary_signals_met']}/2 "
              f"trailing_quote_marker_bonus={scored['has_trailing_quote_marker']} "
              f"span_length_bucket={span_length_bucket(span['end'] - span['start'])}")

        results.append({
            "repo_id": repo_id,
            "ann_id": span["ann_id"],
            "start": span["start"],
            "end": span["end"],
            "length": span["end"] - span["start"],
            "span_length_bucket": span_length_bucket(span["end"] - span["start"]),
            "verdict": scored["verdict"],
            "primary_signals_met": scored["primary_signals_met"],
            "num_lines": scored["num_lines"],
            "meter_stdev": scored["meter_stdev"],
            "is_regular_meter": scored["is_regular_meter"],
            "has_trailing_quote_marker_bonus": scored["has_trailing_quote_marker"],
            "trailing_text_preview": scored["trailing_text_preview"],
            "ends_cleanly": scored["ends_cleanly"],
            "span_text": span_text.replace("\n", " "),
        })

    flagged = [r for r in results if r["verdict"] == "FLAG_FOR_REVIEW"]
    ambiguous = [r for r in results if r["verdict"] == "AMBIGUOUS"]
    print(f"\n\n=== SUMMARY: {len(results)} spans checked — "
          f"{len(flagged)} flagged for review, {len(ambiguous)} ambiguous, "
          f"{len(results) - len(flagged) - len(ambiguous)} likely OK ===")

    triage = flagged + ambiguous
    print("\n=== FRAGMENT ANALYSIS (flagged + ambiguous spans only) ===")
    if not triage:
        print("no flagged or ambiguous spans")
    else:
        buckets = ("very_short", "short", "typical", "long")
        labels = {
            "very_short": "very_short (<15 chars)",
            "short": "short (15-29 chars)",
            "typical": "typical (30-99 chars)",
            "long": "long (100+ chars)",
        }
        n = len(triage)
        for bucket in buckets:
            count = sum(1 for r in triage if r["span_length_bucket"] == bucket)
            print(f"{labels[bucket]}: {count} ({100.0 * count / n:.1f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
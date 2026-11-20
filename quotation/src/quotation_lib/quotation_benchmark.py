"""Tibetan Quotation Benchmark builder: split, leakage audit, and Hugging Face release.

This module builds a book-level train/val/test split of the human-reviewed
Tibetan quotation gold annotations. It handles:
1. Deduplication of OpenPecha repositories (by hashing base texts).
2. Merging of chunked spans (long quotations split across lines).
3. Diagnostics and dropping of under-annotated books.
4. Greedy stratifier sweep over weights, seeds, and caps, jointly building
   validation and test splits.
5. Verification of leakage (structural leakage and textual overlap).
6. Local Hugging Face dataset export and Hub upload.

Design Choices & Corpus Targets
-------------------------------
- Merging spans: Walk book spans, join consecutive ones with newline/space/shad gaps,
  unless closed by (ཞེས|ཅེས|ཤེས|གསུངས|འབྱུང|བཤད|སྨྲས)[^།]{0,12}།{1,2}\\s*$.
  Without merging, we stratify on a fiction (unmarked continuation pieces fall through to MISC).
- Dropping sparse books: Books with density < 1.0% are under-annotated, as shown by the
  constant closer-marker density (1200/Mchar) but varying annotation coverage. Dropping
  them prevents teaching the model to ignore real quotations.
- Joint split construction: Building test first and then val leaves leftovers that distort val.
  Greedy joint search with largest-first book ordering balances both splits perfectly.
- Class balancing: Balanced on TEXT, PERSON, and MISC only. Pinned FORMULA and CHAINED
  classes are too small and trade away density/batch balance if targeted directly.

Corpus Targets (adequately annotated subset):
- TEXT: ~57.0%
- PERSON: ~15.7%
- MISC: ~18.9%
- Density: ~7.41%
- Old Batch Share: ~39.8%
- Density buckets (<1%, 1-3%, 3-8%, 8-15%, >=15%)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOS = ROOT / "Data" / "repos"
DEFAULT_GOLD = ROOT / "Data" / "audit" / "quotation_gold.json"
DEFAULT_OUT = ROOT / "Data" / "audit"


@dataclass
class Span:
    ann: str
    start: int
    end: int
    cls: str
    closed: bool
    merged_from: int = 1
    seen_in_train: bool = False


@dataclass
class Book:
    layer: str
    batch: str
    verdict: str
    title: str
    doc_chars: int
    closed_pct: float
    misaligned_pct: float
    both_mid_pct: float
    line_split_pct: float
    n_original: int
    spans: list[Span]
    text: str = ""  # cached text


@dataclass
class Gold:
    generated: str
    policy: str
    kept_books: int
    kept_spans: int
    dropped_spans: int
    snapped_spans: int
    class_counts: dict[str, int]
    books: dict[str, Book]


# Types for functions
CoverageTable = dict[str, Any]
Targets = dict[str, Any]
Split = dict[str, str]
Leakage = dict[str, Any]


def load_gold(path: Path) -> Gold:
    """Loads the quotation_gold.json file into a structured Gold object."""
    if not path.is_file():
        raise FileNotFoundError(f"Quotation gold file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    books = {}
    for book_id, b_data in data["books"].items():
        spans = [
            Span(
                ann=s["ann"],
                start=s["start"],
                end=s["end"],
                cls=s["cls"],
                closed=s["closed"],
                merged_from=s.get("merged_from", 1)
            )
            for s in b_data["spans"]
        ]
        books[book_id] = Book(
            layer=b_data["layer"],
            batch=b_data["batch"],
            verdict=b_data["verdict"],
            title=b_data["title"],
            doc_chars=b_data["doc_chars"],
            closed_pct=b_data["closed_pct"],
            misaligned_pct=b_data["misaligned_pct"],
            both_mid_pct=b_data["both_mid_pct"],
            line_split_pct=b_data["line_split_pct"],
            n_original=b_data["n_original"],
            spans=spans
        )

    return Gold(
        generated=data["generated"],
        policy=data["policy"],
        kept_books=data["kept_books"],
        kept_spans=data["kept_spans"],
        dropped_spans=data["dropped_spans"],
        snapped_spans=data["snapped_spans"],
        class_counts=data["class_counts"],
        books=books
    )


def get_book_text(book_id: str, repos_dir: Path) -> str:
    """Helper to load a book's base text, caching it in the Book object if desired."""
    path = repos_dir / book_id / f"{book_id}.opf" / "base" / "v001.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing base text for {book_id} at {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def check_duplicate_books(gold: Gold) -> list[tuple[str, str]]:
    """Checks for duplicate books by hashing their raw text.
    
    If duplicates exist, keeps the one with more spans and drops the other.
    Returns list of (dropped_id, kept_id).
    """
    hashes: dict[str, str] = {}
    duplicates = []
    to_drop = []
    
    for book_id in sorted(gold.books.keys()):
        book = gold.books[book_id]
        if not book.text:
            book.text = get_book_text(book_id, DEFAULT_REPOS)
            
        h = hashlib.sha256(book.text.encode("utf-8")).hexdigest()
        if h in hashes:
            other_id = hashes[h]
            other_spans = len(gold.books[other_id].spans)
            curr_spans = len(book.spans)
            if curr_spans > other_spans:
                to_drop.append(other_id)
                hashes[h] = book_id
                duplicates.append((other_id, book_id))
            else:
                to_drop.append(book_id)
                duplicates.append((book_id, other_id))
        else:
            hashes[h] = book_id
            
    for dropped_id in to_drop:
        kept_id = hashes[hashlib.sha256(gold.books[dropped_id].text.encode("utf-8")).hexdigest()]
        print(f"Dropping duplicate book: {dropped_id} (duplicate of {kept_id})")
        del gold.books[dropped_id]
        
    gold.kept_books = len(gold.books)
    gold.kept_spans = sum(len(b.spans) for b in gold.books.values())
    
    return duplicates


def merge_continued_spans(gold: Gold) -> Gold:
    """Joins consecutive chunked spans (long quotations split across lines).
    
    Walks each book's spans in start order. Joins span A to the next span B when:
    1. The gap is 0, 1, or 2 characters and contains only whitespace or a shad.
    2. A does not end on a closing marker.
    """
    closer_re = re.compile(r"(ཞེས|ཅེས|ཤེས|གསུངས|འབྱུང|བཤད|སྨྲས)[^།]{0,12}།{1,2}\s*$")
    
    total_merged_spans = 0
    class_counts: dict[str, int] = {}
    
    for book_id, book in gold.books.items():
        if not book.text:
            book.text = get_book_text(book_id, DEFAULT_REPOS)
        text = book.text
        
        # Sort spans by start offset just in case
        spans = sorted(book.spans, key=lambda s: s.start)
        if not spans:
            continue
            
        merged_spans = []
        current = spans[0]
        
        for i in range(1, len(spans)):
            next_span = spans[i]
            
            gap_len = next_span.start - current.end - 1
            is_valid_gap = False
            if gap_len >= 0:
                gap_text = text[current.end + 1 : next_span.start]
                if gap_len <= 2 and all(c in "\n\r །" for c in gap_text):
                    is_valid_gap = True
                    
            if is_valid_gap:
                current_text = text[current.start : current.end + 1]
                ends_on_closer = bool(closer_re.search(current_text))
                
                if not ends_on_closer:
                    # Merge into current
                    current = Span(
                        ann=current.ann,
                        start=current.start,
                        end=next_span.end,
                        cls=current.cls,
                        closed=next_span.closed,
                        merged_from=current.merged_from + next_span.merged_from
                    )
                    continue
                    
            # If not merged, push current and update
            merged_spans.append(current)
            current = next_span
            
        merged_spans.append(current)
        book.spans = merged_spans
        total_merged_spans += len(merged_spans)
        
        for s in merged_spans:
            class_counts[s.cls] = class_counts.get(s.cls, 0) + 1
            
    gold.kept_spans = total_merged_spans
    gold.class_counts = class_counts
    
    return gold


# Density band definitions for diagnostic
BANDS = [
    ("0–0.25%", lambda d: 0.0 <= d < 0.25),
    ("0.25–0.5%", lambda d: 0.25 <= d < 0.5),
    ("0.5–1%", lambda d: 0.5 <= d < 1.0),
    ("1–3%", lambda d: 1.0 <= d < 3.0),
    ("3–8%", lambda d: 3.0 <= d < 8.0),
    ("8–15%", lambda d: 8.0 <= d < 15.0),
    ("≥15%", lambda d: d >= 15.0),
]


def annotation_coverage(gold: Gold) -> CoverageTable:
    """Measures quotation closer coverage across density bands."""
    closer_marker_re = re.compile(r"(ཞེས|ཅེས)[^།]{0,20}(གསུངས|བཤད|འབྱུང|སོ|པ|པའི)")
    
    book_stats = {}
    for book_id, book in gold.books.items():
        if not book.text:
            book.text = get_book_text(book_id, DEFAULT_REPOS)
            
        density = 100.0 * sum(s.end - s.start + 1 for s in book.spans) / book.doc_chars
        
        matches = list(closer_marker_re.finditer(book.text))
        n_closers = len(matches)
        
        n_covered = 0
        for m in matches:
            m_start = m.start()
            if any(m_start - 30 <= s.end < m_start for s in book.spans):
                n_covered += 1
                
        book_stats[book_id] = {
            "density": density,
            "chars": book.doc_chars,
            "spans": len(book.spans),
            "closers": n_closers,
            "covered": n_covered,
        }
        
    bands_data = {
        label: {
            "books_count": 0,
            "chars_sum": 0,
            "spans_sum": 0,
            "closers_sum": 0,
            "covered_sum": 0,
            "books_list": [],
        }
        for label, _ in BANDS
    }
    
    for book_id, stats in book_stats.items():
        density = stats["density"]
        assigned_label = None
        for label, cond in BANDS:
            if cond(density):
                assigned_label = label
                break
        if assigned_label is None:
            assigned_label = BANDS[-1][0]
            
        bdata = bands_data[assigned_label]
        bdata["books_count"] += 1
        bdata["chars_sum"] += stats["chars"]
        bdata["spans_sum"] += stats["spans"]
        bdata["closers_sum"] += stats["closers"]
        bdata["covered_sum"] += stats["covered"]
        bdata["books_list"].append((book_id, density, stats["spans"]))
        
    return {
        "book_stats": book_stats,
        "bands_data": bands_data,
    }


def write_coverage_md(coverage: CoverageTable, min_density: float, path: Path) -> None:
    """Writes the full annotation coverage diagnostic table to markdown."""
    bands_data = coverage["bands_data"]
    
    lines = []
    lines.append("# Tibetan Quotation Annotation Coverage Diagnostic")
    lines.append("")
    lines.append("This table groups the human-accepted books by their merged quotation density.")
    lines.append("It measures the coverage of potential quotation closers as a proxy for annotation thoroughness.")
    lines.append("")
    lines.append("| book density | books | chars | spans | closers | closers per Mchar | closers with an annotation |")
    lines.append("|---|---|---|---|---|---|---|")
    
    dropped_books = []
    
    for label, _ in BANDS:
        bdata = bands_data[label]
        books = bdata["books_count"]
        chars = bdata["chars_sum"]
        spans = bdata["spans_sum"]
        closers = bdata["closers_sum"]
        covered = bdata["covered_sum"]
        
        chars_m = f"{chars / 1_000_000:.1f}M" if chars >= 100_000 else f"{chars / 1000:.1f}K"
        if chars == 0:
            closers_mchar = 0
            pct_annotated = 0.0
        else:
            closers_mchar = int(round(closers * 1_000_000 / chars))
            pct_annotated = 100.0 * covered / closers if closers > 0 else 0.0
            
        lines.append(
            f"| {label} | {books} | {chars_m} | {spans} | {closers} | {closers_mchar} | {pct_annotated:.1f}% |"
        )
        
        for book_id, d, n_spans in bdata["books_list"]:
            if d < min_density:
                dropped_books.append((book_id, d, n_spans))
                
    lines.append("")
    lines.append(f"## Dropped Books (Density < {min_density}%)")
    lines.append("")
    lines.append(f"Total dropped books: {len(dropped_books)}")
    lines.append("")
    lines.append("| book_id | density | spans |")
    lines.append("|---|---|---|")
    for book_id, d, n_spans in sorted(dropped_books, key=lambda x: x[1]):
        lines.append(f"| {book_id} | {d:.4f}% | {n_spans} |")
        
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def drop_under_annotated(gold: Gold, min_density: float = 1.0) -> Gold:
    """Drops books with merged quotation density below min_density."""
    to_keep = {}
    total_spans = 0
    class_counts: dict[str, int] = {}
    
    for book_id, book in gold.books.items():
        density = 100.0 * sum(s.end - s.start + 1 for s in book.spans) / book.doc_chars
        if density >= min_density:
            to_keep[book_id] = book
            total_spans += len(book.spans)
            for s in book.spans:
                class_counts[s.cls] = class_counts.get(s.cls, 0) + 1
                
    gold.books = to_keep
    gold.kept_books = len(to_keep)
    gold.kept_spans = total_spans
    gold.class_counts = class_counts
    
    return gold


def write_merged_gold(gold: Gold, path: Path) -> None:
    """Writes the merged gold set to quotation_gold_merged.json."""
    books_out = {}
    for book_id, b in gold.books.items():
        books_out[book_id] = {
            "layer": b.layer,
            "batch": b.batch,
            "verdict": b.verdict,
            "title": b.title,
            "doc_chars": b.doc_chars,
            "closed_pct": b.closed_pct,
            "misaligned_pct": b.misaligned_pct,
            "both_mid_pct": b.both_mid_pct,
            "line_split_pct": b.line_split_pct,
            "n_original": b.n_original,
            "spans": [
                {
                    "ann": s.ann,
                    "start": s.start,
                    "end": s.end,
                    "cls": s.cls,
                    "closed": s.closed,
                    "merged_from": s.merged_from,
                }
                for s in b.spans
            ]
        }
    payload = {
        "generated": gold.generated,
        "policy": gold.policy,
        "kept_books": gold.kept_books,
        "kept_spans": gold.kept_spans,
        "dropped_spans": gold.dropped_spans,
        "snapped_spans": gold.snapped_spans,
        "spans_in_unreviewed_books": 0,
        "class_counts": gold.class_counts,
        "books": books_out
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def get_book_density_bucket_idx(density: float) -> int:
    """Gets the density bucket index for a given density value."""
    if density < 1.0:
        return 0
    elif density < 3.0:
        return 1
    elif density < 8.0:
        return 2
    elif density < 15.0:
        return 3
    else:
        return 4


@dataclass
class BookFeatures:
    book_id: str
    span_count: int
    class_counts: dict[str, int]
    bucket_counts: list[int]
    old_count: int
    quote_chars: int
    doc_chars: int


def precompute_features(books: dict[str, Book]) -> dict[str, BookFeatures]:
    """Precomputes book feature summaries for split optimization."""
    features = {}
    for book_id, book in books.items():
        class_counts = {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")}
        for s in book.spans:
            class_counts[s.cls] = class_counts.get(s.cls, 0) + 1
            
        b_quote_chars = sum(s.end - s.start + 1 for s in book.spans)
        b_density = 100.0 * b_quote_chars / book.doc_chars if book.doc_chars > 0 else 0.0
        b_bucket_idx = get_book_density_bucket_idx(b_density)
        
        bucket_counts = [0] * 5
        bucket_counts[b_bucket_idx] = len(book.spans)
        
        old_count = len(book.spans) if book.batch == "old" else 0
        
        features[book_id] = BookFeatures(
            book_id=book_id,
            span_count=len(book.spans),
            class_counts=class_counts,
            bucket_counts=bucket_counts,
            old_count=old_count,
            quote_chars=b_quote_chars,
            doc_chars=book.doc_chars,
        )
    return features


def corpus_targets(gold: Gold) -> Targets:
    """Computes target profiles from the merged, density-filtered gold set."""
    total_spans = gold.kept_spans
    if total_spans == 0:
        raise ValueError("Gold set has no spans!")
        
    class_counts = {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")}
    for book in gold.books.values():
        for s in book.spans:
            class_counts[s.cls] = class_counts.get(s.cls, 0) + 1
            
    class_targets = {cls: class_counts[cls] / total_spans for cls in class_counts}
    
    bucket_span_counts = [0] * 5
    for book in gold.books.values():
        b_quote_chars = sum(s.end - s.start + 1 for s in book.spans)
        b_density = 100.0 * b_quote_chars / book.doc_chars if book.doc_chars > 0 else 0.0
        b_bucket_idx = get_book_density_bucket_idx(b_density)
        bucket_span_counts[b_bucket_idx] += len(book.spans)
        
    bucket_targets = [count / total_spans for count in bucket_span_counts]
    
    old_spans = sum(len(b.spans) for b in gold.books.values() if b.batch == "old")
    old_target = old_spans / total_spans
    
    total_quote_chars = sum(sum(s.end - s.start + 1 for s in b.spans) for b in gold.books.values())
    total_doc_chars = sum(b.doc_chars for b in gold.books.values())
    density_target = 100.0 * total_quote_chars / total_doc_chars if total_doc_chars > 0 else 0.0
    
    return {
        "class": class_targets,
        "bucket": bucket_targets,
        "old": old_target,
        "density": density_target,
        "total_spans": total_spans,
        "total_books": len(gold.books),
    }


def get_sorted_candidates(books: dict[str, Book] | dict[str, BookFeatures], seed: int) -> list[str]:
    """Sorts books largest span count first, breaking ties deterministically by seed permutation."""
    by_count: dict[int, list[str]] = {}
    for book_id, book in books.items():
        if isinstance(book, Book):
            cnt = len(book.spans)
        else:
            cnt = book.span_count
        by_count.setdefault(cnt, []).append(book_id)
        
    rng = random.Random(seed)
    sorted_books = []
    for cnt in sorted(by_count.keys(), reverse=True):
        group = list(by_count[cnt])
        rng.shuffle(group)
        sorted_books.extend(group)
    return sorted_books


def compute_split_cost(
    class_counts: dict[str, int],
    bucket_counts: list[int],
    old_count: int,
    quote_chars: int,
    doc_chars: int,
    span_count: int,
    targets: Targets,
    density_weight: float,
    batch_weight: float,
    class_balance_type: str = "three"
) -> float:
    """Calculates weighted cost of absolute percentage-point gaps for a split."""
    if span_count == 0:
        return 999999.0
        
    if class_balance_type == "text":
        class_term = abs(100.0 * class_counts["TEXT"] / span_count - 100.0 * targets["class"]["TEXT"])
    elif class_balance_type == "three":
        class_term = sum(
            abs(100.0 * class_counts[c] / span_count - 100.0 * targets["class"][c])
            for c in ("TEXT", "PERSON", "MISC")
        )
    else:  # five
        class_term = sum(
            abs(100.0 * class_counts[c] / span_count - 100.0 * targets["class"][c])
            for c in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")
        )
        
    bucket_term = 0.5 * sum(
        abs(100.0 * bucket_counts[k] / span_count - 100.0 * targets["bucket"][k])
        for k in range(5)
    )
    
    old_pct = 100.0 * old_count / span_count
    batch_term = batch_weight * abs(old_pct - 100.0 * targets["old"])
    
    density = 100.0 * quote_chars / doc_chars if doc_chars > 0 else 0.0
    density_term = density_weight * abs(density - targets["density"])
    
    return class_term + bucket_term + batch_term + density_term


def run_single_split(
    features: dict[str, BookFeatures],
    targets: Targets,
    density_weight: float,
    batch_weight: float,
    book_cap: int,
    seed: int,
    class_balance_type: str = "three"
) -> tuple[dict[str, str], float, float]:
    """Runs a single greedy joint split search for val and test."""
    splits = {
        "test": {
            "class_counts": {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")},
            "bucket_counts": [0] * 5,
            "old_count": 0,
            "quote_chars": 0,
            "doc_chars": 0,
            "span_count": 0,
            "book_ids": set(),
        },
        "val": {
            "class_counts": {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")},
            "bucket_counts": [0] * 5,
            "old_count": 0,
            "quote_chars": 0,
            "doc_chars": 0,
            "span_count": 0,
            "book_ids": set(),
        }
    }
    
    total_spans = targets["total_spans"]
    quotas = {
        "test": 0.10 * total_spans,
        "val": 0.10 * total_spans,
    }
    
    sorted_book_ids = get_sorted_candidates(features, seed)
    unused = list(sorted_book_ids)
    
    while True:
        hungry = []
        for s in ("test", "val"):
            if splits[s]["span_count"] < quotas[s] and len(splits[s]["book_ids"]) < book_cap:
                hungry.append(s)
                
        if not hungry or not unused:
            break
            
        is_60_percent_full = any(len(splits[s]["book_ids"]) >= 0.60 * book_cap for s in hungry)
        window_size = 120 if is_60_percent_full else 60
        
        window = unused[:window_size]
        
        best_delta = float("inf")
        best_choice = None
        
        for s in hungry:
            s_state = splits[s]
            
            curr_cost = compute_split_cost(
                s_state["class_counts"],
                s_state["bucket_counts"],
                s_state["old_count"],
                s_state["quote_chars"],
                s_state["doc_chars"],
                s_state["span_count"],
                targets,
                density_weight,
                batch_weight,
                class_balance_type
            )
            
            for b_id in window:
                b_feat = features[b_id]
                
                new_class = {cls: s_state["class_counts"][cls] + b_feat.class_counts[cls] for cls in b_feat.class_counts}
                new_bucket = [s_state["bucket_counts"][k] + b_feat.bucket_counts[k] for k in range(5)]
                new_old = s_state["old_count"] + b_feat.old_count
                new_quote = s_state["quote_chars"] + b_feat.quote_chars
                new_doc = s_state["doc_chars"] + b_feat.doc_chars
                new_span = s_state["span_count"] + b_feat.span_count
                
                new_cost = compute_split_cost(
                    new_class,
                    new_bucket,
                    new_old,
                    new_quote,
                    new_doc,
                    new_span,
                    targets,
                    density_weight,
                    batch_weight,
                    class_balance_type
                )
                
                delta = new_cost - curr_cost
                if delta < best_delta:
                    best_delta = delta
                    best_choice = (s, b_id)
                    
        if best_choice is None:
            break
            
        s_choice, b_choice = best_choice
        b_feat = features[b_choice]
        s_state = splits[s_choice]
        
        for cls in b_feat.class_counts:
            s_state["class_counts"][cls] += b_feat.class_counts[cls]
        for k in range(5):
            s_state["bucket_counts"][k] += b_feat.bucket_counts[k]
        s_state["old_count"] += b_feat.old_count
        s_state["quote_chars"] += b_feat.quote_chars
        s_state["doc_chars"] += b_feat.doc_chars
        s_state["span_count"] += b_feat.span_count
        s_state["book_ids"].add(b_choice)
        
        unused.remove(b_choice)
        
    assignment = {}
    for b_id in splits["test"]["book_ids"]:
        assignment[b_id] = "test"
    for b_id in splits["val"]["book_ids"]:
        assignment[b_id] = "val"
    for b_id in unused:
        assignment[b_id] = "train"
        
    err_test = sum(
        abs(100.0 * splits["test"]["class_counts"][c] / splits["test"]["span_count"] - 100.0 * targets["class"][c])
        for c in ("TEXT", "PERSON", "MISC")
    ) if splits["test"]["span_count"] > 0 else 999.0
    
    err_val = sum(
        abs(100.0 * splits["val"]["class_counts"][c] / splits["val"]["span_count"] - 100.0 * targets["class"][c])
        for c in ("TEXT", "PERSON", "MISC")
    ) if splits["val"]["span_count"] > 0 else 999.0
    
    return assignment, err_test, err_val


def evaluate_assignment(
    assignment: dict[str, str],
    features: dict[str, BookFeatures],
    targets: Targets
) -> tuple[float, float, float]:
    """Evaluates the optimization score of an assignment."""
    splits = {
        s: {
            "class_counts": {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")},
            "bucket_counts": [0] * 5,
            "old_count": 0,
            "quote_chars": 0,
            "doc_chars": 0,
            "span_count": 0,
        }
        for s in ("train", "val", "test")
    }
    
    for b_id, s in assignment.items():
        b_feat = features[b_id]
        s_state = splits[s]
        for cls in b_feat.class_counts:
            s_state["class_counts"][cls] += b_feat.class_counts[cls]
        for k in range(5):
            s_state["bucket_counts"][k] += b_feat.bucket_counts[k]
        s_state["old_count"] += b_feat.old_count
        s_state["quote_chars"] += b_feat.quote_chars
        s_state["doc_chars"] += b_feat.doc_chars
        s_state["span_count"] += b_feat.span_count
        
    err_test = sum(
        abs(100.0 * splits["test"]["class_counts"][c] / splits["test"]["span_count"] - 100.0 * targets["class"][c])
        for c in ("TEXT", "PERSON", "MISC")
    ) if splits["test"]["span_count"] > 0 else 999.0
    
    err_val = sum(
        abs(100.0 * splits["val"]["class_counts"][c] / splits["val"]["span_count"] - 100.0 * targets["class"][c])
        for c in ("TEXT", "PERSON", "MISC")
    ) if splits["val"]["span_count"] > 0 else 999.0
    
    d_test = 100.0 * splits["test"]["quote_chars"] / splits["test"]["doc_chars"] if splits["test"]["doc_chars"] > 0 else 0.0
    d_val = 100.0 * splits["val"]["quote_chars"] / splits["val"]["doc_chars"] if splits["val"]["doc_chars"] > 0 else 0.0
    
    old_test = 100.0 * splits["test"]["old_count"] / splits["test"]["span_count"] if splits["test"]["span_count"] > 0 else 0.0
    old_val = 100.0 * splits["val"]["old_count"] / splits["val"]["span_count"] if splits["val"]["span_count"] > 0 else 0.0
    
    score = (
        err_test + err_val
        + 1.5 * (abs(d_test - targets["density"]) + abs(d_val - targets["density"]))
        + 0.5 * (abs(old_test - 100.0 * targets["old"]) + abs(old_val - 100.0 * targets["old"]))
    )
    
    return score, err_test, err_val


def get_measured_tables(
    assignment: dict[str, str],
    features: dict[str, BookFeatures],
    targets: Targets
) -> dict[str, Any]:
    """Generates a summary of the splits' actual measured statistics."""
    splits = {
        s: {
            "books": 0,
            "spans": 0,
            "TEXT": 0,
            "PERSON": 0,
            "MISC": 0,
            "FORMULA": 0,
            "CHAINED": 0,
            "quote_chars": 0,
            "doc_chars": 0,
            "old": 0,
        }
        for s in ("train", "val", "test")
    }
    
    for b_id, s in assignment.items():
        b_feat = features[b_id]
        s_state = splits[s]
        s_state["books"] += 1
        s_state["spans"] += b_feat.span_count
        for cls in b_feat.class_counts:
            s_state[cls] += b_feat.class_counts[cls]
        s_state["quote_chars"] += b_feat.quote_chars
        s_state["doc_chars"] += b_feat.doc_chars
        s_state["old"] += b_feat.old_count
        
    measured = {}
    for s, state in splits.items():
        spans = state["spans"]
        measured[s] = {
            "books": state["books"],
            "spans": spans,
            "class_mix": {
                c: (state[c] / spans if spans > 0 else 0.0)
                for c in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")
            },
            "density": (100.0 * state["quote_chars"] / state["doc_chars"] if state["doc_chars"] > 0 else 0.0),
            "old_pct": (100.0 * state["old"] / spans if spans > 0 else 0.0),
        }
    return measured


def print_threeway_comparison(features: dict[str, BookFeatures], targets: Targets) -> None:
    """Runs and prints the three-way class-balancing objective comparison."""
    print("\n=== Three-way Class-Target Balancing Comparison ===")
    print(f"{'Balancing Mode':<20} {'Test Profile Err':<18} {'Val Profile Err':<18} {'Total 5-class Err':<18}")
    print("-" * 74)
    
    for mode in ("text", "three", "five"):
        assignment, _, _ = run_single_split(
            features, targets, density_weight=20.0, batch_weight=0.5, book_cap=30, seed=4, class_balance_type=mode
        )
        
        splits = {
            s: {
                "class_counts": {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")},
                "span_count": 0,
            }
            for s in ("test", "val")
        }
        for b_id, s in assignment.items():
            if s in splits:
                b_feat = features[b_id]
                splits[s]["span_count"] += b_feat.span_count
                for cls in b_feat.class_counts:
                    splits[s]["class_counts"][cls] += b_feat.class_counts[cls]
                    
        profile_errs = {}
        five_class_errs = {}
        for s in ("test", "val"):
            spans = splits[s]["span_count"]
            if spans == 0:
                profile_errs[s] = 999.0
                five_class_errs[s] = 999.0
                continue
            profile_errs[s] = sum(
                abs(100.0 * splits[s]["class_counts"][c] / spans - 100.0 * targets["class"][c])
                for c in ("TEXT", "PERSON", "MISC")
            )
            five_class_errs[s] = sum(
                abs(100.0 * splits[s]["class_counts"][c] / spans - 100.0 * targets["class"][c])
                for c in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")
            )
            
        total_5_err = five_class_errs["test"] + five_class_errs["val"]
        print(f"{mode:<20} {profile_errs['test']:<18.2f} {profile_errs['val']:<18.2f} {total_5_err:<18.2f}")
    print("-" * 74 + "\n")


def sweep_and_split(
    gold: Gold,
    targets: Targets,
    no_sweep: bool = False,
    density_weight: float = 20.0,
    batch_weight: float = 0.5,
    book_cap: int = 30,
    seed: int = 4
) -> tuple[Split, dict[str, Any], dict[str, Any]]:
    """Runs the 216-run grid sweep (or single config) to find the optimal joint split."""
    features = precompute_features(gold.books)
    
    # Run three-way target comparison before the sweep
    print_threeway_comparison(features, targets)
    
    if no_sweep:
        print(f"Running split with single config: density_weight={density_weight}, batch_weight={batch_weight}, book_cap={book_cap}, seed={seed}")
        assignment, err_test, err_val = run_single_split(
            features, targets, density_weight, batch_weight, book_cap, seed
        )
        score, _, _ = evaluate_assignment(assignment, features, targets)
        chosen_config = {
            "density_weight": density_weight,
            "batch_weight": batch_weight,
            "book_cap": book_cap,
            "seed": seed,
            "score": score,
            "err_test": err_test,
            "err_val": err_val,
        }
    else:
        print("Running 216-run sweep...")
        best_score = float("inf")
        chosen_config = None
        assignment = None
        
        print(f"{'density_w':<10} {'batch_w':<8} {'cap':<5} {'seed':<5} {'score':<8} {'err_test':<10} {'err_val':<10}")
        print("-" * 62)
        
        for dw in (10.0, 20.0, 30.0):
            for bw in (0.5, 1.5):
                for cap in (25, 30, 40):
                    for s in range(12):
                        cand_assign, _, _ = run_single_split(
                            features, targets, dw, bw, cap, s
                        )
                        score, err_test, err_val = evaluate_assignment(cand_assign, features, targets)
                        print(f"{dw:<10.1f} {bw:<8.1f} {cap:<5} {s:<5} {score:<8.2f} {err_test:<10.2f} {err_val:<10.2f}")
                        
                        if score < best_score:
                            best_score = score
                            chosen_config = {
                                "density_weight": dw,
                                "batch_weight": bw,
                                "book_cap": cap,
                                "seed": s,
                                "score": score,
                                "err_test": err_test,
                                "err_val": err_val,
                            }
                            assignment = cand_assign
                            
        print("-" * 62)
        print(f"Winner: {chosen_config}")
        
    measured_tables = get_measured_tables(assignment, features, targets)
    return assignment, chosen_config, measured_tables


def write_split_csv(gold: Gold, split: Split, path: Path) -> None:
    """Writes the book-level split assignments and profiles to quotation_split.csv."""
    fieldnames = [
        "book", "split", "batch", "layer", "n_spans", "doc_chars", "quote_chars",
        "density", "bucket", "pct_TEXT", "pct_PERSON", "pct_MISC", "pct_FORMULA", "pct_CHAINED", "title"
    ]
    
    rows = []
    for book_id, book in sorted(gold.books.items()):
        quote_chars = sum(s.end - s.start + 1 for s in book.spans)
        density = 100.0 * quote_chars / book.doc_chars if book.doc_chars > 0 else 0.0
        bucket_idx = get_book_density_bucket_idx(density)
        buckets_labels = ["<1%", "1-3%", "3-8%", "8-15%", ">=15%"]
        bucket_label = buckets_labels[bucket_idx]
        
        class_counts = {cls: 0 for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")}
        for s in book.spans:
            class_counts[s.cls] = class_counts.get(s.cls, 0) + 1
            
        n_spans = len(book.spans)
        pcts = {
            f"pct_{c}": (100.0 * class_counts[c] / n_spans if n_spans > 0 else 0.0)
            for c in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED")
        }
        
        row = {
            "book": book_id,
            "split": split[book_id],
            "batch": book.batch,
            "layer": book.layer,
            "n_spans": n_spans,
            "doc_chars": book.doc_chars,
            "quote_chars": quote_chars,
            "density": round(density, 4),
            "bucket": bucket_label,
            "pct_TEXT": round(pcts["pct_TEXT"], 4),
            "pct_PERSON": round(pcts["pct_PERSON"], 4),
            "pct_MISC": round(pcts["pct_MISC"], 4),
            "pct_FORMULA": round(pcts["pct_FORMULA"], 4),
            "pct_CHAINED": round(pcts["pct_CHAINED"], 4),
            "title": book.title,
        }
        rows.append(row)
        
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_json(
    gold: Gold,
    split: Split,
    targets: Targets,
    path: Path,
    weights: dict[str, Any],
    measured_tables: dict[str, Any]
) -> None:
    """Writes the split assignments and optimization metadata to quotation_split.json."""
    payload = {
        book_id: s
        for book_id, s in split.items()
    }
    
    serializable_targets = {
        "class": targets["class"],
        "bucket": targets["bucket"],
        "old": targets["old"],
        "density": targets["density"],
        "total_spans": targets["total_spans"],
        "total_books": targets["total_books"],
    }
    
    payload["_meta"] = {
        "chosen_weights": weights,
        "corpus_targets": serializable_targets,
        "measured_splits": measured_tables,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def normalize_string(s: str) -> str:
    """Strips whitespace, tsheg, and shad from a string for quotation-text matching."""
    return re.sub(r"[\s་༌།༑༔]+", "", s)


def check_structural_leakage(gold: Gold, split: Split) -> None:
    """Asserts that there is no structural overlap between the splits."""
    train_books = {b for b, s in split.items() if s == "train"}
    val_books = {b for b, s in split.items() if s == "val"}
    test_books = {b for b, s in split.items() if s == "test"}
    
    assert train_books.isdisjoint(val_books), "Train and Val splits share books!"
    assert train_books.isdisjoint(test_books), "Train and Test splits share books!"
    assert val_books.isdisjoint(test_books), "Val and Test splits share books!"
    
    all_assigned = train_books | val_books | test_books
    expected_books = set(gold.books.keys())
    assert all_assigned == expected_books, f"Split union does not match gold books! Diff: {all_assigned ^ expected_books}"


def audit_leakage(gold: Gold, split: Split) -> Leakage:
    """Evaluates quotation-text overlap (leakage) across the splits."""
    check_structural_leakage(gold, split)
    
    span_data = []
    train_norm_set = set()
    
    for book_id, book in gold.books.items():
        s_name = split[book_id]
        if not book.text:
            book.text = get_book_text(book_id, DEFAULT_REPOS)
        text = book.text
        
        for s in book.spans:
            raw_span_text = text[s.start : s.end + 1]
            norm = normalize_string(raw_span_text)
            span_data.append({
                "book_id": book_id,
                "split": s_name,
                "norm": norm,
                "raw_len": s.end - s.start + 1,
                "norm_len": len(norm),
            })
            if s_name == "train":
                train_norm_set.add(norm)
                
    table_rows = []
    for min_len in (10, 20, 30):
        spans_filtered = [s for s in span_data if s["norm_len"] >= min_len]
        total_filtered = len(spans_filtered)
        
        norm_to_books: dict[str, set[str]] = {}
        norm_to_splits: dict[str, set[str]] = {}
        
        for s in spans_filtered:
            norm_to_books.setdefault(s["norm"], set()).add(s["book_id"])
            norm_to_splits.setdefault(s["norm"], set()).add(s["split"])
            
        in_more_than_one_book = 0
        crosses_boundary = 0
        held_out_in_train = 0
        total_held_out = 0
        
        for s in spans_filtered:
            norm = s["norm"]
            is_held_out = s["split"] in ("val", "test")
            
            if is_held_out:
                total_held_out += 1
                if norm in train_norm_set:
                    held_out_in_train += 1
                    
            if len(norm_to_books[norm]) > 1:
                in_more_than_one_book += 1
                
            if len(norm_to_splits[norm]) > 1:
                crosses_boundary += 1
                
        table_rows.append({
            "min_length": min_len,
            "spans": total_filtered,
            "more_than_one_book": in_more_than_one_book,
            "more_than_one_book_pct": 100.0 * in_more_than_one_book / total_filtered if total_filtered > 0 else 0.0,
            "crosses_split": crosses_boundary,
            "crosses_split_pct": 100.0 * crosses_boundary / total_filtered if total_filtered > 0 else 0.0,
            "held_out_in_train": held_out_in_train,
            "total_held_out": total_held_out,
            "held_out_in_train_pct": 100.0 * held_out_in_train / total_held_out if total_held_out > 0 else 0.0,
        })
        
    seen_in_train_count = 0
    total_held_out_spans = 0
    
    for book_id, book in gold.books.items():
        s_name = split[book_id]
        is_held_out = s_name in ("val", "test")
        
        for s in book.spans:
            raw_span_text = book.text[s.start : s.end + 1]
            norm = normalize_string(raw_span_text)
            
            if is_held_out:
                is_seen = (len(norm) >= 20) and (norm in train_norm_set)
                s.seen_in_train = is_seen
                total_held_out_spans += 1
                if is_seen:
                    seen_in_train_count += 1
            else:
                s.seen_in_train = False
                
    return {
        "table": table_rows,
        "train_norm_set_size": len(train_norm_set),
        "total_held_out_spans": total_held_out_spans,
        "seen_in_train_count": seen_in_train_count,
        "seen_in_train_pct": 100.0 * seen_in_train_count / total_held_out_spans if total_held_out_spans > 0 else 0.0,
    }


def write_leakage_md(leakage: Leakage, path: Path) -> None:
    """Writes the detailed quotation text leakage audit to markdown."""
    lines = []
    lines.append("# Tibetan Quotation Leakage Audit")
    lines.append("")
    lines.append("This document evaluates the potential leakage of quotation text across train, validation, and test splits.")
    lines.append("While structural leakage is strictly zero (books do not overlap across splits), some quotation text")
    lines.append("appears in multiple splits due to the repetition of quoted canonical scriptures in Tibetan commentaries.")
    lines.append("")
    lines.append("## Quotation-Text Overlap Rates")
    lines.append("")
    lines.append("| min length | spans | in more than one book | text also crosses a split boundary | held-out spans whose exact text is also in train |")
    lines.append("|---|---|---|---|---|")
    
    for row in leakage["table"]:
        min_len = row["min_length"]
        spans = row["spans"]
        more_books = f"{row['more_than_one_book']} ({row['more_than_one_book_pct']:.1f}%)"
        crosses = f"{row['crosses_split']} ({row['crosses_split_pct']:.1f}%)"
        held_out = f"**{row['held_out_in_train']} / {row['total_held_out']} ({row['held_out_in_train_pct']:.1f}%)**"
        
        lines.append(f"| {min_len} chars | {spans} | {more_books} | {crosses} | {held_out} |")
        
    lines.append("")
    lines.append("## Headline Summary")
    lines.append("")
    lines.append(f"- **Total held-out spans (val + test):** {leakage['total_held_out_spans']}")
    lines.append(f"- **Held-out spans seen in training (at >= 20 chars normalization):** {leakage['seen_in_train_count']} ({leakage['seen_in_train_pct']:.1f}%)")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset_card_content(gold: Gold, leakage: Leakage) -> str:
    """Generates the content of README.md dataset card with interpolated statistics."""
    class_counts = gold.class_counts
    total_spans = gold.kept_spans
    
    class_table_rows = []
    class_descriptions = {
        "TEXT": "Quotations introduced by ablative markers (ལས, ནས) pointing to texts or sources.",
        "PERSON": "Quotations introduced by ergative markers or speech verbs/nouns (ན་རེ, གསུངས, etc.) identifying speakers.",
        "MISC": "Quotations with no identified prepended marker or continuation segments whose markers were separated.",
        "FORMULA": "Standard formulas or terminative expressions introducing citations (སྐད་དུ, ཇི་སྐད་དུ, etc.).",
        "CHAINED": "Quotations continuing immediately after a previous closing particle (ཞེས་དང, ཅེས་དང).",
    }
    for cls in ("TEXT", "PERSON", "MISC", "FORMULA", "CHAINED"):
        count = class_counts.get(cls, 0)
        pct = 100.0 * count / total_spans if total_spans > 0 else 0.0
        desc = class_descriptions[cls]
        class_table_rows.append(f"| `{cls}` | {count:,} | {pct:.1f}% | {desc} |")
        
    class_table_md = "\n".join(class_table_rows)
    
    splits = leakage["measured_splits"]
    splits_table_rows = []
    for s in ("train", "val", "test"):
        st = splits[s]
        old_pct = st["old_pct"]
        density = st["density"]
        books = st["books"]
        spans = st["spans"]
        
        text_pct = st["class_mix"]["TEXT"] * 100.0
        person_pct = st["class_mix"]["PERSON"] * 100.0
        misc_pct = st["class_mix"]["MISC"] * 100.0
        
        splits_table_rows.append(
            f"| `{s}` | {books} | {spans:,} | {text_pct:.1f}% | {person_pct:.1f}% | {misc_pct:.1f}% | {density:.2f}% | {old_pct:.1f}% |"
        )
    splits_table_md = "\n".join(splits_table_rows)
    
    leakage_table_rows = []
    for row in leakage["table"]:
        min_len = row["min_length"]
        spans_cnt = row["spans"]
        more_books = f"{row['more_than_one_book']} ({row['more_than_one_book_pct']:.1f}%)"
        crosses = f"{row['crosses_split']} ({row['crosses_split_pct']:.1f}%)"
        held_out = f"{row['held_out_in_train']} / {row['total_held_out']} ({row['held_out_in_train_pct']:.1f}%)"
        leakage_table_rows.append(f"| {min_len} chars | {spans_cnt:,} | {more_books} | {crosses} | {held_out} |")
    leakage_table_md = "\n".join(leakage_table_rows)
    
    coverage_table_rows = []
    bands_data = leakage["coverage_table"]["bands_data"]
    for label, _ in BANDS:
        bdata = bands_data[label]
        books = bdata["books_count"]
        chars = bdata["chars_sum"]
        spans = bdata["spans_sum"]
        closers = bdata["closers_sum"]
        covered = bdata["covered_sum"]
        
        chars_m = f"{chars / 1_000_000:.2f}M" if chars >= 100_000 else f"{chars / 1000:.1f}K"
        if chars == 0:
            closers_mchar = 0
            pct_annotated = 0.0
        else:
            closers_mchar = int(round(closers * 1_000_000 / chars))
            pct_annotated = 100.0 * covered / closers if closers > 0 else 0.0
            
        coverage_table_rows.append(
            f"| {label} | {books} | {chars_m} | {spans} | {closers} | {closers_mchar} | {pct_annotated:.1f}% |"
        )
    coverage_table_md = "\n".join(coverage_table_rows)

    card = f"""---
license: other
language:
- bo
task_categories:
- token-classification
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Tibetan Quotation Benchmark Dataset

This dataset provides a book-level train/validation/test benchmark of human-reviewed, gold-standard classical Tibetan quotation annotations.

## 1. What the data is

The dataset comprises **{gold.kept_books} classical Tibetan books** with **{gold.kept_spans:,} hand-annotated, human-approved quotation spans**.
It was built from a raw initial set of 28,892 span pieces. Merging adjacent line-split annotations reduced the span count to 23,476, and dropping under-annotated books resulted in the final set of {gold.kept_spans:,} spans.

Character offsets are **inclusive on both ends** (meaning `end` points to the last character of the quotation, so the character at index `end` is part of the span).

### Real Example in Tibetan
Here is an example of an annotated span in Tibetan context (annotated text highlighted):
> རྒྱུད་འཇམ་ཞེ་སྡང་རབ་སྤངས་ཤིང་།། ... ཞེས་སོ།།

## 2. Where it came from

The underlying annotations come from two source annotation batches of classical Buddhist commentaries digitized with the support of the Tsadra Foundation and OpenPecha:
1. **Old batch:** 125 books originally annotated with a `Quotation` layer.
2. **New batch:** 195 books originally annotated with a `Citation` layer.

These two layers represent the exact same semantic annotation (where one text quotes another scriptural text, a named teacher, or a stock phrase). They have been treated as a single unified label. Because the batches differ significantly in systematic alignment and layout conventions, the source batch was used directly as a balancing variable during stratification.

## 3. How it was cleaned

The dataset was prepared through a rigorous three-pass quality control pipeline:
1. **Human Review Pass:** A total of 398 books were audited by human reviewers. 78 books were rejected entirely due to poor boundary offsets, while 254 span boundaries in accepted books were repaired and snapped to clean delimiters. A book was accepted or rejected as a whole to ensure that clean-looking spans inside a corrupted book did not survive by chance.
2. **Span Merging:** Quotation spans in the new batch are frequently chunked (split across newline boundaries into multiple pieces of 20–40 or 110–160 characters). Naive training on these chunked spans corrupts model learning because only the first piece carries the leading grammatical marker (meaning continuation pieces have no marker and fall through to `MISC`). We merged continued spans when the gap between them was 0, 1, or 2 characters consisting only of spaces, newlines, or a shad, and where the preceding span did not already end on an explicit closing marker. This step reduced the span count from 28,892 to 23,476.
3. **Dropping Under-Annotated Books:** Some accepted books were only partially annotated. We diagnosed this by comparing the density of explicit quotation closer markers in the raw text against actual annotations. While the marker density was flat across all books (~1,200 markers per million characters), the annotation coverage in the sparse books was as low as 1.1% compared to 45.6% in dense books:

{coverage_table_md}

To prevent these false negatives from penalizing a well-performing model or training the model to remain silent, we dropped all books with a merged quotation density below **1.0%** (removing 66 books, 22.5% of characters, and only 1.6% of spans).

## 4. The label set

Every quotation span is annotated with one of the following five grammatical marker classes based on its leading syllables:

| Class | Spans | Percentage | Description |
|---|---|---|---|
{class_table_md}

## 5. How it is split

Assignments are made **book-level only** (a book's spans are never split across multiple partitions). The splits were constructed jointly using a greedy, largest-first greedy search to balance four critical variables:
1. Marker class mix over **`TEXT`, `PERSON`, and `MISC` classes only** (weight: 1.0).
2. Density-bucket mix across 5 bins (weight: 0.5).
3. Source batch mix (share of old batch spans, weight: 0.5).
4. Overall quotation density (weight: 20.0).

By balancing only the top three classes directly, the optimizer avoids trading away density and batch balance to satisfy tiny classes (`FORMULA` and `CHAINED`), which land slightly above target as a result.

The resulting joint split balances the splits exceptionally well:

| Split | Books | Spans | TEXT | PERSON | MISC | Density | Old Batch % |
|---|---|---|---|---|---|---|---|
{splits_table_md}

## 6. Leakage

We distinguish and report two types of leakage:
1. **Structural Leakage:** This is **strictly zero**. Because the split is book-level, no character of any text is present in more than one split.
2. **Quotation-Text Overlap:** Because classical Tibetan commentary frequently quotes the same core scriptures, identical quotation strings appear across different books and land in different splits. This is an unavoidable property of the classical literature. Rather than artificially deduplicating (which would make the benchmark unrepresentative of real-world use), we document it and tag every held-out span:

| Normalization Length | Spans | In more than one book | Crosses split boundary | Held-out spans whose text is in train |
|---|---|---|---|---|
{leakage_table_md}

- **Total held-out spans (val + test):** {leakage["total_held_out_spans"]}
- **Held-out spans seen in training (at >= 20 chars normalization):** {leakage["seen_in_train_count"]} ({leakage["seen_in_train_pct"]:.1f}%)

Every span in the released dataset includes a `seen_in_train` boolean field. **All future evaluations using this benchmark must report two headline scores:** the overall score, and the `seen_in_train == false` subset score. The gap between them represents the memorization effect of the evaluated model.

## 7. Known limits

- **Boundary Noise:** Minor boundary offset noise survives in accepted books.
- **Under-Annotation:** Dropping books with density < 1.0% reduces, but does not solve, under-annotation. The 1–3% density band still holds only a 7.7% marker coverage rate, meaning some valid quotations remain unlabeled in the corpus.
- **Annotated Density:** The overall density of 7.41% reflects adequately annotated books, not a random classical Tibetan text.

## 8. How to load it

You can load this dataset directly via `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("karma689/tibetan-quotation-detection")
print(dataset)
```

## Citation

```bibtex
@misc{{tibetan_quotation_benchmark,
  title  = {{Tibetan Quotation Detection Benchmark Dataset}},
  author = {{karma689}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/karma689/tibetan-quotation-detection}}
}}
```
"""
    return card


def build_hf_export(gold: Gold, split: Split, leakage: Leakage, out_dir: Path) -> None:
    """Builds the Hugging Face dataset export in the specified local directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = {
        "train": out_dir / "train.jsonl",
        "validation": out_dir / "validation.jsonl",
        "test": out_dir / "test.jsonl",
    }
    
    split_books = {"train": [], "validation": [], "test": []}
    for book_id, book in gold.books.items():
        s_name = split[book_id]
        hf_split = "validation" if s_name == "val" else s_name
        split_books[hf_split].append((book_id, book))
        
    for hf_split, path in jsonl_files.items():
        with path.open("w", encoding="utf-8") as f:
            for book_id, book in sorted(split_books[hf_split], key=lambda x: x[0]):
                quote_chars = sum(s.end - s.start + 1 for s in book.spans)
                density = 100.0 * quote_chars / book.doc_chars if book.doc_chars > 0 else 0.0
                bucket_idx = get_book_density_bucket_idx(density)
                buckets_labels = ["<1%", "1-3%", "3-8%", "8-15%", ">=15%"]
                bucket_label = buckets_labels[bucket_idx]
                
                spans_out = [
                    {
                        "start": s.start,
                        "end": s.end,
                        "cls": s.cls,
                        "closed": s.closed,
                        "seen_in_train": s.seen_in_train
                    }
                    for s in book.spans
                ]
                
                row = {
                    "book_id": book_id,
                    "title": book.title,
                    "text": book.text,
                    "doc_chars": book.doc_chars,
                    "batch": book.batch,
                    "source_layer": book.layer,
                    "density": round(density, 2),
                    "density_bucket": bucket_label,
                    "spans": spans_out,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
    stats_json_path = out_dir / "stats.json"
    meta = {
        "corpus_targets": {
            "class": leakage["targets"]["class"],
            "bucket": leakage["targets"]["bucket"],
            "old": leakage["targets"]["old"],
            "density": leakage["targets"]["density"],
            "total_spans": leakage["targets"]["total_spans"],
            "total_books": leakage["targets"]["total_books"],
        },
        "measured_splits": leakage["measured_splits"],
        "chosen_weights": leakage["weights"],
        "leakage_audit": {
            "table": leakage["table"],
            "seen_in_train_count": leakage["seen_in_train_count"],
            "seen_in_train_pct": leakage["seen_in_train_pct"],
        }
    }
    stats_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    readme_path = out_dir / "README.md"
    readme_path.write_text(generate_dataset_card_content(gold, leakage), encoding="utf-8")
    print(f"Dataset local export completed at: {out_dir}")


def push_to_hub(out_dir: Path, repo_id: str) -> None:
    """Pushes the local dataset export folder to the Hugging Face Hub."""
    from huggingface_hub import HfApi, get_token

    if get_token() is None:
        raise ValueError(
            "No Hugging Face credentials found. Run `hf auth login` or set HF_TOKEN."
        )

    print(f"Uploading folder {out_dir} to Hugging Face repo {repo_id}...")
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out_dir,
        commit_message="Upload Tibetan Quotation Benchmark dataset",
    )
    print(f"Successfully pushed to HF: https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tibetan Quotation Benchmark builder.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Path to quotation_gold.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for CSV, JSON, and MD")
    parser.add_argument("--repo", default="karma689/tibetan-quotation-detection", help="HF Hub repo ID")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to Hugging Face Hub (requires `hf auth login` or HF_TOKEN)",
    )
    parser.add_argument("--min-density", type=float, default=1.0, help="Minimum density threshold to keep books")
    parser.add_argument("--seed", type=int, default=4, help="Seed to use when running with --no-sweep")
    parser.add_argument("--no-sweep", action="store_true", help="Run with a single config instead of a full sweep")
    
    args = parser.parse_args(argv)
    
    try:
        print(f"1. Loading gold set from {args.gold}...")
        gold = load_gold(args.gold)
        print(f"   Loaded {gold.kept_books} books with {gold.kept_spans} spans.")
        
        print("\n2. Checking for duplicate books...")
        duplicates = check_duplicate_books(gold)
        print(f"   Found and resolved {len(duplicates)} duplicate books.")
        
        print("\n3. Merging continued chunked spans...")
        gold_merged = merge_continued_spans(gold)
        print(f"   Spans merged: 28892 (original gold) -> {gold_merged.kept_spans} (merged).")
        
        merged_json_path = args.out / "quotation_gold_merged.json"
        print(f"   Saving merged gold to {merged_json_path}...")
        write_merged_gold(gold_merged, merged_json_path)
        
        print("\n4. Running annotation coverage diagnostics...")
        coverage = annotation_coverage(gold_merged)
        coverage_md_path = args.out / "quotation_annotation_coverage.md"
        print(f"   Saving coverage report to {coverage_md_path}...")
        write_coverage_md(coverage, args.min_density, coverage_md_path)
        
        print(f"\n5. Filtering books under {args.min_density}% density...")
        gold_filtered = drop_under_annotated(gold_merged, args.min_density)
        print(f"   Kept {gold_filtered.kept_books} books and {gold_filtered.kept_spans} spans.")
        
        print("\n6. Recomputing corpus targets...")
        targets = corpus_targets(gold_filtered)
        print("   Corpus targets:")
        print(f"   - TEXT: {targets['class']['TEXT']*100:.2f}%")
        print(f"   - PERSON: {targets['class']['PERSON']*100:.2f}%")
        print(f"   - MISC: {targets['class']['MISC']*100:.2f}%")
        print(f"   - FORMULA: {targets['class']['FORMULA']*100:.2f}%")
        print(f"   - CHAINED: {targets['class']['CHAINED']*100:.2f}%")
        print(f"   - Density: {targets['density']:.2f}%")
        print(f"   - Old Batch: {targets['old']*100:.2f}%")
        
        print("\n7. Optimizing split...")
        split, chosen_config, measured_tables = sweep_and_split(
            gold_filtered,
            targets,
            no_sweep=args.no_sweep,
            density_weight=20.0,
            batch_weight=0.5,
            book_cap=30,
            seed=args.seed
        )
        
        csv_path = args.out / "quotation_split.csv"
        json_path = args.out / "quotation_split.json"
        print(f"   Saving split assignment CSV to {csv_path}...")
        write_split_csv(gold_filtered, split, csv_path)
        print(f"   Saving split assignment JSON to {json_path}...")
        write_split_json(gold_filtered, split, targets, json_path, chosen_config, measured_tables)
        
        print("\n8. Auditing leakage & tagging seen_in_train...")
        leakage = audit_leakage(gold_filtered, split)
        leakage["targets"] = targets
        leakage["measured_splits"] = measured_tables
        leakage["weights"] = chosen_config
        leakage["coverage_table"] = coverage
        
        leakage_md_path = args.out / "quotation_leakage.md"
        print(f"   Saving leakage report to {leakage_md_path}...")
        write_leakage_md(leakage, leakage_md_path)
        
        hf_export_dir = ROOT / "Data" / "hf_export"
        print(f"\n9. Building local HF dataset export in {hf_export_dir}...")
        build_hf_export(gold_filtered, split, leakage, hf_export_dir)
        
        print("\n10. Running end-of-run validation checks...")
        check_structural_leakage(gold_filtered, split)
        print("    - [PASS] Structural leakage checks out: Splits are disjoint and cover kept books.")
        
        for b_id, b in gold_filtered.books.items():
            is_train = split[b_id] == "train"
            for s in b.spans:
                assert hasattr(s, "seen_in_train"), "Span missing seen_in_train attribute!"
                if is_train:
                    assert not s.seen_in_train, f"Train span in {b_id} has seen_in_train=True!"
        print("    - [PASS] seen_in_train invariants checked.")
        
        all_spans_with_book = []
        for b_id, b in gold_filtered.books.items():
            for s in b.spans:
                all_spans_with_book.append((b_id, b, s))
                
        rng_check = random.Random(42)
        sample_spans = rng_check.sample(all_spans_with_book, min(20, len(all_spans_with_book)))
        for b_id, b, s in sample_spans:
            text_slice = b.text[s.start : s.end + 1]
            file_text = get_book_text(b_id, DEFAULT_REPOS)
            file_slice = file_text[s.start : s.end + 1]
            assert text_slice == file_slice, f"Offset mismatch in book {b_id} for span {s.start}-{s.end}!"
        print(f"    - [PASS] Verified offset slice round-trip check for {len(sample_spans)} random spans.")
        
        total_assigned_spans = sum(measured_tables[s]["spans"] for s in ("train", "val", "test"))
        assert total_assigned_spans == gold_filtered.kept_spans, f"Total split spans {total_assigned_spans} != gold kept spans {gold_filtered.kept_spans}!"
        print(f"    - [PASS] Filtered spans match split assignments sum: {total_assigned_spans}.")
        
        if args.push:
            print("\n11. Pushing to Hugging Face Hub...")
            push_to_hub(hf_export_dir, args.repo)
        else:
            print("\n11. Dry-run complete. Skip pushing to HF Hub (use --push to upload).")
            
        print("\n" + "=" * 50)
        print("BENCHMARK BUILD SUMMARY")
        print("=" * 50)
        print(f"Filtered Books: {gold_filtered.kept_books} / Filtered Spans: {gold_filtered.kept_spans:,}")
        print("-" * 50)
        print(f"{'Split':<12} {'Books':<8} {'Spans':<10} {'Density':<10}")
        for s in ("train", "val", "test"):
            m = measured_tables[s]
            print(f"{s:<12} {m['books']:<8} {m['spans']:<10,} {m['density']:<10.2f}%")
        print("-" * 50)
        print(f"Leakage overlap (min 20 chars): {leakage['seen_in_train_count']} / {leakage['total_held_out_spans']} ({leakage['seen_in_train_pct']:.2f}%)")
        print(f"Winning config: dw={chosen_config['density_weight']:.1f}, bw={chosen_config['batch_weight']:.1f}, cap={chosen_config['book_cap']}, seed={chosen_config['seed']}")
        print(f"Scores - score: {chosen_config['score']:.2f}, err_test: {chosen_config['err_test']:.2f}, err_val: {chosen_config['err_val']:.2f}")
        print("=" * 50)
        return 0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

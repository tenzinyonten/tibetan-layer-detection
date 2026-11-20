"""Human-reviewed test set for the Tibetan quotation benchmark.

The 19 test books of ``karma689/tibetan-quotation-detection`` were sent out for
human review, one YAML per book (``<book>-quotation.yaml``). Each reviewer went
through every quotation span and marked it ``accepted`` (kept as is),
``edited`` (boundaries moved) or dropped it, and could ``add`` spans the
original annotators missed. This module turns those YAMLs into a new gold test
set and publishes it next to the old one, so models can be scored on both.

What it builds
--------------
1. **Book-level gold** — ``Data/hf_export_reviewed/test.jsonl``. Same row shape as
   the published ``test.jsonl``; only ``spans`` (and ``density``) change.
2. **Windowed BIOE gold** — ``Data/quotation_windows_w8192_s4579_bioe_reviewed/
   test/*.parquet``. Same windows, same tokens, same ownership mask as the
   published ``windowed_w8192_s4579_bioe`` test split (revision ``w2.0``); only
   ``labels`` change. So a model sees exactly the same input and the two scores
   differ only because the gold changed.
3. **A change report** — ``Data/audit/quotation_review_report.md`` — how much the
   gold moved per book.

Why the windows are re-used, not re-tokenized
---------------------------------------------
The text did not change, so tokenizing it again with ``jhu-clsp/mmBERT-base`` at
the same geometry (8192 / stride 3613) gives the same ids and offsets. Re-using
the published windows makes that a guarantee instead of a hope, and needs no
tokenizer download. The labels are then projected with the *same* function the
original build used (``training_utils.align_window_labels``, first-window
ownership, I→E relabel). Before labelling the reviewed spans, the build runs the
original spans through the same code and checks it reproduces the published
labels exactly — if it does not, nothing is written.

Same rules as the original gold
-------------------------------
* Offsets are **inclusive** on both ends: ``text[start:end + 1]``. Checked on the
  review files: an edited span ends on the shad, the next char is ``ཞེས``.
* Pieces of one long quotation are re-joined with the original
  ``quotation_benchmark.merge_continued_spans`` (gap ≤ 2 chars of newline /
  space / shad, never across a closer).
* ``cls`` comes from the original piece by annotation id. Added spans take the
  class of the original piece they overlap most, else ``MISC``; ``cls_source``
  says which.
* ``seen_in_train`` uses the original rule: normalised text ≥ 20 chars that is
  also a training span.

Run it
------
    python quotation/src/build_quotation_reviewed_benchmark.py              # dry run, local only
    python quotation/src/build_quotation_reviewed_benchmark.py --push       # upload + tag w2.1

``--push`` reads ``HF_TOKEN`` from the environment and uploads only new files:
the reviewed windows under ``windowed_w8192_s4579_bioe_reviewed/test/``, the
book-level set under ``reviewed/``, and a README section. Nothing published at
``w2.0`` is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from quotation_lib.quotation_benchmark import (
    Book,
    Gold,
    Span,
    merge_continued_spans,
    normalize_string,
)
from quotation_lib.training_utils import LABEL2ID as L15
from quotation_lib.training_utils import align_window_labels

ROOT = Path(__file__).resolve().parents[2]

REVIEW_DIR = ROOT / "Data" / "review" / "quotation_test"
BOOK_EXPORT_DIR = ROOT / "Data" / "hf_export"
RAW_GOLD = ROOT / "Data" / "audit" / "quotation_gold.json"
REPOS_DIR = ROOT / "Data" / "repos"
SRC_SHARDS = ROOT / "Data" / "quotation_windows_w8192_s4579_bioe" / "shards" / "test"
OUT_BOOKS_DIR = ROOT / "Data" / "hf_export_reviewed"
OUT_WINDOWS_DIR = ROOT / "Data" / "quotation_windows_w8192_s4579_bioe_reviewed"
OUT_GOLD = ROOT / "Data" / "audit" / "quotation_gold_reviewed_test.json"
OUT_REPORT = ROOT / "Data" / "audit" / "quotation_review_report.md"

HUB_REPO = "karma689/tibetan-quotation-detection"
SRC_CONFIG = "windowed_w8192_s4579_bioe"
SRC_REVISION = "w2.0"
REVIEWED_CONFIG = "windowed_w8192_s4579_bioe_reviewed"
REVIEWED_BOOK_CONFIG = "reviewed_test"
REVIEWED_TAG = "w2.1"

EXPECTED_TEST_BOOKS = 19
SEEN_MIN_CHARS = 20

LABEL_O, LABEL_B, LABEL_I, LABEL_E = 0, 1, 2, 3
CARD_START = "<!-- reviewed-test:start -->"
CARD_END = "<!-- reviewed-test:end -->"


class BuildError(Exception):
    """A check failed; nothing should be pushed."""


# ---------------------------------------------------------------------------
# Review files
# ---------------------------------------------------------------------------


@dataclass
class ReviewedSpan:
    ann: str
    start: int
    end: int  # inclusive
    review: str  # accepted | edited | added
    confirmed: bool


@dataclass
class Review:
    book_id: str
    path: Path
    spans: list[ReviewedSpan]
    dropped: list[dict[str, Any]]
    meta: dict[str, Any]

    @property
    def counts(self) -> dict[str, int]:
        counter = Counter(span.review for span in self.spans)
        return {
            "accepted": counter.get("accepted", 0),
            "edited": counter.get("edited", 0),
            "added": counter.get("added", 0),
            "dropped": len(self.dropped),
            "unconfirmed": sum(
                1 for s in self.spans if s.review in ("edited", "added") and not s.confirmed
            ),
        }


def load_review(path: Path) -> Review:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta = dict(data.get("review") or {})
    book_id = str(meta.get("opf") or path.name.split("-")[0])
    if not path.name.startswith(book_id):
        raise BuildError(f"{path.name}: file name does not match review.opf={book_id}")
    if str(data.get("annotation_type") or meta.get("annotation")) not in ("Quotation", "Citation"):
        raise BuildError(f"{path.name}: not a quotation review")
    spans: list[ReviewedSpan] = []
    for ann, item in (data.get("annotations") or {}).items():
        span = item.get("span") or {}
        review = str(item.get("review") or "accepted")
        if review not in ("accepted", "edited", "added"):
            raise BuildError(f"{path.name}: span {ann} has unknown review {review!r}")
        spans.append(
            ReviewedSpan(
                ann=str(ann),
                start=int(span["start"]),
                end=int(span["end"]),
                review=review,
                confirmed=bool(item.get("confirmed", review == "accepted")),
            )
        )
    dropped = list(meta.get("dropped") or [])
    return Review(book_id=book_id, path=path, spans=spans, dropped=dropped, meta=meta)


def load_reviews(review_dir: Path) -> dict[str, Review]:
    paths = sorted(review_dir.glob("*-quotation.yaml")) + sorted(review_dir.glob("*-quotation.yml"))
    reviews: dict[str, Review] = {}
    for path in paths:
        review = load_review(path)
        if review.book_id in reviews:
            raise BuildError(f"two review files for {review.book_id}")
        reviews[review.book_id] = review
    return reviews


# ---------------------------------------------------------------------------
# Book-level gold
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_pieces(raw_gold_path: Path, book_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Original, unmerged spans per book (ann id, offsets, cls)."""
    if not raw_gold_path.is_file():
        print(f"  note: {raw_gold_path} missing; added/edited classes fall back to overlap with merged gold")
        return {}
    data = json.loads(raw_gold_path.read_text(encoding="utf-8"))
    wanted = set(book_ids)
    return {
        book_id: list(book.get("spans") or [])
        for book_id, book in data.get("books", {}).items()
        if book_id in wanted
    }


def train_norm_set(train_rows: Sequence[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in train_rows:
        text = row["text"]
        for span in row["spans"]:
            out.add(normalize_string(text[span["start"] : span["end"] + 1]))
    return out


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _union_overlaps(spans: list[Span]) -> tuple[list[Span], int]:
    """Reviewed spans must not overlap (BIO cannot express it). Union any that do."""
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    out: list[Span] = []
    joined = 0
    for span in spans:
        if out and span.start <= out[-1].end:
            last = out[-1]
            last.end = max(last.end, span.end)
            last.merged_from += span.merged_from
            joined += 1
        else:
            out.append(span)
    return out, joined


def merge_pieces(book_id: str, text: str, spans: list[Span]) -> list[Span]:
    """Run the original merge rule on one book."""
    book = Book(
        layer="Quotation",
        batch="",
        verdict="accept",
        title="",
        doc_chars=len(text),
        closed_pct=0.0,
        misaligned_pct=0.0,
        both_mid_pct=0.0,
        line_split_pct=0.0,
        n_original=len(spans),
        spans=spans,
        text=text,
    )
    gold = Gold(
        generated="",
        policy="",
        kept_books=1,
        kept_spans=len(spans),
        dropped_spans=0,
        snapped_spans=0,
        class_counts={},
        books={book_id: book},
    )
    return merge_continued_spans(gold).books[book_id].spans


@dataclass
class BookResult:
    row: dict[str, Any]
    old_spans: list[tuple[int, int]]
    new_spans: list[tuple[int, int]]
    counts: dict[str, int]
    overlaps_joined: int
    cls_sources: Counter = field(default_factory=Counter)


def build_reviewed_book(
    review: Review,
    orig_row: dict[str, Any],
    pieces: list[dict[str, Any]],
    train_norms: set[str],
    *,
    require_confirmed: bool,
) -> BookResult:
    book_id = review.book_id
    text = orig_row["text"]
    n = len(text)
    base = REPOS_DIR / book_id / f"{book_id}.opf" / "base" / "v001.txt"
    if base.is_file() and base.read_text(encoding="utf-8") != text:
        raise BuildError(f"{book_id}: base/v001.txt differs from the published text")
    if review.meta.get("base_file") and not str(review.meta["base_file"]).endswith("base/v001.txt"):
        raise BuildError(f"{book_id}: review was made on {review.meta['base_file']}, not base/v001.txt")

    piece_by_id = {str(p.get("ann")): p for p in pieces if p.get("ann")}
    # fall back to the merged published spans when the raw gold file is absent
    overlap_pool = pieces or [
        {"start": s["start"], "end": s["end"], "cls": s["cls"]} for s in orig_row["spans"]
    ]

    spans: list[Span] = []
    piece_meta: list[tuple[int, int, str, str]] = []
    cls_sources: Counter = Counter()
    for item in review.spans:
        if require_confirmed and item.review in ("edited", "added") and not item.confirmed:
            continue
        if not (0 <= item.start <= item.end < n):
            raise BuildError(
                f"{book_id}: span {item.ann} [{item.start}, {item.end}] is outside the text (len {n})"
            )
        if item.ann in piece_by_id:
            cls, source = str(piece_by_id[item.ann]["cls"]), "original"
        else:
            best, best_ov = None, 0
            for piece in overlap_pool:
                ov = _overlap(item.start, item.end, int(piece["start"]), int(piece["end"]))
                if ov > best_ov:
                    best, best_ov = piece, ov
            cls, source = (str(best["cls"]), "overlap") if best else ("MISC", "default")
        cls_sources[source] += 1
        spans.append(
            Span(ann=item.ann, start=item.start, end=item.end, cls=cls, closed=False)
        )
        piece_meta.append((item.start, item.end, item.review, source))

    spans, joined = _union_overlaps(spans)
    merged = merge_pieces(book_id, text, spans)

    rank = {"accepted": 0, "edited": 1, "added": 2}
    out_spans: list[dict[str, Any]] = []
    quote_chars = 0
    for span in merged:
        norm = normalize_string(text[span.start : span.end + 1])
        quote_chars += span.end - span.start + 1
        inside = [m for m in piece_meta if span.start <= m[0] and m[1] <= span.end]
        first = min(inside, key=lambda m: m[0]) if inside else (0, 0, "accepted", "original")
        # a merged quotation is "accepted" only if every piece was accepted
        span_review = max((m[2] for m in inside), key=lambda r: rank[r], default="accepted")
        out_spans.append(
            {
                "start": span.start,
                "end": span.end,
                "cls": span.cls,
                "cls_source": first[3],
                "review": span_review,
                "merged_from": span.merged_from,
                "seen_in_train": len(norm) >= SEEN_MIN_CHARS and norm in train_norms,
            }
        )

    row = dict(orig_row)
    row["spans"] = out_spans
    row["density"] = round(100.0 * quote_chars / max(n, 1), 4)
    row["review"] = {**review.counts, "reviewed_at": str(review.meta.get("updated") or "")}

    return BookResult(
        row=row,
        old_spans=[(int(s["start"]), int(s["end"])) for s in orig_row["spans"]],
        new_spans=[(s["start"], s["end"]) for s in out_spans],
        counts=review.counts,
        overlaps_joined=joined,
        cls_sources=cls_sources,
    )


def gold_shift(old: list[tuple[int, int]], new: list[tuple[int, int]]) -> dict[str, int]:
    """How the merged gold moved: exact, IoU>=0.5, only-old, only-new."""
    exact = len(set(old) & set(new))
    used: set[int] = set()
    iou50 = 0
    for a in old:
        for j, b in enumerate(new):
            if j in used:
                continue
            ov = _overlap(a[0], a[1], b[0], b[1])
            if not ov:
                continue
            union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - ov
            if ov / union >= 0.5:
                used.add(j)
                iou50 += 1
                break
    return {
        "old": len(old),
        "new": len(new),
        "exact": exact,
        "iou50": iou50,
        "old_unmatched": len(old) - iou50,
        "new_unmatched": len(new) - iou50,
    }


# ---------------------------------------------------------------------------
# Windowed BIOE labels
# ---------------------------------------------------------------------------


def _spans_for_align(spans: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"start": int(s["start"]), "end": int(s["end"]), "layer": "Quotation", "_identity": i}
        for i, s in enumerate(spans)
    ]


def _collapse(labels15: Sequence[int]) -> list[int]:
    b, i = L15["B-QUOTE"], L15["I-QUOTE"]
    return [-100 if x == -100 else LABEL_B if x == b else LABEL_I if x == i else LABEL_O for x in labels15]


def project_bioe(
    spans: Sequence[dict[str, Any]],
    offsets: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """BIO with first-window ownership, then I→E on the stitched owned sequence.

    Identical to quotation_windows.project_labels(mode="first") followed by
    quotation_bioe.relabel_book. Specials are the (0, 0) offsets.
    """
    align = _spans_for_align(spans)
    covered = 0
    rows: list[np.ndarray] = []
    for offs in offsets:
        special = [int(a == b) for a, b in offs]
        labels15, covered = align_window_labels(
            [(int(a), int(b)) for a, b in offs], special, align, already_covered_through=covered
        )
        rows.append(np.asarray(_collapse(labels15), dtype=np.int64))

    masks = [row != -100 for row in rows]
    owned = np.concatenate([row[m] for row, m in zip(rows, masks)]) if rows else np.empty(0, np.int64)
    nxt = np.empty_like(owned)
    if len(owned):
        nxt[:-1] = owned[1:]
        nxt[-1] = LABEL_O
    single = np.flatnonzero((owned == LABEL_B) & (nxt != LABEL_I))
    if len(single):
        raise BuildError(f"single-token span at owned token {int(single[0])}; BIOE cannot encode it")
    owned[(owned == LABEL_I) & (nxt != LABEL_I)] = LABEL_E

    out: list[np.ndarray] = []
    cursor = 0
    for row, mask in zip(rows, masks):
        k = int(mask.sum())
        new = row.copy()
        new[mask] = owned[cursor : cursor + k]
        cursor += k
        out.append(new)
    return out


def _read_shard(path: Path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    table = table.take(pc.sort_indices(table, sort_keys=[("window_index", "ascending")]))
    labels = [np.asarray(x, dtype=np.int64) for x in table.column("labels").to_pylist()]
    offsets = [np.asarray(x, dtype=np.int64).reshape(-1, 2) for x in table.column("offset_mapping").to_pylist()]
    return table, labels, offsets


def _with_labels(table, labels: Sequence[np.ndarray]):
    import pyarrow as pa

    idx = table.schema.get_field_index("labels")
    fld = table.schema.field(idx)
    lengths = np.asarray([len(x) for x in labels], dtype=np.int32)
    offs = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offs[1:])
    values = pa.array(np.concatenate(labels) if labels else np.empty(0, np.int64), type=fld.type.value_type)
    arr = pa.ListArray.from_arrays(pa.array(offs), values).cast(fld.type)
    out = table.set_column(idx, fld, arr)
    if table.schema.metadata:
        out = out.replace_schema_metadata(table.schema.metadata)
    return out


def fetch_source_shards(dest: Path, book_order: Sequence[str]) -> None:
    """Download the published w2.0 test shards when they are not on disk."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for index, book_id in enumerate(book_order):
        name = f"{index:04d}_{book_id}.parquet"
        if (dest / name).is_file():
            continue
        local = hf_hub_download(
            HUB_REPO,
            f"{SRC_CONFIG}/test/{name}",
            repo_type="dataset",
            revision=SRC_REVISION,
        )
        (dest / name).write_bytes(Path(local).read_bytes())


@dataclass
class WindowResult:
    book_id: str
    windows: int
    n_spans: int
    b_count: int
    e_count: int
    positive_tokens: int
    supervised_tokens: int


def relabel_book_windows(
    shard: Path,
    book_id: str,
    orig_spans: Sequence[dict[str, Any]],
    new_spans: Sequence[dict[str, Any]],
    out_path: Path,
) -> WindowResult:
    table, old_labels, offsets = _read_shard(shard)
    ids = set(table.column("book_id").to_pylist())
    if ids != {book_id}:
        raise BuildError(f"{shard.name}: holds {ids}, expected {book_id}")

    # 1) the original spans through this code must give the published labels
    again = project_bioe(orig_spans, offsets)
    for w, (a, b) in enumerate(zip(again, old_labels)):
        if not np.array_equal(a, b):
            bad = int(np.flatnonzero(a != b)[0])
            raise BuildError(
                f"{book_id} window {w}: re-projecting the published gold does not reproduce "
                f"the published labels (first difference at token {bad}: {a[bad]} vs {b[bad]}). "
                "The relabel path has drifted from the original build; refusing to continue."
            )

    # 2) the reviewed spans
    new = project_bioe(new_spans, offsets)
    for w, (a, b) in enumerate(zip(new, old_labels)):
        if not np.array_equal(a == -100, b == -100):
            raise BuildError(f"{book_id} window {w}: ownership mask changed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    pq.write_table(_with_labels(table, new), out_path)

    flat = np.concatenate(new)
    owned = flat[flat != -100]
    return WindowResult(
        book_id=book_id,
        windows=len(new),
        n_spans=len(new_spans),
        b_count=int((owned == LABEL_B).sum()),
        e_count=int((owned == LABEL_E).sum()),
        positive_tokens=int((owned > 0).sum()),
        supervised_tokens=int(len(owned)),
    )


# ---------------------------------------------------------------------------
# Report, stats, card
# ---------------------------------------------------------------------------


def write_report(
    results: Sequence[BookResult],
    windows: Sequence[WindowResult],
    path: Path,
    *,
    require_confirmed: bool,
) -> dict[str, Any]:
    win = {w.book_id: w for w in windows}
    tot = Counter()
    lines = [
        "# Quotation test set — human review",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Unconfirmed edits/additions {'excluded' if require_confirmed else 'included'}.",
        "",
        "Counts in the first five columns are review decisions on the original *pieces* "
        "(before re-joining). `old`/`new` are merged quotations; `same` = identical "
        "offsets; `IoU≥.5` = old and new quotations that still match at the benchmark's "
        "IoU threshold.",
        "",
        "| book | accepted | edited | added | dropped | unconfirmed | old | new | same | IoU≥.5 | windows |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    per_book = {}
    for r in results:
        shift = gold_shift(r.old_spans, r.new_spans)
        c = r.counts
        book_id = r.row["book_id"]
        lines.append(
            f"| {book_id} | {c['accepted']} | {c['edited']} | {c['added']} | {c['dropped']} | "
            f"{c['unconfirmed']} | {shift['old']} | {shift['new']} | {shift['exact']} | "
            f"{shift['iou50']} | {win[book_id].windows if book_id in win else '-'} |"
        )
        tot.update(c)
        tot.update({f"shift_{k}": v for k, v in shift.items()})
        tot["overlaps_joined"] += r.overlaps_joined
        for k, v in r.cls_sources.items():
            tot[f"cls_{k}"] += v
        per_book[book_id] = {**c, **shift, "density": r.row["density"]}
    lines.append(
        f"| **total** | {tot['accepted']} | {tot['edited']} | {tot['added']} | {tot['dropped']} | "
        f"{tot['unconfirmed']} | {tot['shift_old']} | {tot['shift_new']} | {tot['shift_exact']} | "
        f"{tot['shift_iou50']} | {sum(w.windows for w in windows)} |"
    )
    seen = sum(s["seen_in_train"] for r in results for s in r.row["spans"])
    n_new = tot["shift_new"]
    cls_counts = Counter(s["cls"] for r in results for s in r.row["spans"])
    lines += [
        "",
        f"- Overlapping reviewed spans joined: {tot['overlaps_joined']}",
        f"- Class source of reviewed pieces: original id {tot['cls_original']}, "
        f"overlap {tot['cls_overlap']}, default MISC {tot['cls_default']}",
        f"- `seen_in_train`: {seen} / {n_new} ({100 * seen / max(n_new, 1):.1f}%)",
        "- Classes: " + ", ".join(f"{k} {v}" for k, v in sorted(cls_counts.items())),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "totals": {k: int(v) for k, v in tot.items()},
        "per_book": per_book,
        "seen_in_train": int(seen),
        "class_counts": dict(cls_counts),
    }


def card_section(stats: dict[str, Any]) -> str:
    t = stats["totals"]
    return f"""{CARD_START}
## Reviewed test set (`{REVIEWED_TAG}`)

All {stats['books']} test books were checked by human reviewers, span by span. Each
original span was accepted as is, had its edges corrected, or was removed, and
reviewers added quotations the first annotators missed. Train and validation are
unchanged.

| | count |
|---|---|
| spans accepted unchanged | {t['accepted']} |
| spans with corrected edges | {t['edited']} |
| spans added | {t['added']} |
| spans removed | {t['dropped']} |
| quotations before review (merged) | {t['shift_old']} |
| quotations after review (merged) | {t['shift_new']} |
| identical before and after | {t['shift_exact']} |

Same conventions as the original set: `end` is **inclusive**, pieces of one long
quotation are re-joined, `seen_in_train` uses the same rule. Each span also has
`review` (`accepted` / `edited` / `added`) and `cls_source` (where its marker class
came from — added spans have no original class).

Two new configs, both test-only:

* `{REVIEWED_BOOK_CONFIG}` — book-level rows, `reviewed/test.jsonl`
* `{REVIEWED_CONFIG}` — the `{SRC_CONFIG}` test windows with reviewed labels.
  Input ids, offsets and the ownership mask are identical to `{SRC_CONFIG}`
  test at `{SRC_REVISION}`; only `labels` differ, so a score on the two tells you
  exactly how much the gold change moved it.

```python
from datasets import load_dataset
test = load_dataset("{HUB_REPO}",
                    data_files={{"test": "{REVIEWED_CONFIG}/test/*.parquet"}},
                    revision="{REVIEWED_TAG}")["test"]
```

Per-book changes: `reviewed/review_report.md`; raw numbers: `reviewed/stats.json`.
{CARD_END}
"""


def update_readme(readme: str, section: str) -> str:
    if readme.startswith("---"):
        _, front, body = readme.split("---", 2)
    else:
        front, body = "", "\n" + readme
    meta = yaml.safe_load(front) or {}
    configs = [c for c in (meta.get("configs") or []) if isinstance(c, dict)]
    wanted = [
        {"config_name": REVIEWED_BOOK_CONFIG, "data_files": [{"split": "test", "path": "reviewed/test.jsonl"}]},
        {"config_name": REVIEWED_CONFIG, "data_files": [{"split": "test", "path": f"{REVIEWED_CONFIG}/test/*.parquet"}]},
    ]
    names = {c["config_name"] for c in wanted}
    configs = [c for c in configs if c.get("config_name") not in names] + wanted
    meta["configs"] = configs
    if CARD_START in body and CARD_END in body:
        head, rest = body.split(CARD_START, 1)
        _, tail = rest.split(CARD_END, 1)
        body = head + section.strip() + tail
    else:
        body = body.rstrip() + "\n\n" + section.strip() + "\n"
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---" + body


def push(out_books: Path, out_windows: Path, report: Path, stats_path: Path, stats: dict, *, repo: str, tag: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise BuildError("HF_TOKEN is not set")
    api = HfApi(token=token)
    tags = {t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags}
    if tag in tags:
        raise BuildError(f"tag {tag} already exists on {repo}; pass --tag with a new name")

    readme_path = hf_hub_download(repo, "README.md", repo_type="dataset", token=token)
    readme = update_readme(Path(readme_path).read_text(encoding="utf-8"), card_section(stats))

    ops = [
        CommitOperationAdd("reviewed/test.jsonl", str(out_books / "test.jsonl")),
        CommitOperationAdd("reviewed/review_report.md", str(report)),
        CommitOperationAdd("reviewed/stats.json", str(stats_path)),
        CommitOperationAdd("README.md", readme.encode("utf-8")),
    ]
    for shard in sorted((out_windows / "test").glob("*.parquet")):
        ops.append(CommitOperationAdd(f"{REVIEWED_CONFIG}/test/{shard.name}", str(shard)))
    info = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Add human-reviewed test set ({stats['books']} books)",
    )
    api.create_tag(repo, tag=tag, revision=info.oid, repo_type="dataset",
                   tag_message="human-reviewed quotation test set")
    print(f"pushed {len(ops)} files to {repo} @ {info.oid[:10]}, tagged {tag}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=REVIEW_DIR)
    parser.add_argument("--book-export", type=Path, default=BOOK_EXPORT_DIR,
                        help="folder with the published train.jsonl / test.jsonl")
    parser.add_argument("--src-shards", type=Path, default=SRC_SHARDS,
                        help="published w2.0 BIOE test shards (downloaded if missing)")
    parser.add_argument("--require-confirmed", action="store_true",
                        help="leave out edited/added spans the reviewer did not confirm")
    parser.add_argument("--allow-partial", action="store_true",
                        help="build with fewer than 19 review files (never pushes)")
    parser.add_argument("--push", action="store_true", help="upload to the Hub (needs HF_TOKEN)")
    parser.add_argument("--repo", default=HUB_REPO)
    parser.add_argument("--tag", default=REVIEWED_TAG)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    test_rows = read_jsonl(args.book_export / "test.jsonl")
    order = [row["book_id"] for row in test_rows]
    by_id = {row["book_id"]: row for row in test_rows}

    reviews = load_reviews(args.reviews)
    unknown = sorted(set(reviews) - set(by_id))
    if unknown:
        raise BuildError(f"review files for books not in the test split: {unknown}")
    missing = [b for b in order if b not in reviews]
    if missing:
        msg = f"{len(missing)} test books have no review file: {missing}"
        if not args.allow_partial:
            raise BuildError(msg + "  (put them in " + str(args.reviews) + ")")
        if args.push:
            raise BuildError("refusing to push a partial review set")
        print("WARNING: " + msg)
    print(f"{len(reviews)} review files")

    pieces = raw_pieces(RAW_GOLD, order)
    train_norms = train_norm_set(read_jsonl(args.book_export / "train.jsonl"))

    results: list[BookResult] = []
    for book_id in order:
        if book_id not in reviews:
            continue
        results.append(
            build_reviewed_book(
                reviews[book_id], by_id[book_id], pieces.get(book_id, []), train_norms,
                require_confirmed=args.require_confirmed,
            )
        )
    write_jsonl([r.row for r in results], OUT_BOOKS_DIR / "test.jsonl")
    OUT_GOLD.write_text(
        json.dumps(
            {"source": "human review", "books": {r.row["book_id"]: {"spans": r.row["spans"]} for r in results}},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )

    # windows
    if any(not (args.src_shards / f"{i:04d}_{b}.parquet").is_file() for i, b in enumerate(order)):
        print(f"downloading published test windows ({SRC_CONFIG} @ {SRC_REVISION})")
        fetch_source_shards(args.src_shards, order)
    windows: list[WindowResult] = []
    for index, book_id in enumerate(order):
        if book_id not in reviews:
            continue
        name = f"{index:04d}_{book_id}.parquet"
        row = next(r.row for r in results if r.row["book_id"] == book_id)
        w = relabel_book_windows(
            args.src_shards / name, book_id, by_id[book_id]["spans"], row["spans"],
            OUT_WINDOWS_DIR / "test" / name,
        )
        flag = "" if w.b_count == w.e_count == w.n_spans else "  <-- span/B/E mismatch"
        print(f"  {book_id}: {w.windows} windows, {w.n_spans} spans, B={w.b_count} E={w.e_count}{flag}")
        windows.append(w)

    bad = [w for w in windows if not (w.b_count == w.e_count == w.n_spans)]
    if bad:
        raise BuildError(
            "token labels do not hold one B and one E per reviewed span in: "
            + ", ".join(f"{w.book_id} (spans {w.n_spans}, B {w.b_count}, E {w.e_count})" for w in bad)
        )

    stats = write_report(results, windows, OUT_REPORT, require_confirmed=args.require_confirmed)
    stats.update(
        {
            "books": len(results),
            "windows": sum(w.windows for w in windows),
            "supervised_tokens": sum(w.supervised_tokens for w in windows),
            "positive_tokens": sum(w.positive_tokens for w in windows),
            "source": {"repo": args.repo, "config": SRC_CONFIG, "revision": SRC_REVISION},
            "require_confirmed": args.require_confirmed,
            "reviewed_jsonl_sha256": hashlib.sha256((OUT_BOOKS_DIR / "test.jsonl").read_bytes()).hexdigest(),
        }
    )
    stats_path = OUT_WINDOWS_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_WINDOWS_DIR / "README_card_snippet.md").write_text(card_section(stats), encoding="utf-8")

    t = stats["totals"]
    report_text = OUT_REPORT.read_text(encoding="utf-8")
    print("\n" + report_text[report_text.index("| book"):])
    print(
        f"books {stats['books']}  quotations {t['shift_old']} -> {t['shift_new']}  "
        f"(identical {t['shift_exact']}, still IoU>=0.5 {t['shift_iou50']})"
    )
    print(f"windows {stats['windows']}  positive token rate "
          f"{100 * stats['positive_tokens'] / max(stats['supervised_tokens'], 1):.3f}%")
    print(f"wrote {OUT_BOOKS_DIR / 'test.jsonl'}\n      {OUT_WINDOWS_DIR}/test/*.parquet\n      {OUT_REPORT}\n      {OUT_GOLD}")

    if args.push:
        push(OUT_BOOKS_DIR, OUT_WINDOWS_DIR, OUT_REPORT, stats_path, stats, repo=args.repo, tag=args.tag)
    else:
        print("dry run: nothing uploaded (add --push)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

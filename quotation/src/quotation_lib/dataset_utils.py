"""Book-level {text, spans} row building, shared by upload_to_hf.py and any
training script that needs the local-fallback path (e.g. experiments/*/train.py
before the Hub push, or with no network).

Deliberately produces rows that are NOT pre-combined into BIO tags and NOT
tokenized — that keeps the data reusable across checkpoints (mmBERT now,
bert-base-tibetan if we ever fall back) without regenerating it. Tokenizing,
windowing, overlap-priority resolution, and BIO conversion are training-time
steps (workflow step 5 / Task #12), not here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .audit_utils import (
    MODEL_LAYERS,
    classify_span,
    find_books,
    in_scope_layers,
    load_book,
    parse_layer_spans,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOS_DIR = ROOT / "Data" / "repos"
DEFAULT_SPLIT_CSV = ROOT / "Data" / "audit" / "split.csv"
MASK_SOURCE = "confirmed"

# Known declared-but-missing-file mismatches in this fixed corpus. This also
# lets training correct datasets uploaded before the confirmed-negative scope
# policy was implemented.
KNOWN_SCOPE_EXCLUSIONS = {
    "P000010": {"Quotation"},
    "P000100": {"Tsawa"},
}

# Local split.csv uses "val"; Hub DatasetDict uses HF's "validation".
SPLIT_TO_HUB = {"train": "train", "val": "validation", "test": "test"}


def confirmed_scope_for_book(book_id: str) -> list[str]:
    """All confirmed layers, less known unassessed book/layer mismatches."""
    excluded = KNOWN_SCOPE_EXCLUSIONS.get(book_id, set())
    return [layer for layer in MODEL_LAYERS if layer not in excluded]


def apply_confirmed_scope(rows: list[dict]) -> int:
    """Repair legacy uploaded rows in memory; return the number changed."""
    changed = 0
    for row in rows:
        expected = confirmed_scope_for_book(row["book_id"])
        if set(row.get("in_scope_layers", [])) != set(expected):
            row["in_scope_layers"] = expected
            changed += 1
    return changed

DATASET_CARD = """\
---
language:
- bo
license: other
task_categories:
- token-classification
pretty_name: Tibetan Annotation Layer Detection
tags:
- tibetan
- openpecha
- tsadra
- ner
- quotation
- span-detection
- buddhist-studies
size_categories:
- n<1K
source_datasets:
- tsadra/OpenPecha
---

# Tibetan Annotation Layer Detection

Book-level dataset for multi-layer span detection in classical Tibetan Buddhist \
texts. Each row is one book: its full raw text, plus a flat list of cleaned \
character-offset spans across seven annotation layers — Quotation, Sabche, \
Tsawa, Yigchung, Chapter, Author, and BookTitle. The underlying texts and their \
original annotations come from the [OpenPecha](https://openpecha.org/) corpus, \
built on digitized editions curated with [Tsadra Foundation](https://www.tsadra.org/).

The dataset is intentionally **raw text + spans, not pre-tokenized and not \
pre-combined into BIO tags** — that keeps it reusable across tokenizers/model \
checkpoints and preserves overlapping annotations at full fidelity. Overlap \
resolution (e.g. a Quotation nested inside a Sabche) and BIO conversion are \
left to whoever trains on it, so the choice of priority rule doesn't have to \
be baked into the data itself.

## Schema

| Column | Type | Description |
|---|---|---|
| `book_id` | string | OpenPecha book id (e.g. `P000010`) |
| `text` | string | Full book text (`base/v001.txt`) |
| `spans` | list of `{start, end, label}` | Inclusive character offsets; label is one of `QUOTATION`, `SABCHE`, `TSAWA`, `YIGCHUNG`, `CHAPTER`, `AUTHOR`, `BOOKTITLE` |
| `in_scope_layers` | list of string | Layers this book is confirmed in-scope for. A layer absent from a book's spans is only a trustworthy negative if that layer is also listed here — otherwise it simply wasn't assessed for that book, and should not be scored as a negative. |

Splits: `train` / `validation` / `test` — 217 / 25 / 24 books. Book-level \
stratified split (weighted iterative stratification balancing both labeled \
character volume and per-layer book coverage); a book is never split across \
more than one partition, and `test` is held out for final evaluation only.

## Corpus statistics

- **266 books**, ~66.2M characters of raw text, ~44.0M tokens (mmBERT tokenizer).
- **61,661 total spans** across all seven layers:

| Layer | Spans | Notes |
|---|---:|---|
| Quotation | 24,730 | longest span in the corpus: 2,940 tokens |
| Sabche | 18,847 | |
| Yigchung | 9,025 | |
| Tsawa | 6,845 | |
| Chapter | 1,637 | |
| Author | 311 | ~1 span/book — experimental scope |
| BookTitle | 266 | ~1 span/book — experimental scope |

- 23 spans overlap another layer in the raw annotations (preserved as-is in `spans`, not resolved).

## Data quality notes

- **512 invalid spans dropped** (480 Quotation, 30 Chapter, 2 Yigchung) — all \
  `inverted_empty_start_eq_end_plus_1`; none recoverable.
- **Citation layer excluded entirely** — only 2 books carry it, 88% of those spans invalid.
- **Author / BookTitle** are in scope experimentally (~1 span/book); may be \
  demoted to metadata-only if a trained model underperforms on them.
- **`in_scope_layers` uses corpus-confirmed-negative semantics**: a layer is \
  in-scope for a book unless that book declared the layer but its annotation \
  file is missing (a known data mismatch, not a real "not applicable" \
  assessment) — currently `P000010` (Quotation) and `P000100` (Tsawa).
- Character offsets are inclusive on both ends (`text[start:end+1]`, not \
  Python's default exclusive-end slicing) — verified against the raw \
  annotation source, not assumed.

## Citation

If you use this dataset, please cite it and the underlying OpenPecha / Tsadra \
Foundation source corpus it's built from:

```bibtex
@misc{tibetan_layer_detection,
  title  = {Tibetan Annotation Layer Detection Dataset},
  author = {karma689},
  year   = {2026},
  url    = {https://huggingface.co/datasets/karma689/layer_detection},
  note   = {Source texts and annotations from OpenPecha, built on editions curated with Tsadra Foundation}
}
```

Please also acknowledge [OpenPecha](https://openpecha.org/) and \
[Tsadra Foundation](https://www.tsadra.org/) as the source of the underlying \
digitized texts and their original annotation layers, which this dataset \
cleans, splits, and repackages for token-classification training but did not \
create.

## License

The annotation cleaning, book-level train/validation/test split, and this \
packaged `{book_id, text, spans, in_scope_layers}` format are original work \
and are released under **CC0 1.0 Universal (Public Domain Dedication)**.

The underlying Tibetan texts and their original annotations are sourced from \
individual [OpenPecha-Data](https://github.com/OpenPecha-Data) repositories; \
licensing for those source texts varies by repository (many are CC0, per \
OpenPecha's general practice, but this has not been individually verified \
for every one of the 266 books here). Anyone reusing this dataset beyond \
research/training purposes should check the license of the specific source \
repositories before redistribution.

## Acknowledgements

The source texts and their original Quotation/Sabche/Tsawa/Yigchung/Chapter/\
Author/BookTitle annotations were produced through [OpenPecha](https://openpecha.org/), \
a collaborative platform for open Buddhist-text corpora, working from digitized \
editions curated with [Tsadra Foundation](https://www.tsadra.org/), a nonprofit \
supporting the study, practice, and translation of Tibetan Buddhism. This \
dataset builds on their work by auditing and cleaning the raw annotation \
spans, computing a book-level stratified split, and repackaging the result \
for token-classification model training — it does not represent new \
annotation or translation effort on the underlying texts.
"""


def read_split_assignment(split_csv: Path) -> dict[str, str]:
    with split_csv.open(encoding="utf-8", newline="") as fh:
        return {row["book"]: row["split"] for row in csv.DictReader(fh)}


def book_spans(rec, layer: str) -> list[dict]:
    """Valid (cleaned) spans for one layer, as {start, end, label} dicts.

    Applies the same keep/fix/drop decision as the audit step (classify_span)
    directly from the raw layer YAML, so this doesn't depend on
    write_cleaned_layers() having been run first.
    """
    path = rec.layer_files.get(layer)
    if not path:
        return []
    out = []
    for _ann_id, start, end, _line in parse_layer_spans(path):
        reasons, action, fixed_start, fixed_end, _note = classify_span(start, end, rec.doc_len)
        if action == "drop":
            continue
        s = fixed_start if fixed_start is not None else start
        e = fixed_end if fixed_end is not None else end
        out.append({"start": s, "end": e, "label": layer.upper()})
    return out


def build_rows(repos_dir: Path, split_assignment: dict[str, str]) -> list[dict]:
    """One row per book: book_id, text, spans (all 7 layers), in_scope_layers, split."""
    rows = []
    missing_split = []
    for book_dir in find_books(repos_dir):
        book_id = book_dir.name
        if book_id not in split_assignment:
            missing_split.append(book_id)
            continue
        rec = load_book(book_dir)
        if not rec.base_path or not rec.base_path.is_file():
            continue

        text = rec.base_path.read_text(encoding="utf-8")
        spans: list[dict] = []
        for layer in MODEL_LAYERS:
            spans.extend(book_spans(rec, layer))
        spans.sort(key=lambda sp: (sp["start"], sp["end"]))

        scope = sorted(in_scope_layers(rec, MASK_SOURCE))

        rows.append(
            {
                "book_id": book_id,
                "text": text,
                "spans": spans,
                "in_scope_layers": scope,
                "split": split_assignment[book_id],
            }
        )

    if missing_split:
        print(
            f"Warning: {len(missing_split)} books have no split assignment, skipped: "
            f"{missing_split[:10]}{'...' if len(missing_split) > 10 else ''}",
            file=sys.stderr,
        )
    return rows

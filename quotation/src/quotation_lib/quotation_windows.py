"""Build mmBERT token windows from the Tibetan Quotation Benchmark and publish them.

Geometry (config ``windowed_w8192_s4579``)
------------------------------------------
| setting | value |
|---|---|
| ``max_length`` | 8192 |
| HF tokenizer ``stride`` (overlap) | 3613 |
| step (hop) | 4579 |
| overlap | 44.104% |

A 42% / step-4751 setting assumed a ~3399-token longest span. Measured on
``IB522F095`` with ``jhu-clsp/mmBERT-base``: **3613 tokens** for the 5007-char
chain — gate 4 therefore requires ``hf_stride >= 3613`` (step 4579).

The user-facing config name carries the **step**, not the HF ``stride`` argument.
``tokenizer(..., max_length=8192, stride=3441)`` yields step 4751. Passing 4751 as
HF ``stride`` would give 58% overlap — ``resolve_geometry`` refuses ``stride >= max_length``.

Why 42% overlap
---------------
A span is guaranteed whole inside some window only when overlap ≥ span length. The
longest merged quotation is ~5,007 characters (~3,399 tokens). At 42% overlap the HF
stride is 3,441 tokens — the first setting that guarantees containment for any span
placement. 30% and 40% cut nothing *today* but do not clear the bar; re-merging or
adding books can move the longest chain. The build asserts ``max_span_tokens <= hf_stride``
and fails loudly if a future merge breaks the guarantee.

Columns (minimal)
-----------------
``book_id``, ``window_index``, ``input_ids``, ``attention_mask``, ``labels`` (3-class BIO,
``-100`` on specials and duplicated stride prefix), ``offset_mapping``. All five marker
classes collapse to ``B-QUOTE`` / ``I-QUOTE``. ``cls`` and ``seen_in_train`` stay on the
book-level set only.

Ownership
---------
Default ``first``: within each book, walk windows in order; a token is supervised once
(``token_end <= already_covered_through`` → ``-100``). Optional ``centre``: assign each
token to the window whose character midpoint is nearest. Both modes must satisfy
contiguity — concatenating owned regions in ``window_index`` order reconstructs the book
token sequence exactly.

Gold spans use **inclusive** character offsets on both ends: ``text[start:end+1]``.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import yaml
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from quotation_lib.training_utils import LABEL2ID as L15
from quotation_lib.training_utils import align_window_labels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "build_quotation_windows.yaml"

LABEL_O = 0
LABEL_B_QUOTE = 1
LABEL_I_QUOTE = 2
ID2LABEL = {LABEL_O: "O", LABEL_B_QUOTE: "B-QUOTE", LABEL_I_QUOTE: "I-QUOTE"}

EXPECTED_TOTAL_SPANS = 23_095
EXPECTED_TOTAL_BOOKS = 254
ROUND_TRIP_SAMPLES = 200

OwnershipMode = Literal["first", "centre"]


class GateFailure(Exception):
    """A hard verification gate failed; do not push."""


@dataclass(frozen=True)
class Geometry:
    max_length: int
    step: int
    hf_stride: int

    @property
    def overlap_percent(self) -> float:
        return 100.0 * self.hf_stride / self.max_length


@dataclass
class BookRow:
    book_id: str
    text: str
    spans: list[dict[str, Any]]
    split: str
    title: str = ""
    doc_chars: int = 0


@dataclass
class Benchmark:
    splits: dict[str, list[BookRow]]
    source_repo: str
    source_revision: str
    source_commit: str = ""

    @property
    def all_books(self) -> list[BookRow]:
        out: list[BookRow] = []
        for split_name in ("train", "validation", "test"):
            out.extend(self.splits.get(split_name, []))
        return out

    def book_by_id(self) -> dict[str, BookRow]:
        return {book.book_id: book for book in self.all_books}


@dataclass
class Window:
    book_id: str
    window_index: int
    input_ids: Sequence[int]
    attention_mask: Sequence[int]
    labels: Sequence[int]
    offset_mapping: Sequence[Sequence[int]]
    special_tokens_mask: Sequence[int] = field(default_factory=list)


@dataclass
class Report:
    ok: bool
    failures: list[str]
    window_counts: dict[str, int]
    b_quote_counts: dict[str, int]
    severance: int
    class_balance: dict[str, float]
    max_span_tokens: int
    max_span_meta: dict[str, Any] | None
    contiguity_overlap_tokens: int
    contiguity_total_tokens: int
    n_short_books: int
    shortest_book: str
    shortest_tokens: int
    round_trip_ok: int
    round_trip_total: int
    gate_lines: list[str] = field(default_factory=list)
    measured_windows: int = 0
    expected_windows: int = 0
    window_count_warning: str | None = None


def load_job_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Job config must be a mapping: {path}")
    return cfg


def resolve_geometry(cfg: dict[str, Any]) -> Geometry:
    max_length = int(cfg["max_length"])
    step = int(cfg["step"])
    hf_stride = int(cfg.get("hf_stride") or (max_length - step))
    if step <= 0 or step >= max_length:
        raise ValueError(
            f"step must satisfy 0 < step < max_length; got step={step} max_length={max_length}"
        )
    if hf_stride >= max_length:
        raise ValueError(
            f"hf_stride (overlap) must be < max_length; got hf_stride={hf_stride}. "
            f"Did you pass step={step} as HF stride? Use hf_stride={max_length - step}."
        )
    expected_stride = max_length - step
    if hf_stride != expected_stride:
        raise ValueError(
            f"hf_stride={hf_stride} inconsistent with step={step} "
            f"(expected hf_stride={expected_stride})"
        )
    return Geometry(max_length=max_length, step=step, hf_stride=hf_stride)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark(
    repo: str,
    revision: str,
    *,
    local_dir: Path | None = None,
    local_files_only: bool = False,
) -> Benchmark:
    """Load book-level benchmark from Hub or local ``Data/hf_export/`` jsonl."""
    splits: dict[str, list[BookRow]] = {}
    source_commit = ""

    local_root = local_dir
    if local_files_only and (not local_root or not (local_root / "train.jsonl").is_file()):
        raise GateFailure(
            f"dry-run requires local benchmark at {local_root}; network disabled"
        )
    if local_root and (local_root / "train.jsonl").is_file():
        for hf_split in ("train", "validation", "test"):
            path = local_root / f"{hf_split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing local split file: {path}")
            rows = _read_jsonl(path)
            splits[hf_split] = [
                BookRow(
                    book_id=str(row["book_id"]),
                    text=str(row["text"]),
                    spans=list(row.get("spans") or []),
                    split=hf_split,
                    title=str(row.get("title") or ""),
                    doc_chars=int(row.get("doc_chars") or len(row["text"])),
                )
                for row in rows
            ]
    else:
        from datasets import load_dataset

        ds = load_dataset(repo, revision=revision)
        for hf_split in ("train", "validation", "test"):
            if hf_split not in ds:
                raise GateFailure(f"split {hf_split!r} missing from {repo}")
            splits[hf_split] = [
                BookRow(
                    book_id=str(row["book_id"]),
                    text=str(row["text"]),
                    spans=list(row.get("spans") or []),
                    split=hf_split,
                    title=str(row.get("title") or ""),
                    doc_chars=int(row.get("doc_chars") or len(row["text"])),
                )
                for row in ds[hf_split]
            ]
        try:
            api = HfApi()
            refs = api.list_repo_refs(repo, repo_type="dataset")
            for branch in refs.branches:
                if branch.name == revision or branch.name == "main":
                    source_commit = branch.target_commit
                    break
        except Exception:
            source_commit = ""

    _audit_benchmark(splits)
    return Benchmark(
        splits=splits,
        source_repo=repo,
        source_revision=revision,
        source_commit=source_commit,
    )


def _audit_benchmark(splits: dict[str, list[BookRow]]) -> None:
    expected_books = {"train": 210, "validation": 25, "test": 19}
    expected_spans = {"train": 18_753, "validation": 2_005, "test": 2_337}
    all_ids: set[str] = set()
    total_spans = 0
    for split_name, books in splits.items():
        if len(books) != expected_books[split_name]:
            raise GateFailure(
                f"{split_name}: expected {expected_books[split_name]} books, got {len(books)}"
            )
        split_spans = sum(len(book.spans) for book in books)
        if split_spans != expected_spans[split_name]:
            raise GateFailure(
                f"{split_name}: expected {expected_spans[split_name]} spans, got {split_spans}"
            )
        total_spans += split_spans
        for book in books:
            if book.book_id in all_ids:
                raise GateFailure(f"duplicate book_id {book.book_id}")
            all_ids.add(book.book_id)
            for span in book.spans:
                start, end = int(span["start"]), int(span["end"])
                if start > end or start < 0 or end >= len(book.text):
                    raise GateFailure(
                        f"{book.book_id}: invalid span [{start},{end}] text_len={len(book.text)}"
                    )
    if len(all_ids) != EXPECTED_TOTAL_BOOKS:
        raise GateFailure(f"expected {EXPECTED_TOTAL_BOOKS} books, got {len(all_ids)}")
    if total_spans != EXPECTED_TOTAL_SPANS:
        raise GateFailure(f"expected {EXPECTED_TOTAL_SPANS} spans, got {total_spans}")


def hash_tokenizer_files(repo: str, *, local_files_only: bool = False) -> tuple[str, dict[str, str]]:
    names = ("tokenizer.json", "tokenizer_config.json")
    per_file: dict[str, str] = {}
    blobs: list[bytes] = []
    for name in names:
        path = Path(
            hf_hub_download(
                repo,
                name,
                repo_type="model",
                local_files_only=local_files_only,
            )
        )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        per_file[name] = digest
        blobs.append(name.encode() + b"\0" + bytes.fromhex(digest))
    combined = hashlib.sha256(b"".join(blobs)).hexdigest()
    return combined, per_file


def load_tokenizer(
    repo: str,
    *,
    local_files_only: bool = False,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=local_files_only)
    if tokenizer.is_fast is not True:
        raise GateFailure("tokenizer.is_fast must be True for offset_mapping")
    if len(tokenizer) != 256_000:
        raise GateFailure(f"expected vocab_size 256000, got {len(tokenizer)}")
    combined, per_file = hash_tokenizer_files(repo, local_files_only=local_files_only)
    tokenizer._qw_file_hash = combined  # type: ignore[attr-defined]
    tokenizer._qw_file_hashes = per_file  # type: ignore[attr-defined]
    return tokenizer


def _as_offset_pairs(offset_mapping: Sequence[Sequence[int]]) -> Any:
    if len(offset_mapping) == 0:
        return np.empty((0, 2), dtype=np.int64)
    offs = np.asarray(offset_mapping)
    if offs.ndim != 2 or offs.shape[-1] != 2:
        raise GateFailure(
            f"offset_mapping expected (n, 2) pairs, got shape {getattr(offs, 'shape', None)}"
        )
    return offs


def _special_id_set(tokenizer: Any) -> set[int]:
    special_ids = getattr(tokenizer, "_qw_special_ids", None)
    if special_ids is None:
        special_ids = {int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])}
        tokenizer._qw_special_ids = special_ids  # type: ignore[attr-defined]
    return special_ids


def _content_offsets(
    offset_mapping: Sequence[Sequence[int]],
    input_ids: Sequence[int],
    tokenizer: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Return (token_index, char_start, char_end) for content tokens."""
    if len(offset_mapping) == 0:
        return []
    offs = _as_offset_pairs(offset_mapping)
    specials = np.asarray(
        _special_mask(
            offset_mapping, input_ids, tokenizer, special_tokens_mask=special_tokens_mask
        ),
        dtype=bool,
    )
    keep = ~specials & (offs[:, 0] != offs[:, 1])
    idx = np.flatnonzero(keep)
    return [(int(i), int(offs[i, 0]), int(offs[i, 1])) for i in idx]


def _is_special_token(tokenizer: Any, token_id: int) -> bool:
    return int(token_id) in _special_id_set(tokenizer)


def _special_mask(
    offset_mapping: Sequence[Sequence[int]],
    input_ids: Sequence[int],
    tokenizer: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> Any:
    n = len(offset_mapping)
    if special_tokens_mask is not None and len(special_tokens_mask) == n:
        return np.asarray(special_tokens_mask, dtype=bool)
    if n == 0:
        return np.zeros(0, dtype=bool)
    offs = _as_offset_pairs(offset_mapping)
    ids = np.asarray(input_ids)
    return np.isin(ids, list(_special_id_set(tokenizer))) | (offs[:, 0] == offs[:, 1])


def _spans_for_align(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert benchmark spans to ``align_window_labels`` format (inclusive offsets)."""
    return [
        {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "layer": "Quotation",
            "_identity": index,
        }
        for index, span in enumerate(spans)
    ]


def _collapse_quote_labels(labels15: Iterable[int]) -> list[int]:
    """Map 15-class BIO labels to 3-class quotation BIO."""
    quote_b, quote_i, o_id = L15["B-QUOTE"], L15["I-QUOTE"], L15["O"]
    out: list[int] = []
    for label in labels15:
        if label == -100:
            out.append(-100)
        elif label == quote_b:
            out.append(LABEL_B_QUOTE)
        elif label == quote_i:
            out.append(LABEL_I_QUOTE)
        elif label == o_id:
            out.append(LABEL_O)
        else:
            out.append(LABEL_O)
    return out


def _normalize_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        out.append(
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "_identity": index,
            }
        )
    return sorted(out, key=lambda item: (item["start"], item["_identity"]))


def _bio_label_for_token(
    token_start: int,
    token_end: int,
    spans: list[dict[str, Any]],
    *,
    active_span: int | None,
    by_start: list[dict[str, Any]],
    next_span: int,
    active: list[dict[str, Any]],
) -> tuple[int, int | None, int, list[dict[str, Any]]]:
    while next_span < len(by_start) and by_start[next_span]["start"] < token_end:
        active.append(by_start[next_span])
        next_span += 1
    if active:
        active = [span for span in active if token_start < span["end"] + 1]
    if not active:
        return LABEL_O, None, next_span, active
    winner = active[0]
    tag = LABEL_I_QUOTE if active_span == winner["_identity"] else LABEL_B_QUOTE
    return tag, winner["_identity"], next_span, active


def _align_window_bio(
    offset_mapping: Sequence[list[int]],
    input_ids: Sequence[int],
    spans: list[dict[str, Any]],
    tokenizer: Any,
    *,
    already_covered_through: int = 0,
    apply_first_mask: bool = True,
    special_tokens_mask: Sequence[int] | None = None,
) -> list[int]:
    labels: list[int] = []
    active_span: int | None = None
    by_start = _normalize_spans(spans)
    next_span = 0
    active: list[dict[str, Any]] = []
    specials = _special_mask(
        offset_mapping, input_ids, tokenizer, special_tokens_mask=special_tokens_mask
    )

    for (token_start, token_end), is_special in zip(offset_mapping, specials, strict=True):
        token_start, token_end = int(token_start), int(token_end)
        if is_special:
            labels.append(-100)
            active_span = None
            continue

        label_id, active_span, next_span, active = _bio_label_for_token(
            token_start,
            token_end,
            by_start,
            active_span=active_span,
            by_start=by_start,
            next_span=next_span,
            active=active,
        )
        if apply_first_mask and token_end <= already_covered_through:
            labels.append(-100)
        else:
            labels.append(label_id)

    return labels


def _window_char_midpoint(
    offset_mapping: Sequence[list[int]],
    input_ids: Sequence[int],
    tokenizer: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> float:
    content = _content_offsets(
        offset_mapping, input_ids, tokenizer, special_tokens_mask=special_tokens_mask
    )
    if not content:
        return 0.0
    return (content[0][1] + content[-1][2]) / 2.0


def tokenize_book_windows(
    book: BookRow,
    tok: Any,
    geom: Geometry,
) -> list[Window]:
    encoded = tok(
        book.text,
        truncation=True,
        max_length=geom.max_length,
        stride=geom.hf_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=False,
    )
    windows: list[Window] = []
    for window_index, input_ids_row in enumerate(encoded["input_ids"]):
        input_ids = [int(x) for x in input_ids_row]
        attention_mask = [1] * len(input_ids)
        offset_mapping = [
            [int(a), int(b)] for a, b in encoded["offset_mapping"][window_index]
        ]
        special_tokens_mask = [
            int(x) for x in encoded["special_tokens_mask"][window_index]
        ]
        windows.append(
            Window(
                book_id=book.book_id,
                window_index=window_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=[],
                offset_mapping=offset_mapping,
                special_tokens_mask=special_tokens_mask,
            )
        )
    return windows


def project_labels(
    book: BookRow,
    windows: list[Window],
    mode: OwnershipMode,
    tok: Any,
) -> None:
    """Apply BIO labels and ownership masking in place."""
    align_spans = _spans_for_align(book.spans)
    if mode == "first":
        covered_through = 0
        for window in windows:
            labels15, covered_through = align_window_labels(
                window.offset_mapping,
                window.special_tokens_mask,
                align_spans,
                already_covered_through=covered_through,
            )
            window.labels = _collapse_quote_labels(labels15)
        return

    # centre: raw BIO then assign each content token to nearest window midpoint
    raw_windows: list[list[int]] = []
    for window in windows:
        labels15, _furthest = align_window_labels(
            window.offset_mapping,
            window.special_tokens_mask,
            align_spans,
            already_covered_through=0,
        )
        raw_windows.append(_collapse_quote_labels(labels15))
    window_mids = [
        _window_char_midpoint(
            w.offset_mapping,
            w.input_ids,
            tok,
            special_tokens_mask=w.special_tokens_mask,
        )
        for w in windows
    ]
    owners: dict[tuple[int, int], int] = {}
    for window_index, window in enumerate(windows):
        for token_index, (ts, te) in enumerate(window.offset_mapping):
            ts, te = int(ts), int(te)
            if _is_special_token(tok, window.input_ids[token_index]) or ts == te:
                continue
            key = (ts, te)
            if key not in owners:
                tok_mid = (ts + te) / 2.0
                best = min(
                    range(len(windows)),
                    key=lambda wi: (
                        abs(tok_mid - window_mids[wi]),
                        wi,
                    ),
                )
                owners[key] = best

    for window_index, window in enumerate(windows):
        labels: list[int] = []
        for token_index, (ts, te) in enumerate(window.offset_mapping):
            ts, te = int(ts), int(te)
            if _is_special_token(tok, window.input_ids[token_index]) or ts == te:
                labels.append(-100)
                continue
            key = (ts, te)
            if owners.get(key) != window_index:
                labels.append(-100)
            else:
                labels.append(raw_windows[window_index][token_index])
        window.labels = labels


def _owned_content_tokens(window: Window, tok: Any) -> list[tuple[int, int, int, int]]:
    """Return (token_index, char_start, char_end, label) for owned content tokens."""
    if len(window.offset_mapping) == 0:
        return []
    offs = _as_offset_pairs(window.offset_mapping)
    labels = np.asarray(window.labels)
    ids = np.asarray(window.input_ids)
    if not (len(offs) == len(labels) == len(ids)):
        raise GateFailure(
            f"{window.book_id} w{window.window_index}: column length mismatch "
            f"offsets={len(offs)} labels={len(labels)} ids={len(ids)}"
        )
    specials = np.isin(ids, list(_special_id_set(tok)))
    keep = (labels != -100) & (offs[:, 0] != offs[:, 1]) & ~specials
    idx = np.flatnonzero(keep)
    return [(int(i), int(offs[i, 0]), int(offs[i, 1]), int(labels[i])) for i in idx]


def _stitch_owned(book: BookRow, windows: list[Window], tok: Any) -> list[tuple[int, int, int]]:
    stitched: list[tuple[int, int, int]] = []
    for window in sorted(windows, key=lambda w: w.window_index):
        stitched.extend(_owned_content_tokens(window, tok))
    return [(ts, te, label) for _idx, ts, te, label in stitched]


def _expected_bio_sequence(
    book: BookRow,
    stitched: list[tuple[int, int, int]],
) -> list[int]:
    by_start = _normalize_spans(book.spans)
    active_span: int | None = None
    next_span = 0
    active: list[dict[str, Any]] = []
    expected: list[int] = []
    for token_start, token_end, _label in stitched:
        label_id, active_span, next_span, active = _bio_label_for_token(
            token_start,
            token_end,
            by_start,
            active_span=active_span,
            by_start=by_start,
            next_span=next_span,
            active=active,
        )
        expected.append(label_id)
    return expected


def _span_fully_in_window(
    start: int,
    end: int,
    offset_mapping: Sequence[list[int]],
    input_ids: Sequence[int],
    tok: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> bool:
    content = _content_offsets(
        offset_mapping, input_ids, tok, special_tokens_mask=special_tokens_mask
    )
    if not content:
        return False
    w_start = content[0][1]
    w_end = content[-1][2]
    return w_start <= start and end < w_end


def _tokens_for_span_in_window(
    start: int,
    end: int,
    offset_mapping: Sequence[list[int]],
    input_ids: Sequence[int],
    tok: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> list[int]:
    specials = _special_mask(
        offset_mapping, input_ids, tok, special_tokens_mask=special_tokens_mask
    )
    indices: list[int] = []
    for index, (ts, te) in enumerate(offset_mapping):
        ts, te = int(ts), int(te)
        if specials[index] or ts == te:
            continue
        if ts < end + 1 and te > start:
            indices.append(index)
    return indices


def _char_range_from_tokens(
    indices: Sequence[int],
    offset_mapping: Sequence[list[int]],
    input_ids: Sequence[int],
    tok: Any,
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> tuple[int, int] | None:
    specials = _special_mask(
        offset_mapping, input_ids, tok, special_tokens_mask=special_tokens_mask
    )
    starts: list[int] = []
    ends: list[int] = []
    for index in indices:
        ts, te = int(offset_mapping[index][0]), int(offset_mapping[index][1])
        if specials[index] or ts == te:
            continue
        starts.append(ts)
        ends.append(te)
    if not starts:
        return None
    return min(starts), max(ends)


def verify(
    benchmark: Benchmark,
    windows_by_book: dict[str, list[Window]],
    tok: Any,
    geom: Geometry,
    *,
    ownership: OwnershipMode,
    expected_windows: int,
    window_tolerance: float,
    limit_books: bool = False,
    rng: random.Random | None = None,
) -> Report:
    rng = rng or random.Random(42)
    failures: list[str] = []
    gate_lines: list[str] = []
    window_counts = {name: 0 for name in ("train", "validation", "test")}
    b_quote_counts = {name: 0 for name in ("train", "validation", "test")}
    label_counter: Counter[int] = Counter()
    severance = 0
    max_span_tokens = 0
    max_span_meta: dict[str, Any] | None = None
    contiguity_overlap_tokens = 0
    contiguity_total_tokens = 0
    n_short_books = 0
    shortest_tokens = 10**9
    shortest_book = ""
    round_trip_ok = 0
    round_trip_total = 0

    books_by_id = benchmark.book_by_id()
    bio_failures = 0

    # Gate 7: split disjointness (already checked at load; reaffirm)
    if not limit_books:
        all_ids: set[str] = set()
        for split_name, books in benchmark.splits.items():
            for book in books:
                if book.book_id in all_ids:
                    failures.append(f"split disjointness: duplicate {book.book_id}")
                all_ids.add(book.book_id)
        if len(all_ids) != EXPECTED_TOTAL_BOOKS:
            failures.append(f"split union: expected {EXPECTED_TOTAL_BOOKS} books, got {len(all_ids)}")
        gate_lines.append(
            f"Gate 7 split disjointness: {len(all_ids)} books, union OK"
            if len(all_ids) == EXPECTED_TOTAL_BOOKS
            else "Gate 7 split disjointness: FAILED"
        )
    else:
        gate_lines.append("Gate 7 split disjointness: skipped (limit-books smoke run)")

    all_spans_for_roundtrip: list[tuple[BookRow, dict[str, Any]]] = []
    cut_spans: list[dict[str, Any]] = []
    window_bounds_by_book: dict[str, list[tuple[Window, int | None, int | None]]] = {}

    books_done = 0
    books_total = sum(len(books) for books in benchmark.splits.values())
    for split_name, books in benchmark.splits.items():
        for book in books:
            windows = windows_by_book.get(book.book_id)
            if not windows:
                if not limit_books:
                    failures.append(f"{book.book_id}: no windows")
                continue
            books_done += 1
            if books_done == 1 or books_done % 25 == 0 or books_done == books_total:
                print(
                    f"verify: {books_done}/{books_total} books "
                    f"({book.book_id}, {len(windows)} windows)",
                    flush=True,
                )
            window_counts[split_name] += len(windows)
            windows = sorted(windows, key=lambda w: w.window_index)

            # Gate 8: window_index integrity
            indices = [w.window_index for w in windows]
            expected_idx = list(range(len(windows)))
            if indices != expected_idx or len(set(indices)) != len(indices):
                failures.append(f"{book.book_id}: window_index not 0..n-1 contiguous")

            # One token walk per window; reuse for contiguity / ownership / severance.
            owned_by_window = [_owned_content_tokens(window, tok) for window in windows]
            stitched = [
                (ts, te, label)
                for owned in owned_by_window
                for _idx, ts, te, label in owned
            ]
            window_char_ranges = [
                _content_offsets(
                    w.offset_mapping,
                    w.input_ids,
                    tok,
                    special_tokens_mask=w.special_tokens_mask,
                )
                for w in windows
            ]
            window_bounds_by_book[book.book_id] = [
                (window, content[0][1], content[-1][2]) if content else (window, None, None)
                for window, content in zip(windows, window_char_ranges, strict=True)
            ]

            # Gate 2: contiguity
            contiguity_total_tokens += len(stitched)
            prev_end = -1
            for ts, te, _label in stitched:
                if prev_end >= 0 and ts < prev_end:
                    contiguity_overlap_tokens += 1
                if prev_end >= 0 and ts > prev_end:
                    failures.append(
                        f"{book.book_id}: character gap in owned tokens "
                        f"({prev_end} -> {ts})"
                    )
                prev_end = te

            # mmBERT may map multiple token ids to the same char span within one window.
            # Flag only cross-window double ownership of the same span.
            span_owner_window: dict[tuple[int, int], int] = {}
            cross_window_dupes = 0
            for window, owned in zip(windows, owned_by_window, strict=True):
                for _idx, ts, te, _label in owned:
                    key = (ts, te)
                    prev = span_owner_window.get(key)
                    if prev is not None and prev != window.window_index:
                        cross_window_dupes += 1
                    span_owner_window[key] = window.window_index
            if cross_window_dupes:
                failures.append(
                    f"{book.book_id}: {cross_window_dupes} char spans owned in "
                    f"more than one window"
                )

            if stitched:
                if stitched[0][0] != 0:
                    failures.append(
                        f"{book.book_id}: owned tokens start at char {stitched[0][0]}, not 0"
                    )
                last_char_end = stitched[-1][1]
                if last_char_end < len(book.text):
                    failures.append(
                        f"{book.book_id}: owned tokens end at char {last_char_end}, "
                        f"text length {len(book.text)}"
                    )

            # Gate 5: BIO integrity
            expected = _expected_bio_sequence(book, stitched)
            actual = [label for _ts, _te, label in stitched]
            if expected != actual:
                bio_failures += 1
                failures.append(f"{book.book_id}: stitched BIO != span projection")

            # Gate 6: B-QUOTE counts
            for _ts, _te, label in stitched:
                label_counter[label] += 1
                if label == LABEL_B_QUOTE:
                    b_quote_counts[split_name] += 1

            # Gate 3: containment + gate 4 max span
            token_starts = [ts for ts, _te, _l in stitched]
            token_ends = [te for _ts, te, _l in stitched]
            bounds = window_bounds_by_book[book.book_id]
            for span in book.spans:
                start, end = int(span["start"]), int(span["end"])
                inside = any(
                    w_start is not None and w_start <= start and end < w_end
                    for _window, w_start, w_end in bounds
                )
                if not inside:
                    cut_spans.append(
                        {"book_id": book.book_id, "start": start, "end": end}
                    )
                left = bisect.bisect_right(token_ends, start)
                right = bisect.bisect_left(token_starts, end + 1)
                n_tok = max(0, right - left)
                if n_tok > max_span_tokens:
                    max_span_tokens = n_tok
                    max_span_meta = {
                        "book_id": book.book_id,
                        "start": start,
                        "end": end,
                        "tokens": n_tok,
                    }
                all_spans_for_roundtrip.append((book, span))

            # severance: span crosses owned-window boundary
            owned_boundaries = [
                owned[-1][2] for owned in owned_by_window[:-1] if owned
            ]

            for span in book.spans:
                start, end = int(span["start"]), int(span["end"])
                for boundary in owned_boundaries:
                    if start < boundary < end + 1:
                        severance += 1
                        break

            # short books
            if len(windows) == 1:
                n_content = len(stitched)
                if n_content < geom.max_length:
                    n_short_books += 1
                    if n_content < shortest_tokens:
                        shortest_tokens = n_content
                        shortest_book = book.book_id

    if cut_spans:
        failures.append(
            f"Gate 3 containment: {len(cut_spans)} spans not whole in any window"
        )
    gate_lines.append(
        f"Gate 3 containment: {len(cut_spans)} cut spans (must be 0)"
    )

    # Gate 4
    if max_span_tokens > geom.hf_stride:
        failures.append(
            f"Gate 4 guarantee: max_span_tokens={max_span_tokens} > hf_stride={geom.hf_stride} "
            f"({max_span_meta})"
        )
    gate_lines.append(
        f"Gate 4 guarantee: max_span_tokens={max_span_tokens} hf_stride={geom.hf_stride} "
        f"meta={max_span_meta}"
    )

    gate_lines.append(
        f"Gate 5 BIO integrity: {'OK' if bio_failures == 0 else f'FAILED ({bio_failures} books)'}"
    )

    expected_spans = {
        "train": 18_753,
        "validation": 2_005,
        "test": 2_337,
    }
    if not limit_books:
        for split_name, expected in expected_spans.items():
            if b_quote_counts[split_name] != expected:
                failures.append(
                    f"Gate 6 span counts: {split_name} B-QUOTE={b_quote_counts[split_name]} "
                    f"expected {expected}"
                )
    gate_lines.append(
        "Gate 6 B-QUOTE counts: "
        + ", ".join(f"{k}={b_quote_counts[k]}" for k in b_quote_counts)
    )

    gate_lines.append(f"Gate 8 window_index: checked per book")

    # Gate 1: round-trip
    sample = all_spans_for_roundtrip
    rng.shuffle(sample)
    sample = sample[:ROUND_TRIP_SAMPLES]
    for book, span in sample:
        round_trip_total += 1
        start, end = int(span["start"]), int(span["end"])
        gold_text = book.text[start : end + 1]
        bounds = window_bounds_by_book.get(book.book_id) or [
            (window, None, None) for window in windows_by_book[book.book_id]
        ]
        success = False
        for window, w_start, w_end in bounds:
            if w_start is None or w_end is None or not (w_start <= start and end < w_end):
                continue
            indices = _tokens_for_span_in_window(
                start,
                end,
                window.offset_mapping,
                window.input_ids,
                tok,
                special_tokens_mask=window.special_tokens_mask,
            )
            char_range = _char_range_from_tokens(
                indices,
                window.offset_mapping,
                window.input_ids,
                tok,
                special_tokens_mask=window.special_tokens_mask,
            )
            if char_range is None:
                continue
            c_start, c_end = char_range
            recovered = book.text[c_start:c_end]
            gold_cmp = gold_text.replace(" ", "")
            rec_cmp = recovered.replace(" ", "")
            covers = c_start <= start and c_end >= end + 1
            if covers and (
                gold_cmp == rec_cmp or gold_cmp in rec_cmp or rec_cmp in gold_cmp
            ):
                success = True
                break
        if success:
            round_trip_ok += 1
        else:
            failures.append(
                f"Gate 1 round-trip failed: {book.book_id} [{start},{end}]"
            )
    gate_lines.append(
        f"Gate 1 round-trip: {round_trip_ok}/{round_trip_total} OK"
    )

    measured_windows = sum(window_counts.values())
    window_count_warning = None
    if not limit_books:
        delta = abs(measured_windows - expected_windows) / expected_windows
        if delta > window_tolerance:
            window_count_warning = (
                f"window count {measured_windows} vs expected {expected_windows} "
                f"(rel_delta={delta:.3f}, tolerance={window_tolerance})"
            )
            gate_lines.append(f"Window forecast WARNING: {window_count_warning}")
        else:
            gate_lines.append(
                f"Window forecast: measured={measured_windows} expected≈{expected_windows} "
                f"rel_delta={delta:.3f}"
            )

    total_labels = sum(label_counter.values()) or 1
    class_balance = {
        "O": label_counter[LABEL_O] / total_labels,
        "I-QUOTE": label_counter[LABEL_I_QUOTE] / total_labels,
        "B-QUOTE": label_counter[LABEL_B_QUOTE] / total_labels,
    }
    gate_lines.append(
        f"Gate 2 contiguity: overlap artefact tokens={contiguity_overlap_tokens} "
        f"of {contiguity_total_tokens} "
        f"({100.0 * contiguity_overlap_tokens / max(contiguity_total_tokens, 1):.4f}%)"
    )
    gate_lines.append(f"Report severance: {severance} spans cross owned-window boundary")
    gate_lines.append(
        "Report class balance: "
        + ", ".join(f"{k}={v * 100:.3f}%" for k, v in class_balance.items())
    )

    return Report(
        ok=not failures,
        failures=failures,
        window_counts=window_counts,
        b_quote_counts=b_quote_counts,
        severance=severance,
        class_balance=class_balance,
        max_span_tokens=max_span_tokens,
        max_span_meta=max_span_meta,
        contiguity_overlap_tokens=contiguity_overlap_tokens,
        contiguity_total_tokens=contiguity_total_tokens,
        n_short_books=n_short_books,
        shortest_book=shortest_book,
        shortest_tokens=shortest_tokens if shortest_book else 0,
        round_trip_ok=round_trip_ok,
        round_trip_total=round_trip_total,
        gate_lines=gate_lines,
        measured_windows=measured_windows,
        expected_windows=expected_windows,
        window_count_warning=window_count_warning,
    )


def _window_to_dict(window: Window) -> dict[str, Any]:
    return {
        "book_id": window.book_id,
        "window_index": window.window_index,
        "input_ids": window.input_ids,
        "attention_mask": window.attention_mask,
        "labels": window.labels,
        "offset_mapping": window.offset_mapping,
    }


def write_shards(
    windows_by_split: dict[str, list[tuple[int, list[Window]]]],
    out_dir: Path,
) -> None:
    from datasets import Dataset

    shard_root = out_dir / "shards"
    for split_name, book_windows in windows_by_split.items():
        split_dir = shard_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for book_index, windows in book_windows:
            book_id = windows[0].book_id
            path = split_dir / f"{book_index:04d}_{book_id}.parquet"
            rows = [_window_to_dict(w) for w in windows]
            columns: dict[str, list[Any]] = {key: [] for key in rows[0]}
            for row in rows:
                for key, value in row.items():
                    columns[key].append(value)
            Dataset.from_dict(columns).to_parquet(str(path))


def write_stats(
    report: Report,
    out_dir: Path,
    *,
    cfg: dict[str, Any],
    geom: Geometry,
    tokenizer: Any,
    benchmark: Benchmark,
    ownership: OwnershipMode,
) -> None:
    stats = {
        "source_repo": benchmark.source_repo,
        "source_revision": benchmark.source_revision,
        "source_commit": benchmark.source_commit,
        "tokenizer_repo": cfg["tokenizer_repo"],
        "tokenizer_commit": getattr(tokenizer, "name_or_path", cfg["tokenizer_repo"]),
        "vocab_size": len(tokenizer),
        "tokenizer_is_fast": tokenizer.is_fast,
        "tokenizer_hashes": getattr(tokenizer, "_qw_file_hashes", {}),
        "tokenizer_hash_combined": getattr(tokenizer, "_qw_file_hash", None),
        "geometry": {
            "max_length": geom.max_length,
            "step": geom.step,
            "hf_stride": geom.hf_stride,
            "overlap_percent": geom.overlap_percent,
            "config_name": cfg.get("hub_config_name"),
        },
        "ownership_mode": ownership,
        "window_counts": report.window_counts,
        "measured_windows": report.measured_windows,
        "expected_windows": report.expected_windows,
        "window_count_warning": report.window_count_warning,
        "b_quote_counts": report.b_quote_counts,
        "severance": report.severance,
        "class_balance": report.class_balance,
        "max_span_tokens": report.max_span_tokens,
        "max_span_meta": report.max_span_meta,
        "contiguity_overlap_tokens": report.contiguity_overlap_tokens,
        "contiguity_total_tokens": report.contiguity_total_tokens,
        "n_short_books": report.n_short_books,
        "shortest_book": report.shortest_book,
        "shortest_tokens": report.shortest_tokens,
        "round_trip": {
            "ok": report.round_trip_ok,
            "total": report.round_trip_total,
        },
        "gate_lines": report.gate_lines,
        "verification_ok": report.ok,
        "failures": report.failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "window_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def build_card_section(report: Report, cfg: dict[str, Any], geom: Geometry, tokenizer: Any) -> str:
    hashes = getattr(tokenizer, "_qw_file_hashes", {})
    hash_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in hashes.items())
    combined = getattr(tokenizer, "_qw_file_hash", "undetermined")
    config_name = cfg.get("hub_config_name", "windowed_w8192_s4751")
    tag = cfg.get("hub_tag", "w1.0")
    return f"""\
<!-- quotation-windows-card -->
# Config `{config_name}` (tag {tag})

## WARNING — derived data

This dataset is **derived** from a specific tokenizer, window length, and step.
**Regenerate it** if any of those change. Never mix windows from different tokenizers
or strides in one training run.

Load the windowed parquet shards — not the unpinned default book-level jsonl.
``load_dataset(repo, "{config_name}")`` uses the JSON builder because the
default README config is ``*.jsonl``; ``datasets`` infers one builder for the
whole repo from that first config.

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={{
        "train": "hf://datasets/{cfg['hub_repo']}@{tag}/{config_name}/train/*.parquet",
        "validation": "hf://datasets/{cfg['hub_repo']}@{tag}/{config_name}/validation/*.parquet",
        "test": "hf://datasets/{cfg['hub_repo']}@{tag}/{config_name}/test/*.parquet",
    }},
)
```

## Geometry

| Setting | Value |
|---|---:|
| Window length (`max_length`) | {geom.max_length} |
| Step (name `s{geom.step}`) | {geom.step} |
| HF tokenizer `stride` (overlap) | {geom.hf_stride} |
| Overlap | {geom.overlap_percent:.3f}% |

Hub name `s{geom.step}` is the **step**, not the HuggingFace `stride` argument.
`tokenizer(..., max_length={geom.max_length}, stride={geom.hf_stride})` yields step {geom.step}.

## Offsets and ownership

**Gold spans use inclusive character offsets on both ends** (`text[start:end+1]`).

Overlapped tokens receive label ``-100`` in every window except the one that owns them.
Concatenating each book's **owned** token regions in ``window_index`` order reconstructs
the book's token sequence exactly — no token is double-counted and none are missing.
Stitching owned windows for inference or character-space scoring is therefore correct.

Ownership mode for this build: `{cfg.get('_ownership', 'first')}`.

## Label set

All five book-level marker classes (TEXT, PERSON, MISC, FORMULA, CHAINED) collapse to
``B-QUOTE`` (id 1) and ``I-QUOTE`` (id 2). Out-of-span tokens are ``O`` (id 0).

| id | label |
|---:|---|
| 0 | `O` |
| 1 | `B-QUOTE` |
| 2 | `I-QUOTE` |

## Measured statistics

| Split | Windows | B-QUOTE (spans) |
|---|---:|---:|
| train | {report.window_counts.get('train', 0)} | {report.b_quote_counts.get('train', 0)} |
| validation | {report.window_counts.get('validation', 0)} | {report.b_quote_counts.get('validation', 0)} |
| test | {report.window_counts.get('test', 0)} | {report.b_quote_counts.get('test', 0)} |
| **total** | **{report.measured_windows}** | **{sum(report.b_quote_counts.values())}** |

- Severance (spans crossing an owned-window boundary): **{report.severance}**
- Longest span: **{report.max_span_tokens}** tokens ({report.max_span_meta})
- Class balance (owned supervised tokens): O={report.class_balance.get('O', 0) * 100:.3f}%, \
I-QUOTE={report.class_balance.get('I-QUOTE', 0) * 100:.3f}%, \
B-QUOTE={report.class_balance.get('B-QUOTE', 0) * 100:.3f}%
- Short books (&lt; one window): {report.n_short_books} (shortest: {report.shortest_book} \
at {report.shortest_tokens} content tokens)

## Tokenizer provenance

| Field | Value |
|---|---|
| Repo | `{cfg['tokenizer_repo']}` |
| vocab_size | {len(tokenizer)} |
| is_fast | {tokenizer.is_fast} |
| Combined sha256 | `{combined}` |

| File | sha256 |
|---|---|
{hash_rows}
"""


DEFAULT_BOOK_CONFIG = {
    "config_name": "default",
    "data_files": [
        {"split": "train", "path": "train.jsonl"},
        {"split": "validation", "path": "validation.jsonl"},
        {"split": "test", "path": "test.jsonl"},
    ],
}


def windowed_parquet_data_files(repo: str, config_name: str, revision: str) -> dict[str, str]:
    """Hub paths for the named windowed config. One builder, parquet only."""
    root = f"hf://datasets/{repo}@{revision}/{config_name}"
    return {
        split: f"{root}/{split}/*.parquet"
        for split in ("train", "validation", "test")
    }


def load_windowed_dataset(
    repo: str,
    config_name: str,
    revision: str,
    *,
    streaming: bool = True,
):
    """Load windowed shards with the parquet builder.

    ``load_dataset(repo, config_name)`` is wrong here: the default book-level
    config is jsonl, and ``datasets`` infers a single builder for the repo from
    the first README ``configs:`` entry.
    """
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files=windowed_parquet_data_files(repo, config_name, revision),
        streaming=streaming,
    )


def _ensure_windowed_config_yaml(readme: str, config_name: str) -> str:
    """Record the windowed parquet glob in the card YAML.

    Hub ``load_dataset(repo, config_name)`` still cannot mix this with the
    book-level jsonl default — use ``load_windowed_dataset``.
    """
    if readme.startswith("---"):
        _, front, body = readme.split("---", 2)
    else:
        front, body = "", "\n" + readme
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict):
        meta = {}
    configs = list(meta.get("configs") or [])
    has_default = any(
        isinstance(item, dict) and item.get("config_name") == "default"
        for item in configs
    )
    if not has_default:
        configs.insert(0, dict(DEFAULT_BOOK_CONFIG))
    entry = {
        "config_name": config_name,
        "data_files": [
            {"split": "train", "path": f"{config_name}/train/*.parquet"},
            {"split": "validation", "path": f"{config_name}/validation/*.parquet"},
            {"split": "test", "path": f"{config_name}/test/*.parquet"},
        ],
    }
    replaced = False
    for index, item in enumerate(configs):
        if isinstance(item, dict) and item.get("config_name") == config_name:
            configs[index] = entry
            replaced = True
            break
    if not replaced:
        configs.append(entry)
    meta["configs"] = configs
    dumped = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    return f"---\n{dumped}---{body}"


def push_to_hub(
    out_dir: Path,
    repo: str,
    config_name: str,
    tag: str,
    *,
    card_section: str,
    default_train_books: int = 210,
) -> None:
    import os

    from datasets import load_dataset

    if not config_name or config_name in {"default", "data"}:
        raise GateFailure(
            "refusing push_to_hub without a real config_name "
            "(would overwrite the default book-level dataset)"
        )
    if os.environ.get("HF_TOKEN") is None:
        from huggingface_hub import get_token

        if get_token() is None:
            raise GateFailure("HF_TOKEN or huggingface-cli login required for --push")

    api = HfApi()
    refs = api.list_repo_refs(repo, repo_type="dataset")
    tags_before = {t.name: t.target_commit for t in refs.tags}
    main_before = next((b.target_commit for b in refs.branches if b.name == "main"), "")

    default_ds = load_dataset(repo)
    if "text" not in default_ds["train"].column_names:
        raise GateFailure("default config is not book-level; refusing to push")
    if len(default_ds["train"]) != default_train_books:
        raise GateFailure(
            f"default train has {len(default_ds['train'])} rows, expected {default_train_books}"
        )

    shard_root = out_dir / "shards"
    if not shard_root.is_dir():
        raise GateFailure(f"no shards at {shard_root}")

    print(f"pre-push: main={main_before} default train books={len(default_ds['train'])}")
    api.upload_folder(
        folder_path=str(shard_root),
        path_in_repo=config_name,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="*.parquet",
        commit_message=(
            f"Add named config {config_name} (quotation windows). "
            "Default book-level config unchanged."
        ),
    )

    refs2 = api.list_repo_refs(repo, repo_type="dataset")
    main_after = next((b.target_commit for b in refs2.branches if b.name == "main"), "")
    default_after = load_dataset(repo)
    if "text" not in default_after["train"].column_names:
        raise GateFailure("default config lost book-level text after push")
    if len(default_after["train"]) != default_train_books:
        raise GateFailure("default train book count changed after push")

    try:
        readme_path = hf_hub_download(repo, "README.md", repo_type="dataset")
        existing = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        existing = (out_dir / "README_card_snippet.md").read_text(encoding="utf-8")

    marker = "<!-- quotation-windows-card -->"
    existing = _ensure_windowed_config_yaml(existing, config_name)
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start] + card_section.strip() + "\n"
    else:
        existing = existing.rstrip() + "\n\n" + card_section.strip() + "\n"

    api.upload_file(
        path_or_fileobj=existing.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Document named config {config_name}",
    )

    refs3 = api.list_repo_refs(repo, repo_type="dataset")
    tags3 = {t.name: t.target_commit for t in refs3.tags}
    branches3 = {b.name: b.target_commit for b in refs3.branches}
    revision = branches3.get("main")
    if not revision:
        raise GateFailure("main branch missing after README upload")
    if tags3.get(tag) != revision:
        if tag in tags3:
            api.delete_tag(repo, tag=tag, repo_type="dataset")
        api.create_tag(
            repo,
            tag=tag,
            revision=revision,
            repo_type="dataset",
            tag_message=f"{config_name} quotation windows",
        )
    windowed = load_windowed_dataset(repo, config_name, tag, streaming=True)
    row = next(iter(windowed["train"]))
    if "labels" not in row or "offset_mapping" not in row:
        raise GateFailure(f"{config_name}@{tag} missing expected columns")
    print(
        f"post-push: default still book-level ({len(default_after['train'])} train books); "
        f"streamed {config_name} has labels"
    )


def _list_column_arrays(col: Any) -> list[Any]:
    """Split a ListArray into per-row 1-d numpy views without ``to_pylist``."""
    chunk = col.combine_chunks()
    values = chunk.values.to_numpy(zero_copy_only=False)
    offsets = chunk.offsets.to_numpy()
    return [values[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]


def _offset_mapping_arrays(col: Any) -> list[Any]:
    """Split ``list<list<int>>`` offset_mapping into ``(n_tokens, 2)`` arrays.

    ``RecordBatch.to_pylist()`` materializes ~16k Python ints per 8192-token
    window. Flattening Arrow children is ~100× faster (the previous windowed
    builder streamed ``iter_batches`` for the same reason; quotation verify
    still needs the book grouped, so we keep arrays instead of nested lists).
    """
    outer = col.combine_chunks()
    inner = outer.values
    inner_vals = inner.values.to_numpy(zero_copy_only=False)
    inner_off = inner.offsets.to_numpy()
    outer_off = outer.offsets.to_numpy()
    out: list[Any] = []
    for i in range(len(outer_off) - 1):
        start_list = int(outer_off[i])
        end_list = int(outer_off[i + 1])
        start = int(inner_off[start_list])
        end = int(inner_off[end_list])
        pairs = inner_vals[start:end]
        if pairs.size % 2:
            raise GateFailure(
                f"offset_mapping row {i} has {pairs.size} scalars, expected pairs"
            )
        out.append(pairs.reshape(-1, 2))
    return out


def _windows_from_parquet_table(table: Any) -> list[Window]:
    if table.num_rows == 0:
        return []
    book_ids = table.column("book_id").to_pylist()
    window_indices = table.column("window_index").to_pylist()
    input_ids = _list_column_arrays(table.column("input_ids"))
    attention_mask = _list_column_arrays(table.column("attention_mask"))
    labels = _list_column_arrays(table.column("labels"))
    offset_mapping = _offset_mapping_arrays(table.column("offset_mapping"))
    windows: list[Window] = []
    for i, book_id in enumerate(book_ids):
        pairs = offset_mapping[i]
        special = (pairs[:, 0] == pairs[:, 1]).astype(pairs.dtype) if len(pairs) else []
        windows.append(
            Window(
                book_id=str(book_id),
                window_index=int(window_indices[i]),
                input_ids=input_ids[i],
                attention_mask=attention_mask[i],
                labels=labels[i],
                offset_mapping=pairs,
                special_tokens_mask=special,
            )
        )
    return windows


def _load_shards(out_dir: Path) -> dict[str, dict[str, list[Window]]]:
    import pyarrow.parquet as pq

    result: dict[str, dict[str, list[Window]]] = {}
    shard_root = out_dir / "shards"
    for split_name in ("train", "validation", "test"):
        split_dir = shard_root / split_name
        if not split_dir.is_dir():
            continue
        by_book: dict[str, list[Window]] = defaultdict(list)
        paths = sorted(split_dir.glob("*.parquet"))
        for index, path in enumerate(paths, start=1):
            table = pq.read_table(path)
            for window in _windows_from_parquet_table(table):
                by_book[window.book_id].append(window)
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(
                    f"load shards {split_name}: {index}/{len(paths)} files",
                    flush=True,
                )
        for book_id in by_book:
            by_book[book_id].sort(key=lambda w: w.window_index)
        result[split_name] = dict(by_book)
    return result


def expected_book_counts(cfg: dict[str, Any]) -> dict[str, int]:
    raw = cfg.get("expected_books") or {}
    defaults = {"train": 210, "validation": 25, "test": 19}
    return {name: int(raw.get(name) or defaults[name]) for name in defaults}


def shard_file_counts(out_dir: Path) -> dict[str, int]:
    shard_root = out_dir / "shards"
    return {
        name: len(list((shard_root / name).glob("*.parquet")))
        for name in ("train", "validation", "test")
    }


def existing_shards_ready(
    out_dir: Path,
    expected_books: dict[str, int],
    *,
    geom: Geometry | None = None,
) -> bool:
    """True when local parquet shards look like a finished build for this job."""
    shard_root = out_dir / "shards"
    if not shard_root.is_dir():
        return False
    counts = shard_file_counts(out_dir)
    if any(counts.get(name, 0) != expected_books[name] for name in expected_books):
        return False
    stats_path = out_dir / "window_stats.json"
    if not stats_path.is_file():
        return True
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("verification_ok") is False:
        return False
    if geom is None:
        return True
    saved = stats.get("geometry") or {}
    return (
        int(saved.get("max_length") or -1) == geom.max_length
        and int(saved.get("step") or -1) == geom.step
        and int(saved.get("hf_stride") or -1) == geom.hf_stride
    )


def _card_from_build(out_dir: Path) -> str:
    path = out_dir / "README_card_snippet.md"
    if not path.is_file():
        raise GateFailure(
            f"missing data card {path}; run a local build or --verify-only first"
        )
    return path.read_text(encoding="utf-8")


def _print_report(report: Report) -> None:
    print("\n" + "=" * 72)
    print("VERIFICATION GATES")
    print("=" * 72)
    for line in report.gate_lines:
        print(line)
    if report.failures:
        print("\nFAILURES:")
        for failure in report.failures:
            print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--config-name", default=None)
    parser.add_argument(
        "--push",
        action="store_true",
        help=(
            "Upload existing local shards to Hub. Skips tokenize/write when a "
            "complete build is already on disk. Combine with --verify-only to "
            "re-check gates before upload."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-run verification gates on existing local shards (no tokenize)",
    )
    parser.add_argument(
        "--ownership",
        choices=("first", "centre"),
        default="first",
        help="Token ownership rule (default: first)",
    )
    parser.add_argument("--limit-books", type=int, default=0)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cfg = load_job_config(args.config)
    geom = resolve_geometry(cfg)
    print(
        f"GEOMETRY: max_length={geom.max_length} step={geom.step} "
        f"hf_stride(overlap)={geom.hf_stride} overlap={geom.overlap_percent:.3f}%"
    )

    out_dir = Path(args.out or cfg["local_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    repo = str(args.repo or cfg["hub_repo"])
    config_name = str(args.config_name or cfg["hub_config_name"])
    local_source = ROOT / str(cfg.get("local_source_dir") or "Data/hf_export")
    dry_run = not args.push
    local_files_only = dry_run and not args.verify_only

    cfg["_ownership"] = args.ownership

    if cfg.get("damaged_books"):
        raise GateFailure(
            f"damaged_books must be empty for quotation windows; got {cfg['damaged_books']}"
        )

    expected_windows = int(cfg.get("expected_total_windows") or 7402)
    tolerance = float(cfg.get("window_count_tolerance") or 0.05)
    expected_books = expected_book_counts(cfg)
    reuse_shards = (not args.limit_books) and existing_shards_ready(
        out_dir, expected_books, geom=geom
    )

    def _load_source() -> tuple[Any, Benchmark]:
        tokenizer = load_tokenizer(
            str(cfg["tokenizer_repo"]),
            local_files_only=local_files_only,
        )
        benchmark = load_benchmark(
            str(cfg["source_repo"]),
            str(cfg.get("source_revision") or "main"),
            local_dir=local_source if local_source.is_dir() else None,
            local_files_only=dry_run,
        )
        return tokenizer, benchmark

    def _verify_existing(tokenizer: Any, benchmark: Benchmark) -> Report:
        shards = _load_shards(out_dir)
        windows_by_book: dict[str, list[Window]] = {}
        for split_books in shards.values():
            windows_by_book.update(split_books)
        report = verify(
            benchmark,
            windows_by_book,
            tokenizer,
            geom,
            ownership=args.ownership,
            expected_windows=expected_windows,
            window_tolerance=tolerance,
            limit_books=bool(args.limit_books),
        )
        _print_report(report)
        if not report.ok:
            raise GateFailure("; ".join(report.failures))
        return report

    if args.push and reuse_shards:
        counts = shard_file_counts(out_dir)
        print(
            f"Reusing existing shards at {out_dir / 'shards'} "
            f"(train={counts['train']} validation={counts['validation']} "
            f"test={counts['test']}); skip tokenize/write"
        )
        if args.verify_only:
            tokenizer, benchmark = _load_source()
            report = _verify_existing(tokenizer, benchmark)
            card = build_card_section(report, cfg, geom, tokenizer)
            (out_dir / "README_card_snippet.md").write_text(card, encoding="utf-8")
        else:
            card = _card_from_build(out_dir)
        push_to_hub(
            out_dir,
            repo,
            config_name,
            str(cfg.get("hub_tag") or "w1.0"),
            card_section=card,
        )
        return 0

    if args.verify_only:
        tokenizer, benchmark = _load_source()
        _verify_existing(tokenizer, benchmark)
        print("verify-only passed.")
        return 0

    if args.push and not reuse_shards:
        print(
            f"No complete local shards for this geometry at {out_dir / 'shards'}; "
            "building first, then pushing"
        )

    tokenizer, benchmark = _load_source()

    windows_by_split: dict[str, list[tuple[int, list[Window]]]] = {}
    windows_by_book = {}

    for split_name, books in benchmark.splits.items():
        if args.limit_books:
            books = books[: args.limit_books]
        split_entries: list[tuple[int, list[Window]]] = []
        for book_index, book in enumerate(books):
            windows = tokenize_book_windows(book, tokenizer, geom)
            project_labels(book, windows, args.ownership, tokenizer)
            split_entries.append((book_index, windows))
            windows_by_book[book.book_id] = windows
            print(
                f"[{split_name} {book_index + 1}/{len(books)}] {book.book_id} "
                f"windows={len(windows)}",
                flush=True,
            )
        windows_by_split[split_name] = split_entries

    report = verify(
        benchmark,
        windows_by_book,
        tokenizer,
        geom,
        ownership=args.ownership,
        expected_windows=expected_windows,
        window_tolerance=tolerance,
        limit_books=bool(args.limit_books),
    )
    _print_report(report)
    if not report.ok:
        raise GateFailure("; ".join(report.failures))

    write_shards(windows_by_split, out_dir)
    write_stats(
        report,
        out_dir,
        cfg=cfg,
        geom=geom,
        tokenizer=tokenizer,
        benchmark=benchmark,
        ownership=args.ownership,
    )
    card = build_card_section(report, cfg, geom, tokenizer)
    (out_dir / "README_card_snippet.md").write_text(card, encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"Windows: train={report.window_counts['train']} "
        f"validation={report.window_counts['validation']} "
        f"test={report.window_counts['test']} "
        f"total={report.measured_windows}"
    )
    print(
        "Class balance: "
        + ", ".join(f"{k}={v * 100:.3f}%" for k, v in report.class_balance.items())
    )
    print(f"Longest span: {report.max_span_tokens} tokens ({report.max_span_meta})")
    print(f"Severance: {report.severance} spans")
    print(f"Output: {out_dir}")

    if args.push:
        push_to_hub(
            out_dir,
            repo,
            config_name,
            str(cfg.get("hub_tag") or "w1.0"),
            card_section=card,
        )
    else:
        print("Dry-run: local shards written; no Hub upload.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as exc:
        print(f"GATE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

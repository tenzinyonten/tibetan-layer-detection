"""Training-time span resolution, token alignment, and window construction."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence

from .audit_utils import MODEL_LAYERS, PRIORITY_RANK, span_length

LAYER_TO_TAG = {
    "Quotation": "QUOTE",
    "Sabche": "SABCHE",
    "Tsawa": "TSAWA",
    "Yigchung": "YIGCHUNG",
    "Chapter": "CHAPTER",
    "Author": "AUTHOR",
    "BookTitle": "BOOKTITLE",
}
TAG_TO_LAYER = {tag: layer for layer, tag in LAYER_TO_TAG.items()}
UPPER_TO_LAYER = {layer.upper(): layer for layer in MODEL_LAYERS}

LABELS = ["O"] + [
    f"{prefix}-{tag}" for tag in LAYER_TO_TAG.values() for prefix in ("B", "I")
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def bio_spans(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Convert BIO labels to inclusive token spans.

    An orphaned ``I`` starts a new span. This is important for validation
    windows whose duplicated stride prefix is masked with ``-100``.
    """
    spans: list[tuple[str, int, int]] = []
    active_tag: str | None = None
    active_start = 0

    for index, label in enumerate([*labels, "O"]):
        if label == "O":
            prefix, tag = "O", None
        else:
            try:
                prefix, tag = label.split("-", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid BIO label: {label!r}") from exc
            if prefix not in {"B", "I"} or not tag:
                raise ValueError(f"Invalid BIO label: {label!r}")

        continues = prefix == "I" and tag == active_tag
        if active_tag is not None and not continues:
            spans.append((active_tag, active_start, index - 1))
            active_tag = None
        if tag is not None and active_tag is None:
            active_tag = tag
            active_start = index

    return spans


def soft_span_metrics(
    true_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    *,
    tolerance: int,
) -> dict[str, float]:
    """Return micro span metrics with boundary tolerance in tokens.

    A prediction matches at most one gold span of the same layer when both its
    start and end boundaries are within ``tolerance`` tokens.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if len(true_sequences) != len(predicted_sequences):
        raise ValueError("true and predicted sequence counts must match")

    true_positive = 0
    predicted_total = 0
    gold_total = 0
    for true_labels, predicted_labels in zip(
        true_sequences, predicted_sequences, strict=True
    ):
        gold = bio_spans(true_labels)
        predicted = bio_spans(predicted_labels)
        gold_total += len(gold)
        predicted_total += len(predicted)
        unmatched = set(range(len(gold)))

        for predicted_tag, predicted_start, predicted_end in predicted:
            candidates = [
                index
                for index in unmatched
                if gold[index][0] == predicted_tag
                and abs(gold[index][1] - predicted_start) <= tolerance
                and abs(gold[index][2] - predicted_end) <= tolerance
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda index: (
                    abs(gold[index][1] - predicted_start)
                    + abs(gold[index][2] - predicted_end),
                    index,
                ),
            )
            unmatched.remove(best)
            true_positive += 1

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def canonical_layer(label: str) -> str:
    """Convert a stored span label (for example ``QUOTATION``) to its layer name."""
    if label in MODEL_LAYERS:
        return label
    try:
        return UPPER_TO_LAYER[label.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown modeled layer label: {label!r}") from exc


def normalize_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize inclusive character spans."""
    normalized = []
    for index, span in enumerate(spans):
        start, end = int(span["start"]), int(span["end"])
        if start < 0 or end < start:
            raise ValueError(f"Invalid training span at index {index}: [{start}, {end}]")
        normalized.append(
            {
                "start": start,
                "end": end,
                "layer": canonical_layer(str(span["label"])),
                "_identity": index,
            }
        )
    return normalized


def pick_winning_span(
    spans: Iterable[dict[str, Any]], token_start: int, token_end: int
) -> dict[str, Any] | None:
    """Choose the label for one half-open token offset.

    Priority is applied only to the shared characters: Quotation wins first,
    otherwise the longer span wins, followed by the stable layer priority.
    This preserves the non-overlapping remainder of a lower-priority span.
    """
    candidates = [
        span
        for span in spans
        if token_start < span["end"] + 1 and span["start"] < token_end
    ]
    if not candidates:
        return None
    quotations = [span for span in candidates if span["layer"] == "Quotation"]
    pool = quotations or candidates
    return min(
        pool,
        key=lambda span: (
            -span_length(span["start"], span["end"]),
            PRIORITY_RANK.get(span["layer"], 99),
            span["_identity"],
        ),
    )


def scope_mask(in_scope_layers: Iterable[str]) -> list[int]:
    """Return labels allowed to contribute to this book's softmax/loss."""
    scope = {canonical_layer(layer) for layer in in_scope_layers}
    return [
        int(label == "O" or TAG_TO_LAYER[label[2:]] in scope)
        for label in LABELS
    ]


def align_window_labels(
    offset_mapping: Iterable[tuple[int, int]],
    special_tokens_mask: Iterable[int],
    spans: list[dict[str, Any]],
    *,
    already_covered_through: int = 0,
) -> tuple[list[int], int]:
    """Align one tokenizer window and suppress duplicated stride-prefix loss.

    Tokens repeated from an earlier window remain in the model input as
    context, but receive ``-100`` so overlapping windows do not count the same
    target twice.

    Content tokens are assumed to walk left-to-right in character space (true
    for Hugging Face overflow windows). Spans are swept once: a start-sorted
    cursor admits new candidates, and an active list drops those that have
    ended — O(tokens + spans) instead of O(tokens × spans).
    """
    labels: list[int] = []
    active_span: int | None = None
    furthest_end = already_covered_through

    by_start = sorted(spans, key=lambda span: (span["start"], span["_identity"]))
    next_span = 0
    active: list[dict[str, Any]] = []

    for (token_start, token_end), is_special in zip(
        offset_mapping, special_tokens_mask, strict=True
    ):
        token_start, token_end = int(token_start), int(token_end)
        if is_special or token_start == token_end:
            labels.append(-100)
            if is_special:
                active_span = None
            continue

        while next_span < len(by_start) and by_start[next_span]["start"] < token_end:
            active.append(by_start[next_span])
            next_span += 1
        if active:
            active = [span for span in active if token_start < span["end"] + 1]

        winner = pick_winning_span(active, token_start, token_end) if active else None
        if winner is None:
            label_id = LABEL2ID["O"]
            active_span = None
        else:
            tag = "I" if active_span == winner["_identity"] else "B"
            label_id = LABEL2ID[f"{tag}-{LAYER_TO_TAG[winner['layer']]}"]
            active_span = winner["_identity"]

        labels.append(-100 if token_end <= already_covered_through else label_id)
        furthest_end = max(furthest_end, token_end)

    return labels, furthest_end


def masked_label_ids(masked_layers: Iterable[str]) -> set[int]:
    """Return B/I label ids whose gold targets should be ignored in the loss."""
    ids: set[int] = set()
    for layer in masked_layers:
        tag = LAYER_TO_TAG[canonical_layer(layer)]
        ids.add(LABEL2ID[f"B-{tag}"])
        ids.add(LABEL2ID[f"I-{tag}"])
    return ids


def apply_masked_labels(
    labels: list[int], masked_layers: Iterable[str]
) -> list[int]:
    """Replace selected layer gold labels with ``-100`` (kept in the label set)."""
    ignore = masked_label_ids(masked_layers)
    if not ignore:
        return labels
    return [-100 if label in ignore else label for label in labels]


def tokenize_book_windows(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    stride: int,
    *,
    masked_layers: Iterable[str] = (),
    include_metadata: bool = False,
) -> dict[str, list[Any]]:
    """Tokenize one book into windows with aligned labels and scope masks.

    ``include_metadata`` adds per-window ``book_id``, ``window_index``,
    ``offset_mapping``, and ``special_tokens_mask``. Training must leave this
    false — those columns are not model inputs and would change the collator
    batch. Alignment and ``-100`` stride-prefix masking are unchanged.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride < 0 or stride >= max_length:
        raise ValueError("stride must satisfy 0 <= stride < max_length")

    spans = normalize_spans(row.get("spans", []))
    scope = {canonical_layer(layer) for layer in row.get("in_scope_layers", [])}
    out_of_scope_positives = sorted({span["layer"] for span in spans} - scope)
    if out_of_scope_positives:
        raise ValueError(
            f"{row.get('book_id', '<unknown>')} has positive spans outside its "
            f"in_scope_layers: {out_of_scope_positives}"
        )

    encoded = tokenizer(
        row["text"],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=False,
    )

    offsets_by_window = encoded["offset_mapping"]
    specials_by_window = encoded["special_tokens_mask"]
    labels_by_window = []
    covered_through = 0
    for offsets, specials in zip(offsets_by_window, specials_by_window, strict=True):
        labels, covered_through = align_window_labels(
            offsets,
            specials,
            spans,
            already_covered_through=covered_through,
        )
        labels_by_window.append(apply_masked_labels(labels, masked_layers))

    ignored_keys = {
        "overflow_to_sample_mapping",
        "num_truncated_tokens",
    }
    if not include_metadata:
        ignored_keys.update({"offset_mapping", "special_tokens_mask"})
    result = {
        key: value
        for key, value in encoded.items()
        if key not in ignored_keys
    }
    result["labels"] = labels_by_window
    result["scope_mask"] = [scope_mask(scope) for _ in labels_by_window]
    if include_metadata:
        n_windows = len(labels_by_window)
        result["book_id"] = [str(row.get("book_id", ""))] * n_windows
        result["window_index"] = list(range(n_windows))
    return result


def window_cache_key(
    row: dict[str, Any],
    *,
    tokenizer_name: str,
    max_length: int,
    stride: int,
    masked_layers: Iterable[str] = (),
) -> str:
    """Stable fingerprint for one book's tokenized windows on disk."""
    payload = {
        "book_id": row.get("book_id"),
        "text_sha256": hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest(),
        "spans": row.get("spans", []),
        "in_scope_layers": sorted(row.get("in_scope_layers", [])),
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "stride": stride,
        "masked_layers": sorted(canonical_layer(layer) for layer in masked_layers),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def book_window_cache_path(cache_dir: Path, book_id: str, cache_key: str) -> Path:
    """Per-book cache directory: ``cache_dir/<book_id>/<fingerprint>/``."""
    return Path(cache_dir) / str(book_id) / cache_key


def load_cached_book_windows(path: Path) -> dict[str, list[Any]] | None:
    """Load one book's windows from disk, or ``None`` if missing/corrupt."""
    from datasets import Dataset

    if not path.is_dir():
        return None
    try:
        dataset = Dataset.load_from_disk(str(path))
    except Exception:  # noqa: BLE001 — treat any cache read failure as a miss
        return None
    return {column: list(dataset[column]) for column in dataset.column_names}


def save_book_windows(path: Path, windows: dict[str, list[Any]]) -> None:
    """Atomically write one book's windows; drop stale fingerprints for the book."""
    from datasets import Dataset

    path = Path(path)
    book_dir = path.parent
    book_dir.mkdir(parents=True, exist_ok=True)
    tmp = book_dir / f".{path.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    Dataset.from_dict(windows).save_to_disk(str(tmp))
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    for sibling in book_dir.iterdir():
        if sibling.name == path.name or sibling.name.startswith("."):
            continue
        if sibling.is_dir():
            shutil.rmtree(sibling)


def _tokenize_row_with_cache(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    stride: int,
    *,
    masked_layers: Sequence[str],
    cache_dir: Path | None,
    tokenizer_name: str | None,
    include_metadata: bool = False,
) -> tuple[dict[str, list[Any]], bool]:
    """Return ``(windows, cache_hit)`` for one book."""
    cache_path: Path | None = None
    if cache_dir is not None:
        if include_metadata:
            raise ValueError(
                "include_metadata is incompatible with window cache: the "
                "fingerprint does not record that flag"
            )
        if not tokenizer_name:
            raise ValueError("tokenizer_name is required when cache_dir is set")
        cache_path = book_window_cache_path(
            cache_dir,
            str(row.get("book_id", "unknown")),
            window_cache_key(
                row,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                stride=stride,
                masked_layers=masked_layers,
            ),
        )
        cached = load_cached_book_windows(cache_path)
        if cached is not None:
            return cached, True

    windows = tokenize_book_windows(
        row,
        tokenizer,
        max_length,
        stride,
        masked_layers=masked_layers,
        include_metadata=include_metadata,
    )
    if cache_path is not None:
        save_book_windows(cache_path, windows)
    return windows, False


def build_windowed_dataset(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    stride: int,
    *,
    masked_layers: Iterable[str] = (),
    cache_dir: str | Path | None = None,
    tokenizer_name: str | None = None,
    num_workers: int = 1,
    desc: str | None = None,
    verbose: bool = True,
    include_metadata: bool = False,
):
    """Build a Hugging Face Dataset of token windows.

    When ``cache_dir`` is set, each book is stored under
    ``cache_dir/<book_id>/<fingerprint>/`` so a partial run can resume without
    re-tokenizing finished books. ``num_workers > 1`` tokenizes books in
    parallel (thread pool; HF tokenizers release the GIL).

    ``verbose`` reports each book as it lands, since alignment over long books
    can run for many minutes with no other output.
    """
    from datasets import Dataset
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar

    row_list = list(rows)
    masked = tuple(masked_layers)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    workers = max(1, int(num_workers))
    indexed = list(enumerate(row_list))
    total = len(row_list)
    label = f"{desc} " if desc else ""
    started = time.monotonic()
    done = 0
    progress_lock = Lock()

    def report(row: dict[str, Any], hit: bool, windows: dict[str, list[Any]]) -> None:
        nonlocal done
        with progress_lock:
            done += 1
            position = done
        elapsed = time.monotonic() - started
        eta = (elapsed / position) * (total - position) if position else 0.0
        print(
            f"[{label}tokenize {position}/{total}] "
            f"{row.get('book_id', '<unknown>')} "
            f"{'cached' if hit else 'tokenized'} "
            f"windows={len(windows.get('input_ids', []))} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    def process(index: int, row: dict[str, Any]) -> tuple[int, dict[str, list[Any]], bool]:
        windows, hit = _tokenize_row_with_cache(
            row,
            tokenizer,
            max_length,
            stride,
            masked_layers=masked,
            cache_dir=cache_path,
            tokenizer_name=tokenizer_name,
            include_metadata=include_metadata,
        )
        if verbose:
            report(row, hit, windows)
        return index, windows, hit

    if verbose:
        print(
            f"[{label}tokenize] {total} books, {workers} worker(s), "
            f"max_length={max_length} stride={stride}",
            file=sys.stderr,
            flush=True,
        )
    if cache_path is not None:
        disable_progress_bar()
    try:
        if workers == 1 or len(row_list) <= 1:
            ordered = [process(index, row) for index, row in indexed]
        else:
            ordered = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process, index, row) for index, row in indexed]
                for future in as_completed(futures):
                    ordered.append(future.result())
            ordered.sort(key=lambda item: item[0])
    finally:
        if cache_path is not None:
            enable_progress_bar()

    columns: dict[str, list[Any]] = {}
    cache_hits = 0
    for _index, windows, hit in ordered:
        cache_hits += int(hit)
        for key, values in windows.items():
            columns.setdefault(key, []).extend(values)

    if cache_path is not None:
        misses = len(row_list) - cache_hits
        print(
            f"window cache ({cache_path}): "
            f"{cache_hits} hit / {misses} miss of {len(row_list)} books"
        )
    return Dataset.from_dict(columns)


def class_weights_from_dataset(
    dataset: Any,
    cap: float = 5.0,
    *,
    o_weight: float | None = 0.07,
) -> list[float]:
    """Balanced inverse-frequency weights from aligned, non-duplicate targets.

    Frequency is pooled per layer, so ``B-X`` and ``I-X`` share one weight.
    Weighting them independently makes ``B`` tens of times heavier than ``I``
    (a span contributes one ``B`` and many ``I``), which rewards restarting a
    span over continuing it and shreds long layers into token-sized fragments.

    Weights are scaled so the *most common* layer sits at ``1.0`` and rarer
    layers rise from there. Scaling by the mean instead lets the rarest layers,
    whose inverse frequency is orders of magnitude above the rest, drag every
    common layer toward zero.

    ``o_weight`` is read on that same scale, so ``0.07`` means ``O`` counts for
    7% of a token in the most common layer. Anchoring to the minimum keeps
    ``o_weight < 1`` sufficient to hold ``O`` below *every* layer weight, no
    matter how skewed the corpus is. Unobserved layers stay neutral at ``1.0``
    — they are never targets, so their weight is inert.
    """
    counts: Counter[int] = Counter()
    for labels in dataset["labels"]:
        counts.update(label for label in labels if label != -100)

    total = sum(counts.values())
    if total == 0:
        raise ValueError("Training dataset contains no supervised tokens")

    o_id = LABEL2ID["O"]
    layer_counts = {
        tag: counts[LABEL2ID[f"B-{tag}"]] + counts[LABEL2ID[f"I-{tag}"]]
        for tag in LAYER_TO_TAG.values()
    }
    observed = {tag: count for tag, count in layer_counts.items() if count > 0}
    groups = len(observed) + (1 if counts[o_id] else 0)
    if not groups:
        raise ValueError("Training dataset contains no supervised tokens")

    raw = {tag: total / (groups * count) for tag, count in observed.items()}
    scale = min(raw.values()) if raw else 1.0

    weights = [1.0] * len(LABELS)
    for tag, value in raw.items():
        normalized = value / scale
        weights[LABEL2ID[f"B-{tag}"]] = normalized
        weights[LABEL2ID[f"I-{tag}"]] = normalized

    if o_weight is not None:
        weights[o_id] = float(o_weight)
    elif counts[o_id]:
        weights[o_id] = total / (groups * counts[o_id]) / scale

    return [min(weight, cap) for weight in weights]

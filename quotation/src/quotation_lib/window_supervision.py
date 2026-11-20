"""Window-level supervision fields for the derived dataset.

Builds ``spans``, ``start_targets``, ``end_targets``, and ``loss_mask`` on top
of an unchanged ``tokenize_book_windows`` BIO path. Gold character offsets stay
inclusive ``[start, end]`` (``text[start:end+1]``). Window token indices are
half-open.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit_utils import MODEL_LAYERS, PRIORITY_RANK
from .training_utils import (
    ID2LABEL,
    LABEL2ID,
    LAYER_TO_TAG,
    TAG_TO_LAYER,
    canonical_layer,
)

DEFAULT_WINDOW_LEN = 8192
DEFAULT_STEP = 4916  # 40% overlap; HF stride = window_len - step
DEFAULT_SMOOTHING = (1.0, 0.3, 0.1)  # center, ±1, ±2
DEFAULT_MIN_SPAN_CHARS = 5
DEFAULT_DAMAGED_BOOKS = ("P000218", "P000271", "P000078", "P000126")
MAX_EXPECTED_SPAN_TOKENS = 3125

# Earlier joint-model pipeline (unpinned Hub default, HF overlap 5120).
JOINT_MODEL_HF_STRIDE = 5120
JOINT_MODEL_STEP = 3072
JOINT_MODEL_WINDOWS = 14338
JOINT_MODEL_DATASET_REVISION = None

# Target for this job. 9103 and 8906 are known-wrong formulas — do not use.
TARGET_WINDOWS = 9060
WRONG_MODEL_WINDOWS = 9103
NAIVE_SCALE_WINDOWS = 8906

LAYER_INDEX = {layer: index for index, layer in enumerate(MODEL_LAYERS)}
HUB_LABEL = {layer: layer.upper() for layer in MODEL_LAYERS}
HUB_LABEL["Quotation"] = "QUOTATION"
BOOKTITLE_LABEL_IDS = {LABEL2ID["B-BOOKTITLE"], LABEL2ID["I-BOOKTITLE"]}
O_ID = LABEL2ID["O"]


class VerificationError(Exception):
    """A verification gate failed; the caller must not push."""


def parse_smoothing(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parts = [float(item) for item in value]
    if not parts:
        raise ValueError("smoothing must have at least a center weight")
    return tuple(parts)


def parse_damaged_books(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_DAMAGED_BOOKS
    if isinstance(value, str):
        books = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        books = tuple(str(item).strip() for item in value if str(item).strip())
    return books or DEFAULT_DAMAGED_BOOKS


def resolve_windowing(
    window_len: int,
    step: int,
) -> tuple[int, int, int]:
    """Return ``(window_len, step, hf_stride)``. ``step`` is the hop, not HF overlap."""
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    if step >= window_len or step <= 0:
        raise ValueError(
            f"stride/step must satisfy 0 < step < window_len; "
            f"got step={step} window_len={window_len}"
        )
    hf_stride = window_len - step
    if hf_stride < 0:
        raise ValueError(f"hf_stride={hf_stride} is negative")
    return window_len, step, hf_stride


def inclusive_length(start: int, end: int) -> int:
    return end - start + 1


def span_id_for(
    book_id: str,
    hub_label: str,
    char_start: int,
    char_end: int,
    source_index: int,
) -> str:
    return f"{book_id}:{hub_label}:{char_start}:{char_end}:{source_index}"


def parse_span_id(span_id: str) -> tuple[str, str, int, int, int]:
    book_id, rest = span_id.split(":", 1)
    hub_label, char_start, char_end, source_index = rest.rsplit(":", 3)
    return book_id, hub_label, int(char_start), int(char_end), int(source_index)


def filter_supervision_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    min_span_chars: int = DEFAULT_MIN_SPAN_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Count then drop invalid / short gold spans. Does not merge overlaps."""
    kept: list[dict[str, Any]] = []
    counts = {"invalid": 0, "short": 0, "kept": 0, "total": len(spans)}
    for index, span in enumerate(spans):
        start, end = int(span["start"]), int(span["end"])
        if start > end or start < 0:
            counts["invalid"] += 1
            continue
        if inclusive_length(start, end) < min_span_chars:
            counts["short"] += 1
            continue
        label = str(span["label"])
        kept.append(
            {
                "source_index": index,
                "char_start": start,
                "char_end": end,
                "hub_label": label,
                "layer": canonical_layer(label),
            }
        )
        counts["kept"] += 1
    return kept, counts


def content_token_offsets(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Return ``(token_index, char_start, char_end_excl)`` for content tokens."""
    out: list[tuple[int, int, int]] = []
    for index, ((start, end), is_special) in enumerate(zip(offsets, specials, strict=True)):
        start, end = int(start), int(end)
        if is_special or start == end:
            continue
        out.append((index, start, end))
    return out


def window_char_range(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
) -> tuple[int, int] | None:
    content = content_token_offsets(offsets, specials)
    if not content:
        return None
    return content[0][1], content[-1][2]


def _content_index(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    content = content_token_offsets(offsets, specials)
    starts = [tok_s for _index, tok_s, _tok_e in content]
    ends = [tok_e for _index, _tok_s, tok_e in content]
    return content, starts, ends


def _token_containing_indexed(
    content: Sequence[tuple[int, int, int]],
    starts: Sequence[int],
    char_pos: int,
) -> int | None:
    if not content:
        return None
    index = bisect.bisect_right(starts, char_pos) - 1
    if index < 0:
        return None
    tok_i, tok_s, tok_e = content[index]
    if tok_s <= char_pos < tok_e:
        return tok_i
    return None


def _overlapping_range_indexed(
    content: Sequence[tuple[int, int, int]],
    starts: Sequence[int],
    ends: Sequence[int],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    if not content:
        return None
    left = bisect.bisect_right(ends, char_start)
    right = bisect.bisect_left(starts, char_end + 1)
    if left >= right:
        return None
    return content[left][0], content[right - 1][0] + 1


def token_containing_char(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    char_pos: int,
) -> int | None:
    content, starts, _ends = _content_index(offsets, specials)
    return _token_containing_indexed(content, starts, char_pos)


def overlapping_token_range(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    """Half-open window token indices overlapping inclusive ``[char_start, char_end]``.

    Mid-token gold boundaries round outward: the start token is the one
    containing the start char, and the exclusive end is one past the token
    containing the end char.
    """
    content, starts, ends = _content_index(offsets, specials)
    return _overlapping_range_indexed(content, starts, ends, char_start, char_end)


def mid_token_rounds(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    char_start: int,
    char_end: int,
) -> int:
    """Count start/end boundaries that fall inside a token rather than on its edge."""
    content, starts, _ends = _content_index(offsets, specials)
    rounds = 0
    start_tok = _token_containing_indexed(content, starts, char_start)
    if start_tok is not None:
        tok_s = int(offsets[start_tok][0])
        if char_start != tok_s:
            rounds += 1
    end_tok = _token_containing_indexed(content, starts, char_end)
    if end_tok is not None:
        tok_e = int(offsets[end_tok][1])
        if char_end != tok_e - 1:
            rounds += 1
    return rounds


def merge_char_intervals(
    gold_spans: Sequence[Mapping[str, Any]],
    *,
    layer: str | None = None,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for gold in gold_spans:
        if layer is not None and gold["layer"] != layer:
            continue
        intervals.append((int(gold["char_start"]), int(gold["char_end"]) + 1))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_overlaps(merged: Sequence[tuple[int, int]], tok_s: int, tok_e: int) -> bool:
    if not merged:
        return False
    starts = [start for start, _end in merged]
    index = bisect.bisect_left(starts, tok_e) - 1
    return index >= 0 and merged[index][1] > tok_s


def gold_overlaps_window(
    char_start: int,
    char_end: int,
    window_start: int,
    window_end_excl: int,
) -> bool:
    return char_start < window_end_excl and char_end >= window_start


def gold_fully_inside(
    char_start: int,
    char_end: int,
    window_start: int,
    window_end_excl: int,
) -> bool:
    return window_start <= char_start and char_end < window_end_excl


def paint_edge(
    row: Any,
    center: int,
    smoothing: Sequence[float],
) -> None:
    """Paint center / ±1 / ±2 (or longer) with ``max()``, clipped to the row."""
    n = len(row)
    for radius, weight in enumerate(smoothing):
        weight = float(weight)
        positions = (center,) if radius == 0 else (center - radius, center + radius)
        for pos in positions:
            if 0 <= pos < n:
                row[pos] = min(1.0, max(float(row[pos]), weight))


def count_same_layer_overlap_pairs(window_spans: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for i, left in enumerate(window_spans):
        for right in window_spans[i + 1 :]:
            if left["layer"] != right["layer"]:
                continue
            if left["start"] < right["end"] and right["start"] < left["end"]:
                count += 1
    return count


def token_overlaps_any_gold(
    tok_s: int,
    tok_e: int,
    gold_spans: Sequence[Mapping[str, Any]],
    *,
    layer: str | None = None,
) -> bool:
    return interval_overlaps(merge_char_intervals(gold_spans, layer=layer), tok_s, tok_e)


def build_loss_mask(
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    labels: Sequence[int],
    gold_spans: Sequence[Mapping[str, Any]],
    *,
    damaged: bool,
) -> tuple[list[float], dict[str, int]]:
    n = len(labels)
    mask = np.ones(n, dtype=np.float32)
    zeros = {"special_pad": 0, "booktitle": 0, "damaged_o": 0}
    booktitle_iv = merge_char_intervals(gold_spans, layer="BookTitle")
    any_iv = merge_char_intervals(gold_spans) if damaged else []
    for index, ((tok_s, tok_e), is_special, label) in enumerate(
        zip(offsets, specials, labels, strict=True)
    ):
        tok_s, tok_e = int(tok_s), int(tok_e)
        if is_special or tok_s == tok_e:
            mask[index] = 0.0
            zeros["special_pad"] += 1
            continue
        is_booktitle = int(label) in BOOKTITLE_LABEL_IDS or interval_overlaps(
            booktitle_iv, tok_s, tok_e
        )
        if is_booktitle:
            mask[index] = 0.0
            zeros["booktitle"] += 1
            continue
        if damaged and not interval_overlaps(any_iv, tok_s, tok_e):
            mask[index] = 0.0
            zeros["damaged_o"] += 1
    return mask, zeros


def build_window_fields(
    *,
    book_id: str,
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    labels: Sequence[int],
    gold_spans: Sequence[Mapping[str, Any]],
    damaged: bool,
    smoothing: Sequence[float] = DEFAULT_SMOOTHING,
) -> dict[str, Any]:
    """Build supervision columns for one window. ``gold_spans`` are already filtered."""
    window_len = len(labels)
    char_range = window_char_range(offsets, specials)
    start_targets = np.zeros((len(MODEL_LAYERS), window_len), dtype=np.float32)
    end_targets = np.zeros((len(MODEL_LAYERS), window_len), dtype=np.float32)
    window_spans: list[dict[str, Any]] = []
    n_mid_rounds = 0
    n_overlong = 0

    if char_range is not None:
        window_start, window_end_excl = char_range
        content, tok_starts, tok_ends = _content_index(offsets, specials)
        for gold in gold_spans:
            char_start = int(gold["char_start"])
            char_end = int(gold["char_end"])
            if not gold_overlaps_window(
                char_start, char_end, window_start, window_end_excl
            ):
                continue
            tok_range = _overlapping_range_indexed(
                content, tok_starts, tok_ends, char_start, char_end
            )
            if tok_range is None:
                continue
            tok_start, tok_end = tok_range
            n_tokens = tok_end - tok_start
            if n_tokens > MAX_EXPECTED_SPAN_TOKENS:
                n_overlong += 1
            if n_tokens > window_len:
                raise VerificationError(
                    f"{book_id}: gold span tokens {n_tokens} exceed window {window_len}"
                )
            truncated = not gold_fully_inside(
                char_start, char_end, window_start, window_end_excl
            )
            start_tok = _token_containing_indexed(content, tok_starts, char_start)
            end_tok = _token_containing_indexed(content, tok_starts, char_end)
            if start_tok is not None and char_start != int(offsets[start_tok][0]):
                n_mid_rounds += 1
            if end_tok is not None and char_end != int(offsets[end_tok][1]) - 1:
                n_mid_rounds += 1
            hub_label = str(gold["hub_label"])
            window_spans.append(
                {
                    "layer": hub_label,
                    "start": tok_start,
                    "end": tok_end,
                    "truncated": truncated,
                    "span_id": span_id_for(
                        book_id,
                        hub_label,
                        char_start,
                        char_end,
                        int(gold["source_index"]),
                    ),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            layer_i = LAYER_INDEX[gold["layer"]]
            if start_tok is not None:
                paint_edge(start_targets[layer_i], start_tok, smoothing)
            if end_tok is not None:
                paint_edge(end_targets[layer_i], end_tok, smoothing)

    loss_mask, zero_counts = build_loss_mask(
        offsets, specials, labels, gold_spans, damaged=damaged
    )
    return {
        "spans": window_spans,
        "start_targets": start_targets,
        "end_targets": end_targets,
        "loss_mask": loss_mask,
        "n_mid_token_rounds": n_mid_rounds,
        "n_same_layer_overlaps": count_same_layer_overlap_pairs(window_spans),
        "n_truncated": sum(1 for span in window_spans if span["truncated"]),
        "n_overlong_spans": n_overlong,
        "loss_mask_zeros": zero_counts,
    }


def attach_window_supervision(
    windows: dict[str, list[Any]],
    row: Mapping[str, Any],
    *,
    damaged_books: Iterable[str] = DEFAULT_DAMAGED_BOOKS,
    smoothing: Sequence[float] = DEFAULT_SMOOTHING,
    min_span_chars: int = DEFAULT_MIN_SPAN_CHARS,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Attach supervision fields to one book's windows. Does not touch BIO ``labels``."""
    book_id = str(row.get("book_id", ""))
    kept, drop_counts = filter_supervision_spans(
        list(row.get("spans") or []),
        min_span_chars=min_span_chars,
    )
    damaged = book_id in set(damaged_books)
    n_windows = len(windows["input_ids"])
    spans_col: list[list[dict[str, Any]]] = []
    start_col: list[list[list[float]]] = []
    end_col: list[list[list[float]]] = []
    mask_col: list[list[float]] = []
    zeros = {"special_pad": 0, "booktitle": 0, "damaged_o": 0}
    n_mid = 0
    n_same_layer = 0
    n_truncated = 0
    n_overlong = 0
    for index in range(n_windows):
        fields = build_window_fields(
            book_id=book_id,
            offsets=windows["offset_mapping"][index],
            specials=windows["special_tokens_mask"][index],
            labels=windows["labels"][index],
            gold_spans=kept,
            damaged=damaged,
            smoothing=smoothing,
        )
        spans_col.append(fields["spans"])
        start_col.append(fields["start_targets"])
        end_col.append(fields["end_targets"])
        mask_col.append(fields["loss_mask"])
        n_mid += fields["n_mid_token_rounds"]
        n_same_layer += fields["n_same_layer_overlaps"]
        n_truncated += fields["n_truncated"]
        n_overlong += fields["n_overlong_spans"]
        for key, value in fields["loss_mask_zeros"].items():
            zeros[key] += value
    windows["spans"] = spans_col
    windows["start_targets"] = start_col
    windows["end_targets"] = end_col
    windows["loss_mask"] = mask_col
    if "book_id" not in windows:
        windows["book_id"] = [book_id] * n_windows
    stats = {
        "book_id": book_id,
        "drop_counts": drop_counts,
        "n_mid_token_rounds": n_mid,
        "n_same_layer_overlaps": n_same_layer,
        "n_truncated_span_rows": n_truncated,
        "n_overlong_spans": n_overlong,
        "loss_mask_zeros": zeros,
        "n_gold_kept": drop_counts["kept"],
        "damaged": damaged,
    }
    return windows, stats


def windows_for_length(token_length: int, max_length: int, step: int) -> int:
    if token_length <= max_length:
        return 1
    return 1 + math.ceil((token_length - max_length) / step)


def predicted_window_total(
    token_lengths: Sequence[int],
    max_length: int,
    step: int,
) -> int:
    return sum(windows_for_length(length, max_length, step) for length in token_lengths)


def offset_integrity_by_book_layer(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (book, hub-label) fraction of spans with ``end >= len(text)`` or ``start < 0``."""
    report: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        book_id = str(row["book_id"])
        text_len = len(row["text"])
        totals: dict[str, int] = defaultdict(int)
        fails: dict[str, int] = defaultdict(int)
        for span in row.get("spans") or []:
            label = str(span["label"])
            start, end = int(span["start"]), int(span["end"])
            totals[label] += 1
            if start < 0 or end >= text_len:
                fails[label] += 1
        for label, total in totals.items():
            fail = fails[label]
            report[(book_id, label)] = {
                "total": total,
                "fail": fail,
                "rate": fail / total if total else 0.0,
            }
    return report


def bio_agreement_for_window(
    window_spans: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    specials: Sequence[int],
) -> dict[str, int]:
    """Compare non-collapsed ``spans`` to winner-takes-all BIO ``labels``.

    A token labelled a higher-priority layer is a collision, not a failure.
    Contradiction: supervised ``O`` (or a lower-priority layer) on a span token.
    """
    collisions = 0
    contradictions = 0
    for span in window_spans:
        layer = canonical_layer(str(span["layer"]))
        tag = LAYER_TO_TAG[layer]
        expected = {LABEL2ID[f"B-{tag}"], LABEL2ID[f"I-{tag}"]}
        rank = PRIORITY_RANK[layer]
        for tok in range(int(span["start"]), int(span["end"])):
            if tok < 0 or tok >= len(labels):
                contradictions += 1
                continue
            if specials[tok]:
                continue
            label = int(labels[tok])
            if label == -100 or label in expected:
                continue
            if label == O_ID:
                contradictions += 1
                continue
            other_name = ID2LABEL.get(label, "")
            if "-" not in other_name:
                contradictions += 1
                continue
            other_tag = other_name.split("-", 1)[1]
            other_layer = TAG_TO_LAYER.get(other_tag)
            if other_layer is None:
                contradictions += 1
                continue
            if PRIORITY_RANK.get(other_layer, 99) < rank:
                collisions += 1
            else:
                contradictions += 1
    return {"collisions": collisions, "contradictions": contradictions}


def peak_positions(row: Sequence[float], threshold: float = 1.0) -> list[int]:
    return [index for index, value in enumerate(row) if float(value) >= threshold]


def check_target_peaks_for_window(
    window_spans: Sequence[Mapping[str, Any]],
    start_targets: Sequence[Sequence[float]],
    end_targets: Sequence[Sequence[float]],
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
) -> dict[str, int]:
    """In-window edges must produce a 1.0 peak; extra 1.0 peaks are a mismatch."""
    expected_starts: dict[int, set[int]] = defaultdict(set)
    expected_ends: dict[int, set[int]] = defaultdict(set)
    for span in window_spans:
        layer_i = LAYER_INDEX[canonical_layer(str(span["layer"]))]
        start_tok = token_containing_char(offsets, specials, int(span["char_start"]))
        end_tok = token_containing_char(offsets, specials, int(span["char_end"]))
        if start_tok is not None:
            expected_starts[layer_i].add(start_tok)
        if end_tok is not None:
            expected_ends[layer_i].add(end_tok)
    matched = 0
    missing = 0
    extra = 0
    for layer_i in range(len(MODEL_LAYERS)):
        got_s = set(peak_positions(start_targets[layer_i]))
        got_e = set(peak_positions(end_targets[layer_i]))
        want_s = expected_starts[layer_i]
        want_e = expected_ends[layer_i]
        matched += len(got_s & want_s) + len(got_e & want_e)
        missing += len(want_s - got_s) + len(want_e - got_e)
        extra += len(got_s - want_s) + len(got_e - want_e)
    return {"matched": matched, "missing": missing, "extra": extra}


def round_trip_span(
    text: str,
    offsets: Sequence[Sequence[int]],
    specials: Sequence[int],
    span: Mapping[str, Any],
) -> tuple[bool, str, str]:
    char_start = int(span["char_start"])
    char_end = int(span["char_end"])
    gold = text[char_start : char_end + 1]
    decoded_start = None
    decoded_end = None
    for tok in range(int(span["start"]), int(span["end"])):
        if tok < 0 or tok >= len(offsets):
            continue
        if specials[tok]:
            continue
        tok_s, tok_e = int(offsets[tok][0]), int(offsets[tok][1])
        if tok_s == tok_e:
            continue
        if decoded_start is None:
            decoded_start = tok_s
        decoded_end = tok_e
    reconstructed = (
        text[decoded_start:decoded_end] if decoded_start is not None and decoded_end is not None else ""
    )
    gold_cmp = gold.replace(" ", "")
    recon_cmp = reconstructed.replace(" ", "")
    # Outward mid-token rounding makes the token slice a superstring of gold.
    covers = (
        decoded_start is not None
        and decoded_end is not None
        and decoded_start <= char_start
        and decoded_end >= char_end + 1
    )
    ok = covers and (gold_cmp == recon_cmp or gold_cmp in recon_cmp or recon_cmp in gold_cmp)
    return ok, gold, reconstructed


def format_verification_table(rows: Sequence[tuple[str, str, str]]) -> str:
    width_gate = max(len("Gate"), *(len(row[0]) for row in rows)) if rows else 4
    width_value = max(len("Value"), *(len(row[1]) for row in rows)) if rows else 5
    header = f"{'Gate':<{width_gate}}  {'Value':<{width_value}}  Status"
    rule = "-" * len(header)
    body = "\n".join(
        f"{gate:<{width_gate}}  {value:<{width_value}}  {status}"
        for gate, value, status in rows
    )
    return f"{header}\n{rule}\n{body}"


def window_count_status(
    measured: int,
    *,
    predicted: int | None,
    expected: int = TARGET_WINDOWS,
    tolerance: float = 0.03,
    limit_books: bool = False,
) -> tuple[str, str | None]:
    """Return ``(status, failure_or_none)``. Predicted formula explains ceil drift."""
    if limit_books:
        return "SKIP (limit-books)", None
    rel = abs(measured - expected) / expected
    explained = predicted is not None and measured == predicted
    d_good = abs(measured - expected)
    closer_wrong = abs(measured - WRONG_MODEL_WINDOWS) < d_good
    closer_naive = abs(measured - NAIVE_SCALE_WINDOWS) < d_good
    if rel > tolerance:
        return (
            "FAIL",
            f"window count {measured} is more than {tolerance:.0%} off {expected}",
        )
    if closer_wrong and not explained:
        return (
            "FAIL",
            f"window count {measured} is closer to false-model {WRONG_MODEL_WINDOWS} "
            f"than to {expected} (no per-book formula match)",
        )
    if closer_naive and not explained:
        return (
            "FAIL",
            f"window count {measured} is closer to naive-scale {NAIVE_SCALE_WINDOWS} "
            f"than to {expected} (no per-book HF overflow match)",
        )
    # A measured total that equals the naive 8906 is allowed when it is the
    # actual HF overflow count (per-book L_i), not 14338×3072/4916.
    return "PASS", None


def verify_window_supervision(
    split_windows: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    token_lengths: Sequence[int],
    window_len: int = DEFAULT_WINDOW_LEN,
    step: int = DEFAULT_STEP,
    expected_windows: int = TARGET_WINDOWS,
    tolerance: float = 0.03,
    min_span_chars: int = DEFAULT_MIN_SPAN_CHARS,
    rng_seed: int = 42,
    limit_books: bool = False,
    n_round_trip: int = 5,
) -> dict[str, Any]:
    """Run dataset-level gates. ``split_windows`` maps split name → iterable of rows."""
    table: list[tuple[str, str, str]] = []
    failures: list[str] = []

    def fail(gate: str, value: str, message: str) -> None:
        table.append((gate, value, "FAIL"))
        failures.append(message)

    def ok(gate: str, value: str) -> None:
        table.append((gate, value, "PASS"))

    def info(gate: str, value: str) -> None:
        table.append((gate, value, "INFO"))

    pred_new = predicted_window_total(token_lengths, window_len, step)
    pred_old = predicted_window_total(token_lengths, window_len, JOINT_MODEL_STEP)
    source_by_id = {str(row["book_id"]): row for row in source_rows}
    gold_total = sum(len(row.get("spans") or []) for row in source_rows)
    kept_total = 0
    dropped_short = 0
    dropped_invalid = 0
    for row in source_rows:
        _kept, counts = filter_supervision_spans(
            list(row.get("spans") or []), min_span_chars=min_span_chars
        )
        kept_total += counts["kept"]
        dropped_short += counts["short"]
        dropped_invalid += counts["invalid"]

    by_id: dict[str, list[bool]] = defaultdict(list)
    n_trunc_rows = 0
    n_windows_with_trunc = 0
    zeros = {"special_pad": 0, "booktitle": 0, "damaged_o": 0}
    peak_matched = 0
    peak_missing = 0
    peak_extra = 0
    collisions = 0
    contradictions = 0
    reservoir: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen_candidates = 0
    rng = random.Random(rng_seed)
    same_layer = 0
    bio_window_used = False
    measured = 0

    for split_name in ("train", "validation", "test"):
        for window in split_windows.get(split_name, []):
            measured += 1
            spans = list(window.get("spans") or [])
            if any(span.get("truncated") for span in spans):
                n_windows_with_trunc += 1
            n_trunc_rows += sum(1 for span in spans if span.get("truncated"))
            for span in spans:
                by_id[str(span["span_id"])].append(bool(span.get("truncated")))
                if span.get("truncated"):
                    continue
                seen_candidates += 1
                slim = {
                    "book_id": window.get("book_id"),
                    "offset_mapping": window.get("offset_mapping"),
                    "special_tokens_mask": window.get("special_tokens_mask"),
                }
                item = (slim, span)
                if len(reservoir) < n_round_trip:
                    reservoir.append(item)
                else:
                    replace_at = rng.randrange(seen_candidates)
                    if replace_at < n_round_trip:
                        reservoir[replace_at] = item
            same_layer += count_same_layer_overlap_pairs(spans)
            mask = window.get("loss_mask") or []
            labels = window.get("labels") or []
            specials = window.get("special_tokens_mask") or [0] * len(mask)
            offsets = window.get("offset_mapping") or []
            for index, weight in enumerate(mask):
                if float(weight) != 0.0:
                    continue
                tok_s, tok_e = (
                    (int(offsets[index][0]), int(offsets[index][1])) if offsets else (0, 0)
                )
                if index < len(specials) and (specials[index] or tok_s == tok_e):
                    zeros["special_pad"] += 1
                elif index < len(labels) and int(labels[index]) in BOOKTITLE_LABEL_IDS:
                    zeros["booktitle"] += 1
                else:
                    zeros["damaged_o"] += 1
            if offsets and "start_targets" in window:
                peaks = check_target_peaks_for_window(
                    spans,
                    window["start_targets"],
                    window["end_targets"],
                    offsets,
                    specials,
                )
                peak_matched += peaks["matched"]
                peak_missing += peaks["missing"]
                peak_extra += peaks["extra"]
            if spans and not bio_window_used:
                agree = bio_agreement_for_window(spans, labels, specials)
                collisions += agree["collisions"]
                contradictions += agree["contradictions"]
                bio_window_used = True

    info("Joint-model windows (step 3072, measured live)", str(JOINT_MODEL_WINDOWS))
    info("Joint-model windows (step 3072, predicted from L_i)", str(pred_old))
    info(f"new windows (step {step}, measured)", str(measured))
    info(f"new windows (step {step}, predicted from L_i)", str(pred_new))
    info("target (~9060, not 9103 / 8906)", str(expected_windows))
    status, err = window_count_status(
        measured,
        predicted=pred_new,
        expected=expected_windows,
        tolerance=tolerance,
        limit_books=limit_books,
    )
    if err:
        fail("window count", f"{measured} vs {expected_windows}", err)
    else:
        table.append(("window count", f"{measured} vs {expected_windows}", status))
    info("source gold spans", str(gold_total))
    info(
        "supervision spans kept / short-drop / invalid-drop",
        f"{kept_total} / {dropped_short} / {dropped_invalid}",
    )

    recovered = sum(1 for flags in by_id.values() if any(not flag for flag in flags))
    unrecoverable = sorted(
        span_id for span_id, flags in by_id.items() if flags and all(flags)
    )
    n_ids = len(by_id)
    info("windows with a truncated=True span", str(n_windows_with_trunc))
    info("truncated span-rows", str(n_trunc_rows))
    info("unique gold span_ids in windows", str(n_ids))
    if n_ids == 0 and not limit_books:
        fail("unrecoverable spans", "0 span_ids", "no spans were emitted")
    elif unrecoverable:
        fail(
            "unrecoverable spans",
            f"{len(unrecoverable)} / {n_ids}",
            f"{len(unrecoverable)} gold span_id(s) truncated in every window "
            f"(stride is wrong). First: {unrecoverable[:3]}",
        )
    else:
        ok("span_ids non-truncated in ≥1 window", f"{recovered}/{n_ids} (100%)")

    integrity = offset_integrity_by_book_layer(source_rows)
    worst_rate = 0.0
    worst_key = ("", "")
    if integrity:
        worst_key, worst_item = max(integrity.items(), key=lambda kv: kv[1]["rate"])
        worst_rate = worst_item["rate"]
    info(
        "offset-integrity worst book/layer",
        f"{worst_key[0]} {worst_key[1]} {worst_rate:.4%}" if worst_key[0] else "n/a",
    )
    bad = {key: item for key, item in integrity.items() if item["rate"] > 0.01}
    if bad:
        preview = next(iter(bad.items()))
        fail(
            "offset integrity",
            f"{len(bad)} book/layer >1%",
            f"offset-integrity failure rate >1% at {preview[0]}: {preview[1]}",
        )
    else:
        ok("offset integrity (all book/layer ≤1%)", f"worst={worst_rate:.4%}")

    info(
        "loss_mask==0 special/pad / booktitle / damaged_O",
        f"{zeros['special_pad']} / {zeros['booktitle']} / {zeros['damaged_o']}",
    )
    info("same-layer overlap pairs kept", str(same_layer))

    if peak_missing or peak_extra:
        fail(
            "start/end target peaks",
            f"matched={peak_matched} missing={peak_missing} extra={peak_extra}",
            "start/end 1.0 peaks do not match in-window edges",
        )
    else:
        ok("start/end target peaks vs in-window edges", f"matched={peak_matched}")

    info("BIO collisions (higher-priority winner)", str(collisions))
    if contradictions:
        fail("BIO agreement", f"contradictions={contradictions}", "BIO contradiction")
    else:
        ok("BIO agreement (contradictions)", "0")

    checked = 0
    failed_rt = 0
    reports: list[str] = []
    for window, span in reservoir:
        if checked >= n_round_trip:
            break
        book_id = str(window.get("book_id") or parse_span_id(str(span["span_id"]))[0])
        row = source_by_id.get(book_id)
        if row is None:
            continue
        ok_rt, gold, reconstructed = round_trip_span(
            row["text"],
            window["offset_mapping"],
            window["special_tokens_mask"],
            span,
        )
        line = (
            f"{book_id} {span['span_id']} gold={gold!r} "
            f"offsets={reconstructed!r} match={ok_rt}"
        )
        reports.append(line)
        checked += 1
        if not ok_rt:
            failed_rt += 1
    if checked < n_round_trip and not limit_books:
        fail(
            "round-trip",
            f"{checked}/{n_round_trip}",
            f"only {checked} non-truncated spans available for round-trip",
        )
    elif failed_rt:
        failed_line = next((line for line in reports if "match=False" in line), reports[0])
        fail("round-trip", f"{checked - failed_rt}/{checked}", failed_line)
    else:
        ok("round-trip non-truncated spans", f"{checked}/{n_round_trip}")

    extra_drop = 100.0 * (JOINT_MODEL_WINDOWS - measured) / JOINT_MODEL_WINDOWS if measured else 0.0
    info(
        "windows vs joint model 14338 (extras drop ~37.5%; 266 first windows stay)",
        f"{extra_drop:.1f}% fewer",
    )
    info(
        "LR / warmup",
        "rescale warmup and schedule — same nominal epoch count undertrains",
    )

    return {
        "table": table,
        "table_text": format_verification_table(table),
        "failures": failures,
        "ok": not failures,
        "measured_windows": measured,
        "predicted_windows": pred_new,
        "predicted_joint_model_windows": pred_old,
        "round_trip": reports,
        "unrecoverable": unrecoverable,
        "loss_mask_zeros": zeros,
        "integrity_worst_rate": worst_rate,
        "collisions": collisions,
        "contradictions": contradictions,
        "n_gold_source": gold_total,
        "n_gold_kept": kept_total,
        "n_dropped_short": dropped_short,
        "n_dropped_invalid": dropped_invalid,
        "n_windows_with_trunc": n_windows_with_trunc,
    }

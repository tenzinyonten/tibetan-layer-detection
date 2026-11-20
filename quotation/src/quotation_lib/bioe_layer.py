"""BIOE annotation layer utilities and metrics for Tibetan Quotation Detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .training_utils import bio_spans
from .span_metrics import match_iou, prf

NEG = -1.0e9
DEFAULT_BREAK_PENALTY = 2.0
DEFAULT_IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# label scheme
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BioeScheme:
    """The 4-label scheme for BIOE quotation detection."""

    layer: str = "Quotation"
    tag: str = "QUOTE"
    labels: tuple[str, ...] = ("O", "B-QUOTE", "I-QUOTE", "E-QUOTE")
    label2id: dict[str, int] = field(default_factory=lambda: {"O": 0, "B-QUOTE": 1, "I-QUOTE": 2, "E-QUOTE": 3})
    id2label: dict[int, str] = field(default_factory=lambda: {0: "O", 1: "B-QUOTE", 2: "I-QUOTE", 3: "E-QUOTE"})

    @property
    def o_id(self) -> int:
        return 0

    @property
    def b_id(self) -> int:
        return 1

    @property
    def i_id(self) -> int:
        return 2

    @property
    def e_id(self) -> int:
        return 3

    @property
    def n_labels(self) -> int:
        return 4


def build_scheme(layer: str = "Quotation") -> BioeScheme:
    """Return a BioeScheme instance."""
    return BioeScheme()


# --------------------------------------------------------------------------
# SplitBundle and Split Loading
# --------------------------------------------------------------------------


@dataclass
class SplitBundle:
    """A split dataset and its book id / window index alignment."""

    dataset: Any
    book_ids: list[str]
    window_indices: list[int]
    offset_mappings: Sequence[Any] | None
    stats: dict[str, float]

    def __len__(self) -> int:
        return len(self.dataset)

    def select(self, count: int) -> SplitBundle:
        count = max(0, min(int(count), len(self.dataset)))
        return SplitBundle(
            dataset=self.dataset.select(range(count)),
            book_ids=self.book_ids[:count],
            window_indices=self.window_indices[:count],
            offset_mappings=self.offset_mappings[:count] if self.offset_mappings is not None else None,
            stats=self.stats,
        )

    def check_alignment(self) -> None:
        if self.book_ids and len(self.book_ids) != len(self.dataset):
            raise ValueError(
                f"book_ids ({len(self.book_ids)}) and dataset "
                f"({len(self.dataset)}) are out of step"
            )


DELETED_CONFIG_KEYS = (
    "positive_class_weight",
    "class_weight_cap",
    "masked_layers",
    "damaged_books",
)

MODEL_COLUMNS = ("input_ids", "attention_mask", "labels")


class LazyColumn:
    """One dataset column, read a row at a time.

    Holding the whole train offset_mapping as Python lists costs about 9 GB,
    and only the offline eval script ever reads it.
    """

    def __init__(self, dataset: Any, column: str) -> None:
        self._dataset = dataset
        self._column = column

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            rows = range(*index.indices(len(self._dataset)))
            return LazyColumn(self._dataset.select(rows), self._column)
        if index < 0:
            index += len(self._dataset)
        return self._dataset[int(index)][self._column]

    def __iter__(self):
        for row in range(len(self._dataset)):
            yield self[row]


def flat_labels(split_ds: Any) -> np.ndarray:
    """Every label of a split as one 1-D array.

    Windows are ragged — the last window of a book is short and the collator
    pads at batch time — so the column has no rectangular shape to take.
    """
    import pyarrow as pa

    column = split_ds.with_format("arrow")[:].column("labels")
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    pieces = [chunk.flatten().to_numpy(zero_copy_only=False) for chunk in chunks]
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(pieces)


def load_bioe_splits(cfg: Mapping[str, Any]) -> dict[str, SplitBundle]:
    """Load the pinned windowed Hub config for BIOE scheme."""
    from datasets import load_dataset
    from .window_data import require_pinned_revision

    stale = [key for key in DELETED_CONFIG_KEYS if key in cfg]
    if stale:
        raise ValueError(
            "stale config keys are not no-ops in this trainer: "
            + ", ".join(stale)
        )

    revision = require_pinned_revision(cfg.get("dataset_revision"))
    repo = str(cfg["dataset_repo"])
    config_name = str(cfg.get("dataset_config_name") or "windowed_w8192_s4579_bioe")
    
    data_files = {
        "train": f"{config_name}/train/*.parquet",
        "validation": f"{config_name}/validation/*.parquet",
        "test": f"{config_name}/test/*.parquet",
    }
    
    dataset = load_dataset(
        repo,
        data_files=data_files,
        revision=revision,
    )
    
    splits = {}
    for name in ("train", "validation", "test"):
        if name not in dataset:
            continue
            
        split_ds = dataset[name]
        
        book_ids = [str(val) for val in split_ds["book_id"]] if "book_id" in split_ds.column_names else []
        window_indices = [int(val) for val in split_ds["window_index"]] if "window_index" in split_ds.column_names else []
        offset_mappings = None
        if "offset_mapping" in split_ds.column_names:
            offset_mappings = LazyColumn(
                split_ds.select_columns(["offset_mapping"]), "offset_mapping"
            )

        # The trainer runs with remove_unused_columns=False, so anything left
        # here reaches the collator, which can only tensorise these three.
        bookkeeping = [
            column for column in split_ds.column_names if column not in MODEL_COLUMNS
        ]
        if bookkeeping:
            split_ds = split_ds.remove_columns(bookkeeping)

        # Compute stats (B == E check, etc.) using fast numpy operations
        labels_np = flat_labels(split_ds)
        valid_labels = labels_np[labels_np != -100]
        unique, counts_np = np.unique(valid_labels, return_counts=True)
        counts = Counter(dict(zip(unique.tolist(), counts_np.tolist())))
            
        supervised = sum(counts.values())
        positives = counts[1] + counts[2] + counts[3] # B, I, E
        
        stats = {
            "windows": float(len(split_ds)),
            "supervised_tokens": float(supervised),
            "positive_tokens": float(positives),
            "positive_token_rate": positives / supervised if supervised else 0.0,
            "n_begin": float(counts[1]),
            "n_inside": float(counts[2]),
            "n_end": float(counts[3]),
        }
        
        # Assert B == E per split
        if counts[1] != counts[3]:
            raise ValueError(f"Split {name} B/E count mismatch: B={counts[1]}, E={counts[3]}")
            
        bundle = SplitBundle(
            dataset=split_ds,
            book_ids=book_ids,
            window_indices=window_indices,
            offset_mappings=offset_mappings,
            stats=stats,
        )
        bundle.check_alignment()
        splits["validation" if name == "validation" else name] = bundle
        
    if "train" not in splits:
        raise ValueError("windowed dataset has no train split")
        
    return splits


def format_split_stats(desc: str, stats: Mapping[str, float]) -> str:
    return (
        f"{desc}: {int(stats['windows'])} windows; "
        f"{int(stats['n_begin'])} B / {int(stats['n_inside'])} I / "
        f"{int(stats['n_end'])} E; "
        f"{int(stats['positive_tokens'])} positive tokens "
        f"({100 * stats['positive_token_rate']:.3f}% of supervised)"
    )


# --------------------------------------------------------------------------
# Class Weights
# --------------------------------------------------------------------------


def pinned_class_weights(
    class_weights_cfg: Mapping[str, float],
    scheme: BioeScheme,
) -> list[float]:
    """Read the literal class_weights mapping, order it by label id,
    refuse if any of the four keys is missing.
    """
    weights = [0.0] * scheme.n_labels
    for label_name, label_id in scheme.label2id.items():
        if label_name not in class_weights_cfg:
            raise ValueError(f"Missing class weight for label: {label_name}")
        weights[label_id] = float(class_weights_cfg[label_name])
    return weights


# --------------------------------------------------------------------------
# Viterbi Decode
# --------------------------------------------------------------------------


def transition_matrix(
    labels: Sequence[str],
    break_penalty: float = 0.0,
) -> np.ndarray:
    """Legal BIOE transitions; break_penalty is subtracted on E -> B only."""
    size = len(labels)
    matrix = np.zeros((size, size), dtype=np.float64)
    
    def simplify(label: str) -> str:
        if label == "O":
            return "O"
        if label.startswith("B-"):
            return "B"
        if label.startswith("I-"):
            return "I"
        if label.startswith("E-"):
            return "E"
        raise ValueError(f"Unknown label: {label}")

    LEGAL = {
        ("O", "O"), ("O", "B"),
        ("B", "I"), ("B", "E"),
        ("I", "I"), ("I", "E"),
        ("E", "O"), ("E", "B")
    }

    for i, previous in enumerate(labels):
        p_simple = simplify(previous)
        for j, nxt in enumerate(labels):
            n_simple = simplify(nxt)
            if (p_simple, n_simple) not in LEGAL:
                matrix[i, j] = NEG
            elif p_simple == "E" and n_simple == "B":
                matrix[i, j] -= break_penalty
    return matrix


def log_softmax(scores: np.ndarray) -> np.ndarray:
    """log_softmax over the last axis, on a fresh writeable array."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def viterbi_decode(
    logits: Any,
    scheme: BioeScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    attention_mask: Any | None = None,
    chunk: int = 64,
) -> np.ndarray:
    """Batched Viterbi over [B, T, 4] logits. Returns [B, T] ids.

    The recursion is one step per token and each step touches only a 4x4 score
    table, so it runs in numpy: a torch op that small costs far more in
    dispatch than in arithmetic, and a stitched book is ~10^5 tokens long.
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"logits must be [B, T, C], got {tuple(scores.shape)}")
    if scores.shape[-1] != scheme.n_labels:
        raise ValueError(
            f"expected {scheme.n_labels} classes, got {scores.shape[-1]}"
        )
    transitions = np.asarray(
        transition_matrix(scheme.labels, break_penalty), dtype=np.float32
    )
    mask = None
    if attention_mask is not None:
        mask = np.asarray(attention_mask) <= 0

    batch, seq, n_labels = scores.shape
    paths = np.zeros((batch, seq), dtype=np.int64)
    for start in range(0, batch, chunk):
        stop = min(start + chunk, batch)
        log_probs = log_softmax(scores[start:stop])
        if mask is not None:
            pad = mask[start:stop]
            log_probs[pad] = NEG
            log_probs[pad, scheme.o_id] = 0.0

        # Enforce start constraint: must start with O or B
        score = log_probs[:, 0].copy()
        score[:, scheme.i_id] = NEG
        score[:, scheme.e_id] = NEG

        rows = stop - start
        back = np.zeros((rows, seq, n_labels), dtype=np.int8)
        for time in range(1, seq):
            candidates = score[:, :, None] + transitions[None]
            back[:, time] = candidates.argmax(axis=1)
            score = candidates.max(axis=1) + log_probs[:, time]

        path = np.zeros((rows, seq), dtype=np.int64)

        # Enforce end constraint: must end with O or E
        score_final = score.copy()
        score_final[:, scheme.b_id] = NEG
        score_final[:, scheme.i_id] = NEG
        path[:, -1] = score_final.argmax(axis=-1)

        row_index = np.arange(rows)
        for time in range(seq - 1, 0, -1):
            path[:, time - 1] = back[row_index, time, path[:, time]]
        paths[start:stop] = path
    return paths


# --------------------------------------------------------------------------
# BIOE Spans
# --------------------------------------------------------------------------


def bioe_spans(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Convert BIOE labels to inclusive token spans.
    
    A valid span is B I* E.
    Orphan I, orphan E, and unterminated B I* are dropped.
    """
    spans: list[tuple[str, int, int]] = []
    active_start: int | None = None
    active_tag: str | None = None
    
    for index, label in enumerate(labels):
        if label == "O":
            active_start = None
            active_tag = None
        else:
            try:
                prefix, tag = label.split("-", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid BIOE label: {label!r}") from exc
            
            if prefix == "B":
                active_start = index
                active_tag = tag
            elif prefix == "I":
                if active_start is not None and tag == active_tag:
                    pass
                else:
                    active_start = None
                    active_tag = None
            elif prefix == "E":
                if active_start is not None and tag == active_tag:
                    spans.append((active_tag, active_start, index))
                active_start = None
                active_tag = None
            else:
                raise ValueError(f"Invalid BIOE label: {label!r}")
                
    return spans


def downgrade_to_bio(labels: Sequence[str]) -> list[str]:
    """Map E-QUOTE to I-QUOTE in a BIOE label sequence."""
    return ["I-QUOTE" if label == "E-QUOTE" else label for label in labels]


# --------------------------------------------------------------------------
# Stitch Book Logits
# --------------------------------------------------------------------------


def stitch_book_logits(
    logits: np.ndarray,
    gold_ids: np.ndarray,
    book_ids: Sequence[str],
    window_indices: Sequence[int],
    scheme: BioeScheme,
    break_penalty: float = 0.0,
    book_owned_counts: Mapping[str, int] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Group logits and gold_ids by book_id, sort by window_index,
    concatenate where gold_ids != -100, run Viterbi once per book,
    and return pred_seqs and gold_seqs.
    """
    from collections import defaultdict
    books = defaultdict(list)
    for i, (book_id, win_idx) in enumerate(zip(book_ids, window_indices, strict=True)):
        books[book_id].append((win_idx, logits[i], gold_ids[i]))
        
    pred_seqs: list[list[str]] = []
    gold_seqs: list[list[str]] = []
    
    for book_id, windows in sorted(books.items()):
        windows.sort(key=lambda x: x[0])
        
        # Check for gaps in window_index
        win_indices = [w[0] for w in windows]
        if win_indices != list(range(len(windows))):
            raise ValueError(f"Gap detected in window indices for book {book_id}: {win_indices}")
            
        book_logits_list = []
        book_golds_list = []
        
        for win_idx, win_logits, win_golds in windows:
            mask = (win_golds != -100)
            book_logits_list.append(win_logits[mask])
            book_golds_list.append(win_golds[mask])
            
        book_logits = np.concatenate(book_logits_list, axis=0)
        book_golds = np.concatenate(book_golds_list, axis=0)
        
        assert book_logits.shape[0] == book_golds.shape[0], (
            f"Logits shape {book_logits.shape[0]} and gold shape {book_golds.shape[0]} "
            f"mismatch for book {book_id}"
        )
        
        if book_owned_counts is not None:
            if book_id in book_owned_counts:
                assert len(book_golds) == book_owned_counts[book_id], (
                    f"Stitched length {len(book_golds)} does not match "
                    f"owned-token count {book_owned_counts[book_id]} for book {book_id}"
                )
        
        book_logits_batched = np.expand_dims(book_logits, axis=0)
        book_preds_batched = viterbi_decode(book_logits_batched, scheme, break_penalty=break_penalty)
        book_preds = book_preds_batched[0]
        
        pred_seq = [scheme.id2label[int(p)] for p in book_preds]
        gold_seq = [scheme.id2label[int(g)] for g in book_golds]
        
        pred_seqs.append(pred_seq)
        gold_seqs.append(gold_seq)
        
    return pred_seqs, gold_seqs


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def sequences_from_ids(
    pred_ids: Any,
    gold_ids: Any,
    scheme: BioeScheme,
) -> tuple[list[list[str]], list[list[str]]]:
    pred_seqs: list[list[str]] = []
    gold_seqs: list[list[str]] = []
    for preds, golds in zip(pred_ids, gold_ids, strict=True):
        pred_seqs.append(
            [
                scheme.id2label[int(pred)]
                for pred, gold in zip(preds, golds, strict=True)
                if int(gold) != -100
            ]
        )
        gold_seqs.append(
            [scheme.id2label[int(gold)] for gold in golds if int(gold) != -100]
        )
    return pred_seqs, gold_seqs


def iou50_metrics(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    prefix: str = "iou50",
    span_reader=bioe_spans,
) -> dict[str, float]:
    true_positive = 0
    n_pred = 0
    n_gold = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        gold = span_reader(gold_seq)
        pred = span_reader(pred_seq)
        n_gold += len(gold)
        n_pred += len(pred)
        true_positive += len(match_iou(gold, pred, threshold))
    precision, recall, f1 = prf(true_positive, n_pred, n_gold)
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_n_pred": float(n_pred),
        f"{prefix}_n_gold": float(n_gold),
        f"{prefix}_overproposal": (n_pred / n_gold) if n_gold else 0.0,
    }


def token_recall_inside_gold(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
) -> dict[str, float]:
    hit = 0
    total = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        for pred, gold in zip(pred_seq, gold_seq, strict=True):
            if gold == "O":
                continue
            total += 1
            hit += int(pred != "O")
    return {
        "token_recall_inside_gold": hit / total if total else 0.0,
        "token_recall_n": float(total),
    }


def e_recall(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
) -> float:
    """Fraction of gold E-QUOTE tokens predicted as E-QUOTE."""
    hit = 0
    total = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        for pred, gold in zip(pred_seq, gold_seq, strict=True):
            if gold == "E-QUOTE":
                total += 1
                if pred == "E-QUOTE":
                    hit += 1
    return hit / total if total else 0.0


def mean_iou_of_matches(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    span_reader=bioe_spans,
) -> float:
    values: list[float] = []
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        gold = span_reader(gold_seq)
        pred = span_reader(pred_seq)
        for _g, _p, value in match_iou(gold, pred, threshold):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def per_book_metrics(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    book_ids: Sequence[str],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    span_reader=bioe_spans,
) -> dict[str, float]:
    if not book_ids:
        return {}
    tally: dict[str, list[int]] = {}  # book -> [tp, n_pred, n_gold]
    for pred_seq, gold_seq, book in zip(pred_seqs, gold_seqs, book_ids, strict=True):
        gold = span_reader(gold_seq)
        pred = span_reader(pred_seq)
        row = tally.setdefault(str(book), [0, 0, 0])
        row[0] += len(match_iou(gold, pred, threshold))
        row[1] += len(pred)
        row[2] += len(gold)

    metrics: dict[str, float] = {}
    with_gold = {book: row for book, row in tally.items() if row[2] > 0}
    for book, (true_positive, n_pred, n_gold) in sorted(with_gold.items()):
        precision, recall, f1 = prf(true_positive, n_pred, n_gold)
        metrics[f"book_{book}_iou50_f1"] = f1
        metrics[f"book_{book}_iou50_precision"] = precision
        metrics[f"book_{book}_iou50_recall"] = recall
        metrics[f"book_{book}_n_gold"] = float(n_gold)
        metrics[f"book_{book}_n_pred"] = float(n_pred)
    metrics["n_books_with_gold"] = float(len(with_gold))

    if with_gold:
        top = max(with_gold, key=lambda book: with_gold[book][2])
        rest = [row for book, row in with_gold.items() if book != top]
        totals = [sum(column) for column in zip(*rest)] if rest else [0, 0, 0]
        _p, _r, f1 = prf(totals[0], totals[1], totals[2])
        metrics["iou50_f1_excl_top_book"] = f1
        metrics["excl_top_book_n_gold"] = float(totals[2])
    return metrics


def evaluate_logits(
    logits: Any,
    gold_ids: Any,
    scheme: BioeScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    label_bias: Any | None = None,
    book_ids: Sequence[str] | None = None,
    window_indices: Sequence[int] | None = None,
    include_argmax: bool = True,
    include_per_book: bool = True,
) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float32)
    gold_ids = np.asarray(gold_ids)
    biased = logits
    if label_bias is not None:
        biased = logits + np.asarray(label_bias, dtype=np.float32).reshape(1, 1, -1)

    viterbi_ids = viterbi_decode(biased, scheme, break_penalty=break_penalty)
    pred_seqs, gold_seqs = sequences_from_ids(viterbi_ids, gold_ids, scheme)
    
    metrics = iou50_metrics(pred_seqs, gold_seqs)
    metrics.update(token_recall_inside_gold(pred_seqs, gold_seqs))
    metrics["iou50_mean_iou"] = mean_iou_of_matches(pred_seqs, gold_seqs)
    metrics["viterbi_break_penalty"] = float(break_penalty)
    metrics["e_recall"] = e_recall(pred_seqs, gold_seqs)
    
    if label_bias is not None:
        metrics["label_bias"] = float(np.asarray(label_bias).reshape(-1)[1])

    # Downgrade metrics
    pred_seqs_bio = [downgrade_to_bio(p) for p in pred_seqs]
    gold_seqs_bio = [downgrade_to_bio(g) for g in gold_seqs]
    downgraded_metrics = iou50_metrics(
        pred_seqs_bio,
        gold_seqs_bio,
        prefix="iou50_downgrade",
        span_reader=bio_spans,
    )
    metrics["iou50_f1_downgrade"] = downgraded_metrics["iou50_downgrade_f1"]

    # Stitched metrics
    if book_ids is not None and window_indices is not None:
        stitched_preds, stitched_golds = stitch_book_logits(
            biased,
            gold_ids,
            book_ids,
            window_indices,
            scheme,
            break_penalty=break_penalty,
        )
        stitched_metrics = iou50_metrics(
            stitched_preds,
            stitched_golds,
            prefix="iou50_stitched",
            span_reader=bioe_spans,
        )
        metrics["iou50_f1_stitched"] = stitched_metrics["iou50_stitched_f1"]

    if include_argmax:
        argmax_pred, argmax_gold = sequences_from_ids(
            np.argmax(logits, axis=-1), gold_ids, scheme
        )
        metrics.update(
            iou50_metrics(argmax_pred, argmax_gold, prefix="argmax_iou50")
        )
    if include_per_book and book_ids:
        metrics.update(per_book_metrics(pred_seqs, gold_seqs, book_ids))
    return metrics


def compute_metrics_factory(
    scheme: BioeScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    book_ids: Sequence[str] | None = None,
    window_indices: Sequence[int] | None = None,
):
    ids = list(book_ids) if book_ids else None
    win_indices = list(window_indices) if window_indices else None

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        logits = np.asarray(predictions, dtype=np.float32)
        if ids is not None and len(ids) != logits.shape[0]:
            raise ValueError(
                f"book_ids has {len(ids)} entries but the eval batch has "
                f"{logits.shape[0]} rows"
            )
        return evaluate_logits(
            logits,
            label_ids,
            scheme,
            break_penalty=break_penalty,
            book_ids=ids,
            window_indices=win_indices,
        )

    return compute_metrics


def bias_vector(scheme: BioeScheme, value: float) -> np.ndarray:
    """Add value to both positive classes, nothing to O."""
    bias = np.zeros(scheme.n_labels, dtype=np.float32)
    bias[scheme.b_id] = float(value)
    bias[scheme.i_id] = float(value)
    bias[scheme.e_id] = float(value)
    return bias


DEFAULT_PENALTY_SWEEP = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_BIAS_SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0)


def sweep_decode(
    logits: Any,
    gold_ids: Any,
    scheme: BioeScheme,
    *,
    penalties: Sequence[float] = DEFAULT_PENALTY_SWEEP,
    biases: Sequence[float] = DEFAULT_BIAS_SWEEP,
    book_ids: Sequence[str] | None = None,
    window_indices: Sequence[int] | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for penalty in penalties:
        for bias in biases:
            row = evaluate_logits(
                logits,
                gold_ids,
                scheme,
                break_penalty=float(penalty),
                label_bias=bias_vector(scheme, bias) if bias else None,
                book_ids=book_ids,
                window_indices=window_indices,
                include_argmax=False,
                include_per_book=False,
            )
            row["break_penalty"] = float(penalty)
            row["bias"] = float(bias)
            rows.append(row)
    return rows


def best_decode(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("decode sweep produced no rows")
    return dict(
        max(
            rows,
            key=lambda row: (
                row.get("iou50_f1", 0.0),
                -row.get("break_penalty", 0.0),
                -row.get("bias", 0.0),
            ),
        )
    )

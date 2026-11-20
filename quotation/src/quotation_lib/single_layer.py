"""Single-annotation (one-layer) BIO training on the v2 windowed dataset.

Why this module exists
----------------------
v2 changed twelve things at once and could not be read. This path changes
exactly two:

1. the label set shrinks from 15 BIO classes to 3 (``O``, ``B-<TAG>``,
   ``I-<TAG>``) for one annotation layer, and
2. selection / early stopping move to the v2 metric (Viterbi + IoU@0.5).

**The loss is v1.3's, untouched**: plain ``CrossEntropyLoss`` with the
inverse-frequency class weights produced by
:func:`quotation_lib.training_utils.class_weights_from_dataset`. No focal
loss, no Tversky, no boundary head, no EER term, no ``loss_mask``.

Data comes from the pinned 40%-overlap windowed config
(``windowed_w8192_s4916``, revision ``v2.0``), so no re-tokenization happens
here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .training_utils import (
    ID2LABEL as ID2LABEL15,
    LABEL2ID as LABEL2ID15,
    LABELS as LABELS15,
    LAYER_TO_TAG,
    bio_spans,
    canonical_layer,
)
from .span_metrics import inclusive_iou, match_iou, prf

NEG = -1.0e9
DEFAULT_BREAK_PENALTY = 2.0
DEFAULT_IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# label scheme
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleLayerScheme:
    """The 3-label scheme for one annotation layer, plus its 15-class origin."""

    layer: str
    tag: str
    labels: tuple[str, ...]
    label2id: dict[str, int]
    id2label: dict[int, str]
    source_b_id: int
    source_i_id: int

    @property
    def o_id(self) -> int:
        return self.label2id["O"]

    @property
    def b_id(self) -> int:
        return self.label2id[f"B-{self.tag}"]

    @property
    def i_id(self) -> int:
        return self.label2id[f"I-{self.tag}"]

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def build_scheme(layer: str) -> SingleLayerScheme:
    """``build_scheme("Yigchung")`` → O / B-YIGCHUNG / I-YIGCHUNG."""
    canonical = canonical_layer(layer)
    tag = LAYER_TO_TAG[canonical]
    labels = ("O", f"B-{tag}", f"I-{tag}")
    return SingleLayerScheme(
        layer=canonical,
        tag=tag,
        labels=labels,
        label2id={name: index for index, name in enumerate(labels)},
        id2label=dict(enumerate(labels)),
        source_b_id=LABEL2ID15[f"B-{tag}"],
        source_i_id=LABEL2ID15[f"I-{tag}"],
    )


# --------------------------------------------------------------------------
# label projection: 15-class window labels -> 3-class, recovering nested gold
# --------------------------------------------------------------------------


def project_labels(
    labels15: Sequence[int],
    spans: Sequence[Mapping[str, Any]] | None,
    scheme: SingleLayerScheme,
) -> tuple[list[int], int]:
    """Project one window's 15-class labels onto the single-layer scheme.

    Two sources are combined, in this order:

    * The 15-class ``labels`` column, which already carries the correct BIO
      continuation across the duplicated stride prefix and the ``-100`` mask
      on repeated / special tokens. Its ``B-<TAG>`` / ``I-<TAG>`` prefixes are
      preserved exactly, so the supervision matches what v1.3 saw.
    * The ``spans`` column, which lists *every* overlapping gold span rather
      than the winner-takes-all collapse. A layer span nested inside a
      higher-priority one (Quotation wins the collapse) is invisible in
      ``labels`` and would silently train as a negative. Those tokens are
      recovered here.

    ``-100`` positions are never written to: they are either special tokens or
    tokens already supervised by the previous window.

    Returns the new label list and the number of tokens recovered from
    ``spans`` that the collapse had hidden.
    """
    out: list[int] = []
    for value in labels15:
        value = int(value)
        if value == -100:
            out.append(-100)
        elif value == scheme.source_b_id:
            out.append(scheme.b_id)
        elif value == scheme.source_i_id:
            out.append(scheme.i_id)
        else:
            out.append(scheme.o_id)

    recovered = 0
    if spans:
        length = len(out)
        for span in spans:
            if str(span["layer"]).upper() != scheme.tag:
                continue
            start = max(0, int(span["start"]))
            end = min(length, int(span["end"]))  # half-open
            for index in range(start, end):
                if out[index] == -100 or out[index] != scheme.o_id:
                    continue
                previous = out[index - 1] if index > 0 else scheme.o_id
                out[index] = (
                    scheme.i_id
                    if previous in (scheme.b_id, scheme.i_id)
                    else scheme.b_id
                )
                recovered += 1
    return out, recovered


def in_scope(scope_mask: Sequence[int], scheme: SingleLayerScheme) -> bool:
    """True when this window's book has the layer inside its confirmed scope."""
    mask = list(scope_mask)
    if len(mask) != len(LABELS15):
        raise ValueError(
            f"scope_mask has {len(mask)} entries, expected {len(LABELS15)}"
        )
    return bool(mask[scheme.source_b_id]) and bool(mask[scheme.source_i_id])


DROPPABLE_COLUMNS = (
    "spans",
    "scope_mask",
    "offset_mapping",
    "special_tokens_mask",
    "start_targets",
    "end_targets",
    "loss_mask",
    "book_id",
    "window_index",
    "token_type_ids",
)


@dataclass
class SplitBundle:
    """A projected split and the book id of each of its windows, kept together.

    The book ids are only meaningful while they line up index-for-index with
    the dataset rows, and evaluation slices the dataset (``--smoke``). Slicing
    through :meth:`select` keeps the two in step; slicing the dataset alone
    cannot, which is why the dataset is not handed out bare.
    """

    dataset: Any
    book_ids: list[str]
    stats: dict[str, float]

    def __len__(self) -> int:
        return len(self.dataset)

    def select(self, count: int) -> "SplitBundle":
        count = max(0, min(int(count), len(self.dataset)))
        return SplitBundle(
            dataset=self.dataset.select(range(count)),
            book_ids=self.book_ids[:count],
            stats=self.stats,
        )

    def check_alignment(self) -> None:
        if self.book_ids and len(self.book_ids) != len(self.dataset):
            raise ValueError(
                f"book_ids ({len(self.book_ids)}) and dataset "
                f"({len(self.dataset)}) are out of step; per-book metrics "
                "would be attributed to the wrong books"
            )


def project_split(
    dataset: Any,
    scheme: SingleLayerScheme,
    *,
    desc: str,
    num_proc: int | None = None,
) -> SplitBundle:
    """Filter out-of-scope windows, project labels, drop unused columns.

    Returns a :class:`SplitBundle`: the projected dataset, the per-window book
    ids in dataset order, and a stats dict for the run log.
    """
    n_before = len(dataset)
    if "scope_mask" in dataset.column_names:
        dataset = dataset.filter(
            lambda scope_mask: in_scope(scope_mask, scheme),
            input_columns=["scope_mask"],
            desc=f"{desc}: scope filter",
        )
    n_after = len(dataset)
    # Captured after the filter and before map drops the column.
    book_ids = (
        [str(value) for value in dataset["book_id"]]
        if "book_id" in dataset.column_names
        else []
    )

    has_spans = "spans" in dataset.column_names
    input_columns = ["labels", "spans"] if has_spans else ["labels"]

    def project(labels, spans=None):
        new_labels, recovered = project_labels(labels, spans, scheme)
        return {"labels": new_labels, "n_recovered": recovered}

    drop = [name for name in DROPPABLE_COLUMNS if name in dataset.column_names]
    projected = dataset.map(
        project,
        input_columns=input_columns,
        remove_columns=drop,
        num_proc=num_proc,
        desc=f"{desc}: project to {scheme.tag}",
    )
    recovered_total = int(sum(projected["n_recovered"]))
    projected = projected.remove_columns(["n_recovered"])

    counts = Counter()
    windows_with_positive = 0
    for labels in projected["labels"]:
        local = Counter(int(value) for value in labels if int(value) != -100)
        counts.update(local)
        if local[scheme.b_id] or local[scheme.i_id]:
            windows_with_positive += 1
    supervised = sum(counts.values())
    positives = counts[scheme.b_id] + counts[scheme.i_id]
    stats = {
        "windows_in": float(n_before),
        "windows_out_of_scope": float(n_before - n_after),
        "windows": float(len(projected)),
        "windows_with_positive": float(windows_with_positive),
        "windows_with_positive_pct": (
            100.0 * windows_with_positive / len(projected) if len(projected) else 0.0
        ),
        "supervised_tokens": float(supervised),
        "positive_tokens": float(positives),
        "positive_token_rate": positives / supervised if supervised else 0.0,
        "n_begin": float(counts[scheme.b_id]),
        "recovered_tokens": float(recovered_total),
    }
    bundle = SplitBundle(dataset=projected, book_ids=book_ids, stats=stats)
    bundle.check_alignment()
    return bundle


def format_split_stats(desc: str, stats: Mapping[str, float]) -> str:
    return (
        f"{desc}: {int(stats['windows'])} windows "
        f"({int(stats['windows_out_of_scope'])} dropped as out of scope); "
        f"{int(stats['windows_with_positive'])} "
        f"({stats['windows_with_positive_pct']:.1f}%) contain a gold span; "
        f"{int(stats['n_begin'])} spans, "
        f"{int(stats['positive_tokens'])} positive tokens "
        f"({100 * stats['positive_token_rate']:.3f}% of supervised); "
        f"{int(stats['recovered_tokens'])} tokens recovered from nested gold"
    )


# --------------------------------------------------------------------------
# class weights (v1.3 formula, restricted to one layer)
# --------------------------------------------------------------------------


def single_layer_class_weights(
    dataset: Any,
    scheme: SingleLayerScheme,
    *,
    cap: float = 5.0,
    o_weight: float | None = 0.07,
    positive_weight: float | None = None,
) -> list[float]:
    """v1.3's ``class_weights_from_dataset`` on a 3-label scheme.

    With one layer observed the formula pins that layer at 1.0 and ``O`` at
    ``o_weight`` — the same *relative* scale v1.3 gave its most common layer.
    ``positive_weight`` overrides the 1.0 if you want to reproduce the
    absolute weight Yigchung carried inside the 15-class run.
    """
    counts: Counter[int] = Counter()
    for labels in dataset["labels"]:
        counts.update(int(value) for value in labels if int(value) != -100)
    positives = counts[scheme.b_id] + counts[scheme.i_id]
    if positives == 0:
        raise ValueError(
            f"training split has no {scheme.tag} tokens; nothing to learn"
        )
    weights = [1.0] * scheme.n_labels
    value = 1.0 if positive_weight is None else float(positive_weight)
    weights[scheme.b_id] = value
    weights[scheme.i_id] = value
    if o_weight is not None:
        weights[scheme.o_id] = float(o_weight)
    elif counts[scheme.o_id]:
        total = sum(counts.values())
        weights[scheme.o_id] = total / (2 * counts[scheme.o_id]) / (
            total / (2 * positives)
        )
    return [min(weight, cap) for weight in weights]


def joint_model_equivalent_weight(
    counts_15: Mapping[int, int],
    scheme: SingleLayerScheme,
    *,
    cap: float = 5.0,
) -> float:
    """What weight this layer carried inside the 15-class v1.3 run.

    Printed at startup so the change in the positive:O ratio is visible rather
    than implicit.
    """
    total = sum(counts_15.values())
    layer_counts = {
        tag: counts_15.get(LABEL2ID15[f"B-{tag}"], 0)
        + counts_15.get(LABEL2ID15[f"I-{tag}"], 0)
        for tag in LAYER_TO_TAG.values()
    }
    observed = {tag: count for tag, count in layer_counts.items() if count > 0}
    groups = len(observed) + (1 if counts_15.get(LABEL2ID15["O"], 0) else 0)
    raw = {tag: total / (groups * count) for tag, count in observed.items()}
    scale = min(raw.values()) if raw else 1.0
    return min(raw.get(scheme.tag, 1.0) / scale, cap)


def count_labels_15(dataset: Any) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for labels in dataset["labels"]:
        counts.update(int(value) for value in labels if int(value) != -100)
    return dict(counts)


# --------------------------------------------------------------------------
# Viterbi decode over an arbitrary BIO label set
# --------------------------------------------------------------------------


def transition_matrix(
    labels: Sequence[str],
    break_penalty: float = 0.0,
) -> np.ndarray:
    """Legal BIO transitions; ``break_penalty`` is subtracted on span exit.

    Same rule as :func:`quotation_lib.span_metrics.bio_transition_matrix`,
    parameterised by the label list instead of the 15-class global.
    """
    size = len(labels)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, previous in enumerate(labels):
        for j, nxt in enumerate(labels):
            if nxt.startswith("I-"):
                tag = nxt[2:]
                if previous not in (f"B-{tag}", f"I-{tag}"):
                    matrix[i, j] = NEG
                    continue
            if previous == "O":
                continue
            continuing = nxt.startswith("I-") and nxt[2:] == previous[2:]
            if not continuing:
                matrix[i, j] -= break_penalty
    return matrix


def viterbi_decode(
    logits: Any,
    scheme: SingleLayerScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    attention_mask: Any | None = None,
    chunk: int = 64,
) -> np.ndarray:
    """Batched Viterbi over ``[B, T, 3]`` logits. Returns ``[B, T]`` ids."""
    import torch
    import torch.nn.functional as F

    if isinstance(logits, np.ndarray):
        tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    else:
        tensor = logits.detach().float().cpu()
    if tensor.dim() != 3:
        raise ValueError(f"logits must be [B, T, C], got {tuple(tensor.shape)}")
    if tensor.size(-1) != scheme.n_labels:
        raise ValueError(
            f"expected {scheme.n_labels} classes, got {tensor.size(-1)}"
        )
    transitions = torch.as_tensor(
        transition_matrix(scheme.labels, break_penalty), dtype=torch.float32
    )
    mask = None
    if attention_mask is not None:
        mask = torch.as_tensor(np.asarray(attention_mask), dtype=torch.float32)

    batch, seq, _ = tensor.shape
    paths = torch.zeros(batch, seq, dtype=torch.long)
    for start in range(0, batch, chunk):
        sl = slice(start, min(start + chunk, batch))
        log_probs = F.log_softmax(tensor[sl], dim=-1)
        if mask is not None:
            pad = mask[sl] <= 0
            extra = torch.full_like(log_probs, NEG)
            extra[..., scheme.o_id] = 0.0
            log_probs = torch.where(pad.unsqueeze(-1), extra, log_probs)
        score = log_probs[:, 0]
        back = torch.zeros(
            log_probs.size(0), seq, scheme.n_labels, dtype=torch.int16
        )
        for time in range(1, seq):
            candidates = score.unsqueeze(-1) + transitions.unsqueeze(0)
            best, previous = candidates.max(dim=1)
            back[:, time] = previous.to(torch.int16)
            score = best + log_probs[:, time]
        path = torch.zeros(log_probs.size(0), seq, dtype=torch.long)
        path[:, -1] = score.argmax(dim=-1)
        for time in range(seq - 1, 0, -1):
            path[:, time - 1] = (
                back[:, time]
                .gather(1, path[:, time].unsqueeze(1).to(torch.int64))
                .squeeze(1)
            )
        paths[sl] = path
    return paths.numpy()


# --------------------------------------------------------------------------
# metrics: IoU@0.5 span F1 (the selection metric) + diagnostics
# --------------------------------------------------------------------------


def sequences_from_ids(
    pred_ids: Any,
    gold_ids: Any,
    scheme: SingleLayerScheme,
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
) -> dict[str, float]:
    true_positive = 0
    n_pred = 0
    n_gold = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        gold = bio_spans(gold_seq)
        pred = bio_spans(pred_seq)
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


def mean_iou_of_matches(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    values: list[float] = []
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        gold = bio_spans(gold_seq)
        pred = bio_spans(pred_seq)
        for _g, _p, value in match_iou(gold, pred, threshold):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def per_book_metrics(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    book_ids: Sequence[str],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    """Pooled IoU@0.5 counts split by book, plus the leave-the-biggest-out score.

    Validation Yigchung is extremely concentrated — two of twenty-five books
    hold 93% of the gold spans — so a pooled score can move because one book
    moved. ``iou50_f1_excl_top_book`` re-pools with the single largest
    contributor removed and answers that directly.

    Books are keyed by id, so counts pool correctly across a book's windows.
    """
    if not book_ids:
        return {}
    tally: dict[str, list[int]] = {}  # book -> [tp, n_pred, n_gold]
    for pred_seq, gold_seq, book in zip(pred_seqs, gold_seqs, book_ids, strict=True):
        gold = bio_spans(gold_seq)
        pred = bio_spans(pred_seq)
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
    scheme: SingleLayerScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    label_bias: Any | None = None,
    book_ids: Sequence[str] | None = None,
    include_argmax: bool = True,
    include_per_book: bool = True,
) -> dict[str, float]:
    """The one scoring path. Used in-loop, for the test split, and offline.

    ``label_bias`` is a length-3 vector added to every token's logits before
    decoding — the per-layer bias sweep. It is applied to the Viterbi decode
    only, so ``argmax_*`` stays an unbiased reference.
    """
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
    if label_bias is not None:
        metrics["label_bias"] = float(np.asarray(label_bias).reshape(-1)[1])

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
    scheme: SingleLayerScheme,
    *,
    break_penalty: float = DEFAULT_BREAK_PENALTY,
    book_ids: Sequence[str] | None = None,
):
    """HuggingFace ``compute_metrics``: Viterbi IoU@0.5 F1 plus diagnostics.

    ``iou50_f1`` is the value ``metric_for_best_model`` and early stopping read;
    every added key is a sibling of it, never a replacement. ``book_ids`` must
    line up with the eval dataset row for row — evaluation uses a sequential
    sampler, so dataset order is preserved, and :class:`SplitBundle` keeps the
    two in step through slicing.
    """
    ids = list(book_ids) if book_ids else None

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        logits = np.asarray(predictions, dtype=np.float32)
        if ids is not None and len(ids) != logits.shape[0]:
            raise ValueError(
                f"book_ids has {len(ids)} entries but the eval batch has "
                f"{logits.shape[0]} rows; per-book metrics would be wrong"
            )
        return evaluate_logits(
            logits,
            label_ids,
            scheme,
            break_penalty=break_penalty,
            book_ids=ids,
        )

    return compute_metrics


def bias_vector(scheme: SingleLayerScheme, value: float) -> np.ndarray:
    """Add ``value`` to both positive classes, nothing to ``O``."""
    bias = np.zeros(scheme.n_labels, dtype=np.float32)
    bias[scheme.b_id] = float(value)
    bias[scheme.i_id] = float(value)
    return bias


DEFAULT_PENALTY_SWEEP = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_BIAS_SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0)


def sweep_decode(
    logits: Any,
    gold_ids: Any,
    scheme: SingleLayerScheme,
    *,
    penalties: Sequence[float] = DEFAULT_PENALTY_SWEEP,
    biases: Sequence[float] = DEFAULT_BIAS_SWEEP,
    book_ids: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    """Score every (break penalty × logit bias) pair. Rows carry both settings.

    The label set dropped from 15 classes to 3, so the transition structure
    changed and neither v1.3's 4.0 nor v2's 2.0 carries over. This has to be
    re-swept rather than inherited.
    """
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
                include_argmax=False,
                include_per_book=False,
            )
            row["break_penalty"] = float(penalty)
            row["bias"] = float(bias)
            rows.append(row)
    return rows


def best_decode(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Pick the sweep row with the highest ``iou50_f1``, ties to lower settings."""
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

"""In-loop and offline metrics: Viterbi IoU@0.5, token recall, ASRF, MUC."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .training_utils import ID2LABEL, LABEL2ID, LABELS, bio_spans
from .loss_objective import N_BIO

DEFAULT_IN_LOOP_BREAK_PENALTY = 2.0
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_ASRF_THRESHOLD = 0.5
NEG = -1.0e9
VITERBI_CHUNK = 32

TAG_NAMES = [label[2:] for label in LABELS if label.startswith("B-")]


def bio_transition_matrix(break_penalty: float = 0.0) -> np.ndarray:
    """Legal BIO transitions; ``break_penalty`` is subtracted on span exit."""
    size = len(LABELS)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, previous in enumerate(LABELS):
        for j, nxt in enumerate(LABELS):
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


def bio_transition_torch(
    break_penalty: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(
        bio_transition_matrix(break_penalty), device=device, dtype=dtype
    )


def viterbi_decode_torch(
    logits: torch.Tensor,
    *,
    break_penalty: float = DEFAULT_IN_LOOP_BREAK_PENALTY,
    attention_mask: torch.Tensor | None = None,
    chunk: int = VITERBI_CHUNK,
) -> torch.Tensor:
    """Batched Viterbi. ``logits`` is ``[B, T, C]``; returns ``[B, T]`` ids."""
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, C], got {tuple(logits.shape)}")
    batch, seq, n_labels = logits.shape
    if n_labels != N_BIO:
        raise ValueError(f"expected {N_BIO} classes, got {n_labels}")
    transitions = bio_transition_torch(break_penalty, logits.device, torch.float32)
    paths = torch.zeros(batch, seq, dtype=torch.long, device=logits.device)
    for start in range(0, batch, chunk):
        sl = slice(start, min(start + chunk, batch))
        paths[sl] = _viterbi_chunk(logits[sl].float(), transitions, attention_mask[sl] if attention_mask is not None else None)
    return paths


def _viterbi_chunk(
    logits: torch.Tensor,
    transitions: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    if attention_mask is not None:
        pad = attention_mask <= 0
        extra = torch.full_like(log_probs, NEG)
        extra[..., LABEL2ID["O"]] = 0.0
        log_probs = torch.where(pad.unsqueeze(-1), extra, log_probs)
    batch, seq, n_labels = log_probs.shape
    score = log_probs[:, 0]
    back = torch.zeros(batch, seq, n_labels, dtype=torch.int16, device=logits.device)
    for time in range(1, seq):
        candidates = score.unsqueeze(-1) + transitions.unsqueeze(0)
        best, prev = candidates.max(dim=1)
        back[:, time] = prev.to(torch.int16)
        score = best + log_probs[:, time]
    path = torch.zeros(batch, seq, dtype=torch.long, device=logits.device)
    path[:, -1] = score.argmax(dim=-1)
    for time in range(seq - 1, 0, -1):
        path[:, time - 1] = back[:, time].gather(
            1, path[:, time].unsqueeze(1).to(torch.int64)
        ).squeeze(1)
    return path


def viterbi_decode_numpy(
    log_probs: np.ndarray,
    transitions: np.ndarray,
) -> np.ndarray:
    """Single-sequence Viterbi. ``log_probs`` is ``[T, C]``."""
    n_tokens, n_labels = log_probs.shape
    if n_tokens == 0:
        return np.zeros(0, dtype=np.int16)
    best = log_probs[0].copy()
    back = np.zeros((n_tokens, n_labels), dtype=np.int16)
    for time in range(1, n_tokens):
        scores = best[:, None] + transitions
        back[time] = np.argmax(scores, axis=0)
        best = scores[back[time], np.arange(n_labels)] + log_probs[time]
    path = np.zeros(n_tokens, dtype=np.int16)
    path[-1] = int(np.argmax(best))
    for time in range(n_tokens - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    return path


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def apply_label_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """``bias`` is ``[C]`` added to every token."""
    return logits + bias.reshape(1, -1)


def layer_bias_vector(per_layer: Mapping[str, float]) -> np.ndarray:
    bias = np.zeros(len(LABELS), dtype=np.float32)
    for index, label in enumerate(LABELS):
        if label != "O":
            bias[index] = float(per_layer.get(label[2:], 0.0))
    return bias


def inclusive_iou(
    left: tuple[str, int, int],
    right: tuple[str, int, int],
) -> float:
    lo, hi = max(left[1], right[1]), min(left[2], right[2])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    union = (left[2] - left[1] + 1) + (right[2] - right[1] + 1) - inter
    return inter / union if union else 0.0


def match_iou(
    gold: Sequence[tuple[str, int, int]],
    pred: Sequence[tuple[str, int, int]],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    by_tag: dict[str, list[int]] = defaultdict(list)
    for index, span in enumerate(gold):
        by_tag[span[0]].append(index)
    candidates: list[tuple[float, int, int]] = []
    for p_index, p_span in enumerate(pred):
        for g_index in by_tag.get(p_span[0], ()):
            value = inclusive_iou(gold[g_index], p_span)
            if value >= threshold:
                candidates.append((value, g_index, p_index))
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, g_index, p_index in candidates:
        if g_index in used_gold or p_index in used_pred:
            continue
        used_gold.add(g_index)
        used_pred.add(p_index)
        matches.append((g_index, p_index, value))
    return matches


def prf(true_positive: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = true_positive / n_pred if n_pred else 0.0
    recall = true_positive / n_gold if n_gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def sequences_from_ids(
    pred_ids: Sequence[Sequence[int]] | np.ndarray,
    gold_ids: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    pred_seqs: list[list[str]] = []
    gold_seqs: list[list[str]] = []
    for preds, golds in zip(pred_ids, gold_ids, strict=True):
        pred_seqs.append(
            [
                ID2LABEL[int(pred)]
                for pred, gold in zip(preds, golds, strict=True)
                if int(gold) != -100
            ]
        )
        gold_seqs.append(
            [ID2LABEL[int(gold)] for gold in golds if int(gold) != -100]
        )
    return pred_seqs, gold_seqs


def token_recall_inside_gold(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
) -> dict[str, float]:
    hit = 0
    total = 0
    per_tag_hit: dict[str, int] = defaultdict(int)
    per_tag_total: dict[str, int] = defaultdict(int)
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        for pred, gold in zip(pred_seq, gold_seq, strict=True):
            if gold == "O":
                continue
            tag = gold[2:]
            total += 1
            per_tag_total[tag] += 1
            matched = pred != "O" and pred[2:] == tag
            hit += int(matched)
            per_tag_hit[tag] += int(matched)
    out = {
        "token_recall_inside_gold": hit / total if total else 0.0,
        "token_recall_n": float(total),
    }
    for tag in TAG_NAMES:
        denom = per_tag_total[tag]
        out[f"token_recall_{tag.lower()}"] = (
            per_tag_hit[tag] / denom if denom else 0.0
        )
    return out


def iou50_from_sequences(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    tp = 0
    n_pred = 0
    n_gold = 0
    per_tag_tp: dict[str, int] = defaultdict(int)
    per_tag_pred: dict[str, int] = defaultdict(int)
    per_tag_gold: dict[str, int] = defaultdict(int)
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        gold = bio_spans(gold_seq)
        pred = bio_spans(pred_seq)
        n_gold += len(gold)
        n_pred += len(pred)
        matches = match_iou(gold, pred, threshold)
        tp += len(matches)
        for tag, _start, _end in gold:
            per_tag_gold[tag] += 1
        for tag, _start, _end in pred:
            per_tag_pred[tag] += 1
        for g_index, _p_index, _value in matches:
            per_tag_tp[gold[g_index][0]] += 1
    precision, recall, f1 = prf(tp, n_pred, n_gold)
    metrics = {
        "iou50_precision": precision,
        "iou50_recall": recall,
        "iou50_f1": f1,
        "iou50_n_pred": float(n_pred),
        "iou50_n_gold": float(n_gold),
    }
    for tag in TAG_NAMES:
        _p, _r, tag_f1 = prf(per_tag_tp[tag], per_tag_pred[tag], per_tag_gold[tag])
        metrics[f"iou50_f1_{tag.lower()}"] = tag_f1
    return metrics


def compute_eval_metrics(
    logits: np.ndarray | torch.Tensor,
    label_ids: np.ndarray | torch.Tensor,
    *,
    break_penalty: float = DEFAULT_IN_LOOP_BREAK_PENALTY,
    attention_mask: np.ndarray | torch.Tensor | None = None,
    label_bias: np.ndarray | None = None,
) -> dict[str, float]:
    """Viterbi IoU@0.5 plus token-recall-inside-gold. No per-layer bias by default."""
    if isinstance(logits, np.ndarray):
        logit_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    else:
        logit_t = logits.detach().float().cpu()
    if label_bias is not None:
        logit_t = logit_t + torch.as_tensor(label_bias, dtype=logit_t.dtype).view(1, 1, -1)
    mask_t = None
    if attention_mask is not None:
        mask_t = torch.as_tensor(attention_mask, dtype=torch.float32)
    pred_ids = viterbi_decode_torch(
        logit_t, break_penalty=break_penalty, attention_mask=mask_t
    ).numpy()
    gold_ids = np.asarray(label_ids)
    pred_seqs, gold_seqs = sequences_from_ids(pred_ids, gold_ids)
    metrics = iou50_from_sequences(pred_seqs, gold_seqs)
    metrics.update(token_recall_inside_gold(pred_seqs, gold_seqs))
    metrics["viterbi_break_penalty"] = float(break_penalty)
    return metrics


def compute_metrics_for_trainer(
    eval_pred: Any,
    *,
    break_penalty: float = DEFAULT_IN_LOOP_BREAK_PENALTY,
) -> dict[str, float]:
    """HuggingFace ``compute_metrics`` entry point."""
    predictions, label_ids = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return compute_eval_metrics(
        predictions, label_ids, break_penalty=break_penalty
    )


def argmax_decode(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=-1).astype(np.int16)


def _local_maxima(values: np.ndarray, threshold: float) -> np.ndarray:
    """True at strict local maxima above ``threshold``. ``values`` is ``[T]``."""
    n = values.shape[0]
    peaks = np.zeros(n, dtype=bool)
    if n == 0:
        return peaks
    for index in range(n):
        current = float(values[index])
        if current <= threshold:
            continue
        left = float(values[index - 1]) if index > 0 else -np.inf
        right = float(values[index + 1]) if index + 1 < n else -np.inf
        if current >= left and current >= right:
            peaks[index] = True
    return peaks


def asrf_decode(
    bio_logits: np.ndarray,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    *,
    attention_mask: np.ndarray | None = None,
    threshold: float = DEFAULT_ASRF_THRESHOLD,
) -> np.ndarray:
    """Cut at start/end local maxima > 0.5; majority-vote BIO label per segment.

    ``bio_logits`` / ``start_logits`` / ``end_logits`` are ``[T, C]``.
    Returns a ``[T]`` label-id path. The joint model has no aux heads — callers skip this.
    """
    seq = bio_logits.shape[0]
    token_ids = argmax_decode(bio_logits)
    start_p = 1.0 / (1.0 + np.exp(-np.asarray(start_logits, dtype=np.float64)))
    end_p = 1.0 / (1.0 + np.exp(-np.asarray(end_logits, dtype=np.float64)))
    start_score = start_p.max(axis=-1)
    end_score = end_p.max(axis=-1)
    if attention_mask is not None:
        start_score = np.where(attention_mask > 0, start_score, 0.0)
        end_score = np.where(attention_mask > 0, end_score, 0.0)
    cuts = _local_maxima(start_score, threshold) | _local_maxima(end_score, threshold)
    bounds = [0]
    bounds.extend(int(index) for index in np.flatnonzero(cuts))
    bounds.append(seq)
    unique_bounds = sorted(set(bounds))
    path = np.zeros(seq, dtype=np.int16)
    for left, right in zip(unique_bounds, unique_bounds[1:]):
        if right <= left:
            continue
        votes = Counter_safe(token_ids[left:right])
        majority = max(votes, key=lambda key: (votes[key], -int(key)))
        label = int(majority)
        if label == LABEL2ID["O"]:
            path[left:right] = LABEL2ID["O"]
            continue
        tag = ID2LABEL[label][2:] if ID2LABEL[label] != "O" else None
        if tag is None:
            path[left:right] = LABEL2ID["O"]
            continue
        path[left] = LABEL2ID[f"B-{tag}"]
        if right - left > 1:
            path[left + 1 : right] = LABEL2ID[f"I-{tag}"]
    return path


def Counter_safe(values: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[int(value)] += 1
    return counts


def muc_taxonomy(
    gold: Sequence[tuple[str, int, int]],
    pred: Sequence[tuple[str, int, int]],
) -> dict[str, int]:
    """MUC COR / PAR / INC / MIS / SPU over inclusive token spans."""
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    counts = {"COR": 0, "PAR": 0, "INC": 0, "MIS": 0, "SPU": 0}
    pairs: list[tuple[float, int, int]] = []
    for g_index, g_span in enumerate(gold):
        for p_index, p_span in enumerate(pred):
            overlap = inclusive_iou(
                (g_span[0], g_span[1], g_span[2]),
                (g_span[0], p_span[1], p_span[2]),
            )
            if overlap <= 0:
                continue
            pairs.append((overlap, g_index, p_index))
    pairs.sort(reverse=True)
    for _overlap, g_index, p_index in pairs:
        if g_index in used_gold or p_index in used_pred:
            continue
        used_gold.add(g_index)
        used_pred.add(p_index)
        g_span, p_span = gold[g_index], pred[p_index]
        exact = g_span[1] == p_span[1] and g_span[2] == p_span[2]
        if g_span[0] == p_span[0] and exact:
            counts["COR"] += 1
        elif g_span[0] == p_span[0]:
            counts["PAR"] += 1
        else:
            counts["INC"] += 1
    counts["MIS"] = len(gold) - len(used_gold)
    counts["SPU"] = len(pred) - len(used_pred)
    return counts


def boundary_mae(
    gold: Sequence[tuple[str, int, int]],
    pred: Sequence[tuple[str, int, int]],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    errors: list[int] = []
    for g_index, p_index, _value in match_iou(gold, pred, threshold):
        errors.append(
            abs(gold[g_index][1] - pred[p_index][1])
            + abs(gold[g_index][2] - pred[p_index][2])
        )
    if not errors:
        return {"boundary_mae": float("nan"), "boundary_median": float("nan"), "n": 0.0}
    return {
        "boundary_mae": float(np.mean(errors)),
        "boundary_median": float(np.median(errors)),
        "n": float(len(errors)),
    }


def coverage_histogram(
    gold_seqs: Sequence[Sequence[str]],
    pred_seqs: Sequence[Sequence[str]],
    bins: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, int]:
    """Per gold span, fraction of tokens predicted as that layer."""
    edges = list(bins)
    labels = []
    for index, left in enumerate(edges[:-1]):
        right = edges[index + 1]
        labels.append(f"[{left:.2f},{right:.2f})")
    labels.append("1.0")
    hist = {name: 0 for name in labels}
    hist["n_gold_spans"] = 0
    for gold_seq, pred_seq in zip(gold_seqs, pred_seqs, strict=True):
        for tag, start, end in bio_spans(gold_seq):
            hist["n_gold_spans"] += 1
            length = end - start + 1
            hit = 0
            for index in range(start, end + 1):
                pred = pred_seq[index]
                if pred != "O" and pred[2:] == tag:
                    hit += 1
            frac = hit / length if length else 0.0
            if frac >= 1.0:
                hist["1.0"] += 1
                continue
            placed = False
            for index, left in enumerate(edges[:-1]):
                right = edges[index + 1]
                if left <= frac < right:
                    hist[f"[{left:.2f},{right:.2f})"] += 1
                    placed = True
                    break
            if not placed:
                hist["1.0"] += 1
    return hist


def quote_overproposal(pred_seqs: Sequence[Sequence[str]], gold_seqs: Sequence[Sequence[str]]) -> float:
    n_pred = n_gold = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        n_pred += sum(1 for span in bio_spans(pred_seq) if span[0] == "QUOTE")
        n_gold += sum(1 for span in bio_spans(gold_seq) if span[0] == "QUOTE")
    return n_pred / n_gold if n_gold else 0.0


def flatten_window_metrics(
    pred_ids: Sequence[Sequence[int]],
    gold_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    pred_seqs, gold_seqs = sequences_from_ids(pred_ids, gold_ids)
    gold_spans: list[tuple[str, int, int]] = []
    pred_spans: list[tuple[str, int, int]] = []
    # Offset spans so they don't collide across windows when pooling MUC.
    cursor = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        for tag, start, end in bio_spans(gold_seq):
            gold_spans.append((tag, start + cursor, end + cursor))
        for tag, start, end in bio_spans(pred_seq):
            pred_spans.append((tag, start + cursor, end + cursor))
        cursor += max(len(gold_seq), 1)
    metrics = iou50_from_sequences(pred_seqs, gold_seqs)
    metrics.update(token_recall_inside_gold(pred_seqs, gold_seqs))
    metrics.update({f"muc_{key.lower()}": value for key, value in muc_taxonomy(gold_spans, pred_spans).items()})
    metrics.update(boundary_mae(gold_spans, pred_spans))
    metrics["quote_overproposal"] = quote_overproposal(pred_seqs, gold_seqs)
    metrics["coverage"] = coverage_histogram(pred_seqs=pred_seqs, gold_seqs=gold_seqs)
    return metrics

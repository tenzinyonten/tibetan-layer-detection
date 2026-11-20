"""Training objective: focal token CE + span Tversky + boundary BCE + EER.

The 15-class BIO softmax is unchanged. Per-layer probability is

    p_layer[c, t] = softmax(logits[t])[B_c] + softmax(logits[t])[I_c]

``L_span`` is gathered into a padded ``[n_spans, max_region_len]`` tensor and
reduced in one shot — do not loop over region tokens in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .audit_utils import MODEL_LAYERS
from .training_utils import ID2LABEL, LABEL2ID, LABELS, canonical_layer

N_BIO = len(LABELS)
N_LAYERS = len(MODEL_LAYERS)
BOOKTITLE_INDEX = MODEL_LAYERS.index("BookTitle")
EPS = 1e-6

B_INDEX = torch.tensor(
    [LABEL2ID[f"B-{_tag}"] for _tag in (
        "QUOTE", "SABCHE", "TSAWA", "YIGCHUNG", "CHAPTER", "AUTHOR", "BOOKTITLE"
    )],
    dtype=torch.long,
)
I_INDEX = torch.tensor(
    [LABEL2ID[f"I-{_tag}"] for _tag in (
        "QUOTE", "SABCHE", "TSAWA", "YIGCHUNG", "CHAPTER", "AUTHOR", "BOOKTITLE"
    )],
    dtype=torch.long,
)

DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_LABEL_SMOOTHING = 0.05
DEFAULT_TOKEN_LOSS_CLIP = 4.0
DEFAULT_TVERSKY_ALPHA = 0.3
DEFAULT_TVERSKY_BETA = 0.7
DEFAULT_POS_WEIGHT_CAP = 100.0
DEFAULT_AUX_RAMP_STEPS = 2000
DEFAULT_LAMBDAS = {"span": 1.0, "boundary": 0.1, "eer": 1.0}


def _move_index(index: torch.Tensor, device: torch.device) -> torch.Tensor:
    if index.device != device:
        return index.to(device)
    return index


def layer_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """``p_layer`` of shape ``[batch, n_layers, seq]`` from 15-class BIO logits."""
    probs = F.softmax(logits.float(), dim=-1)
    b_index = _move_index(B_INDEX, logits.device)
    i_index = _move_index(I_INDEX, logits.device)
    return probs.index_select(-1, b_index) + probs.index_select(-1, i_index)
    # [B, T, C] — caller may want [B, C, T]
    # keep [B, T, C] internally via permute at use sites


def layer_probs_bct(logits: torch.Tensor) -> torch.Tensor:
    """Per-layer probability ``[batch, n_layers, seq]``."""
    return layer_probs_from_logits(logits).permute(0, 2, 1).contiguous()


def lambda_at_step(target: float, step: int, ramp_steps: int) -> float:
    """Linear ramp from 0 to ``target`` over ``ramp_steps``. No ramp if ``<= 0``."""
    if ramp_steps <= 0:
        return float(target)
    if step <= 0:
        return 0.0
    return float(target) * min(1.0, step / ramp_steps)


@dataclass(frozen=True)
class EerPrior:
    rho: float
    gamma: float


def parse_eer_priors(raw: Mapping[str, Mapping[str, float]]) -> list[EerPrior]:
    """One prior per ``MODEL_LAYERS`` entry. Missing layers default to (0, 0)."""
    priors: list[EerPrior] = []
    for layer in MODEL_LAYERS:
        hub = layer.upper()
        if layer == "Quotation":
            hub = "QUOTATION"
        entry = raw.get(hub) or raw.get(layer) or {}
        priors.append(
            EerPrior(rho=float(entry.get("rho", 0.0)), gamma=float(entry.get("gamma", 0.0)))
        )
    return priors


def focal_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    clip: float = DEFAULT_TOKEN_LOSS_CLIP,
    n_classes: int = N_BIO,
) -> torch.Tensor:
    """Masked, clipped focal CE. No class weights.

    Reduce as ``sum(loss * mask) / sum(mask).clamp(min=1)``.
    """
    logits = logits.float()
    mask = loss_mask.float()
    valid = mask > 0
    safe_labels = labels.clamp(min=0)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    ce = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    p_t = torch.exp(-nll).clamp(min=EPS, max=1.0)
    focal = ((1.0 - p_t) ** gamma) * ce
    focal = focal.clamp(max=clip)
    focal = torch.where(valid, focal, torch.zeros_like(focal))
    denom = mask.sum().clamp(min=1.0)
    return (focal * mask).sum() / denom


def _layer_id_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map BIO ids to layer index; ``O`` / ``-100`` become ``-1``."""
    layer_id = torch.full_like(labels, -1)
    b_index = _move_index(B_INDEX, labels.device)
    i_index = _move_index(I_INDEX, labels.device)
    for layer, (b_i, i_i) in enumerate(zip(b_index.tolist(), i_index.tolist())):
        layer_id = torch.where(
            (labels == b_i) | (labels == i_i),
            torch.full_like(layer_id, layer),
            layer_id,
        )
    return layer_id


def _span_layer_index(layer: str) -> int:
    return MODEL_LAYERS.index(canonical_layer(str(layer)))


def pack_span_regions(
    spans_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    seq_len: int,
    include_truncated: bool,
    skip_booktitle: bool = True,
) -> dict[str, list]:
    """Collect span metadata. Cheap O(n_spans) Python; no per-token loop."""
    batch_idx: list[int] = []
    layer_idx: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    region_starts: list[int] = []
    region_ends: list[int] = []
    for b, window_spans in enumerate(spans_batch):
        for span in window_spans or []:
            if not include_truncated and bool(span.get("truncated")):
                continue
            layer = _span_layer_index(span["layer"])
            if skip_booktitle and layer == BOOKTITLE_INDEX:
                continue
            i = int(span["start"])
            j = int(span["end"])
            if j <= i:
                continue
            i = max(0, min(i, seq_len))
            j = max(0, min(j, seq_len))
            if j <= i:
                continue
            length = j - i
            margin = min(200, max(50, length // 2))
            r0 = max(0, i - margin)
            r1 = min(seq_len, j + margin)
            if r1 <= r0:
                continue
            batch_idx.append(b)
            layer_idx.append(layer)
            starts.append(i)
            ends.append(j)
            region_starts.append(r0)
            region_ends.append(r1)
    return {
        "batch_idx": batch_idx,
        "layer_idx": layer_idx,
        "starts": starts,
        "ends": ends,
        "region_starts": region_starts,
        "region_ends": region_ends,
    }


def _paint_occupancy(
    spans_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """``[B, C, T]`` bool: token lies in any gold span of that layer."""
    occ = torch.zeros(batch_size, N_LAYERS, seq_len, dtype=torch.bool, device=device)
    packed = pack_span_regions(
        spans_batch, seq_len=seq_len, include_truncated=True, skip_booktitle=True
    )
    for b, c, i, j in zip(
        packed["batch_idx"], packed["layer_idx"], packed["starts"], packed["ends"]
    ):
        occ[b, c, i:j] = True
    return occ


def span_tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spans_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    attention_mask: torch.Tensor | None = None,
    alpha: float = DEFAULT_TVERSKY_ALPHA,
    beta: float = DEFAULT_TVERSKY_BETA,
) -> torch.Tensor:
    """Mean ``1 - Tversky`` over non-truncated, non-BOOKTITLE gold spans.

    ``fp_keep`` drops tokens that sit in a *different* same-layer gold span.
    ``fn_keep`` drops tokens the 15-class collapse assigned to another layer.
    """
    p_layer = layer_probs_bct(logits)  # [B, C, T]
    batch_size, _, seq_len = p_layer.shape
    device = p_layer.device
    packed = pack_span_regions(
        spans_batch, seq_len=seq_len, include_truncated=False, skip_booktitle=True
    )
    n_spans = len(packed["batch_idx"])
    if n_spans == 0:
        return p_layer.sum() * 0.0

    region_lens = [
        r1 - r0 for r0, r1 in zip(packed["region_starts"], packed["region_ends"])
    ]
    max_len = max(region_lens)
    batch_t = torch.tensor(packed["batch_idx"], device=device, dtype=torch.long)
    layer_t = torch.tensor(packed["layer_idx"], device=device, dtype=torch.long)
    start_t = torch.tensor(packed["starts"], device=device, dtype=torch.long)
    end_t = torch.tensor(packed["ends"], device=device, dtype=torch.long)
    r0_t = torch.tensor(packed["region_starts"], device=device, dtype=torch.long)
    r1_t = torch.tensor(packed["region_ends"], device=device, dtype=torch.long)

    offsets = torch.arange(max_len, device=device)
    positions = r0_t.unsqueeze(1) + offsets.unsqueeze(0)  # [S, R]
    valid = positions < r1_t.unsqueeze(1)
    pos_clamped = positions.clamp(0, seq_len - 1)

    p_t = p_layer[batch_t.unsqueeze(1), layer_t.unsqueeze(1), pos_clamped]
    g_t = (pos_clamped >= start_t.unsqueeze(1)) & (pos_clamped < end_t.unsqueeze(1))

    occupancy = _paint_occupancy(
        spans_batch, batch_size=batch_size, seq_len=seq_len, device=device
    )
    occ_t = occupancy[batch_t.unsqueeze(1), layer_t.unsqueeze(1), pos_clamped]
    fp_keep = ~(occ_t & ~g_t)

    layer_id = _layer_id_from_labels(labels)
    assigned = layer_id[batch_t.unsqueeze(1), pos_clamped]
    fn_keep = (assigned < 0) | (assigned == layer_t.unsqueeze(1))

    if attention_mask is not None:
        content = attention_mask[batch_t.unsqueeze(1), pos_clamped].bool()
        valid = valid & content

    valid_f = valid.float()
    p_t = p_t * valid_f
    g_f = g_t.float() * valid_f
    fp_k = fp_keep.float() * valid_f
    fn_k = fn_keep.float() * valid_f

    tp = (p_t * g_f * fn_k).sum(dim=1)
    fn = ((1.0 - p_t) * g_f * fn_k).sum(dim=1)
    fp = (p_t * (1.0 - g_f) * fp_k).sum(dim=1)
    mass = tp + alpha * fp + beta * fn
    active = mass > EPS
    if not bool(active.any()):
        return p_layer.sum() * 0.0
    tversky = tp / (mass + EPS)
    return (1.0 - tversky)[active].mean()


def boundary_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_targets: torch.Tensor,
    end_targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Independent sigmoid BCE on start/end heads. Targets stay soft.

    ``start_logits`` / ``end_logits``: ``[B, T, C]``.
    ``start_targets`` / ``end_targets``: ``[B, C, T]`` (dataset layout) or ``[B, T, C]``.
    ``pos_weight``: ``[C]``, already clamped.
    BOOKTITLE channel is zeroed.
    """
    start_logits = start_logits.float()
    end_logits = end_logits.float()
    start_t = _as_btc(start_targets).float()
    end_t = _as_btc(end_targets).float()
    mask = loss_mask.float().unsqueeze(-1)  # [B, T, 1]
    channel = torch.ones(
        1, 1, start_logits.size(-1), device=start_logits.device, dtype=start_logits.dtype
    )
    channel[..., BOOKTITLE_INDEX] = 0.0
    weight = mask * channel
    pw = pos_weight.to(device=start_logits.device, dtype=start_logits.dtype)

    def _bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pw
        )
        per = per * weight
        denom = weight.sum().clamp(min=1.0)
        return per.sum() / denom

    return _bce(start_logits, start_t) + _bce(end_logits, end_t)


def _as_btc(targets: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, C, T]`` (dataset) or ``[B, T, C]`` (heads)."""
    if targets.dim() != 3:
        raise ValueError(f"targets must be 3D, got {tuple(targets.shape)}")
    if targets.size(1) == N_LAYERS and targets.size(-1) != N_LAYERS:
        return targets.permute(0, 2, 1).contiguous()
    return targets


def eer_loss(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    priors: Sequence[EerPrior],
) -> torch.Tensor:
    """Hinge on |ρ_c - mean p_layer[c]| over *all* non-padding tokens.

    ``loss_mask == 0`` tokens are included — that is the point of the term.
    BOOKTITLE is excluded.
    """
    p_layer = layer_probs_bct(logits)  # [B, C, T]
    content = attention_mask.float()  # [B, T]
    denom = content.sum().clamp(min=1.0)
    rho_hat = (p_layer * content.unsqueeze(1)).sum(dim=(0, 2)) / denom
    total = p_layer.sum() * 0.0
    for layer, prior in enumerate(priors):
        if layer == BOOKTITLE_INDEX:
            continue
        total = total + torch.relu((rho_hat[layer] - prior.rho).abs() - prior.gamma)
    return total


def predicted_density(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """ρ̂ per layer over non-padding tokens. Includes ``loss_mask == 0``."""
    p_layer = layer_probs_bct(logits)
    content = attention_mask.float()
    denom = content.sum().clamp(min=1.0)
    return (p_layer * content.unsqueeze(1)).sum(dim=(0, 2)) / denom


@dataclass
class V2LossWeights:
    token: float = 1.0
    span: float = DEFAULT_LAMBDAS["span"]
    boundary: float = DEFAULT_LAMBDAS["boundary"]
    eer: float = DEFAULT_LAMBDAS["eer"]
    ramp_span_eer_steps: int = DEFAULT_AUX_RAMP_STEPS


def compute_loss_objective(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    spans_batch: Sequence[Sequence[Mapping[str, Any]]],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_targets: torch.Tensor,
    end_targets: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_weight: torch.Tensor,
    priors: Sequence[EerPrior],
    weights: V2LossWeights,
    step: int,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    token_loss_clip: float = DEFAULT_TOKEN_LOSS_CLIP,
    tversky_alpha: float = DEFAULT_TVERSKY_ALPHA,
    tversky_beta: float = DEFAULT_TVERSKY_BETA,
) -> dict[str, torch.Tensor]:
    """Return unweighted terms, applied lambdas, and the scalar ``loss``."""
    l_token = focal_token_loss(
        logits,
        labels,
        loss_mask,
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
        clip=token_loss_clip,
    )
    l_span = span_tversky_loss(
        logits,
        labels,
        spans_batch,
        attention_mask=attention_mask,
        alpha=tversky_alpha,
        beta=tversky_beta,
    )
    l_bnd = boundary_loss(
        start_logits,
        end_logits,
        start_targets,
        end_targets,
        loss_mask,
        pos_weight,
    )
    l_eer = eer_loss(logits, attention_mask, priors)
    lam_span = lambda_at_step(weights.span, step, weights.ramp_span_eer_steps)
    lam_eer = lambda_at_step(weights.eer, step, weights.ramp_span_eer_steps)
    lam_bnd = weights.boundary
    loss = (
        weights.token * l_token
        + lam_span * l_span
        + lam_bnd * l_bnd
        + lam_eer * l_eer
    )
    return {
        "loss": loss,
        "loss_token": l_token,
        "loss_span": l_span,
        "loss_boundary": l_bnd,
        "loss_eer": l_eer,
        "lambda_span": torch.tensor(lam_span, device=loss.device),
        "lambda_boundary": torch.tensor(lam_bnd, device=loss.device),
        "lambda_eer": torch.tensor(lam_eer, device=loss.device),
    }

"""Pinned Hub load, column asserts, collator, and train-split statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .audit_utils import MODEL_LAYERS
from .training_utils import ID2LABEL, LABEL2ID, LABELS
from .loss_objective import BOOKTITLE_INDEX, N_LAYERS
from .window_supervision import JOINT_MODEL_WINDOWS

REQUIRED_COLUMNS = (
    "input_ids",
    "attention_mask",
    "labels",
    "spans",
    "start_targets",
    "end_targets",
    "loss_mask",
    "book_id",
)

HUB_LAYER_KEYS = {
    "Quotation": ("QUOTATION", "Quotation"),
    "Sabche": ("SABCHE", "Sabche"),
    "Tsawa": ("TSAWA", "Tsawa"),
    "Yigchung": ("YIGCHUNG", "Yigchung"),
    "Chapter": ("CHAPTER", "Chapter"),
    "Author": ("AUTHOR", "Author"),
    "BookTitle": ("BOOKTITLE", "BookTitle"),
}


class DatasetRevisionError(ValueError):
    """``dataset_revision`` was missing, null, or otherwise unpinned."""


def require_pinned_revision(revision: Any) -> str:
    if revision is None:
        raise DatasetRevisionError(
            "dataset_revision must be pinned (e.g. w2.0); unpinned loads are refused"
        )
    text = str(revision).strip()
    if not text or text.lower() in {"null", "none"}:
        raise DatasetRevisionError(
            "dataset_revision must be pinned (e.g. w2.0); unpinned loads are refused"
        )
    return text


def assert_columns(column_names: Sequence[str]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in set(column_names)]
    if missing:
        raise ValueError(
            "windowed dataset is missing required columns: " + ", ".join(missing)
        )


def load_windowed_splits(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load the pinned windowed Hub config. Does not rebuild windows."""
    from datasets import DownloadMode, load_dataset

    revision = require_pinned_revision(cfg.get("dataset_revision"))
    repo = str(cfg["dataset_repo"])
    config_name = str(cfg.get("dataset_config_name") or cfg.get("hub_config_name"))
    if not config_name:
        raise ValueError("dataset_config_name is required (windowed_w8192_s4916)")
    dataset = load_dataset(
        repo,
        config_name,
        revision=revision,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    splits = {}
    for name in ("train", "validation", "test"):
        if name not in dataset:
            continue
        assert_columns(dataset[name].column_names)
        splits["val" if name == "validation" else name] = dataset[name]
    if "train" not in splits:
        raise ValueError("windowed dataset has no train split")
    return splits


def _as_layer_time(array: Any, seq_len: int | None = None) -> np.ndarray:
    """Accept ``[C, T]`` (dataset) or ``[T, C]``. Return ``[C, T]``."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"targets must be 2D, got {values.shape}")
    if values.shape[0] == N_LAYERS:
        return values
    if values.shape[1] == N_LAYERS:
        return values.T
    if seq_len is not None and values.shape[1] == seq_len and values.shape[0] != N_LAYERS:
        raise ValueError(f"cannot interpret targets of shape {values.shape} as [C, T]")
    raise ValueError(f"cannot interpret targets of shape {values.shape} as [C, T]")


def _pad_1d(rows: Sequence[Any], seq_len: int, fill: float) -> torch.Tensor:
    out = torch.full((len(rows), seq_len), fill, dtype=torch.float32)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=torch.float32)
        length = min(int(values.numel()), seq_len)
        if length:
            out[index, :length] = values.reshape(-1)[:length]
    return out


def _pad_targets(rows: Sequence[Any], seq_len: int) -> torch.Tensor:
    out = torch.zeros(len(rows), N_LAYERS, seq_len, dtype=torch.float32)
    for index, row in enumerate(rows):
        values = torch.as_tensor(_as_layer_time(row, seq_len), dtype=torch.float32)
        length = min(int(values.size(-1)), seq_len)
        if length:
            out[index, :, :length] = values[:, :length]
    return out


class V2DataCollator:
    """Pad BIO fields plus ``loss_mask`` / start-end targets; keep ``spans`` as lists."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        pad_to_multiple_of: int | None = 8,
        label_pad_token_id: int = -100,
    ):
        from transformers import DataCollatorForTokenClassification

        self.base = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        spans = [feature.pop("spans", []) or [] for feature in features]
        book_ids = [feature.pop("book_id", "") for feature in features]
        start_targets = [feature.pop("start_targets") for feature in features]
        end_targets = [feature.pop("end_targets") for feature in features]
        loss_masks = [feature.pop("loss_mask") for feature in features]
        for extra in (
            "window_index",
            "offset_mapping",
            "special_tokens_mask",
            "scope_mask",
            "token_type_ids",
        ):
            for feature in features:
                feature.pop(extra, None)
        batch = self.base(features)
        seq_len = int(batch["input_ids"].size(1))
        batch["spans"] = spans
        batch["book_id"] = book_ids
        batch["loss_mask"] = _pad_1d(loss_masks, seq_len, 0.0)
        batch["start_targets"] = _pad_targets(start_targets, seq_len)
        batch["end_targets"] = _pad_targets(end_targets, seq_len)
        return batch


def boundary_pos_weight(
    rows: Iterable[Mapping[str, Any]],
    *,
    cap: float = 100.0,
    positive_threshold: float = 0.5,
) -> torch.Tensor:
    """``n_neg / n_pos`` per layer from the train split, clamped. BOOKTITLE unused."""
    n_pos = np.zeros(N_LAYERS, dtype=np.float64)
    n_neg = np.zeros(N_LAYERS, dtype=np.float64)
    for row in rows:
        mask = np.asarray(row["loss_mask"], dtype=np.float32)
        start = _as_layer_time(row["start_targets"], seq_len=mask.shape[0])
        end = _as_layer_time(row["end_targets"], seq_len=mask.shape[0])
        kept = mask > 0
        for layer in range(N_LAYERS):
            if layer == BOOKTITLE_INDEX:
                continue
            for targets in (start[layer], end[layer]):
                n_pos[layer] += float(((targets >= positive_threshold) & kept).sum())
                n_neg[layer] += float(((targets < positive_threshold) & kept).sum())
    weights = np.ones(N_LAYERS, dtype=np.float32)
    for layer in range(N_LAYERS):
        if layer == BOOKTITLE_INDEX:
            continue
        weights[layer] = float(n_neg[layer] / max(n_pos[layer], 1.0))
    weights = np.clip(weights, 1.0, cap)
    weights[BOOKTITLE_INDEX] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def token_density_by_layer(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Gold layer rate over non-padding tokens, including ``loss_mask == 0``."""
    counts = Counter()
    n_content = 0
    for row in rows:
        attention = np.asarray(row.get("attention_mask", [1] * len(row["labels"])))
        labels = np.asarray(row["labels"])
        length = min(len(attention), len(labels))
        for index in range(length):
            if int(attention[index]) == 0:
                continue
            n_content += 1
            label_id = int(labels[index])
            if label_id < 0:
                continue
            name = ID2LABEL.get(label_id, "O")
            if name == "O" or "-" not in name:
                continue
            tag = name.split("-", 1)[1]
            counts[tag] += 1
    denom = max(n_content, 1)
    out: dict[str, float] = {}
    for layer in MODEL_LAYERS:
        tag = "QUOTE" if layer == "Quotation" else layer.upper()
        out[layer] = counts[tag] / denom
    out["__content_tokens__"] = float(n_content)
    return out


def format_density_report(
    measured: Mapping[str, float],
    configured: Mapping[str, Mapping[str, float]],
) -> str:
    lines = [
        "EER priors vs measured token density "
        "(non-padding tokens, including loss_mask==0):"
    ]
    for layer in MODEL_LAYERS:
        hub, title = HUB_LAYER_KEYS[layer]
        entry = configured.get(hub) or configured.get(title) or configured.get(layer) or {}
        rho = float(entry.get("rho", 0.0))
        gamma = float(entry.get("gamma", 0.0))
        density = float(measured.get(layer, 0.0))
        flag = "  OK" if rho < density or layer == "BookTitle" else "  WARN rho >= measured"
        if layer == "BookTitle":
            flag = "  unused"
        lines.append(
            f"  {layer:10s}  measured={density:.5f}  rho={rho:.5f}  "
            f"gamma={gamma:.5f}{flag}"
        )
    return "\n".join(lines)


def steps_per_epoch(n_windows: int, batch_size: int, grad_accum: int) -> int:
    """Match HF: ``len(dataloader) // gradient_accumulation_steps``."""
    if batch_size <= 0 or grad_accum <= 0:
        raise ValueError("batch_size and grad_accum must be positive")
    loader_len = max(1, math.ceil(n_windows / batch_size))
    return max(1, loader_len // grad_accum)


def report_window_step_counts(
    n_train: int,
    n_val: int,
    *,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
    warmup_ratio: float = 0.2,
) -> dict[str, float | int]:
    """Print joint-model vs current window and step counts; return the computed schedule."""
    joint_steps = steps_per_epoch(JOINT_MODEL_WINDOWS, batch_size, grad_accum)
    train_steps = steps_per_epoch(n_train, batch_size, grad_accum)
    warmup = int(round(train_steps * warmup_ratio * num_epochs))
    eval_steps = max(1, train_steps // 4)
    joint_warmup = int(round(joint_steps * warmup_ratio * num_epochs))
    print(
        f"windows: train={n_train} val={n_val}  "
        f"(joint-model train={JOINT_MODEL_WINDOWS}; delta={n_train - JOINT_MODEL_WINDOWS:+d})"
    )
    print(
        f"steps/epoch: current={train_steps}  joint-model={joint_steps}  "
        f"(batch={batch_size} accum={grad_accum})"
    )
    print(
        f"warmup_steps={warmup}  (joint-model equivalent {joint_warmup}; "
        f"{warmup_ratio} × {num_epochs} epochs × steps/epoch)"
    )
    evals_per_epoch = max(1, train_steps // eval_steps)
    print(f"eval_steps={eval_steps}  ({evals_per_epoch}× per epoch)")
    return {
        "n_train": n_train,
        "n_val": n_val,
        "joint_model_windows": JOINT_MODEL_WINDOWS,
        "joint_model_steps_per_epoch": joint_steps,
        "steps_per_epoch": train_steps,
        "warmup_steps": warmup,
        "eval_steps": eval_steps,
        "evals_per_epoch": evals_per_epoch,
    }


def early_stopping_eval_patience(
    patience_epochs: int,
    *,
    evals_per_epoch: int,
) -> int:
    """HF ``EarlyStoppingCallback`` counts evals, not epochs.

    ``patience_epochs=2`` with 4 evals/epoch must be 8 evals, not 2 — otherwise
    a mid-epoch dip after the first peak stops the run before epoch 2.
    """
    if patience_epochs < 1:
        raise ValueError("early_stopping_patience must be at least 1 epoch")
    if evals_per_epoch < 1:
        raise ValueError("evals_per_epoch must be at least 1")
    return int(patience_epochs) * int(evals_per_epoch)

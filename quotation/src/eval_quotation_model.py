#!/usr/bin/env python3
"""Offline evaluation for a BIOE quotation checkpoint.

    python quotation/src/eval_quotation_model.py \
        --config quotation/configs/train_quotation_model.yaml \
        --checkpoint quotation/outputs/final

    python quotation/src/eval_quotation_model.py --config ... --checkpoint ... \
        --split test --tune-on validation

What it reports:

* a break-penalty x logit-bias sweep
* IoU@0.5 precision / recall / F1, pooled and per book
* the pooled score with the largest book removed
* MUC COR / PAR / INC / MIS / SPU
* boundary MAE on matched spans
* the per-gold-span coverage histogram (BIO-downgraded sequences)
* argmax decode alongside Viterbi
* BIO-downgrade IoU@0.5 and the book-stitched score
* per-marker-class IoU and e_recall, rejoined from quotation_gold_merged.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("quotation_lib") is None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quotation_lib.bioe_layer import (  # noqa: E402
    DEFAULT_BIAS_SWEEP,
    DEFAULT_PENALTY_SWEEP,
    best_decode,
    bias_vector,
    bioe_spans,
    build_scheme,
    downgrade_to_bio,
    evaluate_logits,
    format_split_stats,
    iou50_metrics,
    load_bioe_splits,
    sequences_from_ids,
    stitch_book_logits,
    sweep_decode,
    viterbi_decode,
)
from quotation_lib.training_utils import bio_spans  # noqa: E402
from quotation_lib.window_data import require_pinned_revision  # noqa: E402
from quotation_lib.span_metrics import (  # noqa: E402
    boundary_mae,
    coverage_histogram,
    match_iou,
    muc_taxonomy,
    prf,
)

MARKER_CLASSES = ("TEXT", "PERSON", "MISC", "CHAINED", "FORMULA")
DEFAULT_GOLD = REPO_ROOT / "Data" / "audit" / "quotation_gold_merged.json"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def collect_logits(model, dataset, tokenizer, *, batch_size: int, device) -> tuple:
    """Run the model over a split and return ``(logits, labels)`` as numpy."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
            if (index + 1) % 20 == 0:
                print(f"  {(index + 1) * batch_size} windows", flush=True)

    width = max(array.shape[1] for array in all_logits)
    n_labels = all_logits[0].shape[2]
    rows = sum(array.shape[0] for array in all_logits)
    logits = np.zeros((rows, width, n_labels), dtype=np.float32)
    gold = np.full((rows, width), -100, dtype=np.int64)
    cursor = 0
    for chunk_logits, chunk_labels in zip(all_logits, all_labels, strict=True):
        count, length = chunk_labels.shape
        logits[cursor : cursor + count, :length] = chunk_logits
        gold[cursor : cursor + count, :length] = chunk_labels
        cursor += count
    return logits, gold


def span_level_diagnostics(pred_seqs, gold_seqs) -> dict:
    """MUC, boundary MAE and the coverage histogram, pooled over windows."""
    gold_spans: list[tuple[str, int, int]] = []
    pred_spans: list[tuple[str, int, int]] = []
    cursor = 0
    for pred_seq, gold_seq in zip(pred_seqs, gold_seqs, strict=True):
        for tag, start, end in bioe_spans(gold_seq):
            gold_spans.append((tag, start + cursor, end + cursor))
        for tag, start, end in bioe_spans(pred_seq):
            pred_spans.append((tag, start + cursor, end + cursor))
        cursor += max(len(gold_seq), 1)
    out: dict = {
        f"muc_{key.lower()}": value
        for key, value in muc_taxonomy(gold_spans, pred_spans).items()
    }
    out.update(boundary_mae(gold_spans, pred_spans))
    gold_bio = [downgrade_to_bio(seq) for seq in gold_seqs]
    pred_bio = [downgrade_to_bio(seq) for seq in pred_seqs]
    out["coverage"] = coverage_histogram(gold_seqs=gold_bio, pred_seqs=pred_bio)
    return out


def print_sweep(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'penalty':>8s} {'bias':>6s} {'f1':>8s} {'P':>8s} {'R':>8s} {'pred/gold':>10s}")
    for row in rows:
        print(
            f"  {row['break_penalty']:8g} {row['bias']:6g} "
            f"{row['iou50_f1']:8.4f} {row['iou50_precision']:8.4f} "
            f"{row['iou50_recall']:8.4f} {row['iou50_overproposal']:10.2f}"
        )


def print_per_book(metrics: dict) -> None:
    books = sorted(
        key[len("book_") : -len("_n_gold")]
        for key in metrics
        if key.startswith("book_") and key.endswith("_n_gold")
    )
    if not books:
        return
    print(f"\nper book ({len(books)} with gold spans)")
    print(f"  {'book':12s} {'gold':>6s} {'pred':>6s} {'f1':>8s} {'recall':>8s}")
    for book in sorted(books, key=lambda b: -metrics[f"book_{b}_n_gold"]):
        print(
            f"  {book:12s} {int(metrics[f'book_{book}_n_gold']):6d} "
            f"{int(metrics[f'book_{book}_n_pred']):6d} "
            f"{metrics[f'book_{book}_iou50_f1']:8.4f} "
            f"{metrics[f'book_{book}_iou50_recall']:8.4f}"
        )
    if "iou50_f1_excl_top_book" in metrics:
        print(
            f"  pooled without the largest book: "
            f"{metrics['iou50_f1_excl_top_book']:.4f} "
            f"({int(metrics.get('excl_top_book_n_gold', 0))} gold spans)"
        )


def load_merged_gold(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    books: dict[str, list[tuple[int, int, str]]] = {}
    for book_id, book in data.get("books", {}).items():
        books[str(book_id)] = [
            (int(span["start"]), int(span["end"]), str(span["cls"]))
            for span in book.get("spans", [])
        ]
    return books


def _offset_pairs(offset_mapping: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for pair in offset_mapping:
        pairs.append((int(pair[0]), int(pair[1])))
    return pairs


def owned_char_ranges(
    gold_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    pairs = _offset_pairs(offset_mapping)
    # labels_matrix and the collator right-pad short windows to the batch width
    # with -100. The parquet offset_mapping stays ragged, so a short tail is
    # padding. A supervised label past the offset map is a real mismatch.
    if len(pairs) < len(gold_ids) and any(
        int(label) != -100 for label in gold_ids[len(pairs) :]
    ):
        raise ValueError(
            f"offset_mapping has {len(pairs)} rows but labels has {len(gold_ids)} "
            "with a supervised label past the offset map"
        )
    owned: list[tuple[int, int]] = []
    for label, (start, end) in zip(gold_ids, pairs, strict=False):
        if int(label) == -100:
            continue
        owned.append((int(start), int(end)))
    return owned


def assign_marker_class(
    book_spans: Sequence[tuple[int, int, str]],
    char_start: int,
    char_end_exc: int,
) -> str | None:
    best_cls: str | None = None
    best_overlap = 0
    for start, end_inc, cls in book_spans:
        overlap = min(end_inc + 1, char_end_exc) - max(start, char_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cls = cls
    return best_cls if best_overlap > 0 else None


def marker_class_metrics(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    gold_ids: np.ndarray,
    book_ids: Sequence[str],
    offset_mappings: Sequence[Any],
    merged: dict[str, list[tuple[int, int, str]]],
) -> dict[str, dict[str, float]]:
    gold_by_cls: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    pred_by_cls: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    e_hit: dict[str, int] = defaultdict(int)
    e_total: dict[str, int] = defaultdict(int)
    cursor = 0

    for row, (pred_seq, gold_seq, book_id) in enumerate(
        zip(pred_seqs, gold_seqs, book_ids, strict=True)
    ):
        owned = owned_char_ranges(gold_ids[row], offset_mappings[row])
        if len(owned) != len(gold_seq):
            raise ValueError(
                f"owned tokens ({len(owned)}) != gold sequence ({len(gold_seq)}) "
                f"for {book_id} row {row}"
            )
        book_spans = merged.get(str(book_id), ())

        gold_token_spans = bioe_spans(gold_seq)
        pred_token_spans = bioe_spans(pred_seq)

        gold_classes: list[str | None] = []
        for _tag, start, end in gold_token_spans:
            char_start, char_end = owned[start][0], owned[end][1]
            cls = assign_marker_class(book_spans, char_start, char_end)
            gold_classes.append(cls)
            if cls is not None:
                gold_by_cls[cls].append((_tag, start + cursor, end + cursor))

        for _tag, start, end in pred_token_spans:
            char_start, char_end = owned[start][0], owned[end][1]
            cls = assign_marker_class(book_spans, char_start, char_end)
            if cls is not None:
                pred_by_cls[cls].append((_tag, start + cursor, end + cursor))

        span_class_at_end = {
            end: cls
            for (_tag, _start, end), cls in zip(gold_token_spans, gold_classes, strict=True)
        }
        for index, (pred, gold) in enumerate(zip(pred_seq, gold_seq, strict=True)):
            if gold != "E-QUOTE":
                continue
            cls = span_class_at_end.get(index)
            if cls is None:
                char_start, char_end = owned[index]
                cls = assign_marker_class(book_spans, char_start, char_end)
            if cls is None:
                continue
            e_total[cls] += 1
            e_hit[cls] += int(pred == "E-QUOTE")

        cursor += max(len(gold_seq), 1)

    report: dict[str, dict[str, float]] = {}
    for cls in MARKER_CLASSES:
        gold = gold_by_cls.get(cls, [])
        pred = pred_by_cls.get(cls, [])
        true_positive = len(match_iou(gold, pred, 0.5))
        precision, recall, f1 = prf(true_positive, len(pred), len(gold))
        report[cls] = {
            "iou50_precision": precision,
            "iou50_recall": recall,
            "iou50_f1": f1,
            "n_gold": float(len(gold)),
            "n_pred": float(len(pred)),
            "e_recall": (e_hit[cls] / e_total[cls]) if e_total[cls] else 0.0,
            "e_n": float(e_total[cls]),
        }
    return report


def print_marker_classes(report: dict[str, dict[str, float]]) -> None:
    print("\nper marker class (rejoined on book_id + char offsets)")
    print(
        f"  {'class':10s} {'gold':>6s} {'pred':>6s} {'f1':>8s} "
        f"{'P':>8s} {'R':>8s} {'e_recall':>9s}"
    )
    for cls in MARKER_CLASSES:
        row = report[cls]
        print(
            f"  {cls:10s} {int(row['n_gold']):6d} {int(row['n_pred']):6d} "
            f"{row['iou50_f1']:8.4f} {row['iou50_precision']:8.4f} "
            f"{row['iou50_recall']:8.4f} {row['e_recall']:9.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="split to report. 'test' also requires --tune-on validation.",
    )
    parser.add_argument(
        "--tune-on",
        default=None,
        choices=("train", "validation"),
        help=(
            "choose the decode on this split, then report --split at that "
            "setting. Required when --split is test."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=list(DEFAULT_PENALTY_SWEEP),
    )
    parser.add_argument(
        "--biases", type=float, nargs="+", default=list(DEFAULT_BIAS_SWEEP)
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_pinned_revision(cfg.get("dataset_revision"))
    scheme = build_scheme(str(cfg["target_layer"]))

    if args.split == "test" and args.tune_on is None:
        parser.error(
            "scoring the held-out test split requires --tune-on validation, so "
            "the decode is not chosen on test. This is the one guard that "
            "makes a test number worth reporting."
        )
    if args.split == "test" and args.tune_on != "validation":
        parser.error("--split test requires --tune-on validation")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    model = AutoModelForTokenClassification.from_pretrained(str(args.checkpoint))
    if model.config.num_labels != scheme.n_labels:
        raise SystemExit(
            f"checkpoint has {model.config.num_labels} labels but "
            f"BIOE needs {scheme.n_labels}."
        )
    model.to(device)

    splits = load_bioe_splits(cfg)
    wanted = args.split
    needed = {wanted}
    if args.tune_on:
        needed.add(args.tune_on)

    bundles: dict[str, object] = {}
    logits: dict[str, tuple] = {}
    for name in sorted(needed):
        if name not in splits:
            raise SystemExit(f"windowed dataset has no {name} split")
        bundle = splits[name]
        print(format_split_stats(name, bundle.stats))
        bundles[name] = bundle
        print(f"running the model over {name} ({len(bundle)} windows)")
        logits[name] = collect_logits(
            model, bundle.dataset, tokenizer, batch_size=args.batch_size, device=device
        )

    tune_split = args.tune_on if args.tune_on else wanted
    tune_logits, tune_gold = logits[tune_split]
    tune_bundle = bundles[tune_split]
    sweep = sweep_decode(
        tune_logits,
        tune_gold,
        scheme,
        penalties=args.penalties,
        biases=args.biases,
        book_ids=tune_bundle.book_ids,
        window_indices=tune_bundle.window_indices,
    )
    print_sweep(sweep, f"decode sweep on {tune_split}")
    chosen = best_decode(sweep)
    penalty = float(chosen["break_penalty"])
    bias = float(chosen["bias"])
    print(
        f"\nchosen on {tune_split}: break_penalty={penalty:g} bias={bias:g} "
        f"→ iou50_f1 {chosen['iou50_f1']:.4f}"
    )
    if tune_split != wanted:
        print(f"applying that setting to {wanted}, unchanged")

    bundle = bundles[wanted]
    report_logits, report_gold = logits[wanted]
    label_bias = bias_vector(scheme, bias) if bias else None
    metrics = evaluate_logits(
        report_logits,
        report_gold,
        scheme,
        break_penalty=penalty,
        label_bias=label_bias,
        book_ids=bundle.book_ids,
        window_indices=bundle.window_indices,
    )

    biased = report_logits
    if label_bias is not None:
        biased = report_logits + label_bias.reshape(1, 1, -1)
    pred_seqs, gold_seqs = sequences_from_ids(
        viterbi_decode(biased, scheme, break_penalty=penalty), report_gold, scheme
    )
    diagnostics = span_level_diagnostics(pred_seqs, gold_seqs)

    pred_bio = [downgrade_to_bio(seq) for seq in pred_seqs]
    gold_bio = [downgrade_to_bio(seq) for seq in gold_seqs]
    downgrade = iou50_metrics(
        pred_bio, gold_bio, prefix="iou50_downgrade", span_reader=bio_spans
    )
    stitched_preds, stitched_golds = stitch_book_logits(
        biased,
        report_gold,
        bundle.book_ids,
        bundle.window_indices,
        scheme,
        break_penalty=penalty,
    )
    stitched = iou50_metrics(stitched_preds, stitched_golds, prefix="iou50_stitched")

    marker_report = None
    if bundle.offset_mappings is None:
        print("\nno offset_mapping on this split; skipping per-marker-class metrics")
    elif not args.gold.exists():
        print(f"\ngold file {args.gold} is missing; skipping per-marker-class metrics")
    else:
        merged = load_merged_gold(args.gold)
        marker_report = marker_class_metrics(
            pred_seqs,
            gold_seqs,
            report_gold,
            bundle.book_ids,
            bundle.offset_mappings,
            merged,
        )

    print(f"\n=== {wanted} @ penalty={penalty:g} bias={bias:g} ===")
    print(
        f"  iou50   f1={metrics['iou50_f1']:.4f}  "
        f"P={metrics['iou50_precision']:.4f}  R={metrics['iou50_recall']:.4f}  "
        f"pred/gold={metrics['iou50_overproposal']:.2f}x  "
        f"n_gold={int(metrics['iou50_n_gold'])}"
    )
    print(
        f"  argmax  f1={metrics.get('argmax_iou50_f1', 0.0):.4f}  "
        f"(the Viterbi decode is worth "
        f"{metrics['iou50_f1'] - metrics.get('argmax_iou50_f1', 0.0):+.4f})"
    )
    print(
        f"  token recall inside gold = "
        f"{metrics['token_recall_inside_gold']:.4f}  "
        f"mean IoU of matches = {metrics['iou50_mean_iou']:.4f}  "
        f"e_recall = {metrics['e_recall']:.4f}"
    )
    print(
        f"  BIO downgrade f1={downgrade['iou50_downgrade_f1']:.4f}  "
        f"stitched f1={stitched['iou50_stitched_f1']:.4f}"
    )
    print(
        "  MUC  "
        + "  ".join(
            f"{key.upper()[4:]}={diagnostics[key]}"
            for key in ("muc_cor", "muc_par", "muc_inc", "muc_mis", "muc_spu")
        )
        + "   (MIS=never found, PAR=wrong edges, SPU=fragmentation)"
    )
    print(
        f"  boundary MAE = {diagnostics['boundary_mae']:.2f} tokens "
        f"(median {diagnostics['boundary_median']:.2f}, n={int(diagnostics['n'])})"
    )
    print(f"  coverage histogram = {diagnostics['coverage']}")
    print_per_book(metrics)
    if marker_report is not None:
        print_marker_classes(marker_report)

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": wanted,
        "tuned_on": tune_split,
        "decode": {"break_penalty": penalty, "bias": bias},
        "metrics": metrics,
        "diagnostics": diagnostics,
        "iou50_downgrade": downgrade,
        "iou50_stitched": stitched,
        "marker_classes": marker_report,
        "sweep": sweep,
        "success_criteria": cfg.get("success_criteria", {}),
    }
    target = args.out or (args.checkpoint / f"eval_{wanted}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, (np.floating, np.integer))
            else value,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

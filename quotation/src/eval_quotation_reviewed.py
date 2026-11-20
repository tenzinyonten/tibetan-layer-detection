#!/usr/bin/env python3
"""Score a BIOE quotation checkpoint on the human-reviewed test set.

    # decode chosen on validation (same guard as eval_quotation_model.py)
    python quotation/src/eval_quotation_reviewed.py \
        --config quotation/configs/train_quotation_model.yaml \
        --checkpoint karma689/tibetan-quotation-detection

    # skip the validation pass: reuse the decode from an earlier eval_test.json
    python quotation/src/eval_quotation_reviewed.py --config ... --checkpoint ... \
        --decode-from outputs/final/eval_test.json

    # before pushing: read the reviewed windows from disk instead of the Hub
    python quotation/src/eval_quotation_reviewed.py --config ... --checkpoint ... \
        --reviewed-local Data/quotation_windows_w8192_s4579_bioe_reviewed

The model runs over the test windows once. The reviewed windows have the same
input ids as the published test windows (``windowed_w8192_s4579_bioe`` @ w2.0) —
the script checks this — so the same logits are scored against both golds:

* **original** — the published test labels (the 0.853 headline number)
* **reviewed** — the human-reviewed labels (``*_reviewed`` @ w2.1)

For each it reports IoU@0.5 P/R/F1, the book-stitched score, MUC, boundary MAE,
per book, per marker class, and recall on spans seen / not seen in training.
The decode (break penalty, logit bias) is never chosen on test.

On a Vast.ai box, after ``bash quotation/src/setup_quotation_train_env.sh``. The
dataset is public; the checkpoint is private, so ``huggingface-cli login``
with an account that can read it:

    python quotation/src/eval_quotation_reviewed.py \
        --config quotation/configs/train_quotation_model.yaml \
        --checkpoint karma689/tibetan-quotation-detection

CUDA loads the checkpoint in bf16 and tries Flash Attention 2, then SDPA.
That is the same pairing training uses; fp32 attention at 8,192 tokens does
not fit a 24 GB card. Both golds are scored from one forward pass, so the
original-vs-reviewed gap does not depend on which kernel ran.
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
    SplitBundle,
    best_decode,
    bias_vector,
    bioe_spans,
    build_scheme,
    evaluate_logits,
    flat_labels,
    format_split_stats,
    iou50_metrics,
    sequences_from_ids,
    stitch_book_logits,
    sweep_decode,
    viterbi_decode,
    LazyColumn,
    MODEL_COLUMNS,
)
from quotation_lib.window_data import require_pinned_revision  # noqa: E402
from quotation_lib.span_metrics import match_iou, prf  # noqa: E402

# helpers shared with the main eval script
_spec = importlib.util.spec_from_file_location(
    "eval_quotation_model", REPO_ROOT / "src" / "eval_quotation_model.py"
)
_eval_model = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_model)
collect_logits = _eval_model.collect_logits
span_level_diagnostics = _eval_model.span_level_diagnostics
marker_class_metrics = _eval_model.marker_class_metrics
owned_char_ranges = _eval_model.owned_char_ranges
assign_marker_class = _eval_model.assign_marker_class
print_sweep = _eval_model.print_sweep
print_marker_classes = _eval_model.print_marker_classes

REVIEWED_CONFIG = "windowed_w8192_s4579_bioe_reviewed"
REVIEWED_REVISION = "w2.1"
REVIEWED_BOOKS = "reviewed/test.jsonl"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_bundle(
    *,
    repo: str,
    config_name: str,
    split: str,
    revision: str | None,
    local_dir: Path | None = None,
) -> SplitBundle:
    """One windowed split, from the Hub (pinned) or from a local shard folder."""
    from datasets import load_dataset

    if local_dir is not None:
        files = sorted(str(p) for p in (local_dir / split).glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet shards under {local_dir / split}")
        ds = load_dataset("parquet", data_files={split: files})[split]
    else:
        ds = load_dataset(
            repo,
            data_files={split: f"{config_name}/{split}/*.parquet"},
            revision=require_pinned_revision(revision),
        )[split]

    book_ids = [str(x) for x in ds["book_id"]]
    window_indices = [int(x) for x in ds["window_index"]]
    offsets = LazyColumn(ds.select_columns(["offset_mapping"]), "offset_mapping")
    extra = [c for c in ds.column_names if c not in MODEL_COLUMNS]
    model_ds = ds.remove_columns(extra)

    labels = flat_labels(model_ds)
    valid = labels[labels != -100]
    counts = np.bincount(valid, minlength=4)
    supervised = int(len(valid))
    positives = int(counts[1:].sum())
    if counts[1] != counts[3]:
        raise SystemExit(f"{config_name}/{split}: B={counts[1]} but E={counts[3]}")
    stats = {
        "windows": float(len(model_ds)),
        "supervised_tokens": float(supervised),
        "positive_tokens": float(positives),
        "positive_token_rate": positives / supervised if supervised else 0.0,
        "n_begin": float(counts[1]),
        "n_inside": float(counts[2]),
        "n_end": float(counts[3]),
    }
    bundle = SplitBundle(
        dataset=model_ds,
        book_ids=book_ids,
        window_indices=window_indices,
        offset_mappings=offsets,
        stats=stats,
    )
    bundle.check_alignment()
    return bundle


def labels_matrix(bundle: SplitBundle, width: int) -> np.ndarray:
    """Labels padded to ``width`` with -100, row order of the bundle."""
    out = np.full((len(bundle), width), -100, dtype=np.int64)
    for row, labels in enumerate(bundle.dataset["labels"]):
        out[row, : len(labels)] = labels
    return out


def check_same_inputs(a: SplitBundle, b: SplitBundle) -> None:
    """The reviewed windows must be the published windows with new labels."""
    if a.book_ids != b.book_ids or a.window_indices != b.window_indices:
        raise SystemExit("original and reviewed test windows are not in the same order")
    ids_a = a.dataset.with_format("arrow")[:].column("input_ids")
    ids_b = b.dataset.with_format("arrow")[:].column("input_ids")
    if not ids_a.equals(ids_b):
        raise SystemExit(
            "original and reviewed test windows have different input_ids; "
            "they must come from the same tokenisation"
        )


def load_book_rows(path_or_none: Path | None, *, repo: str, filename: str, revision: str) -> list[dict]:
    if path_or_none is not None:
        path = path_or_none
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def spans_by_book(rows: Sequence[dict], key: str) -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = {}
    for row in rows:
        out[str(row["book_id"])] = [
            (int(s["start"]), int(s["end"]), str(s.get(key)))
            for s in row["spans"]
        ]
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def group_recall(
    pred_seqs: Sequence[Sequence[str]],
    gold_seqs: Sequence[Sequence[str]],
    gold_ids: np.ndarray,
    book_ids: Sequence[str],
    offset_mappings: Any,
    groups: dict[str, list[tuple[int, int, str]]],
    names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """IoU@0.5 recall of gold spans split by a per-span group (e.g. seen_in_train).

    Only recall is clean here: a prediction that overlaps no gold span belongs to
    no group, so per-group precision would be flattering.
    """
    gold_by: dict[str, list] = defaultdict(list)
    pred_by: dict[str, list] = defaultdict(list)
    cursor = 0
    for row, (pred_seq, gold_seq, book_id) in enumerate(zip(pred_seqs, gold_seqs, book_ids, strict=True)):
        owned = owned_char_ranges(gold_ids[row], offset_mappings[row])
        book = groups.get(str(book_id), ())
        for tag, start, end in bioe_spans(gold_seq):
            g = assign_marker_class(book, owned[start][0], owned[end][1])
            if g is not None:
                gold_by[g].append((tag, start + cursor, end + cursor))
        for tag, start, end in bioe_spans(pred_seq):
            g = assign_marker_class(book, owned[start][0], owned[end][1])
            if g is not None:
                pred_by[g].append((tag, start + cursor, end + cursor))
        cursor += max(len(gold_seq), 1)
    out = {}
    for name in names:
        gold, pred = gold_by.get(name, []), pred_by.get(name, [])
        tp = len(match_iou(gold, pred, 0.5))
        _, recall, _ = prf(tp, len(pred), len(gold))
        out[name] = {"n_gold": float(len(gold)), "iou50_recall": recall}
    return out


def score(
    name: str,
    logits: np.ndarray,
    gold: np.ndarray,
    bundle: SplitBundle,
    scheme: Any,
    *,
    penalty: float,
    bias: float,
    book_rows: Sequence[dict],
) -> dict[str, Any]:
    label_bias = bias_vector(scheme, bias) if bias else None
    metrics = evaluate_logits(
        logits, gold, scheme,
        break_penalty=penalty, label_bias=label_bias,
        book_ids=bundle.book_ids, window_indices=bundle.window_indices,
    )
    biased = logits if label_bias is None else logits + label_bias.reshape(1, 1, -1)
    pred_seqs, gold_seqs = sequences_from_ids(
        viterbi_decode(biased, scheme, break_penalty=penalty), gold, scheme
    )
    diagnostics = span_level_diagnostics(pred_seqs, gold_seqs)
    s_pred, s_gold = stitch_book_logits(
        biased, gold, bundle.book_ids, bundle.window_indices, scheme, break_penalty=penalty
    )
    stitched = iou50_metrics(s_pred, s_gold, prefix="iou50_stitched")

    classes = marker_class_metrics(
        pred_seqs, gold_seqs, gold, bundle.book_ids, bundle.offset_mappings,
        spans_by_book(book_rows, "cls"),
    )
    seen = {
        b: [(s, e, "seen" if g == "True" else "unseen") for s, e, g in spans]
        for b, spans in spans_by_book(book_rows, "seen_in_train").items()
    }
    seen_report = group_recall(
        pred_seqs, gold_seqs, gold, bundle.book_ids, bundle.offset_mappings, seen, ("seen", "unseen")
    )
    review_report = None
    if any("review" in s for row in book_rows for s in row["spans"]):
        review_report = group_recall(
            pred_seqs, gold_seqs, gold, bundle.book_ids, bundle.offset_mappings,
            spans_by_book(book_rows, "review"), ("accepted", "edited", "added"),
        )
    return {
        "name": name,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "iou50_stitched": stitched,
        "marker_classes": classes,
        "seen_in_train": seen_report,
        "by_review_status": review_report,
    }


def print_side_by_side(results: Sequence[dict]) -> None:
    names = [r["name"] for r in results]
    print("\n" + " " * 26 + "".join(f"{n:>12s}" for n in names))
    rows = [
        ("IoU@0.5 F1", lambda r: r["metrics"]["iou50_f1"], ".4f"),
        ("  precision", lambda r: r["metrics"]["iou50_precision"], ".4f"),
        ("  recall", lambda r: r["metrics"]["iou50_recall"], ".4f"),
        ("gold spans", lambda r: r["metrics"]["iou50_n_gold"], ".0f"),
        ("pred / gold", lambda r: r["metrics"]["iou50_overproposal"], ".2f"),
        ("stitched F1", lambda r: r["iou50_stitched"]["iou50_stitched_f1"], ".4f"),
        ("argmax F1", lambda r: r["metrics"].get("argmax_iou50_f1", 0.0), ".4f"),
        ("mean IoU of matches", lambda r: r["metrics"]["iou50_mean_iou"], ".4f"),
        ("boundary MAE (tokens)", lambda r: r["diagnostics"]["boundary_mae"], ".2f"),
        ("MUC missed", lambda r: r["diagnostics"]["muc_mis"], ".0f"),
        ("MUC spurious", lambda r: r["diagnostics"]["muc_spu"], ".0f"),
        ("MUC partial", lambda r: r["diagnostics"]["muc_par"], ".0f"),
        ("recall, seen in train", lambda r: r["seen_in_train"]["seen"]["iou50_recall"], ".4f"),
        ("recall, unseen", lambda r: r["seen_in_train"]["unseen"]["iou50_recall"], ".4f"),
    ]
    for label, get, fmt in rows:
        print(f"  {label:24s}" + "".join(f"{format(float(get(r)), fmt):>12s}" for r in results))


def print_per_book(results: Sequence[dict]) -> None:
    m = [r["metrics"] for r in results]
    books = sorted(
        {k[5:-7] for k in m[-1] if k.startswith("book_") and k.endswith("_n_gold")},
        key=lambda b: -m[-1][f"book_{b}_n_gold"],
    )
    print("\nper book (IoU@0.5 F1; gold spans)")
    print(f"  {'book':12s}" + "".join(f"{r['name'] + ' f1':>16s}{'gold':>7s}" for r in results) + f"{'change':>9s}")
    for b in books:
        cells = ""
        for x in m:
            f1 = x.get(f"book_{b}_iou50_f1")
            n = x.get(f"book_{b}_n_gold", 0)
            cells += f"{(f'{f1:.4f}' if f1 is not None else '-'):>16s}{int(n):7d}"
        change = ""
        if len(m) == 2 and f"book_{b}_iou50_f1" in m[0] and f"book_{b}_iou50_f1" in m[1]:
            change = f"{m[1][f'book_{b}_iou50_f1'] - m[0][f'book_{b}_iou50_f1']:+.4f}"
        print(f"  {b:12s}{cells}{change:>9s}")


def load_scorer(checkpoint: str, cfg: dict, scheme: Any, device: Any):
    """Checkpoint + tokenizer, in the dtype and attention kernel a 24 GB GPU can run.

    Flash Attention 2 rejects fp32, and the math attention kernel at 8,192
    tokens does not fit 24 GB. Training already uses bf16 + flash, with SDPA
    if flash-attn is missing (``setup_vast_train_env.sh`` treats that as
    non-fatal). This follows that.
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    use_bf16 = device.type == "cuda" and bool(cfg.get("bf16", True))
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    requested = cfg.get("attn_implementation") if device.type == "cuda" else "sdpa"
    fallback = str(cfg.get("attn_fallback_implementation") or "sdpa")
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if requested:
        kwargs["attn_implementation"] = requested
    try:
        model = AutoModelForTokenClassification.from_pretrained(str(checkpoint), **kwargs)
    except Exception as exc:  # noqa: BLE001 — flash-attn missing or the GPU rejects it
        if requested != "flash_attention_2":
            raise
        print(
            "Flash Attention 2 could not be initialized; "
            f"retrying with {fallback!r}. Original error: {exc}",
            file=sys.stderr,
        )
        kwargs["attn_implementation"] = fallback
        model = AutoModelForTokenClassification.from_pretrained(str(checkpoint), **kwargs)
    if model.config.num_labels != scheme.n_labels:
        raise SystemExit(
            f"checkpoint has {model.config.num_labels} labels, BIOE needs {scheme.n_labels}"
        )
    model.to(device).eval()
    attn = getattr(model.config, "_attn_implementation", requested)
    print(f"device: {device}  dtype: {dtype}  attn: {attn}")
    return tokenizer, model


def print_review_status(report: dict | None) -> None:
    if not report:
        return
    print("\nreviewed gold: recall by what the reviewer did to the span")
    for status in ("accepted", "edited", "added"):
        row = report[status]
        print(f"  {status:9s} n={int(row['n_gold']):5d}  recall={row['iou50_recall']:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", required=True, help="local folder or Hub model id")
    p.add_argument("--reviewed-revision", default=REVIEWED_REVISION)
    p.add_argument("--reviewed-local", type=Path, default=None,
                   help="local folder holding test/*.parquet (skips the Hub for reviewed windows)")
    p.add_argument("--reviewed-books", type=Path, default=None,
                   help="local reviewed test.jsonl (default: reviewed/test.jsonl from the Hub, "
                        "or Data/hf_export_reviewed/test.jsonl with --reviewed-local)")
    p.add_argument("--original-books", type=Path, default=None,
                   help="local published test.jsonl (default: test.jsonl from the Hub at the config's revision)")
    p.add_argument("--no-original", action="store_true", help="score the reviewed gold only")
    p.add_argument("--decode-from", type=Path, default=None,
                   help="eval_*.json from eval_quotation_model.py; reuse its decode, skip the validation pass")
    p.add_argument("--break-penalty", type=float, default=None)
    p.add_argument("--bias", type=float, default=None)
    p.add_argument("--penalties", type=float, nargs="+", default=list(DEFAULT_PENALTY_SWEEP))
    p.add_argument("--biases", type=float, nargs="+", default=list(DEFAULT_BIAS_SWEEP))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--save-logits", type=Path, default=None,
                   help="write test logits (.npy) so the scoring can be redone without the GPU")
    p.add_argument("--out", type=Path, default=Path("outputs/eval_reviewed_test.json"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo = str(cfg["dataset_repo"])
    base_config = str(cfg.get("dataset_config_name") or "windowed_w8192_s4579_bioe")
    base_revision = require_pinned_revision(cfg.get("dataset_revision"))
    scheme = build_scheme(str(cfg["target_layer"]))

    if (args.break_penalty is None) != (args.bias is None):
        p.error("--break-penalty and --bias go together")

    # ---- data ----------------------------------------------------------------
    reviewed = load_bundle(
        repo=repo, config_name=REVIEWED_CONFIG, split="test",
        revision=args.reviewed_revision, local_dir=args.reviewed_local,
    )
    print(format_split_stats("reviewed test", reviewed.stats))
    original = None
    if not args.no_original:
        original = load_bundle(repo=repo, config_name=base_config, split="test", revision=base_revision)
        print(format_split_stats("original test", original.stats))
        check_same_inputs(original, reviewed)
        print("input ids identical between original and reviewed test windows")

    reviewed_books_path = args.reviewed_books
    if reviewed_books_path is None and args.reviewed_local is not None:
        reviewed_books_path = REPO_ROOT / "Data" / "hf_export_reviewed" / "test.jsonl"
    reviewed_rows = load_book_rows(
        reviewed_books_path, repo=repo, filename=REVIEWED_BOOKS, revision=args.reviewed_revision
    )
    original_rows = [] if original is None else load_book_rows(
        args.original_books, repo=repo, filename="test.jsonl", revision=base_revision
    )

    # ---- model ---------------------------------------------------------------
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_scorer(str(args.checkpoint), cfg, scheme, device)

    # ---- decode (never chosen on test) ---------------------------------------
    sweep = None
    if args.decode_from is not None:
        decode = json.loads(args.decode_from.read_text(encoding="utf-8"))["decode"]
        penalty, bias = float(decode["break_penalty"]), float(decode["bias"])
        source = f"from {args.decode_from}"
        tuned_on = json.loads(args.decode_from.read_text(encoding="utf-8")).get("tuned_on")
        if tuned_on == "test":
            raise SystemExit(f"{args.decode_from} was tuned on test; refusing to reuse it")
    elif args.break_penalty is not None:
        penalty, bias = args.break_penalty, args.bias
        source = "given on the command line"
    else:
        val = load_bundle(repo=repo, config_name=base_config, split="validation", revision=base_revision)
        print(format_split_stats("validation", val.stats))
        print(f"running the model over validation ({len(val)} windows)")
        v_logits, v_gold = collect_logits(model, val.dataset, tokenizer, batch_size=args.batch_size, device=device)
        sweep = sweep_decode(
            v_logits, v_gold, scheme, penalties=args.penalties, biases=args.biases,
            book_ids=val.book_ids, window_indices=val.window_indices,
        )
        print_sweep(sweep, "decode sweep on validation")
        chosen = best_decode(sweep)
        penalty, bias = float(chosen["break_penalty"]), float(chosen["bias"])
        source = "tuned on validation"
        del v_logits, v_gold
    print(f"\ndecode: break_penalty={penalty:g} bias={bias:g} ({source})")

    # ---- one pass over test, two golds ---------------------------------------
    print(f"running the model over test ({len(reviewed)} windows)")
    logits, reviewed_gold = collect_logits(
        model, reviewed.dataset, tokenizer, batch_size=args.batch_size, device=device
    )
    if args.save_logits is not None:
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_logits, logits)
        print(f"saved logits to {args.save_logits}")

    results = []
    if original is not None:
        original_gold = labels_matrix(original, logits.shape[1])
        results.append(score("original", logits, original_gold, original, scheme,
                             penalty=penalty, bias=bias, book_rows=original_rows))
    results.append(score("reviewed", logits, reviewed_gold, reviewed, scheme,
                         penalty=penalty, bias=bias, book_rows=reviewed_rows))

    print(f"\n=== test @ penalty={penalty:g} bias={bias:g} ===")
    print_side_by_side(results)
    print_per_book(results)
    for r in results:
        print(f"\n[{r['name']} gold]", end="")
        print_marker_classes(r["marker_classes"])
    print_review_status(results[-1]["by_review_status"])

    payload = {
        "checkpoint": str(args.checkpoint),
        "decode": {"break_penalty": penalty, "bias": bias, "source": source},
        "tuned_on": "validation" if sweep is not None else None,
        "sweep": sweep,
        "datasets": {
            "original": None if original is None else {"config": base_config, "revision": base_revision},
            "reviewed": {
                "config": REVIEWED_CONFIG,
                "revision": None if args.reviewed_local else args.reviewed_revision,
                "local": str(args.reviewed_local) if args.reviewed_local else None,
            },
        },
        "results": {r["name"]: r for r in results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, default=lambda v: v.item() if isinstance(v, (np.floating, np.integer)) else v),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

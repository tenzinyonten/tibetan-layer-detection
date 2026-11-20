#!/usr/bin/env python3
"""Evaluate a trained mmBERT checkpoint on the DharmaBench QUDT task.

This script runs token-level prediction on the QUDT test dataset, collapses non-QUOTE
classes, decodes using argmax and Viterbi (under BIO constraints with custom transition penalties),
and scores the results using MUC-5 (via eval4ner) and character-level IoU@0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# Insert src directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ is None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quotation_lib.bioe_layer import BioeScheme, bioe_spans  # noqa: E402
from quotation_lib.bioe_layer import transition_matrix as bioe_transition_matrix  # noqa: E402
from quotation_lib.training_utils import ID2LABEL, LABELS, bio_spans  # noqa: E402

NEG = -1.0e9
BIOE = BioeScheme()


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_span(span: str | None) -> str | None:
    if span is None:
        return None
    return span.lower().replace("-", "")


# --------------------------------------------------------------------------- #
# Decoding and Viterbi helpers
# --------------------------------------------------------------------------- #
def bio_transition_matrix(break_penalty: float = 0.0) -> np.ndarray:
    """Legal BIO transitions, with a penalty for exiting an open span."""
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


def close_dangling_span(labels: list[str]) -> list[str]:
    """Terminate a BIOE span that runs off the end of the document.

    QUDT chunks anthologies at ~1,000 characters, so roughly half its gold
    quotations are cut by the document edge (see
    docs/dharmabench-qudt-evaluation.md §4.4). A BIOE decoder that believes the
    quotation continues emits no ``E-QUOTE``, and ``bioe_spans`` then drops the
    span entirely. This is a diagnostic variant, not the faithful decode.
    """
    out = list(labels)
    for index in range(len(out) - 1, -1, -1):
        if out[index] == "O":
            return out
        if out[index].startswith("E-"):
            return out
        if out[index].startswith(("B-", "I-")):
            out[index] = "E-" + out[index].split("-", 1)[1]
            return out
    return out


def viterbi(log_probs: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """Best legal label path. log_probs is [tokens, labels]."""
    n_tokens, n_labels = log_probs.shape
    if n_tokens == 0:
        return np.zeros(0, dtype=np.int16)
    best = log_probs[0].copy()
    back = np.zeros((n_tokens, n_labels), dtype=np.int16)
    for t in range(1, n_tokens):
        scores = best[:, None] + transitions
        back[t] = np.argmax(scores, axis=0)
        best = scores[back[t], np.arange(n_labels)] + log_probs[t]
    path = np.zeros(n_tokens, dtype=np.int16)
    path[-1] = int(np.argmax(best))
    for t in range(n_tokens - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


# --------------------------------------------------------------------------- #
# Character Span matching and IoU
# --------------------------------------------------------------------------- #
def char_iou(g_span: tuple[int, int], p_span: tuple[int, int]) -> float:
    g_start, g_end = g_span
    p_start, p_end = p_span
    lo = max(g_start, p_start)
    hi = min(g_end, p_end)
    inter = max(0, hi - lo)
    union = (g_end - g_start) + (p_end - p_start) - inter
    return inter / union if union > 0 else 0.0


def match_char_iou(
    gold: list[tuple[int, int]],
    pred: list[tuple[int, int]],
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedy best-first matching of spans by character-level IoU."""
    candidates = []
    for p_index, p_span in enumerate(pred):
        for g_index, g_span in enumerate(gold):
            val = char_iou(g_span, p_span)
            if val >= threshold:
                candidates.append((val, g_index, p_index))
    candidates.sort(reverse=True)
    used_gold = set()
    used_pred = set()
    matches = []
    for val, g_index, p_index in candidates:
        if g_index in used_gold or p_index in used_pred:
            continue
        used_gold.add(g_index)
        used_pred.add(p_index)
        matches.append((g_index, p_index, val))
    return matches


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path or Hub ID")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=REPO_ROOT / "third_party/DharmaBench/data/Tibetan/QUDT/test.jsonl",
        help="Path to DharmaBench QUDT test.jsonl",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write metrics.json / predictions.jsonl (default <exp>/dharmabench_qudt)",
    )
    parser.add_argument(
        "--close-edge-spans",
        action="store_true",
        help="BIOE heads only: terminate a span left open at the document edge "
        "instead of dropping it. Diagnostic, not the faithful decode.",
    )
    args = parser.parse_args()
    close_edge_spans = args.close_edge_spans

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import eval4ner.muc as muc

    # Load configuration
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    exp_dir = REPO_ROOT

    # Resolve output directory
    output_dir = args.out_dir or (exp_dir / "results" / "dharmabench_qudt")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint/model source
    checkpoint = args.checkpoint
    if not checkpoint:
        local_final = exp_dir / "outputs/final"
        if local_final.exists():
            checkpoint = str(local_final)
        else:
            # Infer from config experiment name or path
            exp_name = cfg.get("experiment_name", exp_dir.name)
            if "v1.2" in exp_name:
                checkpoint = "karma689/mmbert-base-layer-detection-v1.2"
            else:
                checkpoint = "karma689/mmbert-base-layer-detection-v1.3"

    print(f"Loading tokenizer & model from: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    
    device = pick_device(args.device)
    dtype = torch.float32 if device in {"cpu", "mps"} else torch.bfloat16
    print(f"Using device={device} dtype={dtype}")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        attn_implementation="sdpa",
        torch_dtype=dtype,
    )
    model.to(device).eval()

    # v1.x checkpoints carry the 15-label seven-layer BIO head; the quotation
    # The quotation model carries a 4-label BIOE head for Quotation alone. The protocol
    # (MUC-5 type over QUOTE, char IoU@0.5, one forward pass per document) is
    # identical either way; only the decode differs.
    if model.config.num_labels == len(LABELS):
        scheme = "bio15"
        assert LABELS[:3] == ["O", "B-QUOTE", "I-QUOTE"], LABELS[:3]
    elif model.config.num_labels == BIOE.n_labels:
        scheme = "bioe4"
    else:
        raise SystemExit(
            f"Checkpoint has {model.config.num_labels} labels, expected "
            f"{len(LABELS)} (BIO-15) or {BIOE.n_labels} (BIOE-4)"
        )
    print(f"Label scheme: {scheme}")

    # Load test dataset
    if not args.test_file.exists():
        raise FileNotFoundError(f"DharmaBench QUDT test file not found at {args.test_file}")

    print(f"Loading DharmaBench QUDT test file: {args.test_file}")
    test_rows = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))

    if args.limit:
        test_rows = test_rows[:args.limit]
    print(f"Loaded {len(test_rows)} test examples.")

    # 1. Run inference over all test examples
    print(">> Step 1: Running model inference...")
    processed_items = []
    started_time = time.monotonic()

    for i, row in enumerate(test_rows, 1):
        text = row["text"]
        
        # Tokenize sequence
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=8192,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]
        special_tokens_mask = encoded["special_tokens_mask"]

        if len(input_ids) == 0:
            continue

        # Inference
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([input_ids], device=device),
                attention_mask=torch.tensor([attention_mask], device=device),
            )
        logits = outputs.logits[0].float().cpu().numpy()

        # Filter out special tokens & tokens with no mapping (like (0, 0))
        valid_indices = []
        for idx in range(len(logits)):
            if special_tokens_mask[idx] == 1:
                continue
            offset = offset_mapping[idx]
            if offset is None or offset == (0, 0):
                continue
            valid_indices.append(idx)

        valid_logits = logits[valid_indices]
        valid_offsets = [offset_mapping[idx] for idx in valid_indices]

        # Filter gold QUOTE-only annotations
        gold_char_spans = []
        gold_text_spans = []
        for index, (tag, span_text) in enumerate(row["ground_truth"]):
            if tag == "QUOTE":
                start, end = row["offsets"][index]
                gold_char_spans.append((start, end))
                # Normalized gold text span for MUC
                gold_text_spans.append(("QUOTE", normalize_span(span_text)))

        processed_items.append({
            "id": row["id"],
            "text": text,
            "valid_logits": valid_logits,
            "valid_offsets": valid_offsets,
            "gold_char_spans": gold_char_spans,
            "gold_text_spans": gold_text_spans,
        })

        if i % 100 == 0 or i == len(test_rows):
            print(f"  Processed {i}/{len(test_rows)} in {time.monotonic() - started_time:.1f}s")

    # 2. Sweep over decoding strategies
    print("\n>> Step 2: Sweeping decoders and penalties...")
    sweep_results = []
    decoding_strategies = ["argmax"] + [f"viterbi_penalty_{p}" for p in [0, 1, 2, 4, 8, 12]]

    for strategy in decoding_strategies:
        all_pred_text_spans = []
        all_gold_text_spans = []
        all_normalized_texts = []

        total_gold_spans = 0
        total_pred_spans = 0
        total_char_iou_matches = 0

        # 60% of QUDT documents have no gold QUOTE, and empty-vs-empty scores a
        # full 1.0 under MUC. Reporting these separately keeps the headline from
        # being read as detection skill (docs/dharmabench-qudt-evaluation.md §3).
        empty_docs = 0
        empty_docs_left_empty = 0
        gold_spans_touched = 0

        predictions_by_id = {}

        for item in processed_items:
            valid_logits = item["valid_logits"]
            valid_offsets = item["valid_offsets"]
            text = item["text"]
            gold_char_spans = item["gold_char_spans"]

            if scheme == "bio15":
                # Mask non-QUOTE logits so only O, B-QUOTE, and I-QUOTE can be chosen
                allowed_indices = [0, 1, 2]  # O, B-QUOTE, I-QUOTE
                masked_logits = np.full_like(valid_logits, NEG)
                masked_logits[:, allowed_indices] = valid_logits[:, allowed_indices]
            else:
                # A BIOE-4 head is already QUOTE-only; there is nothing to mask.
                masked_logits = valid_logits

            if strategy == "argmax":
                predictions = masked_logits.argmax(axis=1)
            else:
                penalty = float(strategy.split("_")[-1])
                log_probs = log_softmax(masked_logits)
                transitions = (
                    bio_transition_matrix(penalty)
                    if scheme == "bio15"
                    else bioe_transition_matrix(BIOE.labels, penalty)
                )
                predictions = viterbi(log_probs, transitions)

            if scheme == "bio15":
                predicted_labels = [ID2LABEL[pred_id] for pred_id in predictions]
                pred_token_spans = bio_spans(predicted_labels)
            else:
                predicted_labels = [BIOE.id2label[int(p)] for p in predictions]
                if close_edge_spans:
                    predicted_labels = close_dangling_span(predicted_labels)
                pred_token_spans = bioe_spans(predicted_labels)

            pred_char_spans = []
            pred_text_spans = []
            for tag, token_start, token_end in pred_token_spans:
                if tag == "QUOTE":
                    char_start = valid_offsets[token_start][0]
                    char_end = valid_offsets[token_end][1]
                    span_text = text[char_start:char_end]
                    pred_char_spans.append((char_start, char_end))
                    pred_text_spans.append(("QUOTE", normalize_span(span_text)))

            all_pred_text_spans.append(pred_text_spans)
            all_gold_text_spans.append(item["gold_text_spans"])
            all_normalized_texts.append(normalize_span(text))

            # Char IoU evaluation
            matches = match_char_iou(gold_char_spans, pred_char_spans, threshold=0.5)
            total_gold_spans += len(gold_char_spans)
            total_pred_spans += len(pred_char_spans)
            total_char_iou_matches += len(matches)

            if not gold_char_spans:
                empty_docs += 1
                empty_docs_left_empty += int(not pred_char_spans)
            for g_start, g_end in gold_char_spans:
                if any(
                    min(g_end, p_end) > max(g_start, p_start)
                    for p_start, p_end in pred_char_spans
                ):
                    gold_spans_touched += 1

            # Store predictions text spans
            predictions_by_id[item["id"]] = {
                "char_spans": pred_char_spans,
                "text_spans": [(tag, span_text) for tag, span_text in pred_text_spans]
            }

        # Calculate MUC-5 metrics using eval4ner
        try:
            muc_res = muc.evaluate_all(all_pred_text_spans, all_gold_text_spans, all_normalized_texts, verbose=False)
        except Exception as e:
            print(f"Error calling eval4ner: {e}")
            continue

        # Calculate character IoU Precision/Recall/F1
        iou_precision = total_char_iou_matches / total_pred_spans if total_pred_spans > 0 else 0.0
        iou_recall = total_char_iou_matches / total_gold_spans if total_gold_spans > 0 else 0.0
        iou_f1 = (2 * iou_precision * iou_recall / (iou_precision + iou_recall)) if (iou_precision + iou_recall) > 0 else 0.0

        # Save sweep info
        sweep_results.append({
            "strategy": strategy,
            "muc": muc_res,
            "iou_0.5": {
                "precision": iou_precision,
                "recall": iou_recall,
                "f1_score": iou_f1,
                "matches": total_char_iou_matches,
                "gold": total_gold_spans,
                "pred": total_pred_spans
            },
            "documents": {
                "empty_gold": empty_docs,
                "empty_gold_left_empty": empty_docs_left_empty,
                "gold_spans_touched": gold_spans_touched,
                "gold_spans": total_gold_spans,
            },
            "predictions": predictions_by_id
        })

    # Find the best Viterbi penalty based on MUC-type F1 score
    viterbi_runs = [r for r in sweep_results if r["strategy"].startswith("viterbi_")]
    best_viterbi = max(viterbi_runs, key=lambda x: x["muc"]["type"]["f1_score"])
    argmax_run = [r for r in sweep_results if r["strategy"] == "argmax"][0]

    # Save metrics
    best_by_iou = max(sweep_results, key=lambda x: x["iou_0.5"]["f1_score"])
    metrics_data = {
        "model_checkpoint": checkpoint,
        "label_scheme": scheme,
        "close_edge_spans": close_edge_spans,
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_by_iou": {
            "strategy": best_by_iou["strategy"],
            "muc": best_by_iou["muc"],
            "iou_0.5": best_by_iou["iou_0.5"],
        },
        "argmax": {
            "muc": argmax_run["muc"],
            "iou_0.5": argmax_run["iou_0.5"]
        },
        "best_viterbi": {
            "strategy": best_viterbi["strategy"],
            "muc": best_viterbi["muc"],
            "iou_0.5": best_viterbi["iou_0.5"]
        },
        "all_sweep_results": [
            {
                "strategy": r["strategy"],
                "muc": r["muc"],
                "iou_0.5": r["iou_0.5"],
                "documents": r["documents"],
            }
            for r in sweep_results
        ]
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Save predictions.jsonl
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for item in processed_items:
            p_id = item["id"]
            record = {
                "id": p_id,
                "text": item["text"],
                "gold_char_spans": item["gold_char_spans"],
                "gold_text_spans": item["gold_text_spans"],
                "predictions_argmax": argmax_run["predictions"][p_id],
                "predictions_viterbi": best_viterbi["predictions"][p_id]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {output_dir / 'predictions.jsonl'}")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"DharmaBench QUDT Performance for model: {cfg.get('experiment_name', exp_dir.name)}")
    print("=" * 80)
    print(
        f"{'Decoder':18s} | {'MUC Type':>8s} | {'MUC Exact':>9s} | {'IoU F1':>7s} | "
        f"{'pred':>5s} | {'touched':>7s} | {'empty kept':>10s}"
    )
    print("-" * 80)
    for run in sweep_results:
        docs = run["documents"]
        print(
            f"{run['strategy']:18s} | {run['muc']['type']['f1_score']:8.4f} | "
            f"{run['muc']['exact']['f1_score']:9.4f} | "
            f"{run['iou_0.5']['f1_score']:7.4f} | {run['iou_0.5']['pred']:5d} | "
            f"{docs['gold_spans_touched']:3d}/{docs['gold_spans']:<3d} | "
            f"{docs['empty_gold_left_empty']:4d}/{docs['empty_gold']:<5d}"
        )
    print("-" * 80)
    print(
        f"best by MUC type F1: {best_viterbi['strategy']} "
        f"({best_viterbi['muc']['type']['f1_score']:.4f})   "
        f"best by IoU F1: {best_by_iou['strategy']} "
        f"({best_by_iou['iou_0.5']['f1_score']:.4f})"
    )
    print("chance baseline (predict nothing everywhere): 0.6009 MUC type F1")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

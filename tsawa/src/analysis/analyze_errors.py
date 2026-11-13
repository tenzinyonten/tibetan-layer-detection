#!/usr/bin/env python3
"""
analyze_errors.py — diagnose HOW the tsawa model fails, not just how much.

Answers three questions the headline F1 can't:
  1. Is it fragmenting? (one gold span -> several predicted pieces)
  2. Does it fail on short spans specifically?
  3. What does exact-match F1 say, vs the custom tol1 metric?

Runs on CPU. No training, no GPU needed.

Usage
-----
    pip install transformers datasets torch pandas
    python analyze_errors.py \
        --model Yontenn/mmbert-tsawa-binary-v1 \
        --dataset Yontenn/formatting-tsawa-v1 \
        --split test

    # from a local checkpoint instead
    python analyze_errors.py --model ./runs/sqrt_inv_best --dataset ./tsawa/data/processed/tsawa_dataset
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification

O, B, I = 0, 1, 2


def extract_spans(seq: np.ndarray) -> list[tuple[int, int]]:
    """BIO sequence -> [(start, end_exclusive)]."""
    spans, start = [], None
    for i, lab in enumerate(seq):
        if lab == B:
            if start is not None:
                spans.append((start, i))
            start = i
        elif lab == I:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(seq)))
    return spans


def overlaps(a, b) -> int:
    """Token overlap between two spans."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_spans(gold, pred, tol):
    """Greedy one-to-one match within tol on both boundaries."""
    remaining = list(pred)
    tp = 0
    for g in gold:
        for k, p in enumerate(remaining):
            if abs(p[0] - g[0]) <= tol and abs(p[1] - g[1]) <= tol:
                tp += 1
                remaining.pop(k)
                break
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-binary-v1")
    ap.add_argument("--dataset", default="Yontenn/formatting-tsawa-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="scratch/tsawa/analysis/error_analysis")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N windows")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load ------------------------------------------------------------
    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    if args.limit:
        split = split.select(range(args.limit))
    print(f"{args.split}: {len(split)} windows")

    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval().to(args.device)

    # ---- predict ---------------------------------------------------------
    all_gold, all_pred = [], []
    cols = split.column_names
    split_t = split.remove_columns([c for c in cols if c not in
                                    ("input_ids", "attention_mask", "labels")])
    split_t.set_format("torch")

    with torch.no_grad():
        for i in range(0, len(split_t), args.batch_size):
            batch = split_t[i: i + args.batch_size]
            logits = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
            ).logits
            preds = logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].numpy()
            for p_row, l_row in zip(preds, labels):
                m = l_row != -100
                all_gold.append(l_row[m])
                all_pred.append(p_row[m])
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(split_t)}", flush=True)

    # ---- 1. headline metrics, exact vs tolerant --------------------------
    print("\n" + "=" * 62)
    print("1. SPAN F1 — exact match vs boundary tolerance")
    print("=" * 62)
    rows = []
    for tol in (0, 1, 2, 5):
        TP = FP = FN = 0
        for g, p in zip(all_gold, all_pred):
            tp, fp, fn = match_spans(extract_spans(g), extract_spans(p), tol)
            TP += tp; FP += fp; FN += fn
        pr, rc, f1 = prf(TP, FP, FN)
        rows.append({"tolerance": tol, "precision": pr, "recall": rc, "f1": f1,
                     "tp": TP, "fp": FP, "fn": FN})
        label = "exact (seqeval-equivalent)" if tol == 0 else f"+/-{tol} tokens"
        print(f"  tol={tol}  P={pr:.4f}  R={rc:.4f}  F1={f1:.4f}   {label}")
    pd.DataFrame(rows).to_csv(out / "f1_by_tolerance.csv", index=False)
    print("\n  If exact is far below tol1, boundaries are off by a token or two")
    print("  systematically — a different problem from missing spans entirely.")

    # ---- 2. fragmentation ------------------------------------------------
    print("\n" + "=" * 62)
    print("2. FRAGMENTATION — is one gold span becoming several predictions?")
    print("=" * 62)
    frag_counts = Counter()
    merge_counts = Counter()
    orphan_preds = 0
    gold_total = pred_total = 0
    frag_examples = []

    for wi, (g, p) in enumerate(zip(all_gold, all_pred)):
        gspans, pspans = extract_spans(g), extract_spans(p)
        gold_total += len(gspans)
        pred_total += len(pspans)

        for gs in gspans:
            hits = [ps for ps in pspans if overlaps(gs, ps) > 0]
            frag_counts[len(hits)] += 1
            if len(hits) >= 3 and len(frag_examples) < 20:
                frag_examples.append({
                    "window": wi, "gold_start": gs[0], "gold_end": gs[1],
                    "gold_len": gs[1] - gs[0], "n_pieces": len(hits),
                    "pieces": str(hits[:6]),
                })
        for ps in pspans:
            hits = [gs for gs in gspans if overlaps(gs, ps) > 0]
            merge_counts[len(hits)] += 1
            if not hits:
                orphan_preds += 1

    print(f"  gold spans: {gold_total:,}   predicted: {pred_total:,} "
          f"({pred_total/max(gold_total,1):.2f}x)")
    print("\n  predictions overlapping each GOLD span:")
    for n in sorted(frag_counts):
        share = 100 * frag_counts[n] / max(gold_total, 1)
        tag = {0: "  <- missed entirely", 1: "  <- clean"}.get(n, "  <- FRAGMENTED")
        print(f"    {n} piece(s): {frag_counts[n]:>6,}  ({share:5.1f}%){tag}")

    frag = sum(v for k, v in frag_counts.items() if k >= 2)
    print(f"\n  fragmented gold spans: {frag:,} ({100*frag/max(gold_total,1):.1f}%)")
    print(f"  predicted spans overlapping NO gold span: {orphan_preds:,} "
          f"({100*orphan_preds/max(pred_total,1):.1f}% of predictions)")
    print("\n  High fragmentation -> the B weight is too aggressive, or the")
    print("  training data taught it tsawa comes in small pieces.")
    print("  High orphan rate -> it is hallucinating spans in commentary instead.")

    if frag_examples:
        pd.DataFrame(frag_examples).to_csv(out / "fragmentation_examples.csv", index=False)

    # ---- 3. performance by gold span length ------------------------------
    print("\n" + "=" * 62)
    print("3. BY SPAN LENGTH — does it fail on short spans specifically?")
    print("=" * 62)
    buckets = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100),
               (101, 300), (301, 10**9)]
    stats = {b: {"gold": 0, "hit": 0, "frag": 0} for b in buckets}

    def bucket_of(n):
        for b in buckets:
            if b[0] <= n <= b[1]:
                return b
        return buckets[-1]

    for g, p in zip(all_gold, all_pred):
        gspans, pspans = extract_spans(g), extract_spans(p)
        for gs in gspans:
            bk = bucket_of(gs[1] - gs[0])
            stats[bk]["gold"] += 1
            hits = [ps for ps in pspans if overlaps(gs, ps) > 0]
            if any(abs(ps[0] - gs[0]) <= 1 and abs(ps[1] - gs[1]) <= 1 for ps in pspans):
                stats[bk]["hit"] += 1
            if len(hits) >= 2:
                stats[bk]["frag"] += 1

    rows = []
    print(f"  {'tokens':>12} {'gold':>7} {'recall':>8} {'frag%':>7}")
    for b in buckets:
        s = stats[b]
        if not s["gold"]:
            continue
        rec = s["hit"] / s["gold"]
        fr = 100 * s["frag"] / s["gold"]
        name = f"{b[0]}-{b[1]}" if b[1] < 10**9 else f"{b[0]}+"
        print(f"  {name:>12} {s['gold']:>7,} {rec:>8.3f} {fr:>6.1f}%")
        rows.append({"bucket": name, "gold": s["gold"], "recall": rec, "frag_pct": fr})
    pd.DataFrame(rows).to_csv(out / "by_span_length.csv", index=False)

    print("\n  Note: these are TOKEN lengths. The ~30-char fragment question")
    print("  is roughly the 1-10 token buckets. If recall there is near zero,")
    print("  short spans are the failure and dropping them is defensible.")

    print(f"\nCSVs written to {out}/")


if __name__ == "__main__":
    main()
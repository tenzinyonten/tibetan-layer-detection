#!/usr/bin/env python3
"""
check_false_positives.py — where do the wrong tsawa predictions land?

Hypothesis: on old-batch books the model predicts quoted verse (citations) as
tsawa, because the new batch — most of the training data — often labels
quoted verse as BOTH citation and tsawa. Old-batch precision is 0.41 vs 0.65
new.

Uses the v4 dataset only for its citation labels (B-QUOTE=3, I-QUOTE=4); its
windows are identical to v5, which the script verifies window by window.

Every predicted span that does not match gold tsawa at IoU>=0.5 is sorted into:
  ON CITATION   >= half its tokens are citation tokens
  NEAR TSAWA    overlaps gold tsawa but not enough to match (boundary error)
  COMMENTARY    neither

Usage
-----
    python check_false_positives.py --model runs/v5_merged/best \
        --v5 ds_v5 --v4 ds_v4
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification

from eval_particle_rule import iou, spans, viterbi


def matched_preds(gold, pred, thr=0.5):
    c = sorted(((iou(g, p), gi, pi) for pi, p in enumerate(pred)
                for gi, g in enumerate(gold) if iou(g, p) >= thr), reverse=True)
    ug, up = set(), set()
    for _, gi, pi in c:
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi)
    return up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v5_merged/best")
    ap.add_argument("--v5", default="ds_v5")
    ap.add_argument("--v4", default="ds_v4")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--break-penalty", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v5 = load_from_disk(args.v5)[args.split]
    v4 = load_from_disk(args.v4)[args.split]
    if len(v5) != len(v4):
        raise SystemExit(f"window counts differ: v5 {len(v5)} vs v4 {len(v4)}")
    batches = ([str(x) for x in v5["source_batch"]]
               if "source_batch" in v5.column_names else
               ["old" if str(p).startswith("P") else "new" for p in v5["pecha_id"]])

    model = AutoModelForTokenClassification.from_pretrained(args.model).eval().to(dev)
    keep = ["input_ids", "attention_mask", "labels"]
    t5 = v5.remove_columns([c for c in v5.column_names if c not in keep]); t5.set_format("torch")
    t4 = v4.remove_columns([c for c in v4.column_names if c not in keep]); t4.set_format("torch")

    cat = defaultdict(Counter)
    mismatched = 0
    with torch.no_grad():
        for i in range(0, len(t5), args.batch_size):
            b5, b4 = t5[i:i + args.batch_size], t4[i:i + args.batch_size]
            lg = model(input_ids=b5["input_ids"].to(dev),
                       attention_mask=b5["attention_mask"].to(dev)
                       ).logits.float().cpu().numpy()
            for j in range(len(lg)):
                w = i + j
                if not torch.equal(b5["input_ids"][j], b4["input_ids"][j]):
                    mismatched += 1
                    continue
                m = b5["labels"][j].numpy() != -100
                g_lab = b5["labels"][j].numpy()[m]
                q = np.isin(b4["labels"][j].numpy()[m], (3, 4))
                g_tok = np.isin(g_lab, (1, 2))
                gold = spans(g_lab)
                pred = spans(viterbi(lg[j][m], args.break_penalty))
                ok = matched_preds(gold, pred)
                bt = batches[w]
                for k, (s, e) in enumerate(pred):
                    if k in ok:
                        cat[bt]["correct"] += 1
                        continue
                    seg_q = q[s:e + 1].mean()
                    seg_g = g_tok[s:e + 1].any()
                    if seg_q >= 0.5:
                        cat[bt]["on_citation"] += 1
                    elif seg_g:
                        cat[bt]["near_tsawa"] += 1
                    else:
                        cat[bt]["commentary"] += 1
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(t5)}", flush=True)

    if mismatched:
        print(f"\n  WARNING: {mismatched} windows differ between v4 and v5 and were skipped")

    print(f"\n{'='*66}\nWHERE THE WRONG PREDICTIONS LAND — {args.split}\n{'='*66}")
    print(f"  {'batch':<6}{'correct':>9}{'wrong':>8}{'on citation':>14}"
          f"{'near tsawa':>12}{'commentary':>12}")
    for bt in ("old", "new"):
        c = cat[bt]
        wrong = c["on_citation"] + c["near_tsawa"] + c["commentary"]
        if not wrong:
            continue
        pct = lambda x: f"{x} ({100*x/wrong:.0f}%)"
        print(f"  {bt:<6}{c['correct']:>9}{wrong:>8}{pct(c['on_citation']):>14}"
              f"{pct(c['near_tsawa']):>12}{pct(c['commentary']):>12}")

    print("\n  If 'on citation' is large for OLD books, the model is calling quoted")
    print("  verse tsawa — learned from the new batch's double labels — and the")
    print("  citation fix is the top priority.")
    print("  If 'near tsawa' dominates, it's boundary errors instead.")
    print("  If 'commentary' dominates, it's confusing prose for root text.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
eval_merged.py — does merging the new batch's split spans close the gap?

The new batch splits continuous tsawa into several spans at shad boundaries
(27.4% of gaps are shad/whitespace only, vs 0.8% in the old batch). If that is
what costs new-batch recall, then merging adjacent pieces back into one span
should lift new-batch IoU toward the old-batch level — without retraining.

Rule, applied the same way to gold and predictions: two consecutive spans are
merged when the tokens between them decode to nothing but shad / whitespace.

Reports IoU@0.5 by batch, original vs merged. Old batch should barely move;
new batch is the one to watch.

Reuses the decoder and metric from eval_particle_rule.py — keep both files in
the same folder.

Usage
-----
    python eval_merged.py --model Yontenn/mmbert-tsawa-bio-inv-v2 --dataset ds_v2
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval_particle_rule import score, spans, viterbi

JOINERS = set(" \n\t\u0f0d\u0f0e\u0f0b")   # space, newline, tab, shad, double shad, tsheg


def merge(sp, ids, tok):
    """Merge consecutive spans whose gap is only shad / whitespace."""
    if not sp:
        return sp
    out = [list(sp[0])]
    for s, e in sp[1:]:
        gap_ids = ids[out[-1][1] + 1:s]
        gap = tok.decode(gap_ids, skip_special_tokens=True).replace("▁", "")
        if s > out[-1][1] and set(gap) <= JOINERS:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-bio-inv-v2")
    ap.add_argument("--dataset", default="ds_v2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--break-penalty", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    batches = ([str(x) for x in split["source_batch"]]
               if "source_batch" in split.column_names else
               ["old" if str(p).startswith("P") else "new" for p in split["pecha_id"]])

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval().to(dev)

    keep = ["input_ids", "attention_mask", "labels"]
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")
    print(f"{args.split}: {len(t)} windows on {dev}")

    res = {k: defaultdict(list) for k in ("orig_g", "orig_p", "merg_g", "merg_p")}
    n_merged = defaultdict(int)
    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i:i + args.batch_size]
            lg = model(input_ids=b["input_ids"].to(dev),
                       attention_mask=b["attention_mask"].to(dev)
                       ).logits.float().cpu().numpy()
            for j in range(len(lg)):
                w = i + j
                m = b["labels"][j].numpy() != -100
                ids = b["input_ids"][j].numpy()[m]
                g = spans(b["labels"][j].numpy()[m])
                p = spans(viterbi(lg[j][m], args.break_penalty))
                gm, pm = merge(g, ids, tok), merge(p, ids, tok)
                n_merged[batches[w]] += len(g) - len(gm)
                for key in (batches[w], "ALL"):
                    res["orig_g"][key].append(g); res["orig_p"][key].append(p)
                    res["merg_g"][key].append(gm); res["merg_p"][key].append(pm)
            if (i // args.batch_size) % 25 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    print(f"\n{'='*70}\nIoU@0.5 — original gold vs adjacent pieces merged\n{'='*70}")
    print(f"  {'batch':<6}{'gold':>7}{'gold merged':>13}"
          f"{'IoU orig':>10}{'IoU merged':>12}{'R orig':>8}{'R merged':>10}")
    for key in ["old", "new", "ALL"]:
        if key not in res["orig_g"]:
            continue
        f0, _, r0, _ = score(res["orig_g"][key], res["orig_p"][key])
        f1, _, r1, _ = score(res["merg_g"][key], res["merg_p"][key])
        n0 = sum(len(x) for x in res["orig_g"][key])
        n1 = sum(len(x) for x in res["merg_g"][key])
        print(f"  {key:<6}{n0:>7,}{n1:>13,}{f0:>10.4f}{f1:>12.4f}{r0:>8.3f}{r1:>10.3f}")

    print("\n  Old batch should barely move — it has almost no split pieces.")
    print("  If new-batch IoU rises substantially toward the old-batch level,")
    print("  the split convention is what was costing recall, and rebuilding")
    print("  the training data with merged spans is worth doing.")
    print("  If it barely moves, segmentation is not the main cause.")


if __name__ == "__main__":
    main()
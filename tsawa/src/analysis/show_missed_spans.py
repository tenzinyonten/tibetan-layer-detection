#!/usr/bin/env python3
"""
show_missed_spans.py — see exactly what happens where short spans are missed.

Evaluation says recall is 0.000 on gold spans under 10 tokens, yet the model
assigns those same tokens high P(TSAWA). Something is happening between "these
tokens are tsawa" and "here is a span". This prints it.

For every missed gold span it classifies the failure:

  CONTAINED   a single predicted span swallows the gold span whole
              -> the model found it but merged it with neighbours
  OVERLAP     a predicted span overlaps partially
              -> boundaries are wrong but it is in roughly the right place
  SPLIT       several predicted spans overlap this one gold span
  MISSED      nothing overlaps at all
              -> the model genuinely does not see it

CONTAINED dominating means this is a segmentation problem, not a detection or
labelling one, and no amount of data cleaning fixes it.

Usage
-----
    python show_missed_spans.py \
        --model Yontenn/mmbert-tsawa-io-v2 \
        --dataset tsawa/data/processed/tsawa_dataset_v2 \
        --split validation --max-len 10 --examples 15
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer


def spans_io(seq: np.ndarray):
    out, start = [], None
    for i, v in enumerate(seq):
        if v == 1 and start is None:
            start = i
        elif v != 1 and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(seq)))
    return out


def overlap(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def classify(gold, preds, tol=1):
    """How did the model fail on this gold span?"""
    hits = [p for p in preds if overlap(gold, p) > 0]
    if not hits:
        return "MISSED", []
    if any(abs(p[0] - gold[0]) <= tol and abs(p[1] - gold[1]) <= tol for p in hits):
        return "MATCHED", hits
    if len(hits) >= 2:
        return "SPLIT", hits
    p = hits[0]
    if p[0] <= gold[0] and p[1] >= gold[1]:
        return "CONTAINED", hits
    return "OVERLAP", hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-io-v2")
    ap.add_argument("--dataset", default="tsawa/data/processed/tsawa_dataset_v2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-len", type=int, default=10,
                    help="only gold spans up to this many tokens")
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="scratch/tsawa/analysis/missed_spans.csv")
    args = ap.parse_args()

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    if args.limit:
        split = split.select(range(args.limit))
    print(f"{args.split}: {len(split)} windows")

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval().to(args.device)
    n_lab = model.config.num_labels
    print(f"model: {n_lab} labels ({'IO' if n_lab == 2 else 'BIO'})")

    keep = ["input_ids", "attention_mask", "labels"]
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")

    verdicts = Counter()
    by_len = {}
    rows, examples = [], []

    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i: i + args.batch_size]
            logits = model(input_ids=b["input_ids"].to(args.device),
                           attention_mask=b["attention_mask"].to(args.device)).logits
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy()
            labs = b["labels"].numpy()
            ids = b["input_ids"].numpy()

            for j in range(len(labs)):
                m = labs[j] != -100
                gold_io = np.where(labs[j][m] == 2, 1, labs[j][m])
                pr = pred[j][m]
                if n_lab == 3:
                    pr = np.where(pr == 2, 1, pr)
                p_pos = probs[j][m][:, 1:].sum(-1)
                toks = ids[j][m]

                g_spans, p_spans = spans_io(gold_io), spans_io(pr)
                for g in g_spans:
                    L = g[1] - g[0]
                    if L > args.max_len:
                        continue
                    verdict, hits = classify(g, p_spans)
                    verdicts[verdict] += 1
                    by_len.setdefault(L, Counter())[verdict] += 1
                    rows.append({
                        "window": i + j, "gold_start": g[0], "gold_end": g[1],
                        "gold_len": L, "verdict": verdict,
                        "mean_p_tsawa": float(p_pos[g[0]:g[1]].mean()),
                        "n_overlapping_preds": len(hits),
                        "containing_pred_len": (hits[0][1] - hits[0][0]) if hits else 0,
                    })
                    if verdict in ("CONTAINED", "OVERLAP", "MISSED") and \
                            len(examples) < args.examples:
                        ctx0, ctx1 = max(0, g[0] - 25), min(len(toks), g[1] + 25)
                        examples.append({
                            "verdict": verdict, "len": L,
                            "mean_p": float(p_pos[g[0]:g[1]].mean()),
                            "gold_text": tok.decode(toks[g[0]:g[1]],
                                                    skip_special_tokens=True),
                            "pred_span": hits[0] if hits else None,
                            "pred_len": (hits[0][1] - hits[0][0]) if hits else 0,
                            "pred_text": (tok.decode(toks[hits[0][0]:hits[0][1]][:60],
                                                     skip_special_tokens=True)
                                          if hits else ""),
                            "context": tok.decode(toks[ctx0:ctx1],
                                                  skip_special_tokens=True),
                        })
            if (i // args.batch_size) % 50 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    total = sum(verdicts.values())
    print(f"\n{'='*62}\nGOLD SPANS <= {args.max_len} TOKENS: {total:,}\n{'='*62}")
    for v in ("MATCHED", "CONTAINED", "OVERLAP", "SPLIT", "MISSED"):
        n = verdicts[v]
        note = {
            "CONTAINED": "  <- swallowed by a bigger prediction",
            "MISSED": "  <- not seen at all",
            "OVERLAP": "  <- right area, wrong boundaries",
        }.get(v, "")
        print(f"  {v:>10}: {n:>6,} ({100*n/max(total,1):5.1f}%){note}")

    print("\nby gold span length:")
    for L in sorted(by_len):
        c = by_len[L]
        tot = sum(c.values())
        print(f"  {L:>3} tok ({tot:>5,}): " + "  ".join(
            f"{k} {100*c[k]/tot:.0f}%" for k in
            ("MATCHED", "CONTAINED", "OVERLAP", "SPLIT", "MISSED") if c[k]))

    df = pd.DataFrame(rows)
    if len(df):
        cont = df[df.verdict == "CONTAINED"]
        if len(cont):
            print(f"\nwhen contained, the swallowing prediction is typically "
                  f"{cont.containing_pred_len.median():.0f} tokens "
                  f"(vs gold {cont.gold_len.median():.0f})")
        print(f"mean P(TSAWA) on these gold tokens: {df.mean_p_tsawa.mean():.3f}")
        print("  (high P + low MATCHED = detected but not delimited)")

    print(f"\n{'='*62}\nEXAMPLES\n{'='*62}")
    for e in examples:
        print(f"\n[{e['verdict']}] gold {e['len']} tokens, "
              f"P(TSAWA)={e['mean_p']:.3f}")
        print(f"  GOLD      : {e['gold_text']}")
        if e["pred_span"]:
            print(f"  MODEL SAID: {e['pred_len']} tokens -> {e['pred_text']}")
        else:
            print("  MODEL SAID: nothing here")
        print(f"  CONTEXT   : {e['context']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
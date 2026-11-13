#!/usr/bin/env python3
"""
show_old_fp.py — read the old-batch wrong predictions that land on text labelled
neither tsawa nor citation (70% of old-batch errors).

Are they commentary the model mistook for root text, or root verse the old
annotators never marked? Stops after --n examples to save GPU time.

    python show_old_fp.py --model runs/v5_merged/best --v5 ds_v5 --v4 ds_v4
"""

import argparse

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer

from check_false_positives import matched_preds
from eval_particle_rule import spans, viterbi

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/v5_merged/best")
ap.add_argument("--v5", default="ds_v5")
ap.add_argument("--v4", default="ds_v4")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--ctx", type=int, default=25)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
v5 = load_from_disk(args.v5)["validation"]
v4 = load_from_disk(args.v4)["validation"]
pids = v5["pecha_id"]
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained(args.model).eval().to(dev)
dec = lambda ids: tok.decode(ids, skip_special_tokens=True).replace("\n", " ")

found = 0
with torch.no_grad():
    for w in range(len(v5)):
        if not str(pids[w]).startswith("P"):
            continue
        ids = torch.tensor([v5[w]["input_ids"]])
        am = torch.tensor([v5[w]["attention_mask"]])
        lab = np.array(v5[w]["labels"]); lab4 = np.array(v4[w]["labels"])
        m = lab != -100
        lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits[0].float().cpu().numpy()
        x = ids[0].numpy()[m]; g = lab[m]; q = np.isin(lab4[m], (3, 4)); gt = np.isin(g, (1, 2))
        gold = spans(g); pred = spans(viterbi(lg[m], 4.0)); ok = matched_preds(gold, pred)
        for k, (s, e) in enumerate(pred):
            if k in ok or q[s:e+1].mean() >= 0.5 or gt[s:e+1].any():
                continue
            found += 1
            print(f"\n[{found}] {pids[w]}  ({e-s+1} tokens)")
            print(f"  before: …{dec(x[max(0,s-args.ctx):s])}")
            print(f"  PRED  : {dec(x[s:e+1])[:300]}")
            print(f"  after : {dec(x[e+1:e+1+args.ctx])}…")
            if found >= args.n:
                raise SystemExit
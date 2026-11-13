#!/usr/bin/env python3
"""Save every predicted span of a trained layer model, per document, for later analysis.

Uses the same decoding as train_layer.py's stitched evaluation: softmax probabilities of
overlapping windows are averaged per document token, then Viterbi (BIO, --break-penalty)
picks the label path. Output is TOKEN spans (inclusive), plus gold token spans, so the
result can be mapped back to characters wherever the document text is available.

Usage (on the GPU machine):
    python common/predict_layer_spans.py --label-name YIGCHUNG \
        --model /workspace/runs/yigchung_genre_lotsawa/best \
        --dataset Yontenn/yigchung-genre-bio-lotsawa --splits validation \
        --out /workspace/runs/yigchung_genre_lotsawa/predictions_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_layer as tl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-name", default="YIGCHUNG")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="validation", help="comma list, e.g. validation or validation,test")
    ap.add_argument("--break-penalty", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tl.configure_label(args.label_name)
    scheme, tag = "bio", args.label_name.upper()
    ds = load_from_disk(args.dataset) if Path(args.dataset).exists() else load_dataset(args.dataset)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(dev).eval()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    out = {"model": args.model, "dataset": args.dataset, "break_penalty": args.break_penalty,
           "span_end": "inclusive token index", "splits": {}}
    for split in args.splits.split(","):
        d = ds[split]
        logits, labels = [], []
        with torch.no_grad():
            for i, ex in enumerate(d):
                n = int(sum(ex["attention_mask"]))          # windows are padded to max_length
                ids = torch.tensor([ex["input_ids"][:n]], device=dev)
                am = torch.ones_like(ids)
                with torch.autocast(device_type=dev, dtype=dtype, enabled=dev == "cuda"):
                    lg = model(input_ids=ids, attention_mask=am).logits[0].float().cpu().numpy()
                full = np.zeros((len(ex["input_ids"]), lg.shape[-1]), dtype=np.float32)
                full[:n] = lg
                logits.append(full)
                labels.append(np.asarray(ex["labels"]))
                if (i + 1) % 50 == 0:
                    print(f"  {split}: {i + 1}/{len(d)} windows", flush=True)
        meta = {"pecha_id": d["pecha_id"], "token_start": d["token_start"], "token_end": d["token_end"]}
        doc_ids = list(dict.fromkeys(d["pecha_id"]))          # stitch_documents keeps first-seen order
        cats = dict(zip(d["pecha_id"], d["category"])) if "category" in d.column_names else {}
        st_logits, st_labels = tl.stitch_documents(np.stack(logits), np.stack(labels), meta)
        docs = {}
        tp = fp = fn = 0
        for pid, lg, gold in zip(doc_ids, st_logits, st_labels):
            path = tl.viterbi(lg, scheme, args.break_penalty)
            probs = np.exp(lg)
            pred = tl.spans_of(path, scheme, tag)
            g = tl.spans_of(np.where(gold < 0, 0, gold), scheme, tag)
            m = tl.count_matches(g, pred)
            tp += m; fp += len(pred) - m; fn += len(g) - m
            docs[pid] = {
                "category": cats.get(pid, ""),
                "n_tokens": int(len(path)),
                "pred": [[int(s), int(e), round(float(probs[s:e + 1, 1:].sum(-1).mean()), 4)] for s, e in pred],
                "gold": [[int(s), int(e)] for s, e in g],
            }
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f = 2 * p * r / max(p + r, 1e-9)
        out["splits"][split] = {"iou50": {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn},
                                "docs": docs}
        print(f"{split}: IoU@0.5 P {p:.4f} R {r:.4f} F1 {f:.4f} (tp {tp}, fp {fp}, fn {fn})")
    Path(args.out).write_text(json.dumps(out))
    print("saved", args.out)


if __name__ == "__main__":
    main()

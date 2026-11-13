#!/usr/bin/env python3
"""Run the trained model on validation + test and SAVE the raw logits (fp16) per work.
Run on the GPU box (Vast), then copy the .npz files to the Mac to tune Viterbi / analyse errors.
    python3 ta_predict.py MODEL_DIR OUT_DIR [DATASET]
    e.g. python3 ta_predict.py ~/work/ta-run2/best ~/work/preds Yontenn/title-author-training"""
import os, sys, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForTokenClassification

model_dir, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
dataset = sys.argv[3] if len(sys.argv) > 3 else "Yontenn/title-author-training"
ds = load_from_disk(dataset) if Path(dataset).exists() else load_dataset(dataset)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(model_dir, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
META = [c for c in ("work_id", "cid", "source", "region", "region_start", "n_tokens_region") if c in ds["validation"].column_names]

for split in ("validation", "test"):
    d = ds[split]; n = len(d); t0 = time.time()
    order = sorted(range(n), key=lambda i: sum(d[i]["attention_mask"]))   # short first -> little padding
    logits, ids, labels = [None] * n, [None] * n, [None] * n
    B = 4
    for b in range(0, n, B):
        idx = order[b:b + B]; rows = [d[i] for i in idx]
        L = max(sum(r["attention_mask"]) for r in rows)
        x = torch.tensor([r["input_ids"][:L] for r in rows]).to(dev)
        am = torch.tensor([r["attention_mask"][:L] for r in rows]).to(dev)
        with torch.no_grad(): lg = model(input_ids=x, attention_mask=am).logits.float().cpu().numpy()
        for i, r, l in zip(idx, rows, lg):
            k = sum(r["attention_mask"]); logits[i] = l[:k].astype(np.float16)
            ids[i] = np.array(r["input_ids"][:k], dtype=np.int32); labels[i] = np.array(r["labels"][:k], dtype=np.int16)
        if (b // B) % 100 == 0: print(f"{split} {b}/{n}  {time.time()-t0:.0f}s", flush=True)
    meta = {c: np.array([d[i][c] for i in range(n)]) for c in META}
    np.savez_compressed(out_dir / f"{split}_logits.npz",
        logits=np.array(logits, dtype=object), ids=np.array(ids, dtype=object), labels=np.array(labels, dtype=object), **meta)
    print(f"saved {split}: {n} works, {time.time()-t0:.0f}s", flush=True)
print("PRED_DONE")

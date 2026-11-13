#!/usr/bin/env python3
"""
find_label_errors.py — let the model tell you which gold annotations it can't believe.

The v2 IO model reached 0.598 token-level F1 on TSAWA but 0.000 recall on spans
under 10 tokens. That pattern says the model learned one convention well and
refuses another. This script finds the spans it refuses.

Method: run the trained model over the TRAIN split, and for every gold span
compute the mean P(TSAWA) the model assigns to its tokens. Gold spans where the
model is confidently negative are the candidates — either annotation errors, or
a real convention the model failed to learn. Reading the top ones tells you
which.

This is a cheap stand-in for proper out-of-fold analysis. The model has seen
this data in training, which makes it *conservative*: a span it still rejects
after fitting on it is a strong signal. It also means a low score is not proof
of error — read before you delete.

Usage
-----
    python find_label_errors.py \
        --model Yontenn/mmbert-tsawa-io-v2 \
        --dataset Yontenn/formatting-tsawa-v2 \
        --spans tsawa/data/processed/tsawa_spans_resolved.csv \
        --raw-opf data/raw_opf

Outputs scratch/tsawa/analysis/label_errors/ranked_spans.csv plus a readable text dump.
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
from transformers import AutoModelForTokenClassification

# Things worth counting in the candidates, so the taxonomy isn't guesswork.
QUOTE_MARKERS = ["ཞེས་པའི་", "ཞེས་པ་", "ཅེས་པ་", "ཞེས་", "ཅེས་", "གསུངས་", "སོགས་"]
BRACKETS = ["༼", "༽", "༺", "༻", "(", ")", "[", "]"]
# The six books holding ~96% of sub-15-char spans.
INTERLINEAR_BOOKS = ["I881A57E8", "I9FF2B59B", "IF3ACC3E1",
                     "IDAD44BA2", "I1637B774", "IB522F095"]


def extract_spans_io(seq: np.ndarray):
    spans, start = [], None
    for i, lab in enumerate(seq):
        if lab == 1 and start is None:
            start = i
        elif lab != 1 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(seq)))
    return spans


def load_base(raw_opf: Path, pecha: str) -> str | None:
    for p in [raw_opf / f"{pecha}.opf" / f"{pecha}.opf" / "base" / "v001.txt",
              raw_opf / f"{pecha}.opf" / "base" / "v001.txt"]:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Yontenn/mmbert-tsawa-io-v2")
    ap.add_argument("--dataset", default="Yontenn/formatting-tsawa-v2")
    ap.add_argument("--spans", default="tsawa/data/processed/tsawa_spans_resolved.csv")
    ap.add_argument("--raw-opf", default="data/raw_opf")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="scratch/tsawa/analysis/label_errors")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = (load_from_disk(args.dataset) if Path(args.dataset).exists()
          else load_dataset(args.dataset, token=os.environ.get("HF_TOKEN")))
    split = ds[args.split]
    if args.limit:
        split = split.select(range(args.limit))

    # pecha_id per window, if the dataset carries it
    pecha_col = next((c for c in split.column_names
                      if c.lower() in ("pecha_id", "pecha", "doc_id")), None)
    pechas = split[pecha_col] if pecha_col else [None] * len(split)
    print(f"{args.split}: {len(split)} windows"
          f"{'' if pecha_col else '  (no pecha column — book attribution unavailable)'}")

    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval().to(args.device)
    is_io = model.config.num_labels == 2
    print(f"model: {model.config.num_labels} labels ({'IO' if is_io else 'BIO'})")

    keep = ["input_ids", "attention_mask", "labels"]
    t = split.remove_columns([c for c in split.column_names if c not in keep])
    t.set_format("torch")

    rows = []
    with torch.no_grad():
        for i in range(0, len(t), args.batch_size):
            b = t[i: i + args.batch_size]
            logits = model(input_ids=b["input_ids"].to(args.device),
                           attention_mask=b["attention_mask"].to(args.device)).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            labels = b["labels"].numpy()
            ids = b["input_ids"].numpy()

            for j, (pr, lab, tok) in enumerate(zip(probs, labels, ids)):
                widx = i + j
                m = lab != -100
                lab_m, pr_m, tok_m = lab[m], pr[m], tok[m]
                # collapse BIO gold to IO for span extraction
                gold_io = np.where(lab_m == 2, 1, lab_m)
                p_pos = pr_m[:, 1:].sum(axis=1)  # P(any TSAWA class)

                for (s, e) in extract_spans_io(gold_io):
                    rows.append({
                        "window": widx,
                        "pecha_id": pechas[widx] if pecha_col else "",
                        "tok_start": int(s), "tok_end": int(e),
                        "tok_len": int(e - s),
                        "mean_p": float(p_pos[s:e].mean()),
                        "max_p": float(p_pos[s:e].max()),
                        "frac_above_half": float((p_pos[s:e] > 0.5).mean()),
                        "_tokens": tok_m[s:e][:40].tolist(),
                    })
            if (i // args.batch_size) % 50 == 0:
                print(f"  {i}/{len(t)}", flush=True)

    df = pd.DataFrame(rows).sort_values("mean_p")
    print(f"\n{len(df):,} gold spans scored")

    # ---- distribution ----------------------------------------------------
    print("\nmodel confidence on GOLD spans:")
    for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, .8), (.8, 1.01)]:
        n = ((df.mean_p >= lo) & (df.mean_p < hi)).sum()
        print(f"  P {lo:.2f}-{hi:.2f}: {n:>6,} ({100*n/len(df):5.1f}%)")

    rejected = df[df.mean_p < 0.05]
    print(f"\nconfidently rejected (P<0.05): {len(rejected):,} "
          f"({100*len(rejected)/len(df):.1f}%)")

    # ---- who are they ----------------------------------------------------
    print("\nrejected spans by token length:")
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]:
        sub = rejected[(rejected.tok_len >= lo) & (rejected.tok_len <= hi)]
        tot = df[(df.tok_len >= lo) & (df.tok_len <= hi)]
        if len(tot):
            tag = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  {tag:>8}: {len(sub):>6,} of {len(tot):>6,} "
                  f"({100*len(sub)/len(tot):5.1f}% rejected)")

    if pecha_col:
        print("\ntop books by rejected-span count:")
        for p, n in Counter(rejected.pecha_id).most_common(12):
            total = (df.pecha_id == p).sum()
            flag = "  <- known interlinear" if p in INTERLINEAR_BOOKS else ""
            print(f"  {p}: {n:>5,} of {total:>5,} ({100*n/total:5.1f}%){flag}")

    # ---- readable dump ---------------------------------------------------
    spans_csv = Path(args.spans)
    raw_opf = Path(args.raw_opf)
    dump = out / "top_rejected.txt"
    marker_hits = Counter()
    bracket_hits = 0

    if spans_csv.exists() and raw_opf.exists() and pecha_col:
        sp = pd.read_csv(spans_csv)
        sp = sp[~sp.dropped.astype(bool)] if "dropped" in sp.columns else sp
        texts: dict[str, str] = {}
        with dump.open("w", encoding="utf-8") as fh:
            fh.write("Gold spans the model confidently rejects, worst first.\n"
                     "Read these: are they annotation errors, or a real "
                     "convention the model failed to learn?\n\n")
            for _, r in rejected.head(args.top).iterrows():
                p = r.pecha_id
                if p not in texts:
                    texts[p] = load_base(raw_opf, p) or ""
                txt = texts[p]
                cand = sp[sp.pecha_id == p]
                fh.write(f"=== {p}  window {r.window} tokens[{r.tok_start}:{r.tok_end}] "
                         f"len={r.tok_len}  mean_p={r.mean_p:.4f}\n")
                if txt:
                    # char offsets aren't recoverable from token indices here,
                    # so show the book's short spans as the likely referents
                    short = cand[(cand.end - cand.start) <= 20].head(2)
                    for _, s2 in short.iterrows():
                        a, b = int(s2.start), int(s2.end)
                        fh.write(f"    e.g. [{a}:{b}] {txt[max(0,a-60):a]}"
                                 f" <<{txt[a:b]}>> {txt[b:b+60]}\n")
                fh.write("\n")
        print(f"\nreadable dump: {dump}")
    else:
        print("\n(skipping text dump — need --spans, --raw-opf and a pecha column)")

    df.drop(columns=["_tokens"]).to_csv(out / "ranked_spans.csv", index=False)
    print(f"ranked CSV: {out/'ranked_spans.csv'}")
    print("\nNext: read the dump. If the rejected spans are mostly particles,")
    print("editorial brackets, or mid-word slices, that is an annotation-error")
    print("taxonomy you can write rules against. If they look like legitimate")
    print("short citations, the problem is the model, not the labels.")


if __name__ == "__main__":
    main()
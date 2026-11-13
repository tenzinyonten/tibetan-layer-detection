#!/usr/bin/env python3
"""
rescore_masked.py — re-score a --dump-spans file with suspect gold spans masked.

Reads the per-window JSONL written by ``eval_viterbi_iou.py --dump-spans`` and
the suspect pairs in tsawa/data/processed/named_source_pairs.csv (tsawa spans that
sit in a ``ཞེས་དང༌ + named source`` chain and are probably quotations).

For each suspect set, a gold span is dropped when its character range overlaps
a suspect span, and a predicted span is dropped when it overlaps the same
region — the region becomes don't-care, so the model is neither rewarded nor
punished there. Everything else is scored exactly as eval_viterbi_iou.py does
(per window, IoU@0.5, greedy one-to-one).

Token -> character mapping re-tokenizes each base text the way
common/build_tsawa_dataset.py does (no special tokens, offset mapping) and checks
it against every window's char_start / char_end before using it.

Usage
-----
    python tsawa/scripts/eval/rescore_masked.py --dump scratch/tsawa/analysis/v6_nofeat_test_spans.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))  # eval_viterbi_iou
from eval_viterbi_iou import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "tsawa/data/processed/named_source_pairs.csv"
AUDIT = ROOT / "data/shared/tsawa_audit.csv"


def suspect_sets(path: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Both spans of each pair are suspect: a chained citation's first link
    is usually a quotation too. Returns {set_name: {pecha_id: [(s, e)]}}
    with end-exclusive character offsets."""
    sets = {"named_source": defaultdict(set), "named_source+unclear": defaultdict(set)}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] == "outline_heading":
                continue
            spans = {(int(r["a_start"]), int(r["a_end"])),
                     (int(r["b_start"]), int(r["b_end"]))}
            sets["named_source+unclear"][r["pecha_id"]] |= spans
            if r["category"] == "named_source":
                sets["named_source"][r["pecha_id"]] |= spans
    return {k: {p: sorted(v) for p, v in d.items()} for k, d in sets.items()}


def overlaps(a: tuple[int, int], regions: list[tuple[int, int]]) -> bool:
    return any(a[0] < e and s < a[1] for s, e in regions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    pred_keys = [k for k in rows[0] if k.startswith(("argmax", "viterbi"))
                 and not k.endswith("_tok")]
    sets = suspect_sets(Path(args.pairs))
    base = {r["pecha_id"]: r["base_path"]
            for r in csv.DictReader(open(AUDIT, encoding="utf-8"))}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    books = {r["pecha_id"] for r in rows} & set().union(*(s.keys() for s in sets.values()))
    offsets = {}
    for p in sorted(books):
        text = Path(base[p]).read_text(encoding="utf-8")
        enc = tok(text, add_special_tokens=False, truncation=False,
                  return_offsets_mapping=True, return_attention_mask=False)
        offsets[p] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        for r in rows:
            if r["pecha_id"] != p:
                continue
            real = [o for o in offsets[p][r["token_start"]:] if o[1] > o[0]]
            if real[0][0] != r["char_start"]:
                raise SystemExit(f"{p} window {r['window_index']}: tokenization "
                                 f"does not reproduce char_start {r['char_start']}")

    def char_range(p, t):
        return offsets[p][t[0]][0], offsets[p][t[1]][1]

    results = {}
    for key in pred_keys:
        gold = [[tuple(g) for g in r["gold"]] for r in rows]
        pred = [[tuple(x) for x in r[key]] for r in rows]
        results[key] = {"all": score(gold, pred, 0.5)}
        for name, sus in sets.items():
            g2, p2, n_g, n_p = [], [], 0, 0
            for r in rows:
                reg = sus.get(r["pecha_id"])
                if not reg:
                    g2.append([tuple(g) for g in r["gold"]])
                    p2.append([tuple(x) for x in r[key]])
                    continue
                keep_g = [tuple(g) for g, gt in zip(r["gold"], r["gold_tok"])
                          if not overlaps(char_range(r["pecha_id"], gt), reg)]
                keep_p = [tuple(x) for x, xt in zip(r[key], r[f"{key}_tok"])
                          if not overlaps(char_range(r["pecha_id"], xt), reg)]
                n_g += len(r["gold"]) - len(keep_g)
                n_p += len(r[key]) - len(keep_p)
                g2.append(keep_g)
                p2.append(keep_p)
            res = score(g2, p2, 0.5)
            res["masked_gold"], res["masked_pred"] = n_g, n_p
            results[key][f"masked_{name}"] = res

    for key, res in results.items():
        print(f"\n{key}  (IoU@0.5, per window, {len(rows)} windows)")
        print(f"  {'':24} {'F1':>7} {'P':>7} {'R':>7} {'tp':>5} {'fp':>5} {'fn':>5}"
              f" {'-gold':>6} {'-pred':>6}")
        for name, r in res.items():
            print(f"  {name:24} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f}"
                  f" {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}"
                  f" {r.get('masked_gold', 0):6d} {r.get('masked_pred', 0):6d}")
    print("\n  -gold / -pred count window-level spans removed; overlapping "
          "windows can hold the same book span twice.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()

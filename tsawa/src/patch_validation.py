#!/usr/bin/env python3
"""Patch the .opf validation split of tsawa_dataset_v10 with the re-annotated labels.

The 6 re-annotated books that sit in the validation split (I19A08A51, I895C519A, P000028,
P000161, P000175, P000215) get their corrected labels from
tsawa_spans_merged_reannotated_normalized.csv (source == reannotated, split == val), the same
source train uses. The other 12 val books keep exactly the labels they had (tsawa_spans_merged.csv).
Same 18 books, same windows (the text does not change), same masking rules.

Outputs (new directories; v10 is never modified):
  tsawa_dataset_v11                v10 with the patched validation split
  tsawa_dataset_v11_noclassical    v11 without the classical (WB...) docs: train minus WB docs,
                                   validation, test (the classical_* splits are dropped)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tsawa_dataset as B  # noqa: E402
from build_v10_dataset import SCHEMA, read_ignore, read_spans  # noqa: E402

P = ROOT / "tsawa/data/processed"
V10, V11, V11N = P / "tsawa_dataset_v10", P / "tsawa_dataset_v11", P / "tsawa_dataset_v11_noclassical"
SPLIT = P / "split_v3_frozen_excl_set_aside.csv"
NORM = P / "tsawa_spans_merged_reannotated_normalized.csv"
OLD = P / "tsawa_spans_merged.csv"
IGN_V6 = P / "v6_ignore_spans.csv"
AUDIT = P / "tsawa_audit.csv"
OUT_REPORT = P / "tsawa_dataset_v11_patch_report.json"


def main() -> int:
    split_of, _ = B.load_split_file(SPLIT)
    val_ids = sorted(p for p, s in split_of.items() if s == "validation")
    assert len(val_ids) == 18

    # which val books were re-annotated, and their corrected spans
    src_of = {}
    with NORM.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val":
                src_of[r["pecha_id"]] = r["source"]
    patched = sorted(p for p in val_ids if src_of.get(p) == "reannotated")
    kept = [p for p in val_ids if p not in patched]
    assert len(patched) == 6 and set(src_of) == set(val_ids), (patched, sorted(set(val_ids) - set(src_of)))
    new_spans = read_spans(NORM, lambda r: r["split"] == "val" and r["source"] == "reannotated")
    old_spans = read_spans(OLD, lambda r: r["pecha_id"] in set(val_ids))
    spans = {p: (new_spans[p] if p in patched else old_spans[p]) for p in val_ids}
    ign = read_ignore(IGN_V6, set(val_ids))

    audit = pd.read_csv(AUDIT).set_index("pecha_id")
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
    tok.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = B.special_token_ids(tok)
    content_len, stride, max_length = 8190, 5120, 8192
    LAB = B.LABEL2ID

    def gen():
        for pid in val_ids:
            text = Path(audit.loc[pid, "base_path"]).read_text(encoding="utf-8")
            enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
                      return_attention_mask=False)
            ids = list(enc["input_ids"])
            offs = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
            labels = B.label_tokens(offs, spans[pid], LAB)
            ig = B.merge_intervals(ign.get(pid, []))
            if ig:
                for i, (ts, te) in enumerate(offs):
                    if te > ts and B.in_intervals(ig, ts) and labels[i] != LAB["O"]:
                        labels[i] = B.IGNORE_LABEL
            for w_i, (ws, we) in enumerate(B.sliding_windows(len(ids), content_len, stride)):
                pk = B.pack_window(ids, labels, offs, ws, we, cls_id, sep_id, pad_id, max_length)
                pk.update(pecha_id=pid, source_batch=audit.loc[pid, "source_batch"], window_index=w_i,
                          n_tokens_doc=len(ids), coverage_pct=float(audit.loc[pid, "coverage_pct"]))
                yield pk

    new_val = Dataset.from_generator(gen, features=SCHEMA)
    v10 = load_from_disk(str(V10))
    old_val = v10["validation"]

    # ---------------- verification (before saving)
    rep = {"patched_books": patched, "unchanged_books": kept}
    assert sorted(set(new_val["pecha_id"])) == val_ids == sorted(set(old_val["pecha_id"])), "val book ids changed"
    assert len(new_val) == len(old_val), "window count changed"
    assert new_val["pecha_id"] == old_val["pecha_id"] and new_val["window_index"] == old_val["window_index"]
    same_cols = ("input_ids", "attention_mask", "token_start", "token_end", "char_start", "char_end",
                 "n_tokens_doc", "coverage_pct", "source_batch")
    for c in same_cols:
        assert new_val[c] == old_val[c], f"{c} differs"
    pid_col = np.array(old_val["pecha_id"])
    nl, ol = new_val["labels"], old_val["labels"]
    per_book = {}
    for pid in val_ids:
        idx = np.where(pid_col == pid)[0]
        diff = sum(int(np.sum(np.array(nl[i]) != np.array(ol[i]))) for i in idx)
        per_book[pid] = diff
    assert all(per_book[p] == 0 for p in kept), {p: per_book[p] for p in kept if per_book[p]}
    assert all(per_book[p] > 0 for p in patched), {p: per_book[p] for p in patched}
    rep["label_positions_changed"] = per_book

    # span-level change report (exact (start,end) on the final sidecar spans)
    sp_rep = {}
    tot = Counter()
    for pid in patched:
        o, n = {(s, e) for s, e, _ in old_spans[pid]}, {(s, e) for s, e, _ in new_spans[pid]}
        sp_rep[pid] = dict(old=len(o), new=len(n), identical=len(o & n), removed=len(o - n), added=len(n - o))
        tot.update(old=len(o), new=len(n), identical=len(o & n), removed=len(o - n), added=len(n - o))
    for pid in kept:
        assert {(s, e) for s, e, _ in spans[pid]} == {(s, e) for s, e, _ in old_spans[pid]}
    rep["span_changes"] = sp_rep
    rep["span_totals"] = dict(tot)
    rep["val_spans_before"] = sum(len(old_spans[p]) for p in val_ids)
    rep["val_spans_after"] = sum(len(spans[p]) for p in val_ids)

    # ---------------- save v11 and v11_noclassical
    ds11 = DatasetDict({k: (new_val if k == "validation" else v10[k]) for k in v10})
    ds11.save_to_disk(str(V11))
    non_wb = [i for i, p in enumerate(v10["train"]["pecha_id"]) if not p.startswith("WB")]
    ds11n = DatasetDict({"train": v10["train"].select(non_wb), "validation": new_val, "test": v10["test"]})
    ds11n.save_to_disk(str(V11N))
    rep["v11_windows"] = {k: len(ds11[k]) for k in ds11}
    rep["v11_noclassical_windows"] = {k: len(ds11n[k]) for k in ds11n}
    rep["v11_noclassical_train_docs"] = len(set(ds11n["train"]["pecha_id"]))
    OUT_REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

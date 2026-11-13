#!/usr/bin/env python3
"""Build the combined tsawa dataset (v8): existing + re-annotated + classical-train commentaries; frozen .opf val/test;
plus SEPARATE classical_validation / classical_test eval sets (whole-commentary holdout, see make_classical_split.py).

Reuses the labelling / windowing functions of common/build_tsawa_dataset.py (same BIO rule,
same 8192/5120 windows, same mmBERT tokenizer) but does not run its Phase-2 audit totals
check, which cannot hold once the classical docs are added. No training.

Sources (all read-only):
  split          tsawa/data/processed/split_v3_frozen_excl_set_aside.csv   (5 set-aside books absent)
  train spans    tsawa/data/processed/tsawa_spans_merged_reannotated_normalized.csv  (split == train;
                 source = current (53 books) | reannotated (26 books))
  classical      tsawa/data/processed/classical_v3/*                       (25 docs, all train)
  val spans      tsawa/data/processed/tsawa_spans_merged.csv               (frozen: unchanged gold)
  test           scratch/tsawa/analysis/formatting-tsawa-v6_test_patched_full22_final
                 copied row for row; only the unused `features` column is dropped
  ignore spans   v6_ignore_spans.csv (existing books, same rule as v6) + classical ignore_spans
                 -> tokens labelled non-O inside them become -100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
import build_tsawa_dataset as B  # noqa: E402

P = ROOT / "tsawa/data/processed"
SPLIT = P / "split_v3_frozen_excl_set_aside.csv"
TRAIN_SPANS = P / "tsawa_spans_merged_reannotated_normalized.csv"
VAL_SPANS = P / "tsawa_spans_merged.csv"
IGN_V6 = P / "v6_ignore_spans.csv"
CL = P / "classical_v3"
TEST_PATCHED = ROOT / "scratch/tsawa/analysis/formatting-tsawa-v6_test_patched_full22_final"
AUDIT = P / "tsawa_audit.csv"

SCHEMA = Features({
    "pecha_id": Value("string"), "source_batch": Value("string"), "window_index": Value("int32"),
    "n_tokens_doc": Value("int32"), "coverage_pct": Value("float64"), "token_start": Value("int32"),
    "token_end": Value("int32"), "char_start": Value("int32"), "char_end": Value("int32"),
    "input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
    "labels": Sequence(Value("int32")),
})


def read_spans(path: Path, keep) -> dict[str, list[tuple[int, int, str]]]:
    by = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("dropped", "")).lower() == "true" or not keep(r):
                continue
            s, e = int(r["start"]), int(r["end"])
            if e > s:
                by[r["pecha_id"]].append((s, e, r["ann_id"]))
    for v in by.values():
        v.sort()
    return by


def read_ignore(path: Path, ids: set[str]) -> dict[str, list[tuple[int, int]]]:
    by = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["pecha_id"] in ids:
                by[r["pecha_id"]].append((int(r["start"]), int(r["end"])))
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=P / "tsawa_dataset_v8")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=5120)
    args = ap.parse_args()
    content_len = args.max_length - 2

    split_of, _ = B.load_split_file(SPLIT)
    train_ids = sorted(p for p, s in split_of.items() if s == "train")
    val_ids = sorted(p for p, s in split_of.items() if s == "validation")
    test_ids = sorted(p for p, s in split_of.items() if s == "test")
    audit = pd.read_csv(AUDIT).set_index("pecha_id")
    audit_cl = pd.read_csv(CL / "tsawa_audit_classical_v3.csv").set_index("pecha_id")
    cl_ids = sorted(audit_cl.index)
    cl_split = {r["pecha_id"]: r["split"] for r in csv.DictReader((CL / "classical_split.csv").open(encoding="utf-8"))}
    assert set(cl_split) == set(cl_ids), "classical_split.csv does not cover exactly the 25 docs"
    cl_train = sorted(p for p in cl_ids if cl_split[p] == "train")
    cl_val = sorted(p for p in cl_ids if cl_split[p] == "validation")
    cl_test = sorted(p for p in cl_ids if cl_split[p] == "test")
    assert not (set(cl_train) & set(cl_val) | set(cl_train) & set(cl_test) | set(cl_val) & set(cl_test))
    assert not set(cl_ids) & set(split_of), "classical doc ids collide with existing books"

    # ---- spans per source
    tr_rows = read_spans(TRAIN_SPANS, lambda r: r["split"] == "train")
    src_of = {}
    with TRAIN_SPANS.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "train":
                src_of[r["pecha_id"]] = r["source"]
    missing = [p for p in train_ids if p not in tr_rows]
    if missing:
        raise SystemExit(f"train books without spans in the normalized file: {missing}")
    extra = sorted(set(tr_rows) - set(train_ids))
    if extra:
        raise SystemExit(f"normalized file has train spans for books not in the split: {extra}")
    val_spans = read_spans(VAL_SPANS, lambda r: r["pecha_id"] in set(val_ids))
    cl_spans = read_spans(CL / "tsawa_spans_classical_v3.csv", lambda r: True)
    ign = read_ignore(IGN_V6, set(train_ids) | set(val_ids))
    for k, v in read_ignore(CL / "ignore_spans_classical_v3.csv", set(cl_ids)).items():
        ign[k] = v

    mask_all = set(cl_ids)
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
    tok.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = B.special_token_ids(tok)
    LAB = B.LABEL2ID

    stats = {"docs": Counter(), "windows": Counter(), "spans": Counter(), "pos": Counter(),
             "ignored_tokens": Counter(), "total_tokens": Counter(), "spans_by_source": Counter(),
             "docs_by_source": Counter(), "windows_by_source": Counter()}

    def examples(split_name: str, docs: list[tuple[str, str, str, float, list, str]]):
        """docs: (pecha_id, base_path, source_batch, coverage_pct, spans, src_tag)."""
        for pid, base, batch, cov, spans, tag in docs:
            text = Path(base).read_text(encoding="utf-8")
            enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
                      return_attention_mask=False)
            ids = list(enc["input_ids"])
            offs = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
            labels = B.label_tokens(offs, spans, LAB)
            ig = B.merge_intervals(ign.get(pid, []))
            n_ign = 0
            if ig:
                for i, (ts, te) in enumerate(offs):
                    # existing books: mask only tokens that carry a tsawa label (v6 rule).
                    # classical repeats are unlabeled verses: mask every token inside them.
                    if te > ts and B.in_intervals(ig, ts) and (labels[i] != LAB["O"] or pid in mask_all):
                        labels[i] = B.IGNORE_LABEL
                        n_ign += 1
            wins = B.sliding_windows(len(ids), content_len, args.stride)
            stats["docs"][split_name] += 1
            stats["docs_by_source"][tag] += 1
            stats["spans"][split_name] += len(spans)
            stats["spans_by_source"][tag] += len(spans)
            stats["windows"][split_name] += len(wins)
            stats["windows_by_source"][tag] += len(wins)
            stats["ignored_tokens"][tag] += n_ign
            for w_i, (ws, we) in enumerate(wins):
                pk = B.pack_window(ids, labels, offs, ws, we, cls_id, sep_id, pad_id, args.max_length)
                pk.update(pecha_id=pid, source_batch=batch, window_index=w_i, n_tokens_doc=len(ids),
                          coverage_pct=float(cov))
                for lab, att in zip(pk["labels"], pk["attention_mask"]):
                    if att == 1 and lab != B.IGNORE_LABEL:
                        stats["total_tokens"][split_name] += 1
                        stats["pos"][split_name] += lab != LAB["O"]
                yield pk
            print(f"  {split_name:10}{tag:12}{pid}  tokens={len(ids):>8} windows={len(wins):>3} spans={len(spans)}",
                  file=sys.stderr, flush=True)

    def mk(split_name, ids_, spans_by, tag_of):
        docs = []
        for p in ids_:
            if p in audit.index:
                base, batch, cov = audit.loc[p, "base_path"], audit.loc[p, "source_batch"], audit.loc[p, "coverage_pct"]
            else:
                base, batch, cov = audit_cl.loc[p, "base_path"], "classical", audit_cl.loc[p, "coverage_pct"]
            docs.append((p, base, batch, cov, spans_by[p], tag_of(p)))
        return docs

    train_docs = mk("train", train_ids, tr_rows, lambda p: "existing" if src_of[p] == "current" else "reannotated") \
        + mk("train", cl_train, cl_spans, lambda p: "classical_train")
    val_docs = mk("validation", val_ids, val_spans, lambda p: "val_frozen")
    clv_docs = mk("classical_validation", cl_val, cl_spans, lambda p: "classical_val")
    clt_docs = mk("classical_test", cl_test, cl_spans, lambda p: "classical_test")

    out = {}
    out["train"] = Dataset.from_generator(lambda: examples("train", train_docs), features=SCHEMA)
    out["validation"] = Dataset.from_generator(lambda: examples("validation", val_docs), features=SCHEMA)
    out["classical_validation"] = Dataset.from_generator(lambda: examples("classical_validation", clv_docs), features=SCHEMA)
    out["classical_test"] = Dataset.from_generator(lambda: examples("classical_test", clt_docs), features=SCHEMA)
    test = load_from_disk(str(TEST_PATCHED))
    if "features" in test.column_names:
        test = test.remove_columns(["features"])
    out["test"] = test.cast(SCHEMA)
    ds = DatasetDict(out)
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out_dir))

    report = {
        "out_dir": str(args.out_dir),
        "split_file": str(SPLIT),
        "docs": {k: dict(stats["docs"]).get(k, 0) for k in ("train", "validation", "classical_validation", "classical_test")} | {"test": len(set(test["pecha_id"]))},
        "classical_train_docs": cl_train, "classical_validation_docs": cl_val, "classical_test_docs": cl_test,
        "windows": {k: len(ds[k]) for k in ds},
        "spans": dict(stats["spans"]),
        "spans_by_source": dict(stats["spans_by_source"]),
        "docs_by_source": dict(stats["docs_by_source"]),
        "windows_by_source": dict(stats["windows_by_source"]),
        "ignored_tokens_by_source": dict(stats["ignored_tokens"]),
        "positive_tokens": dict(stats["pos"]), "scored_tokens": dict(stats["total_tokens"]),
    }
    (args.out_dir / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

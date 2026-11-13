"""Remove the '_' characters (a syllable-separator artifact of some sources) from the text, and shift the spans.

Only '_' is removed. Reads the *_notrans copies, writes *_notrans_nous copies. Only works with a '_' are re-tokenized
(same code path as ta_build_gold_dataset.py); a self-check re-tokenizes untouched works and compares them with the old rows."""
import random, sys
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from build_tsawa_dataset import IGNORE_LABEL, label_tokens, pack_window, sliding_windows, special_token_ids  # noqa: E402

LABEL2ID = {"O": 0, "B-TITLE": 1, "I-TITLE": 2, "B-AUTHOR": 3, "I-AUTHOR": 4}   # same as ta_build_gold_dataset.py


def skip_leading_ws(offs, text):
    """Copy of the builder's helper: move each token start past leading whitespace (labelling only)."""
    out = []
    for s, e in offs:
        if text[s:e].strip():
            while s < e and text[s].isspace(): s += 1
        out.append((s, e))
    return out

P = ROOT / "title_author/data/processed"
SRC_PQ, SRC_HF = P / "parquet_export_strat_both_notrans", P / "gold_dataset_all_strat_single_v3_both_notrans"
DST_PQ, DST_HF = P / "parquet_export_strat_both_notrans_nous", P / "gold_dataset_all_strat_single_v3_both_notrans_nous"
TOK = Path.home() / ".cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots/c5955035435e2bf121cde7f3c8863ef52ff35d82"
MAXLEN, STRIDE = 8192, 5120
tok = AutoTokenizer.from_pretrained(str(TOK), use_fast=True); tok.model_max_length = int(1e12)
cls_id, sep_id, pad_id = special_token_ids(tok)


def windows_for(r, cid):
    """Rows of the tokenized dataset for one work, built the way the builder builds them (--single)."""
    pieces = [("head", r.text_head)] + ([("tail", r.text_tail)] if (not r.whole_work and isinstance(r.text_tail, str) and r.text_tail) else [])
    parts = []
    for rname, rt in pieces:
        sp = []
        for kind in ("TITLE", "AUTHOR"):
            w = kind.lower()
            if getattr(r, w + "_region") == rname:
                sp.append((int(getattr(r, w + "_start")), int(getattr(r, w + "_end")), r.work_id, kind))
        sp.sort()
        enc = tok(rt, add_special_tokens=False, return_offsets_mapping=True)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        parts.append((list(enc["input_ids"]), list(label_tokens(skip_leading_ws(offs, rt), sp, LABEL2ID)), offs, rt))
    if len(parts) == 2:
        (i1, l1, o1, t1), (i2, l2, o2, _) = parts; shift = len(t1) + 1
        ids, labels, offs, region = i1 + [sep_id] + i2, l1 + [IGNORE_LABEL] + l2, o1 + [(len(t1), len(t1))] + [(x + shift, y + shift) for x, y in o2], "both"
    else:
        ids, labels, offs, region = parts[0][0], parts[0][1], parts[0][2], "full"
    out = []
    for wi, (ws, we) in enumerate(sliding_windows(len(ids), MAXLEN - 2, STRIDE)):
        ex = pack_window(ids, labels, offs, ws, we, cls_id, sep_id, pad_id, MAXLEN)
        ex.update(work_id=r.work_id, cid=cid, source=r.source, region=region, region_start=0, window_index=wi, n_tokens_region=len(ids))
        out.append(ex)
    return out


def strip_row(d):
    """d: dict of one parquet row. Remove '_' from the texts and shift the spans. Returns (row, changed)."""
    texts = {"head": d["text_head"], "tail": d["text_tail"] if isinstance(d["text_tail"], str) else ""}
    if not any("_" in t for t in texts.values()) and "_" not in d["title"] and "_" not in d["author"]:
        return d, False
    d = dict(d)
    for w in ("title", "author"):
        t = texts[d[w + "_region"]]; s = int(d[w + "_start"]); e = int(d[w + "_end"])
        span = t[s:e].replace("_", ""); ns = s - t.count("_", 0, s)
        d[w] = span; d[w + "_start"] = ns; d[w + "_end"] = ns + len(span)
    d["text_head"] = texts["head"].replace("_", "")
    if isinstance(d["text_tail"], str): d["text_tail"] = d["text_tail"].replace("_", "")
    for w in ("title", "author"):
        t = d["text_head"] if d[w + "_region"] == "head" else d["text_tail"]
        assert t[d[w + "_start"]:d[w + "_end"]] == d[w], (d["work_id"], w)
    return d, True


DST_PQ.mkdir(exist_ok=True)
frames, changed_ids, empty = {}, set(), []
for s in ("train", "validation", "test"):
    df = pd.read_parquet(SRC_PQ / f"{s}.parquet"); rows = []
    for d in df.to_dict("records"):
        nd, ch = strip_row(d)
        if ch:
            if not nd["title"].strip() or not nd["author"].strip(): empty.append(d["work_id"]); continue
            changed_ids.add(d["work_id"])
        rows.append(nd)
    out = pd.DataFrame(rows, columns=df.columns).astype(df.dtypes.to_dict(), errors="ignore")
    out.to_parquet(DST_PQ / f"{s}.parquet", index=False); frames[s] = out
    print(s, len(df), "->", len(out), "| works edited:", sum(w in changed_ids for w in out.work_id))
print("works dropped because a span was only underscores:", len(empty), empty[:5])
for f in ("works_index.csv", "README.md"):
    if (SRC_PQ / f).exists(): (DST_PQ / f).write_bytes((SRC_PQ / f).read_bytes())

ds = load_from_disk(str(SRC_HF)); new = {}
for s in ("train", "validation", "test"):
    old = ds[s]; cid_of = dict(zip(old["work_id"], old["cid"]))
    keep = old.filter(lambda r: r["work_id"] not in changed_ids and r["work_id"] not in empty)
    rows = [ex for r in frames[s].itertuples() if r.work_id in changed_ids for ex in windows_for(r, cid_of.get(r.work_id, ""))]
    new[s] = concatenate_datasets([keep, Dataset.from_list(rows, features=old.features)]) if rows else keep
    print(s, "tokenized rows", len(old), "->", len(new[s]), "| re-tokenized windows:", len(rows))
DatasetDict(new).save_to_disk(str(DST_HF))

# self-check: untouched works re-tokenized from the parquet must equal the stored rows
random.seed(0); bad = n = 0
old = ds["train"]; idx = {}
for i, w in enumerate(old["work_id"]): idx.setdefault(w, []).append(i)
pool = [r for r in frames["train"].itertuples() if r.work_id not in changed_ids and r.work_id in idx]
for r in random.sample(pool, 300):
    mine = windows_for(r, "x"); theirs = [old[i] for i in idx[r.work_id]]
    n += 1; bad += not (len(mine) == len(theirs) and all(a["input_ids"] == b["input_ids"] and a["labels"] == b["labels"] for a, b in zip(mine, theirs)))
print(f"self-check: {n - bad}/{n} untouched works re-tokenize identically")

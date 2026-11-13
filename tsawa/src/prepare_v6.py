#!/usr/bin/env python3
"""Prepare split_v3 + ignore-span list for tsawa dataset v6. Does not rewrite
existing sidecars, splits, datasets, or YAML.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratch" / "tsawa" / "scripts"))
sys.path.insert(0, str(ROOT / "tsawa" / "src"))
sys.path.insert(0, str(ROOT / "common"))
from build_stratified_split import greedy_assign, n_windows_from_tokens  # noqa: E402

MERGED = ROOT / "tsawa/data/processed/tsawa_spans_merged.csv"
AUDIT = ROOT / "data/shared/tsawa_audit.csv"
SPLIT_V2 = ROOT / "tsawa/data/processed/split_v2_frozen.csv"
QUOTE_SC = ROOT / "tsawa/data/processed/quotation_spans_snapped.csv"
OUT_SPLIT = ROOT / "data/shared/split_v3_frozen.csv"
OUT_IGNORE = ROOT / "tsawa/data/processed/v6_ignore_spans.csv"
OUT_GROUPS = ROOT / "scratch/tsawa/analysis/v6_root_groups.json"
OUT_REPORT = ROOT / "scratch/tsawa/analysis/v6_prep_report.json"
V5 = ROOT / "tsawa/data/processed/tsawa_dataset_v5_merged"

MIN_SPANS = 10
MATCH_MIN = 40
LINK_SHARE = 0.10
SEED = 123
FORCE_IGNORE = {"I100E7DAD", "I058DD999"}
SOURCE_MARKERS = (
    "གཞུང་ལས",
    "རྒྱུད་ལས",
    "ལུང་ལས",
    "མདོ་ལས",
    "ལས།",
    "ནས།",
    "ལས་",
)
WS = re.compile(r"\s+")


def strip_ws(s: str) -> str:
    return WS.sub("", s)


def load_merged() -> dict[str, list[dict]]:
    by = defaultdict(list)
    with MERGED.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("dropped", "")).lower() == "true":
                continue
            s, e = int(r["start"]), int(r["end"])
            if e <= s:
                continue
            by[r["pecha_id"]].append({"start": s, "end": e, "ann_id": r["ann_id"]})
    return by


def load_audit() -> dict[str, dict]:
    out = {}
    with AUDIT.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["pecha_id"]] = r
    return out


def load_v2_groups(keep: set[str]) -> list[set[str]]:
    members = defaultdict(list)
    body = [l for l in SPLIT_V2.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    for r in csv.DictReader(body):
        if r["pecha_id"] in keep:
            members[r["group_id"]].append(r["pecha_id"])
    return [set(v) for v in members.values() if len(v) > 1]


def load_quote() -> dict[str, list[tuple[int, int]]]:
    by = defaultdict(list)
    if not QUOTE_SC.is_file():
        return by
    with QUOTE_SC.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("dropped", "")).lower() == "true":
                continue
            s, e = int(r["start"]), int(r["end"])
            if e > s:
                by[r["pecha_id"]].append((s, e))
    for v in by.values():
        v.sort()
    return by


def overlap_frac(s, e, quotes) -> float:
    ln = e - s
    if ln <= 0 or not quotes:
        return 0.0
    ov = 0
    for qs, qe in quotes:
        if qe <= s:
            continue
        if qs >= e:
            break
        ov += min(e, qe) - max(s, qs)
    return ov / ln


def ends_with_marker(prefix: str) -> bool:
    return any(prefix.endswith(m) for m in SOURCE_MARKERS)


class UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def ntok_from_v5() -> dict[str, int]:
    from datasets import load_from_disk

    ds = load_from_disk(str(V5))
    out = {}
    for name in ds:
        for ex in ds[name]:
            out[ex["pecha_id"]] = int(ex["n_tokens_doc"])
    return out


def main() -> int:
    merged = load_merged()
    audit = load_audit()
    ntok = ntok_from_v5()

    excluded = []
    keep_ids = []
    for pid, spans in sorted(merged.items()):
        batch = "old" if pid.startswith("P") else "new"
        if len(spans) < MIN_SPANS:
            excluded.append({"pecha_id": pid, "batch": batch, "n_spans": len(spans)})
        else:
            keep_ids.append(pid)
    keep = set(keep_ids)
    print(f"rule1 excluded {len(excluded)}  remaining {len(keep)}")

    texts = {}
    stripped = {}
    needles = {}  # pid -> list[(stripped_span, orig_len, start, end)]
    grams = {}
    for pid in keep_ids:
        base = Path(audit[pid]["base_path"])
        texts[pid] = base.read_text(encoding="utf-8")
        stripped[pid] = strip_ws(texts[pid])
        st = stripped[pid]
        grams[pid] = {hash(st[i : i + 40]) for i in range(0, max(0, len(st) - 39))}
        ns = []
        for sp in merged[pid]:
            raw = texts[pid][sp["start"] : sp["end"]]
            if len(raw) < MATCH_MIN:
                continue
            nd = strip_ws(raw)
            if len(nd) >= 8:
                ns.append((nd, len(raw), sp["start"], sp["end"]))
        needles[pid] = ns

    uf = UF(keep_ids)
    links = []
    n = len(keep_ids)
    for i, a in enumerate(keep_ids):
        na = needles[a]
        if not na:
            continue
        tot = sum(L for _, L, *_ in na) or 1
        for j in range(i + 1, n):
            b = keep_ids[j]
            hit_a = 0
            for nd, L, *_ in na:
                if len(nd) >= 40 and hash(nd[:40]) not in grams[b]:
                    continue
                if nd in stripped[b]:
                    hit_a += L
            share_ab = hit_a / tot
            nb = needles[b]
            tot_b = sum(L for _, L, *_ in nb) or 1
            hit_b = 0
            if nb:
                for nd, L, *_ in nb:
                    if len(nd) >= 40 and hash(nd[:40]) not in grams[a]:
                        continue
                    if nd in stripped[a]:
                        hit_b += L
            share_ba = hit_b / tot_b
            if share_ab >= LINK_SHARE or share_ba >= LINK_SHARE:
                uf.union(a, b)
                links.append((a, b, share_ab, share_ba))
        if (i + 1) % 20 == 0:
            print(f"  pair scan {i+1}/{n}", flush=True)

    for g in load_v2_groups(keep):
        g = [p for p in g if p in keep]
        for p in g[1:]:
            uf.union(g[0], p)

    members = defaultdict(list)
    for pid in keep_ids:
        members[uf.find(pid)].append(pid)
    groups = [{"group_id": gid, "pecha_ids": sorted(ids)} for gid, ids in members.items()]
    multi = [g for g in groups if len(g["pecha_ids"]) > 1]
    largest = max(groups, key=lambda g: len(g["pecha_ids"]))
    if428 = next((g for g in groups if "IF428CDDB" in g["pecha_ids"]), None)
    print(f"root-text groups {len(groups)}  multi {len(multi)}  largest {len(largest['pecha_ids'])}")
    print(f"IF428CDDB group: {if428}")
    print(f"linked pairs >=10%: {len(links)}")

    rows = []
    for pid in keep_ids:
        rec = audit[pid]
        text = texts[pid]
        spans = merged[pid]
        lens = [sp["end"] - sp["start"] for sp in spans]
        tsawa_chars = sum(lens)
        doc_chars = len(text)
        n_short = sum(1 for L in lens if L < 30)
        nt = ntok.get(pid)
        if nt is None:
            nw = max(1, math.ceil(doc_chars / (5120 * 3.5)))
        else:
            nw = n_windows_from_tokens(nt)
        rows.append(
            {
                "pecha_id": pid,
                "batch": "old" if pid.startswith("P") else "new",
                "doc_chars": doc_chars,
                "n_spans": len(spans),
                "tsawa_chars": tsawa_chars,
                "density": tsawa_chars / doc_chars if doc_chars else 0.0,
                "n_short": n_short,
                "pct_short": n_short / len(spans),
                "n_windows": nw,
            }
        )

    tot_w = sum(r["n_windows"] for r in rows)
    tot_c = sum(r["doc_chars"] for r in rows)
    tot_t = sum(r["tsawa_chars"] for r in rows)
    tot_sp = sum(r["n_spans"] for r in rows)
    tot_short = sum(r["n_short"] for r in rows)
    tot_old_w = sum(r["n_windows"] for r in rows if r["batch"] == "old")
    targets = {
        "density": tot_t / tot_c,
        "pct_short": tot_short / tot_sp,
        "batch_old_win": tot_old_w / tot_w,
    }
    weights = {
        "windows": 12.0,
        "density": 80.0,
        "pct_short": 10.0,
        "batch": 8.0,
        "gap": 40.0,
    }
    assignment, sc = greedy_assign(rows, groups, SEED, weights, targets)
    print(f"greedy score {sc:.6f}")

    header = [
        "# tsawa layer-detection document split v3 — FROZEN",
        f"# generated: {date.today().isoformat()}",
        "# generator: tsawa/src/prepare_v6.py (greedy multi-objective, same as split_v2)",
        f"# seed: {SEED}",
        "# targets: 80/10/10 by WINDOW count (8192 / 5120)",
        "# stratified on: batch, n_windows, tsawa density, pct_short",
        "# rule 1: drop pechas with < 10 merged tsawa spans (all splits)",
        "# rule 2: must-link = reprint groups from split_v2_frozen.csv UNION "
        "root-text groups (either-direction >=10% of long (>=40 char) "
        "whitespace-stripped tsawa text found verbatim in the other book's base)",
        "# TEST SPLIT IS FROZEN: documents in it must not change between experiments; "
        "evaluate test once at the end, never for tuning or model selection.",
        "# span source: tsawa/data/processed/tsawa_spans_merged.csv",
    ]
    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    gid_of = {}
    for g in groups:
        for p in g["pecha_ids"]:
            gid_of[p] = g["group_id"]
    with OUT_SPLIT.open("w", newline="", encoding="utf-8") as fh:
        for line in header:
            fh.write(line + "\n")
        w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "group_id"])
        w.writeheader()
        for pid in sorted(keep_ids):
            w.writerow({"pecha_id": pid, "split": assignment[pid], "group_id": gid_of[pid]})
    print(f"wrote {OUT_SPLIT}")

    # leakage: val/test long spans found in any train book
    train_ids = [p for p in keep_ids if assignment[p] == "train"]
    train_blob_grams = set()
    train_stripped = [stripped[p] for p in train_ids]
    for p in train_ids:
        train_blob_grams |= grams[p]

    def found_in_train(nd: str) -> bool:
        if len(nd) >= 40 and hash(nd[:40]) not in train_blob_grams:
            return False
        return any(nd in t for t in train_stripped)

    leak = {}
    for spn in ("val", "test"):
        n_long = n_hit = 0
        for pid in keep_ids:
            if assignment[pid] != spn:
                continue
            for nd, L, *_ in needles[pid]:
                if L < MATCH_MIN:
                    continue
                n_long += 1
                if found_in_train(nd):
                    n_hit += 1
        leak[spn] = {"n_long": n_long, "n_hit": n_hit,
                     "pct": 100.0 * n_hit / n_long if n_long else 0.0}
        flag = " FLAG" if leak[spn]["pct"] >= 5.0 else ""
        print(f"leakage {spn}: {leak[spn]['n_hit']}/{leak[spn]['n_long']} "
              f"= {leak[spn]['pct']:.2f}%{flag}")

    quotes = load_quote()
    ignore_rows = []
    for pid in keep_ids:
        q = quotes.get(pid, [])
        for sp in merged[pid]:
            s, e = sp["start"], sp["end"]
            reason = ""
            if pid in FORCE_IGNORE:
                reason = "force_book"
            else:
                pref = texts[pid][max(0, s - 40) : s]
                if overlap_frac(s, e, q) >= 0.5 and ends_with_marker(pref):
                    reason = "quote_half_and_source_marker"
            if reason:
                ignore_rows.append(
                    {
                        "pecha_id": pid,
                        "ann_id": sp["ann_id"],
                        "start": s,
                        "end": e,
                        "split": assignment[pid],
                        "reason": reason,
                    }
                )
    with OUT_IGNORE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["pecha_id", "ann_id", "start", "end", "split", "reason"]
        )
        w.writeheader()
        w.writerows(ignore_rows)
    ign_split = defaultdict(int)
    ign_ib = 0
    for r in ignore_rows:
        ign_split[r["split"]] += 1
        if r["pecha_id"] == "IB522F095":
            ign_ib += 1
    print(f"ignore spans {len(ignore_rows)}  by split {dict(ign_split)}  IB522F095={ign_ib}")

    # split stats for report
    from collections import Counter

    split_tbl = {}
    for sp in ("train", "val", "test"):
        sub = [r for r in rows if assignment[r["pecha_id"]] == sp]
        split_tbl[sp] = {
            "books": len(sub),
            "windows": sum(r["n_windows"] for r in sub),
            "spans": sum(r["n_spans"] for r in sub),
            "old": sum(1 for r in sub if r["batch"] == "old"),
            "new": sum(1 for r in sub if r["batch"] == "new"),
        }

    report = {
        "excluded": excluded,
        "n_groups": len(groups),
        "n_multi": len(multi),
        "largest_size": len(largest["pecha_ids"]),
        "largest_ids": largest["pecha_ids"],
        "if428": if428["pecha_ids"] if if428 else None,
        "links": len(links),
        "split": split_tbl,
        "leak": leak,
        "ignore": {
            "total": len(ignore_rows),
            "train": ign_split["train"],
            "val": ign_split["val"],
            "test": ign_split["test"],
            "IB522F095": ign_ib,
        },
        "n_keep": len(keep),
    }
    OUT_GROUPS.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Profile tsawa pechas and greedily assign a window-balanced split.

Shared helper of the Tsawa split. ``prepare_tsawa_split.py`` imports ``greedy_assign`` and
``n_windows_from_tokens`` from here; run on its own, this file profiles the pechas and writes a
proposal under ``scratch/tsawa/``. It is the generator named in the header of
``tsawa/data/processed/split_v2_frozen.csv`` (greedy multi-objective, seed 123), copied from the
author's working checkout with only the folder paths changed to repo-relative ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESOLVED = ROOT / "tsawa/data/processed/tsawa_spans_resolved.csv"
AUDIT = ROOT / "data/shared/tsawa_audit.csv"
DS = ROOT / "tsawa/data/processed/tsawa_dataset_v10"
OUT_PROFILE = ROOT / "scratch/tsawa/pecha_profile.csv"
OUT_SPLIT = ROOT / "scratch/tsawa/split_assignment.csv"
OUT_REPORT = ROOT / "scratch/tsawa/split_report.md"

sys.path.insert(0, str(ROOT / "common"))
from bio_dataset import sliding_windows  # noqa: E402

CONTENT_LEN = 8190  # 8192 - CLS - SEP
STRIDE = 5120
SHORT_MAX = 29  # < 30
MED_MAX = 200  # 30-200 inclusive

TIB_HEAD = re.compile(r"^[\s༄༅།༼༽]+")


def n_windows_from_tokens(n_tok: int) -> int:
    return len(sliding_windows(n_tok, CONTENT_LEN, STRIDE))


def load_resolved() -> dict[str, list[tuple[int, int]]]:
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with RESOLVED.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("dropped", "")).lower() == "true":
                continue
            s, e = int(row["start"]), int(row["end"])
            if e > s:
                by[row["pecha_id"]].append((s, e))
    return by


def load_audit_paths() -> dict[str, dict]:
    out = {}
    with AUDIT.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["pecha_id"]] = row
    return out


def find_meta(pecha_id: str, audit: dict) -> Path | None:
    opf = audit.get("opf_root")
    if opf:
        p = Path(opf) / "meta.yml"
        if p.is_file():
            return p
    raw = ROOT / "data/raw_opf"
    for cand in (
        raw / f"{pecha_id}.opf" / f"{pecha_id}.opf" / "meta.yml",
        raw / f"{pecha_id}.opf" / "meta.yml",
    ):
        if cand.is_file():
            return cand
    return None


def _get(d, *path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if cur is None or cur == "" or cur == {}:
        return None
    if isinstance(cur, dict):
        return cur.get("en") or cur.get("bo") or json.dumps(cur, ensure_ascii=False)
    return str(cur).strip() or None


def parse_meta(path: Path | None) -> dict:
    empty = {
        "publisher": "",
        "series": "",
        "book_code": "",
        "title": "",
        "sku": "",
        "source_id": "",
        "meta_keys": "",
        "has_publisher": False,
        "has_series": False,
        "has_book_code_field": False,
        "has_title": False,
    }
    if path is None or not path.is_file():
        return empty
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def walk(obj, prefix=""):
        keys = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                keys.append(p)
                keys.extend(walk(v, p))
        return keys

    keys = walk(data)
    title = (
        _get(data, "source_metadata", "title")
        or _get(data, "ebook_metadata", "title")
        or _get(data, "web_metadata", "title")
        or ""
    )
    # No book_code field observed; sku / source_metadata.id are the identifiers.
    sku = _get(data, "source_metadata", "sku") or _get(data, "ebook_metadata", "sku") or _get(data, "web_metadata", "sku") or ""
    source_id = _get(data, "source_metadata", "id") or ""
    series = _get(data, "source_metadata", "series") or _get(data, "web_metadata", "collection") or ""
    publisher = (
        _get(data, "source_metadata", "publisher")
        or _get(data, "web_metadata", "pusblisher")  # typo in old-batch YAML
        or _get(data, "web_metadata", "publisher")
        or _get(data, "web_metadata", "org")
        or ""
    )
    has_book_code = "source_metadata.book_code" in keys or "book_code" in keys
    return {
        "publisher": publisher,
        "series": series,
        "book_code": "",  # field does not exist
        "title": title,
        "sku": sku,
        "source_id": source_id,
        "meta_keys": "|".join(sorted(keys)),
        "has_publisher": bool(publisher),
        "has_series": bool(series),
        "has_book_code_field": has_book_code,
        "has_title": bool(title),
    }


def norm_title(t: str) -> str:
    t = TIB_HEAD.sub("", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("༄༅།", "").replace("༄༅", "")
    return t


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    a = np.array(xs, dtype=float)
    return float(np.percentile(a, p))


def load_current_split_and_tokens():
    from datasets import load_from_disk

    ds = load_from_disk(str(DS))
    split_of = {}
    ntok = {}
    nwin = Counter()
    for name in ("train", "validation", "test"):
        for ex in ds[name]:
            pid = ex["pecha_id"]
            split_of[pid] = "val" if name == "validation" else name
            ntok[pid] = int(ex["n_tokens_doc"])
            nwin[pid] += 1
    return split_of, ntok, dict(nwin)


def pearson(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def split_stats(rows: list[dict], assignment: dict[str, str]) -> dict[str, dict]:
    out = {}
    for sp in ("train", "val", "test"):
        subset = [r for r in rows if assignment.get(r["pecha_id"]) == sp]
        dc = sum(r["doc_chars"] for r in subset) or 1
        tc = sum(r["tsawa_chars"] for r in subset)
        ns = sum(r["n_spans"] for r in subset) or 1
        nw = sum(r["n_windows"] for r in subset)
        n_old = sum(r["n_windows"] for r in subset if r["batch"] == "old")
        n_short = sum(r["n_short"] for r in subset)
        n_new_docs = sum(1 for r in subset if r["batch"] == "new")
        out[sp] = {
            "n_docs": len(subset),
            "n_windows": nw,
            "n_spans": sum(r["n_spans"] for r in subset),
            "density": tc / dc,
            "pct_short": n_short / ns,
            "batch_old_win": n_old / nw if nw else 0.0,
            "batch_new_docs": n_new_docs,
            "n_old_docs": sum(1 for r in subset if r["batch"] == "old"),
            "mean_pct_short_pecha": float(np.mean([r["pct_short"] for r in subset])) if subset else 0.0,
        }
    tot_w = sum(out[s]["n_windows"] for s in out) or 1
    for sp in out:
        out[sp]["win_share"] = out[sp]["n_windows"] / tot_w
    return out


def objective(stats: dict, g: dict, targets: dict, w: dict) -> float:
    """SSE of deviations after adding group g (or current if g is None)."""
    # stats already includes the assignment
    score = 0.0
    for sp, tgt_share in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        s = stats[sp]
        score += w["windows"] * (s["win_share"] - tgt_share) ** 2
        score += w["density"] * (s["density"] - targets["density"]) ** 2
        score += w["pct_short"] * (s["pct_short"] - targets["pct_short"]) ** 2
        score += w["batch"] * (s["batch_old_win"] - targets["batch_old_win"]) ** 2
    return score


def try_add(rows, assignment, group_ids, split, group_map) -> dict:
    trial = dict(assignment)
    for pid in group_ids:
        trial[pid] = split
    return split_stats(rows, trial)


def score_stats(st: dict, weights: dict, targets: dict,
                shares: dict | None = None) -> float:
    shares = shares or {"train": 0.8, "val": 0.1, "test": 0.1}
    sc = 0.0
    for name, tgt_share in shares.items():
        s = st[name]
        sc += weights["windows"] * (s["win_share"] - tgt_share) ** 2
        sc += weights["density"] * (s["density"] - targets["density"]) ** 2
        sc += weights["pct_short"] * (s["pct_short"] - targets["pct_short"]) ** 2
        sc += weights["batch"] * (s["batch_old_win"] - targets["batch_old_win"]) ** 2
    dens = [st[s]["density"] for s in ("train", "val", "test")]
    sc += weights.get("gap", 20.0) * (max(dens) - min(dens)) ** 2
    return sc


def greedy_assign(rows, groups, seed: int, weights: dict, targets: dict,
                  shares: dict | None = None) -> tuple[dict, float]:
    rng = random.Random(seed)
    row_of = {r["pecha_id"]: r for r in rows}

    def group_nw(g):
        return sum(row_of[p]["n_windows"] for p in g["pecha_ids"])

    # Shuffle among similar sizes so seeds actually differ.
    ordered = sorted(groups, key=lambda g: (-group_nw(g), rng.random()))
    assignment: dict[str, str] = {}
    for g in ordered:
        best_sp, best_sc = None, None
        for sp in ("train", "val", "test"):
            st = try_add(rows, assignment, g["pecha_ids"], sp, None)
            sc = score_stats(st, weights, targets, shares)
            if best_sc is None or sc < best_sc:
                best_sc, best_sp = sc, sp
        for pid in g["pecha_ids"]:
            assignment[pid] = best_sp

    # Local swaps: try moving singleton groups if it lowers the score.
    gid_of = {}
    for g in groups:
        for p in g["pecha_ids"]:
            gid_of[p] = id(g)
    by_g = defaultdict(list)
    for g in groups:
        by_g[id(g)] = g["pecha_ids"]
    for _ in range(3):
        improved = False
        for g in groups:
            cur = assignment[g["pecha_ids"][0]]
            for sp in ("train", "val", "test"):
                if sp == cur:
                    continue
                trial = dict(assignment)
                for p in g["pecha_ids"]:
                    trial[p] = sp
                st0 = split_stats(rows, assignment)
                st1 = split_stats(rows, trial)
                if st1["val"]["n_docs"] < 12 or st1["test"]["n_docs"] < 12:
                    continue
                if score_stats(st1, weights, targets, shares) + 1e-9 < score_stats(st0, weights, targets, shares):
                    assignment = trial
                    improved = True
                    break
        if not improved:
            break

    st = split_stats(rows, assignment)
    return assignment, score_stats(st, weights, targets, shares)


def kfold_assign(rows, groups, n_folds: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    ordered = sorted(
        groups,
        key=lambda g: -sum(
            next(r for r in rows if r["pecha_id"] == p)["n_windows"] for p in g["pecha_ids"]
        )
        + rng.random() * 0.01,
    )
    fold_w = [0] * n_folds
    assignment = {}
    for g in ordered:
        nw = sum(next(r for r in rows if r["pecha_id"] == p)["n_windows"] for p in g["pecha_ids"])
        k = min(range(n_folds), key=lambda i: fold_w[i])
        for p in g["pecha_ids"]:
            assignment[p] = f"fold{k}"
        fold_w[k] += nw
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=1, help="1 = train/val/test; k = CV folds")
    ap.add_argument("--seeds", default="0,1,2,3,4,7,11,13,17,19,42,99,123,256,2026")
    args = ap.parse_args()

    by_spans = load_resolved()
    audit = load_audit_paths()
    current_split, ntok, nwin_actual = load_current_split_and_tokens()

    all_lens: list[int] = []
    rows = []
    meta_key_counter = Counter()
    n_book_code_field = 0

    for pid, spans in sorted(by_spans.items()):
        rec = audit.get(pid, {})
        base = Path(rec["base_path"]) if rec.get("base_path") else None
        if base is None or not base.is_file():
            raise SystemExit(f"missing base for {pid}")
        text = base.read_text(encoding="utf-8")
        doc_chars = len(text)
        lens = [e - s for s, e in spans]
        all_lens.extend(lens)
        tsawa_chars = sum(lens)
        n_short = sum(1 for L in lens if L < 30)
        n_med = sum(1 for L in lens if 30 <= L <= 200)
        n_long = sum(1 for L in lens if L > 200)
        n = len(lens)
        meta_path = find_meta(pid, rec)
        meta = parse_meta(meta_path)
        if meta["has_book_code_field"]:
            n_book_code_field += 1
        for k in meta["meta_keys"].split("|"):
            if k:
                meta_key_counter[k] += 1
        nt = ntok.get(pid)
        if nt is None:
            # should not happen; all tsawa pechas are in the dataset
            n_windows = max(1, math.ceil(doc_chars / (STRIDE * 3.5)))
        else:
            n_windows = n_windows_from_tokens(nt)
            if pid in nwin_actual and nwin_actual[pid] != n_windows:
                n_windows = nwin_actual[pid]
        batch = "old" if pid.startswith("P") else "new"
        dens = tsawa_chars / doc_chars if doc_chars else 0.0
        rows.append(
            {
                "pecha_id": pid,
                "batch": batch,
                "doc_chars": doc_chars,
                "n_spans": n,
                "tsawa_chars": tsawa_chars,
                "density": dens,
                "span_len_median": float(np.median(lens)),
                "span_len_mean": float(np.mean(lens)),
                "span_len_p90": percentile(lens, 90),
                "span_len_max": max(lens),
                "n_short": n_short,
                "n_medium": n_med,
                "n_long": n_long,
                "pct_short": n_short / n,
                "pct_medium": n_med / n,
                "pct_long": n_long / n,
                "n_windows": n_windows,
                "n_tokens_doc": nt if nt is not None else -1,
                "publisher": meta["publisher"],
                "series": meta["series"],
                "book_code": meta["sku"] or meta["source_id"],
                "title": meta["title"],
                "sku": meta["sku"],
                "source_id": meta["source_id"],
                "has_publisher": meta["has_publisher"],
                "has_series": meta["has_series"],
                "current_split": current_split.get(pid, ""),
            }
        )

    dens_vals = [r["density"] for r in rows]
    t1, t2 = percentile(dens_vals, 100 / 3), percentile(dens_vals, 200 / 3)
    for r in rows:
        if r["density"] <= t1:
            r["density_band"] = "sparse"
        elif r["density"] <= t2:
            r["density_band"] = "moderate"
        else:
            r["density_band"] = "dense"
        prof = max(
            ("short", r["n_short"]),
            ("medium", r["n_medium"]),
            ("long", r["n_long"]),
            key=lambda x: x[1],
        )[0]
        r["dominant_length"] = prof

    # PHASE 1 print
    print("=== PHASE 1: PROFILE ===")
    print(f"pechas={len(rows)} spans={sum(r['n_spans'] for r in rows)}")
    print(f"meta.yml keys across all {len(rows)} pechas (count of pechas having key):")
    for k, n in meta_key_counter.most_common():
        if n >= 10 or "title" in k.lower() or "series" in k.lower() or "publish" in k.lower() or "sku" in k.lower() or "book" in k.lower():
            print(f"  {n:3d}/{len(rows)}  {k}")
    print(f"book_code FIELD present on {n_book_code_field}/{len(rows)} (expected 0).")
    print("Using sku/source_metadata.id as book_code proxy in the CSV.")
    print(
        f"usable publisher: {sum(r['has_publisher'] for r in rows)}  "
        f"usable series/collection: {sum(r['has_series'] for r in rows)}  "
        f"nonempty title: {sum(1 for r in rows if r['title'])}"
    )

    arr = np.array(all_lens)
    print("\nSPAN LENGTH DISTRIBUTION (chars, active resolved spans):")
    for p in (0, 10, 25, 50, 75, 90, 95, 99, 100):
        print(f"  p{p:03d}={np.percentile(arr, p):.1f}")
    n_all = len(arr)
    print(
        f"  buckets as specified: short<30 {int((arr<30).sum())} ({100*(arr<30).mean():.1f}%)  "
        f"medium 30-200 {int(((arr>=30)&(arr<=200)).sum())} ({100*((arr>=30)&(arr<=200)).mean():.1f}%)  "
        f"long>200 {int((arr>200).sum())} ({100*(arr>200).mean():.1f}%)"
    )
    # histogram bins
    edges = [0, 15, 30, 50, 80, 120, 200, 400, 800, 2000, 10**9]
    print("  finer histogram:")
    for a, b in zip(edges, edges[1:]):
        c = int(((arr >= a) & (arr < b)).sum())
        print(f"    [{a},{b}): {c} ({100*c/n_all:.1f}%)")

    print(f"\nDENSITY TERCILES: sparse ≤ {t1:.4f}  moderate ≤ {t2:.4f}  else dense")
    print("pechas per batch:", Counter(r["batch"] for r in rows))
    print("pechas per density_band:", Counter(r["density_band"] for r in rows))
    print("pechas per dominant_length:", Counter(r["dominant_length"] for r in rows))
    print("batch × density:")
    for b in ("new", "old"):
        c = Counter(r["density_band"] for r in rows if r["batch"] == b)
        dmean = float(np.mean([r["density"] for r in rows if r["batch"] == b]))
        print(f"  {b}: {dict(c)}  mean density={dmean:.4f}")

    OUT_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pecha_id", "batch", "doc_chars", "n_spans", "tsawa_chars", "density",
        "density_band", "span_len_median", "span_len_mean", "span_len_p90", "span_len_max",
        "n_short", "n_medium", "n_long", "pct_short", "pct_medium", "pct_long",
        "dominant_length", "n_windows", "n_tokens_doc",
        "publisher", "series", "book_code", "title", "sku", "source_id",
        "has_publisher", "has_series", "current_split",
    ]
    with OUT_PROFILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_PROFILE}")

    # PHASE 2
    print("\n=== PHASE 2: WHAT MATTERS ===")
    names = ["density", "pct_short", "span_len_median", "doc_chars", "n_spans", "n_windows", "tsawa_chars"]
    mat = np.array([[r[k] for k in names] for r in rows], dtype=float)
    print("Pearson correlation:")
    print("            " + "".join(f"{n:>16}" for n in names))
    for i, n in enumerate(names):
        line = f"{n:>12}"
        for j in range(len(names)):
            line += f"{np.corrcoef(mat[:, i], mat[:, j])[0,1]:16.3f}"
        print(line)

    old_d = [r["density"] for r in rows if r["batch"] == "old"]
    new_d = [r["density"] for r in rows if r["batch"] == "new"]
    old_s = [r["pct_short"] for r in rows if r["batch"] == "old"]
    new_s = [r["pct_short"] for r in rows if r["batch"] == "new"]
    print(f"\nbatch density  old mean={np.mean(old_d):.4f} med={np.median(old_d):.4f}  "
          f"new mean={np.mean(new_d):.4f} med={np.median(new_d):.4f}")
    print(f"batch pct_short old mean={np.mean(old_s):.4f}  new mean={np.mean(new_s):.4f}")
    print(f"batch n_windows old={sum(r['n_windows'] for r in rows if r['batch']=='old')}  "
          f"new={sum(r['n_windows'] for r in rows if r['batch']=='new')}")

    # groups
    by_title = defaultdict(list)
    by_sku = defaultdict(list)
    by_series = defaultdict(list)
    for r in rows:
        nt = norm_title(r["title"])
        if len(nt) >= 8:
            by_title[nt].append(r["pecha_id"])
        code = r["sku"] or r["source_id"]
        if code:
            by_sku[code].append(r["pecha_id"])
        ser = (r["series"] or "").strip()
        if ser:
            by_series[ser].append(r["pecha_id"])

    print("\nDuplicate titles (norm, ≥2 pechas):")
    # Titles shared by many pechas are publisher imprints, not one work
    # (e.g. 20 P-volumes all titled བློ་གྲོས་མཐའ་ཡས་པའི་མཛོད།).
    GENERIC_TITLE_N = 5
    dup_title_all = {k: v for k, v in by_title.items() if len(set(v)) > 1}
    generic_titles = {k: v for k, v in dup_title_all.items() if len(set(v)) >= GENERIC_TITLE_N}
    dup_title = {k: v for k, v in dup_title_all.items() if len(set(v)) < GENERIC_TITLE_N}
    for k, v in sorted(dup_title_all.items(), key=lambda x: -len(x[1])):
        tag = " GENERIC-not-linked" if k in generic_titles else ""
        print(f"  n={len(v)}{tag}  {k[:80]!r}  {v}")
    print(f"  ({len(dup_title)} must-link title groups, {len(generic_titles)} generic skipped)")
    print("Shared sku/source_id ≥2:")
    dup_sku = {k: v for k, v in by_sku.items() if len(set(v)) > 1}
    for k, v in dup_sku.items():
        print(f"  {k}: {v}")
    print("Shared series/collection ≥2:")
    dup_ser = {k: v for k, v in by_series.items() if len(set(v)) > 1}
    for k, v in sorted(dup_ser.items(), key=lambda x: -len(x[1]))[:30]:
        print(f"  n={len(v)}  {k[:70]!r}")
    print(f"  ({len(dup_ser)} series groups)")

    # suffix -1/-2 in title or id
    print("IDs with volume-like suffixes in title:")
    vol_re = re.compile(r"(?:stod|smad|po[dt]i|vol|volume|དེབ|stod cha|smad cha|\-1|\-2)", re.I)
    for r in rows:
        if vol_re.search(r["title"] or "") or re.search(r"-\d+$", r["pecha_id"]):
            print(f"  {r['pecha_id']}  {r['title'][:60]!r}")

    print("\nOUTLIERS:")
    by_w = sorted(rows, key=lambda r: -r["n_windows"])
    print("  top n_windows:", [(r["pecha_id"], r["n_windows"], f"{r['density']:.3f}") for r in by_w[:8]])
    by_d = sorted(rows, key=lambda r: -r["density"])
    print("  top density:", [(r["pecha_id"], f"{r['density']:.3f}", r["n_windows"]) for r in by_d[:8]])
    by_s = sorted(rows, key=lambda r: -r["n_spans"])
    print("  top n_spans:", [(r["pecha_id"], r["n_spans"], r["n_windows"]) for r in by_s[:8]])

    # Build must-link groups: union-find on title dups + sku dups
    parent = {r["pecha_id"]: r["pecha_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for v in dup_title.values():
        for p in v[1:]:
            union(v[0], p)
    for v in dup_sku.values():
        for p in v[1:]:
            union(v[0], p)
    # Do NOT union entire series - RDI Study Series etc. would glue dozens of unrelated books.
    # Only series groups of size 2 with similar titles already caught.

    group_members = defaultdict(list)
    for r in rows:
        group_members[find(r["pecha_id"])].append(r["pecha_id"])
    groups = [{"group_id": gid, "pecha_ids": ids} for gid, ids in group_members.items()]
    n_multi = sum(1 for g in groups if len(g["pecha_ids"]) > 1)
    print(f"\nMust-link groups: {len(groups)} total, {n_multi} with >1 pecha")
    for g in groups:
        if len(g["pecha_ids"]) > 1:
            print(f"  {g['group_id']}: {g['pecha_ids']}")

    print("\nRECOMMENDATION:")
    print("  Stratify on: n_windows (primary), density, batch, pct_short.")
    print("  Redundant: tsawa_chars ≈ density * doc_chars; n_spans correlates with")
    print("  n_windows/doc size; span_len_median inversely with pct_short.")
    print("  Current split WAS coverage-quartile by DOCUMENT, not unstratified,")
    print("  but not window-weighted - that is why 4.70/4.17/5.53 remains.")
    print("  Series metadata is too coarse (e.g. whole RDI collection) - do not")
    print("  force whole series into one split. Must-link = same title or same sku.")

    # PHASE 3
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
    print(f"\n=== PHASE 3: SPLIT ===")
    print(f"global density={targets['density']:.5f} pct_short={targets['pct_short']:.4f} "
          f"old_win_share={targets['batch_old_win']:.4f} windows={tot_w}")

    if args.n_folds > 1:
        assignment = kfold_assign(rows, groups, args.n_folds, seed=42)
        gid_of = {}
        for g in groups:
            for p in g["pecha_ids"]:
                gid_of[p] = g["group_id"]
        with OUT_SPLIT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "group_id"])
            w.writeheader()
            for r in rows:
                w.writerow({"pecha_id": r["pecha_id"], "split": assignment[r["pecha_id"]], "group_id": gid_of[r["pecha_id"]]})
        print(f"k-fold k={args.n_folds} wrote {OUT_SPLIT}")
        return 0

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    best = None
    for seed in seeds:
        asg, sc = greedy_assign(rows, groups, seed, weights, targets)
        st = split_stats(rows, asg)
        if st["val"]["n_docs"] < 15 or st["test"]["n_docs"] < 15:
            continue
        dens = [st[s]["density"] for s in ("train", "val", "test")]
        gap = max(dens) - min(dens)
        rec = (sc, gap, seed, asg, st)
        if best is None or rec[0] < best[0]:
            best = rec
        print(f"  seed {seed:4d}  obj={sc:.6f}  dens gap={gap:.4f}  "
              f"docs {st['train']['n_docs']}/{st['val']['n_docs']}/{st['test']['n_docs']}  "
              f"win {st['train']['n_windows']}/{st['val']['n_windows']}/{st['test']['n_windows']}")

    if best is None:
        raise SystemExit("no valid assignment")
    sc, gap, seed, assignment, st = best
    print(f"KEEP seed={seed} obj={sc:.6f} dens_gap={gap:.4f}")

    gid_of = {}
    for g in groups:
        for p in g["pecha_ids"]:
            gid_of[p] = g["group_id"]

    with OUT_SPLIT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "group_id"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "pecha_id": r["pecha_id"],
                    "split": assignment[r["pecha_id"]],
                    "group_id": gid_of[r["pecha_id"]],
                }
            )

    # current stats
    cur = {r["pecha_id"]: ("val" if r["current_split"] == "val" else r["current_split"]) for r in rows}
    cur_st = split_stats(rows, cur)

    def fmt_table(st):
        lines = []
        lines.append("| split | docs | windows | win% | density | pct_short | old_win% | new/old docs |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for sp in ("train", "val", "test"):
            s = st[sp]
            lines.append(
                f"| {sp} | {s['n_docs']} | {s['n_windows']} | {100*s['win_share']:.1f}% | "
                f"{100*s['density']:.3f}% | {100*s['pct_short']:.2f}% | "
                f"{100*s['batch_old_win']:.1f}% | {s['batch_new_docs']}/{s['n_old_docs']} |"
            )
        return "\n".join(lines)

    # verify groups
    leaks = []
    for g in groups:
        splits = {assignment[p] for p in g["pecha_ids"]}
        if len(splits) > 1:
            leaks.append((g["pecha_ids"], splits))
    dups = [pid for pid, n in Counter(assignment).items() if False]
    two = [p for p, n in Counter(list(assignment)).items()]
    assert len(assignment) == len(rows)
    assert len(set(assignment)) == len(rows)

    report = []
    report.append("# Stratified split report")
    report.append("")
    report.append("## Note on the current split")
    report.append("")
    report.append("The existing 172/20/20 split is **not** pure random: `build_tsawa_dataset.py`")
    report.append("stratifies by Phase-2 `coverage_pct` quartiles **at document count**, seed 42.")
    report.append("It does **not** balance window mass or span-length mix, which is why token")
    report.append("positive density still landed at 4.70 / 4.17 / 5.53.")
    report.append("")
    report.append("## Metadata fields")
    report.append("")
    report.append("- **No `book_code` field** on these `meta.yml` files.")
    report.append("- Identifiers: `source_metadata.id`, `source_metadata.sku` / `web_metadata.sku`.")
    report.append("- Title: `source_metadata.title` (all 212).")
    report.append("- Publisher: mostly old-batch `web_metadata.pusblisher` (YAML typo) or `org`.")
    report.append("- Series: rare `source_metadata.series`; old batch often `web_metadata.collection`.")
    report.append("")
    report.append("## Span-length cutoffs")
    report.append("")
    report.append("Kept short<30 / medium 30-200 / long>200. Finer histogram is in the script log.")
    report.append("p50 sits in medium; short is a real left cluster (~31%). No silent cutoff change.")
    report.append("")
    report.append(f"## Density terciles")
    report.append("")
    report.append(f"sparse ≤ {t1:.4f}; moderate ≤ {t2:.4f}; else dense.")
    report.append("")
    report.append(f"## Chosen assignment (greedy, seed {seed})")
    report.append("")
    report.append("Must-link: same normalized title or same sku/id. Series collections NOT glued.")
    report.append("")
    report.append("### Current (char-density, window counts)")
    report.append("")
    report.append(fmt_table(cur_st))
    report.append("")
    report.append("### Proposed")
    report.append("")
    report.append(fmt_table(st))
    report.append("")
    report.append("Char density is tsawa_chars/doc_chars (not token pos%). Token pos% needs a rebuild.")
    report.append("")
    report.append("### Deviations from global")
    report.append("")
    report.append(f"Global char density {100*targets['density']:.3f}%; pct_short {100*targets['pct_short']:.2f}%; "
                  f"old window share {100*targets['batch_old_win']:.1f}%.")
    report.append("")
    for sp in ("train", "val", "test"):
        s = st[sp]
        report.append(
            f"- **{sp}**: Δdensity {100*(s['density']-targets['density']):+.3f} pp, "
            f"Δpct_short {100*(s['pct_short']-targets['pct_short']):+.2f} pp, "
            f"Δwin {100*(s['win_share']-[0.8,0.1,0.1][('train','val','test').index(sp)]):+.2f} pp, "
            f"Δold_win {100*(s['batch_old_win']-targets['batch_old_win']):+.2f} pp"
        )
    report.append("")
    report.append("## Leakage checks")
    report.append("")
    report.append(f"- Unique pechas in assignment: {len(assignment)} (expect {len(rows)}).")
    report.append(f"- Group split leaks: {leaks if leaks else 'none'}.")
    report.append("")
    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {OUT_SPLIT}")
    print(f"wrote {OUT_REPORT}")

    print("\n=== PHASE 4: CURRENT vs PROPOSED ===")
    print("CURRENT")
    print(fmt_table(cur_st))
    print("PROPOSED")
    print(fmt_table(st))
    print(f"no pecha in two splits: {len(assignment)==len(rows)}")
    print(f"no group split across splits: {not leaks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

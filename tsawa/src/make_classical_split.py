#!/usr/bin/env python3
"""Whole-commentary holdout for the classical (interleaved_v3) data.

Per root text: commentaries ranked by number of labelled spans (most first);
rank 2 -> classical test, rank 3 -> classical val, all others -> train.
(Deterministic: no random choice; rank 1 stays in train so the largest commentary
is always trained on, rank 2/3 are large enough to give a stable eval.)
Splits at whole-commentary level only. Writes classical_v3/classical_split.csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path.home() / "tsawa_commentary_data" / "interleaved_v3"
OUT = ROOT / "tsawa/data/processed/classical_v3"

man = {r["doc_id"]: r for r in csv.DictReader((SRC / "manifest.csv").open(encoding="utf-8"))}
n = defaultdict(int)
for r in csv.DictReader((OUT / "tsawa_spans_classical_v3.csv").open(encoding="utf-8")):
    n[r["pecha_id"]] += 1
by = defaultdict(list)
for d, m in man.items():
    by[m["repo"]].append(d)
rows = []
for repo in sorted(by):
    docs = sorted(by[repo], key=lambda d: (-n[d], d))
    for rank, d in enumerate(docs, 1):
        test_only = "--test-only" in sys.argv   # rank 3 goes to train, no classical validation
        split = "test" if rank == 2 else "validation" if (rank == 3 and not test_only) else "train"
        rows.append({"pecha_id": d, "root_text_repo": repo, "split": split, "rank_by_spans": rank,
                     "n_spans": n[d], "chars": man[d]["chars"], "commentary": man[d]["commentary"]})
with (OUT / ("classical_split_test_only.csv" if "--test-only" in sys.argv else "classical_split.csv")).open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['root_text_repo'][:34]:36}{r['pecha_id']}  rank {r['rank_by_spans']}  spans {r['n_spans']:>5}  -> {r['split']}")
from collections import Counter
print(Counter(r["split"] for r in rows))

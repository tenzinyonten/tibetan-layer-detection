#!/usr/bin/env python3
"""Score Yigchung models on the benchmark (HF v2 test, 77 books) against the original gold ("before")
and the corrected gold ("after", build_benchmark_after.py).

Usage: python score_benchmark.py NAME=predictions.json [NAME2=predictions2.json ...]
Predictions come from common/predict_layer_spans.py (token spans); they are mapped to characters with the
mmBERT tokenizer and matched one-to-one on IoU >= 0.5 (same rule as the training metric).
Writes benchmark_scores_by_book.csv and benchmark_scores.md next to the corrected benchmark.

Both systems (the general model and the genre model are scored on the same 77 books). NAME is a free label for the
model; the predictions file needs a "test" split. Books missing from the benchmark are skipped.
Groups in the tables: the 5 annotator-reviewed BDRC books, the Lotsawa books (47), and the other reviewed books (25).
Reads:  <local>/hub/test.parquet (before), <local>/benchmark_after/test.parquet (after, from build_benchmark_after.py),
        the catalog (genre per book) and the predictions files.
Writes: <local>/benchmark_after/benchmark_scores_by_book.csv and benchmark_scores.md."""
import paths as P
import csv
import json
import sys
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer

BEFORE = P.HUB / 'test.parquet'
AFTER = P.BENCHMARK_AFTER / 'test.parquet'
OUT = str(P.BENCHMARK_AFTER) + '/'
CAT = {r['book_id']: r['category'] for r in
       csv.DictReader(open(P.CATALOG, encoding='utf-8-sig'))}
REVIEWED = {'B-UT1ER1204_0001', 'B-UT1ER1213_0001', 'B-UT1ER1571_0001', 'B-UT1ER2092_0001', 'B-UT1ER2253_0001'}

tok = AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-base', use_fast=True)
tok.model_max_length = int(1e12)
before = pd.read_parquet(BEFORE).set_index('book_id')
after = pd.read_parquet(AFTER).set_index('book_id')


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return 0.0 if hi < lo else (hi - lo + 1) / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - (hi - lo + 1))


def matches(gold, pred):
    cand = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)
                   if p[0] <= g[1] and p[1] >= g[0]), reverse=True)
    ug, up, n = set(), set(), 0
    for v, gi, pi in cand:
        if v >= 0.5 and gi not in ug and pi not in up:
            ug.add(gi); up.add(pi); n += 1
    return n


def group(b, src):
    if b in REVIEWED:
        return 'BDRC, annotator-reviewed (5)'
    return 'Lotsawa (47)' if src == 'lotsawa' else 'other reviewed books (25)'


offs_cache = {}
rows = []
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    J = json.load(open(path))
    docs = J['splits']['test']['docs']
    for b, D in docs.items():
        if b not in before.index:
            continue
        t = before.loc[b, 'text']
        if b not in offs_cache:
            offs_cache[b] = tok(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        offs = offs_cache[b]
        assert len(offs) == D['n_tokens'], (name, b)
        pred = []
        for s, e, _ in D['pred']:
            a, z = offs[s][0], offs[e][1] - 1
            while a < z and t[a] in ' \n': a += 1
            while z > a and t[z] in ' \n': z -= 1
            pred.append((a, z))
        for gname, G in [('before', before), ('after', after)]:
            gold = [(int(x['start']), int(x['end'])) for x in G.loc[b, 'spans']]
            rows.append({'model': name, 'gold': gname, 'book_id': b, 'source': before.loc[b, 'source'],
                         'group': group(b, before.loc[b, 'source']), 'category': CAT.get(b, 'lotsawa' if before.loc[b, 'source'] == 'lotsawa' else ''),
                         'n_gold': len(gold), 'n_pred': len(pred), 'matched': matches(gold, pred)})
R = pd.DataFrame(rows)
R.to_csv(OUT + 'benchmark_scores_by_book.csv', index=False, encoding='utf-8-sig')


def prf(d):
    g, p, m = d.n_gold.sum(), d.n_pred.sum(), d.matched.sum()
    P = m / p if p else 0.0
    Rc = m / g if g else 0.0
    F = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
    return P, Rc, F, int(g)


lines = ['# Yigchung benchmark (BDRC/yigchung_detection_v2 test, 77 books)', '',
         'IoU@0.5 span matching. "before" = original gold; "after" = corrected gold (annotator review for the 5 BDRC books, '
         'annotator patterns + head-mark rule elsewhere).', '']
for title, key in [('All books', None), ('By group', 'group'), ('By genre', 'category')]:
    lines += [f'## {title}', '', '| ' + ('' if key is None else f'{key} | ') + 'model | gold | spans | precision | recall | F1 |',
              '|' + ('' if key is None else '---|') + '---|---|---|---|---|---|']
    keys = [None] if key is None else sorted(R[key].unique(), key=str)
    for k in keys:
        for (m, gname), d in R.groupby(['model', 'gold'], sort=False):
            if k is not None:
                d = d[d[key] == k]
            if d.empty or d.n_gold.sum() == 0:
                continue
            P, Rc, F, g = prf(d)
            lines.append('| ' + ('' if key is None else f'{k} | ') + f'{m} | {gname} | {g} | {P:.3f} | {Rc:.3f} | {F:.3f} |')
    lines.append('')
md = '\n'.join(lines)
open(OUT + 'benchmark_scores.md', 'w', encoding='utf-8').write(md)
print(md)
print('books scored per model:', R[R.gold == 'before'].groupby('model').size().to_dict())

#!/usr/bin/env python3
"""Propose a book-level train / validation / test split for the genre Yigchung set.

Method
  1. Quality filter: leave out under-annotated books (< MIN_SPANS spans in a book longer than MIN_CHARS chars).
  2. Near-duplicate groups: books sharing a 50-char shingle overlap above DUP_JACCARD go to the same split.
  3. Per genre, search all assignments with 1..MAXK groups in validation and in test. Cost = how far the span
     (and, weighted less, character) shares fall OUTSIDE the acceptable band BAND (default 5-20% for validation
     and for test) + MOVE_COST per book that changes split. Books in ANCHOR keep their split.
  4. Tie-breaks: test books with better labels (annotator-reviewed, reviewed OPF, not borderline), then books the
     previous model (karma689/mmbert-yigchung-v2) never trained on.
Writes a proposal CSV; the catalog is not changed.
"""
import paths as P
import csv
import itertools
import random
from collections import defaultdict

import pandas as pd

G = str(P.GENRE) + '/'
HUB = str(P.HUB) + '/'
OUT = P.SPLIT_PROPOSAL
TARGET = {'train': 0.80, 'validation': 0.10, 'test': 0.10}
MIN_SPANS, MIN_CHARS, DUP_JACCARD, MAXK = 5, 20000, 0.2, 3
BAND = (0.05, 0.20)          # acceptable share of a genre's spans in validation and in test
MOVE_COST = 0.02             # cost per book that changes split (keeps the split stable)
ANCHOR = {'B-UT1ER2253_0001': 'test',   # best gold in the set (annotator-reviewed, 985 spans)
          'I849C345E': 'train',         # bracket-note style the model needs to learn (sadhana swap)
          'P000030': 'test'}            # large liturgy test book, being annotator-reviewed (keeps the test comparable)
REVIEWED = {'B-UT1ER2253_0001', 'B-UT1ER1130_0001', 'B-UT1ER1210_0001', 'B-UT1ER1222_0001', 'B-UT1ER1596_0001'}

books = pd.concat([pd.read_parquet(f'{G}without_lotsawa/{s}.parquet').assign(cur=s) for s in ['train', 'validation', 'test']])
books['n_spans'] = books.spans.map(len)
books['chars'] = books.text.str.len()
old_train = set(pd.read_parquet(HUB + 'train.parquet', columns=['book_id']).book_id) | \
            set(pd.read_parquet(HUB + 'validation.parquet', columns=['book_id']).book_id)
books['old_model_saw'] = books.book_id.isin(old_train)

# 1. quality filter
books['excluded'] = (books.n_spans < MIN_SPANS) & (books.chars > MIN_CHARS)

# 2. near-duplicate groups (50-char shingles from 20 evenly spaced 2,000-char windows)
def shingles(t):
    out = set()
    for k in range(20):
        a = int(len(t) * k / 20)
        w = ''.join(t[a:a + 2000].split())
        out |= {w[i:i + 50] for i in range(0, max(0, len(w) - 50), 25)}
    return out
sh = {r.book_id: shingles(r.text) for r in books.itertuples()}
parent = {b: b for b in sh}
def find(b):
    while parent[b] != b:
        b = parent[b]
    return b
ids = list(sh)
dups = []
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        j = len(sh[a] & sh[b]) / max(1, len(sh[a] | sh[b]))
        if j > DUP_JACCARD:
            parent[find(a)] = find(b); dups.append((a, b, round(j, 3)))
books['group'] = books.book_id.map(find)


def quality(r):   # higher = better test book
    return (r.book_id in REVIEWED) * 2 + (r.source == 'reviewed_opf' or r.source == 'bdrc') * 0.5 - r.borderline * 1


def best_split(g):
    import numpy as np
    grp = g.groupby('group').agg(spans=('n_spans', 'sum'), chars=('chars', 'sum'), n=('book_id', 'size'),
                                 saw=('old_model_saw', 'sum')).reset_index()
    members = g.groupby('group').book_id.apply(list).to_dict()
    cur = g.groupby('group').cur.first().to_dict()
    q = {gp: sum(quality(g[g.book_id == b].iloc[0]) for b in members[gp]) for gp in grp.group}
    fixed = {gp: ANCHOR[b] for gp in grp.group for b in members[gp] if b in ANCHOR}
    S, C = grp.spans.sum(), grp.chars.sum()
    sp, ch = grp.spans.to_numpy() / S, grp.chars.to_numpy() / C
    gl = list(grp.group)
    def out(x):
        return max(0.0, BAND[0] - x) + max(0.0, x - BAND[1])
    best = None
    idx = list(range(len(gl)))
    for kt in range(1, min(MAXK, len(idx) - 2) + 1):
        for T in itertools.combinations(idx, kt):
            if any(fixed.get(gl[i], 'test') != 'test' for i in T) or any(fixed.get(gl[i]) == 'test' and i not in T for i in idx):
                continue
            rest = [i for i in idx if i not in T]
            for kv in range(1, min(MAXK, len(rest) - 1) + 1):
                for V in itertools.combinations(rest, kv):
                    if any(fixed.get(gl[i], 'validation') != 'validation' for i in V) or any(fixed.get(gl[i]) == 'validation' and i not in V for i in idx):
                        continue
                    lab = ['train'] * len(idx)
                    for i in T: lab[i] = 'test'
                    for i in V: lab[i] = 'validation'
                    ts, vs = sp[list(T)].sum(), sp[list(V)].sum()
                    tc, vc = ch[list(T)].sum(), ch[list(V)].sum()
                    moves = sum(grp.n[i] for i in idx if lab[i] != cur[gl[i]])
                    cost = out(ts) + out(vs) + 0.25 * (out(tc) + out(vc)) + MOVE_COST * moves \
                        + 0.2 * abs(ts - TARGET['test']) * 0.1 + 0.2 * abs(vs - TARGET['validation']) * 0.1 \
                        - 0.002 * sum(q[gl[i]] for i in T) - 0.001 * sum(grp.n[i] - grp.saw[i] for i in T)
                    if best is None or cost < best[0]:
                        best = (cost, lab)
    lab = dict(zip(gl, best[1]))
    return {b: lab[gp] for b, gp in zip(g.book_id, g.group)}


assign = {}
for cat, g in books[~books.excluded].groupby('category'):
    assign.update(best_split(g))
books['proposed'] = books.book_id.map(assign).fillna('excluded')
books[['book_id', 'category', 'source', 'n_spans', 'chars', 'borderline', 'old_model_saw', 'group', 'cur', 'proposed']] \
    .sort_values(['category', 'proposed', 'n_spans'], ascending=[True, True, False]).to_csv(OUT, index=False, encoding='utf-8-sig')

print('excluded (under-annotated):', books[books.excluded][['book_id', 'category', 'n_spans', 'chars']].values.tolist())
print('near-duplicate pairs:', dups or 'none')
print('\nspan share per genre  (current -> proposed)')
for cat, g in books.groupby('category'):
    def share(col):
        h = g[g[col].isin(['train', 'validation', 'test'])]
        s = h.groupby(col).n_spans.sum(); t = s.sum()
        return '  '.join(f"{k[:5]} {s.get(k, 0):5} ({100 * s.get(k, 0) / t:3.0f}%)" for k in ['train', 'validation', 'test'])
    print(f'{cat:8} current : {share("cur")}')
    print(f'{"":8} proposed: {share("proposed")}')
print('\nbooks that move:')
mv = books[books.cur != books.proposed][['book_id', 'category', 'n_spans', 'cur', 'proposed', 'old_model_saw']]
print(mv.to_string(index=False))
print('\nwrote', OUT)

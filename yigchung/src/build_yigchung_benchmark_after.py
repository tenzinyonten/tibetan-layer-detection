#!/usr/bin/env python3
"""Build the corrected ("after") Yigchung benchmark from the original HF v2 test split ("before").

  1. The 5 benchmark books the annotator reviewed (THoesam check) take the annotator's spans.
     Annotator offsets are on the text with ༼༽{} brackets; they are mapped back to the HF text.
  2. The other books get the patterns learned from the annotator's 19 reviewed books:
       - trailing unmatched ')' removed                        (annotator convention, ~130 edits)
       - trailing syllable that starts a new phrase removed    ("…བསམ། ཨོཾ" -> "…བསམ།"; 86% agreement on 101 cases)
  3. Every book: number-only spans removed (user rule, same regex as numbers_cleanup/clean_number_spans.py)
     and head marks (༄༅ ༈ ...) trimmed / split (user rule 2026-11-12, same as build_yigchung_genre_datasets.py).
Writes <OUT>/test.parquet (HF schema), <OUT>/changes.csv (one row per changed span), <OUT>/summary_by_book.csv.
Source data is not modified.

General system (its benchmark). Usage: python yigchung/src/build_yigchung_benchmark_after.py     (no arguments; run once,
or when the annotator files change). Head marks are the HM set in the script; the trailing-syllable rule and the
number-only regex are in the code below.
Reads:  <local>/hub/test.parquet, <local>/annotator_review/yaml/<book>-yigchung*.yaml and
        <local>/annotator_review/zips/<book>.zip (the review OPF whose offsets the yaml uses), <local>/annotator_spans/.
Writes: <local>/benchmark_after/{test.parquet,changes.csv,summary_by_book.csv} (results/benchmark/ keeps copies; changes.csv is benchmark_changes.csv there).
The annotator files are not public; ask the layer owner."""
import paths as P
import bisect
import collections
import glob
import hashlib
import os
import re
import sys
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HUB = P.HUB / 'test.parquet'
ANN_DIR = str(P.ANNOTATOR_YAML) + '/'
ZIP_DIR = str(P.ANNOTATOR_ZIPS) + '/'
OUT = str(P.BENCHMARK_AFTER) + '/'
os.makedirs(OUT, exist_ok=True)

BR = '༼༽{}'
HM = '༁༄༅༆༇༈༉༊࿐࿑࿓࿔'
RUN = re.compile(f'[{HM}][{HM}ྃ༔།༎༑ ་\n]*')
LET = re.compile(r'[ཀ-ྼༀ-༃]')
NUMONLY = re.compile(r"^[\s()（）\[\]༼༽.,:;།]*[0-9༠-༩]+[\s()（）\[\]༼༽.,:;།]*$")

log = []


def note(book, a, e, rule, new, t):
    log.append({'book_id': book, 'rule': rule, 'old_start': a, 'old_end': e,
                'new': ' '.join(f'{p}-{q}' for p, q in new) if new else '(dropped)',
                'old_text': t[a:e + 1].replace('\n', '⏎')[:300],
                'new_text': ' | '.join(t[p:q + 1].replace('\n', '⏎')[:150] for p, q in new)})


def annotator_spans(book, text):
    ay = glob.glob(f'{ANN_DIR}{book}-yigchung*.yaml')
    if not ay:
        return None
    with zipfile.ZipFile(f'{ZIP_DIR}{book}.zip') as z:
        base = z.read(next(n for n in z.namelist() if n.endswith('base/v001.txt'))).decode('utf-8')
    keep = [i for i, c in enumerate(base) if c not in BR]
    assert ''.join(base[i] for i in keep) == text, f'{book}: annotator text differs from HF text'
    out = []
    for k, v in (yaml.safe_load(open(ay[0], encoding='utf-8'))['annotations'] or {}).items():
        a, e = v['span']['start'], v['span']['end']
        j0, j1 = bisect.bisect_left(keep, a), bisect.bisect_right(keep, e) - 1
        if j1 >= j0:
            out.append({'id': k, 'start': j0, 'end': j1})
    return sorted(out, key=lambda s: s['start'])


def trim_ws(t, a, e):
    while a < e and t[a] in ' \n':
        a += 1
    while e > a and t[e] in ' \n':
        e -= 1
    return a, e


def rule_paren(book, t, s):
    a, e = s['start'], s['end']
    x = t[a:e + 1]
    if x.rstrip().endswith(')') and '(' not in x:
        e2 = a + len(x.rstrip()) - 2
        a2, e2 = trim_ws(t, a, e2)
        note(book, a, e, 'trailing ) removed', [(a2, e2)], t)
        return [{**s, 'end': e2, 'start': a2}]
    return [s]


def rule_new_phrase(book, t, s):
    a, e = s['start'], s['end']
    if e + 1 >= len(t) or not LET.match(t[e]) or not (LET.match(t[e + 1]) or t[e + 1] == '་'):
        return [s]
    k = e
    while k > a and t[k] not in '་། \n༔':
        k -= 1
    if k <= a or t[k] not in '། \n༔':          # only when the extra syllable follows a shad / space (new phrase)
        return [s]
    a2, e2 = trim_ws(t, a, k)
    if e2 - a2 < 1:
        return [s]
    note(book, a, e, 'extra syllable of next phrase removed', [(a2, e2)], t)
    return [{**s, 'start': a2, 'end': e2}]


def rule_numbers(book, t, s):
    if NUMONLY.match(t[s['start']:s['end'] + 1]):
        note(book, s['start'], s['end'], 'number-only span removed', [], t)
        return []
    return [s]


def rule_headmarks(book, t, s):
    a, e = s['start'], s['end']
    x = t[a:e + 1]
    if not any(c in HM for c in x):
        return [s]
    cuts = [0]
    for m in RUN.finditer(x):
        prev = x[:m.start()].rstrip(' \n')
        if m.start() == 0 or not x[m.end():].strip() or (prev and prev[-1] in '།༎༑༔གཀ'):
            cuts += [m.start(), m.end()]
    cuts.append(len(x))
    pieces = []
    for i in range(0, len(cuts), 2):
        p, q = a + cuts[i], a + cuts[i + 1] - 1
        while p <= q and t[p] in ' \n་':
            p += 1
        while q >= p and t[q] in ' \n':
            q -= 1
        if q - p + 1 >= 3 and t[p:q + 1].strip('།༎༑༔ \n་'):
            pieces.append((p, q))
    if pieces == [(a, e)]:
        return [s]
    note(book, a, e, 'head mark trimmed/split', pieces, t)
    return [{'id': s['id'] if j == 0 else hashlib.md5(f"{s['id']}_{j}".encode()).hexdigest(), 'start': p, 'end': q}
            for j, (p, q) in enumerate(pieces)]


def apply(rule, book, t, spans):
    return [n for s in spans for n in rule(book, t, s)]


hf = pd.read_parquet(HUB)
rows, summ = [], []
for r in hf.itertuples():
    t = r.text
    before = [{'id': s['id'], 'start': int(s['start']), 'end': int(s['end'])} for s in r.spans]
    ann = annotator_spans(r.book_id, t)
    spans = ann if ann is not None else before
    if ann is None:
        spans = apply(rule_paren, r.book_id, t, spans)
        spans = apply(rule_new_phrase, r.book_id, t, spans)
    spans = apply(rule_numbers, r.book_id, t, spans)
    spans = apply(rule_headmarks, r.book_id, t, spans)
    spans = sorted(spans, key=lambda s: s['start'])
    for x, y in zip(spans, spans[1:]):
        assert y['start'] > x['end'], (r.book_id, x, y)
    rows.append({'book_id': r.book_id, 'split': 'test', 'source': r.source, 'text': t, 'spans': spans})
    summ.append({'book_id': r.book_id, 'source': r.source, 'annotator_reviewed': ann is not None,
                 'spans_before': len(before), 'spans_after': len(spans),
                 'changed_by_rules': sum(1 for l in log if l['book_id'] == r.book_id)})

SCHEMA = pa.schema([('book_id', pa.string()), ('split', pa.string()), ('source', pa.string()), ('text', pa.string()),
                    ('spans', pa.list_(pa.struct([('id', pa.string()), ('start', pa.int32()), ('end', pa.int32())])))])
pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT + 'test.parquet')
pd.DataFrame(log).to_csv(OUT + 'changes.csv', index=False, encoding='utf-8-sig')
S = pd.DataFrame(summ)
S.to_csv(OUT + 'summary_by_book.csv', index=False, encoding='utf-8-sig')
print('rules applied:', dict(collections.Counter(l['rule'] for l in log)))
print(f"spans before {S.spans_before.sum()} -> after {S.spans_after.sum()}")
print(S[S.annotator_reviewed | (S.changed_by_rules > 0)].to_string(index=False))
print('wrote', OUT)

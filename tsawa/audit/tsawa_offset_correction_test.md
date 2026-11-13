# Tsawa test labels: offset correction test

Exploratory. Read-only. Script: `tsawa_offset_correction_test.py`; corrected spans (variant A) for the suspect books: `tsawa_offset_corrected_spans.csv`.

## Method

- Suspect books: P000027, P000083, P000144, P000242, P000269. The first four are the test books whose Sabche offsets were realigned; `P000269` is added because its tsawa spans fit the text as badly (start ok 31%, below). Control: the other 16 verse tsawa test books (`IF3ACC3E1` left out: 747 short word-fragment spans, not verse).
- In clean books a tsawa span starts at a line start and ends right before a shad or newline. The correction picks, per span, the shift in [-200, 200] with the best line-start and line-end fit plus verse metre (details in the script docstring). Span length is unchanged (variant A). Variant B also moves each start to the nearest line start within 40 characters.
- Scoring: the saved window-level v6 predictions (Viterbi, break penalty 4.0, token space), IoU >= 0.5, one-to-one matching, as in the reported test F1. Gold per window is rebuilt from character spans with the dataset builder's own `label_tokens` and the mmBERT tokenizer, and the same rebuild is used for original and corrected spans.

## Do the labels fit the text? Line-start / line-end fit before and after the shift

| pecha_id | group | spans | moved | moved_pct | start_ok_before | start_ok_after | end_ok_before | end_ok_after | median_shift_of_moved |
|---|---|---|---|---|---|---|---|---|---|
| P000027 | suspect | 51 | 39 | 76.5 | 0.29 | 0.71 | 0.29 | 0.94 | 30 |
| P000083 | suspect | 190 | 145 | 76.3 | 0.25 | 0.43 | 0.23 | 0.93 | 56 |
| P000144 | suspect | 113 | 90 | 79.6 | 0.23 | 0.56 | 0.25 | 1.0 | 69 |
| P000242 | suspect | 171 | 165 | 96.5 | 0.04 | 0.42 | 0.2 | 0.87 | 82 |
| P000269 | suspect | 39 | 28 | 71.8 | 0.31 | 0.69 | 0.36 | 0.92 | 81 |
| I0FCFA88F | control | 144 | 2 | 1.4 | 1.0 | 1.0 | 0.96 | 0.97 | 42 |
| I319DAFF7 | control | 17 | 2 | 11.8 | 0.88 | 0.88 | 0.88 | 1.0 | 40 |
| I3F4A91F5 | control | 58 | 2 | 3.4 | 0.97 | 0.97 | 0.97 | 1.0 | 84 |
| I575514A8 | control | 33 | 0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| I9AEEF96A | control | 19 | 2 | 10.5 | 0.89 | 0.89 | 0.89 | 1.0 | -34 |
| I9B6A4525 | control | 200 | 1 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | -154 |
| I9D9C7AC9 | control | 43 | 3 | 7.0 | 0.95 | 0.95 | 0.95 | 1.0 | 162 |
| IC05A6BE0 | control | 166 | 0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| IC6F06BCD | control | 14 | 2 | 14.3 | 0.86 | 0.86 | 0.86 | 1.0 | 145 |
| ICDC84458 | control | 61 | 2 | 3.3 | 1.0 | 1.0 | 1.0 | 1.0 | 120 |
| IE5895799 | control | 29 | 2 | 6.9 | 0.93 | 0.93 | 0.93 | 1.0 | 147 |
| P000013 | control | 89 | 1 | 1.1 | 0.99 | 0.99 | 1.0 | 1.0 | 177 |
| P000056 | control | 108 | 0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| P000067 | control | 207 | 1 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | -30 |
| P000118 | control | 52 | 0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| P000164 | control | 183 | 1 | 0.5 | 1.0 | 0.99 | 0.87 | 0.88 | -167 |


Most common shifts per suspect book (shift: spans): `P000027` {30: 15, 0: 12, 31: 5, 2: 2}; `P000083` {0: 45, 57: 31, 56: 19, 58: 18}; `P000144` {0: 23, 99: 15, 100: 9, 98: 6}; `P000242` {113: 21, 112: 19, 111: 14, 82: 13}; `P000269` {0: 11, 37: 3, 7: 2, 106: 2}.

## Test F1

- Windows where the rebuilt gold (original spans) differs from the saved gold: 0 of 639. Saved gold gives F1 0.5517.
| gold | scope | F1 | precision | recall | tp | pred | gold spans |
|---|---|---|---|---|---|---|---|
| saved window gold (as reported) | all 22 test books | 0.5517 | 0.5281 | 0.5774 | 2459 | 4656 | 4259 |
| rebuilt, original spans | all 22 test books | 0.5517 | 0.5281 | 0.5774 | 2459 | 4656 | 4259 |
| rebuilt, suspect books corrected (A: shift only) | all 22 test books | 0.6242 | 0.5979 | 0.6529 | 2784 | 4656 | 4264 |
| rebuilt, suspect books corrected (B: shift + snap start to line start) | all 22 test books | 0.6259 | 0.5994 | 0.6547 | 2791 | 4656 | 4263 |
| rebuilt, original spans | the 5 suspect books | 0.2981 | 0.2324 | 0.4155 | 376 | 1618 | 905 |
| rebuilt, corrected (A: shift only) | the 5 suspect books | 0.5546 | 0.4333 | 0.7703 | 701 | 1618 | 910 |
| rebuilt, corrected (B: shift + snap start to line start) | the 5 suspect books | 0.5603 | 0.4376 | 0.7789 | 708 | 1618 | 909 |


Line-start / line-end fit of the corrected spans in the suspect books: A: shift only: start ok 0.50, end ok 0.93; B: shift + snap start to line start: start ok 0.92, end ok 0.93.

Per suspect book:

| pecha_id | pred | gold_original | tp_original | F1_original | gold_A | tp_A | F1_A | gold_B | tp_B | F1_B |
|---|---|---|---|---|---|---|---|---|---|---|
| P000027 | 83 | 83 | 59 | 0.711 | 83 | 70 | 0.843 | 83 | 70 | 0.843 |
| P000083 | 411 | 301 | 157 | 0.441 | 302 | 252 | 0.707 | 302 | 257 | 0.721 |
| P000144 | 216 | 178 | 57 | 0.289 | 179 | 126 | 0.638 | 179 | 123 | 0.623 |
| P000242 | 569 | 280 | 63 | 0.148 | 282 | 212 | 0.498 | 282 | 216 | 0.508 |
| P000269 | 339 | 63 | 40 | 0.199 | 64 | 41 | 0.203 | 63 | 42 | 0.209 |

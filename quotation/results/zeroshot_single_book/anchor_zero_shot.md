# Anchor zero-shot — QUOTE spans

**Status:** running, one book at a time
**Prompt:** `docs/prompts/gemini_quote_only_anchors.md`
**Model:** gemini-3.8-flash, temperature 0
**Metric:** IoU@0.5, greedy one-to-one (`match_iou`). A prediction is a hit only if it shares a gold Quotation span at inclusive IoU ≥ 0.5.
**Locator:** `script/check_gemini_quotes.py` — forward cursor, `--max-len 2000`, tsheg-before-shad / `༑` normalization.

Offsets are inclusive: `text[start:end+1]`.

How to add the next book: dump Gemini JSON, run the checker, paste a row into the scoreboard, then append a book section with the resolved span table.

```
python script/check_gemini_quotes.py --book P00xxxx \
  --quotes experiments/gemini_flash_zeroshot/p00xxxx_gemini_anchors.json \
  --out experiments/gemini_flash_zeroshot/p00xxxx_anchor_check.json
```

## Scoreboard

| Book | Gold | Parsed | Located | Fail | P | R | F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P000152 | 91 | 80 | 70 | 10 | 0.486 | 0.374 | 0.422 | `experiments/gemini_flash_zeroshot/p000152_anchor_check.json` |

Fail = not located. Of P000152's 10: 9 `tail_beyond_max_len`, 1 hard miss (span 12, tail copied from span 9 and occurs *before* the head).

---

## P000152

**Date:** 2026-09-04
**Chars:** 994,635
**Dump:** `experiments/gemini_flash_zeroshot/p000152_gemini_anchors.json`

Compared with the earlier 14-quote full-text dump (`p000152_quote_check.json`): P stayed ~0.43, R went 0.066 → 0.374. The dump was truncated; anchors let the model finish. Precision did not jump.

Located 70: 34 with IoU ≥ 0.5, 4 partial (0 < IoU < 0.5), 32 with no gold overlap. Median located length 146, max 1994.

Gold misses (57 of 91) are mostly not “the model skipped a quotation”:

1. ~30 tiny `ཞེས་དང༌།` fragments (median miss 30 chars) — gold splits one citation; the prompt asks for one span.
2. Gold sometimes marks the frame (`…མདོ་ལས།`) as the span; the prompt keeps the frame outside.
3. ~11 real holes: long narrative quotes in the first 800k.
4. 4 merge/split fights (IoU 0.10–0.42).
5. At least one gold error (3-char `ཕྱས`).

### Resolved spans

`iou` is the best overlapping gold span (diagnostic). The F1 line only counts `iou ≥ 0.5`. For `tail_beyond_max_len`, the number in parentheses is the unbounded head-to-tail distance.

| id | start | end | len | iou | status |
|---:|------:|----:|----:|----:|--------|
| 1 | 15455 | 15784 | 330 | 0.997 | located |
| 2 | 31010 | 31129 | 120 | 1.000 | located |
| 3 | 34637 | 34847 | 211 | 0.995 | located |
| 4 | 35582 | 35701 | 120 | 0.992 | located |
| 5 | 36268 | 36512 | 245 | 0.000 | located |
| 6 | 821192 | 821303 | 112 | 1.000 | located |
| 7 | 821868 | 821984 | 117 | 0.000 | located |
| 8 | 822447 | 822561 | 115 | 0.991 | located |
| 9 | — | — | — | — | tail_beyond_max_len (8929) |
| 10 | 831772 | 833765 | 1994 | 0.000 | located |
| 11 | 833905 | 835162 | 1258 | 0.000 | located |
| 12 | — | — | — | — | match_failure |
| 13 | — | — | — | — | tail_beyond_max_len (22992) |
| 14 | — | — | — | — | tail_beyond_max_len (220361) |
| 15 | 864009 | 865910 | 1902 | 0.000 | located |
| 16 | — | — | — | — | tail_beyond_max_len (4762) |
| 17 | 872258 | 872990 | 733 | 0.000 | located |
| 18 | 873047 | 873232 | 186 | 0.841 | located |
| 19 | — | — | — | — | tail_beyond_max_len (5335) |
| 20 | 879535 | 880657 | 1123 | 0.889 | located |
| 21 | — | — | — | — | tail_beyond_max_len (6001) |
| 22 | 886747 | 887603 | 857 | 0.404 | located |
| 23 | 887937 | 888049 | 113 | 0.991 | located |
| 24 | 888075 | 888196 | 122 | 0.992 | located |
| 25 | 888215 | 888333 | 119 | 0.000 | located |
| 26 | 888373 | 888486 | 114 | 0.991 | located |
| 27 | 888511 | 888620 | 110 | 0.991 | located |
| 28 | 888838 | 889145 | 308 | 0.997 | located |
| 29 | 889221 | 890293 | 1073 | 0.000 | located |
| 30 | 936546 | 936662 | 117 | 0.000 | located |
| 31 | — | — | — | — | tail_beyond_max_len (14137) |
| 32 | — | — | — | — | tail_beyond_max_len (11181) |
| 33 | 963560 | 964286 | 727 | 0.000 | located |
| 34 | 964474 | 964714 | 241 | 0.996 | located |
| 35 | 965142 | 965249 | 108 | 0.991 | located |
| 36 | 965277 | 965388 | 112 | 0.991 | located |
| 37 | 965421 | 965856 | 436 | 0.998 | located |
| 38 | 965891 | 967221 | 1331 | 0.000 | located |
| 39 | 967245 | 967523 | 279 | 0.996 | located |
| 40 | 967971 | 968076 | 106 | 0.000 | located |
| 41 | 968817 | 969122 | 306 | 0.997 | located |
| 42 | 969156 | 969537 | 382 | 0.997 | located |
| 43 | 969577 | 970105 | 529 | 0.000 | located |
| 44 | 971738 | 972542 | 805 | 0.000 | located |
| 45 | 972579 | 972889 | 311 | 0.000 | located |
| 46 | — | — | — | — | tail_beyond_max_len (4884) |
| 47 | 978384 | 978524 | 141 | 0.993 | located |
| 48 | 978552 | 978669 | 118 | 0.992 | located |
| 49 | 979006 | 979155 | 150 | 0.397 | located |
| 50 | 979269 | 979384 | 116 | 0.303 | located |
| 51 | 979394 | 979530 | 137 | 0.000 | located |
| 52 | 979636 | 979919 | 284 | 0.000 | located |
| 53 | 980034 | 980106 | 73 | 0.000 | located |
| 54 | 980137 | 980332 | 196 | 0.423 | located |
| 55 | 980357 | 980399 | 43 | 0.977 | located |
| 56 | 980427 | 980577 | 151 | 0.000 | located |
| 57 | 980706 | 980818 | 113 | 0.000 | located |
| 58 | 980930 | 981067 | 138 | 0.993 | located |
| 59 | 981104 | 981161 | 58 | 0.983 | located |
| 60 | 981287 | 981401 | 115 | 0.991 | located |
| 61 | 981513 | 981655 | 143 | 0.993 | located |
| 62 | 981691 | 981789 | 99 | 0.990 | located |
| 63 | 981912 | 982110 | 199 | 0.995 | located |
| 64 | 982235 | 982383 | 149 | 0.000 | located |
| 65 | 982491 | 982601 | 111 | 0.000 | located |
| 66 | 982649 | 982937 | 289 | 0.000 | located |
| 67 | 982992 | 983077 | 86 | 0.000 | located |
| 68 | 983175 | 983291 | 117 | 0.000 | located |
| 69 | 983356 | 983652 | 297 | 0.000 | located |
| 70 | 983718 | 983943 | 226 | 0.000 | located |
| 71 | 984034 | 984152 | 119 | 0.000 | located |
| 72 | 984238 | 984378 | 141 | 0.993 | located |
| 73 | 984413 | 984702 | 290 | 0.000 | located |
| 74 | 984793 | 985037 | 245 | 0.000 | located |
| 75 | 985157 | 985270 | 114 | 0.991 | located |
| 76 | 985301 | 985442 | 142 | 0.993 | located |
| 77 | 985565 | 985917 | 353 | 0.997 | located |
| 78 | 985955 | 986091 | 137 | 0.993 | located |
| 79 | 986251 | 986412 | 162 | 0.000 | located |
| 80 | 986456 | 986598 | 143 | 0.000 | located |

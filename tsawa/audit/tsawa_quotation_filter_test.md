# Quotation filter for tsawa predictions: read-only check

Predictions are window-level (neighbouring windows overlap, so a span can be counted in two windows, as in the reported score). A prediction is a true positive when it matches a gold span at IoU >= 0.5 (one-to-one, greedy). Quotation/Citation spans are the books' own gold layer spans.

Baseline: 2930 TP, 1726 FP, 250 FN; P 0.629, R 0.921, F1 0.748. 19 of the 22 test books have a Quotation or Citation layer.

## 1 and 2: how many false positives and true positives overlap quotations

| overlap with a Quotation/Citation span | false positives (1726) | true positives (2930) |
|---|---|---|
| any overlap (coverage > 0) | 1179 (68.3%) | 69 (2.4%) |
| at least 25% of the span is quotation | 1111 (64.4%) | 62 (2.1%) |
| at least 50% of the span is quotation | 1042 (60.4%) | 49 (1.7%) |
| at least 90% of the span is quotation | 938 (54.3%) | 45 (1.5%) |
| IoU >= 0.5 with one Quotation/Citation span | 623 (36.1%) | 27 (0.9%) |


## 3: simulated filter (predictions dropped, matching re-run)

| rule: drop prediction if | dropped | FPs removed | TPs lost | P | R | F1 | dF1 |
|---|---|---|---|---|---|---|---|
| coverage >= 0.01 | 1248 | 1179 | 69 | 0.839 | 0.9 | 0.869 | 0.121 |
| coverage >= 0.25 | 1173 | 1111 | 62 | 0.823 | 0.902 | 0.861 | 0.113 |
| coverage >= 0.5 | 1091 | 1042 | 49 | 0.808 | 0.906 | 0.854 | 0.106 |
| coverage >= 0.75 | 1019 | 971 | 48 | 0.792 | 0.906 | 0.846 | 0.098 |
| coverage >= 0.9 | 983 | 938 | 45 | 0.785 | 0.907 | 0.842 | 0.094 |
| IoU with one span >= 0.3 | 791 | 759 | 32 | 0.75 | 0.911 | 0.823 | 0.075 |
| IoU with one span >= 0.5 | 650 | 623 | 27 | 0.725 | 0.913 | 0.808 | 0.06 |
| IoU with one span >= 0.7 | 562 | 540 | 22 | 0.71 | 0.914 | 0.8 | 0.052 |


## By book (quotation overlap >= 50% of the span)

| pecha_id | preds | fp | fp_cov50 | tp | tp_cov50 | has_quotation_layer |
|---|---|---|---|---|---|---|
| IE5895799 | 287 | 235 | 188 | 52 | 0 | True |
| ICDC84458 | 268 | 221 | 178 | 47 | 15 | True |
| P000269 | 339 | 295 | 170 | 44 | 0 | True |
| I3F4A91F5 | 239 | 154 | 124 | 85 | 0 | True |
| P000242 | 569 | 289 | 111 | 280 | 2 | True |
| P000083 | 411 | 114 | 67 | 297 | 0 | True |
| I9B6A4525 | 393 | 102 | 58 | 291 | 0 | True |
| IC6F06BCD | 76 | 59 | 45 | 17 | 0 | True |
| P000144 | 216 | 71 | 30 | 145 | 1 | True |
| IC05A6BE0 | 277 | 34 | 28 | 243 | 15 | True |
| I319DAFF7 | 27 | 14 | 13 | 13 | 0 | True |
| I9AEEF96A | 32 | 15 | 11 | 17 | 0 | True |

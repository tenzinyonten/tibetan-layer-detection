# Yigchung benchmark (BDRC/yigchung_detection_v2 test, 77 books)

IoU@0.5 span matching. "before" = original gold; "after" = corrected gold (annotator review for the 5 BDRC books, annotator patterns + head-mark rule elsewhere).

## All books

| model | gold | spans | precision | recall | F1 |
|---|---|---|---|---|---|
| karma689_v2 | before | 4059 | 0.518 | 0.432 | 0.471 |
| karma689_v2 | after | 3793 | 0.519 | 0.463 | 0.489 |
| genre_v2 | before | 4059 | 0.511 | 0.351 | 0.417 |
| genre_v2 | after | 3793 | 0.515 | 0.379 | 0.437 |

## By group

| group | model | gold | spans | precision | recall | F1 |
|---|---|---|---|---|---|---|
| BDRC, annotator-reviewed (5) | karma689_v2 | before | 1570 | 0.480 | 0.500 | 0.490 |
| BDRC, annotator-reviewed (5) | karma689_v2 | after | 1297 | 0.481 | 0.607 | 0.536 |
| BDRC, annotator-reviewed (5) | genre_v2 | before | 1570 | 0.559 | 0.574 | 0.566 |
| BDRC, annotator-reviewed (5) | genre_v2 | after | 1297 | 0.561 | 0.698 | 0.622 |
| Lotsawa (47) | karma689_v2 | before | 61 | 0.818 | 0.885 | 0.850 |
| Lotsawa (47) | karma689_v2 | after | 61 | 0.818 | 0.885 | 0.850 |
| Lotsawa (47) | genre_v2 | before | 61 | 0.673 | 0.607 | 0.638 |
| Lotsawa (47) | genre_v2 | after | 61 | 0.673 | 0.607 | 0.638 |
| other reviewed books (25) | karma689_v2 | before | 2428 | 0.544 | 0.376 | 0.445 |
| other reviewed books (25) | karma689_v2 | after | 2435 | 0.545 | 0.375 | 0.445 |
| other reviewed books (25) | genre_v2 | before | 2428 | 0.435 | 0.201 | 0.275 |
| other reviewed books (25) | genre_v2 | after | 2435 | 0.442 | 0.203 | 0.278 |

## By genre

| category | model | gold | spans | precision | recall | F1 |
|---|---|---|---|---|---|---|
| liturgy | karma689_v2 | before | 1583 | 0.539 | 0.717 | 0.616 |
| liturgy | karma689_v2 | after | 1595 | 0.541 | 0.713 | 0.615 |
| liturgy | genre_v2 | before | 1583 | 0.760 | 0.812 | 0.786 |
| liturgy | genre_v2 | after | 1595 | 0.766 | 0.812 | 0.788 |
| lotsawa | karma689_v2 | before | 61 | 0.818 | 0.885 | 0.850 |
| lotsawa | karma689_v2 | after | 61 | 0.818 | 0.885 | 0.850 |
| lotsawa | genre_v2 | before | 61 | 0.673 | 0.607 | 0.638 |
| lotsawa | genre_v2 | after | 61 | 0.673 | 0.607 | 0.638 |
| other | karma689_v2 | before | 2407 | 0.465 | 0.233 | 0.311 |
| other | karma689_v2 | after | 2129 | 0.465 | 0.264 | 0.337 |
| other | genre_v2 | before | 2407 | 0.098 | 0.042 | 0.059 |
| other | genre_v2 | after | 2129 | 0.100 | 0.049 | 0.066 |
| prayer | karma689_v2 | before | 8 | 1.000 | 0.125 | 0.222 |
| prayer | karma689_v2 | after | 8 | 1.000 | 0.125 | 0.222 |
| prayer | genre_v2 | before | 8 | 1.000 | 0.125 | 0.222 |
| prayer | genre_v2 | after | 8 | 1.000 | 0.125 | 0.222 |

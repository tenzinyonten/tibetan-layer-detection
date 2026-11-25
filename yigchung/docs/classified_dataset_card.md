---
license: mit
pretty_name: Bo-Yigchung-Classified-Dataset
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- layer-detection
- yigchung
---

# Bo-Yigchung-Classified-Dataset

Tokenized windows of Tibetan books of four genres (liturgy, prayer, sadhana, songs), labelled for Yigchung, the small-print interlinear note. Books were classified into genres with Gemini; books of the genre "other" are excluded. It trains [BDRC/Bo-Yigchung-Classified-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Classified-Detection). The test documents are published separately as [BDRC/Bo-Yigchung-Classified-Benchmark](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Classified-Benchmark).

## Format

- One row per window, for `jhu-clsp/mmBERT-base`. 3-class BIO labels: 0 `O`, 1 `B-YIGCHUNG`, 2 `I-YIGCHUNG`; `-100` marks special tokens and tokens already labelled by an earlier window.
- Columns: `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `category`, `window_index`, `n_tokens_doc`, `coverage_pct`.
- Windows of 8,192 tokens (8,190 content tokens), stride 5,120. `token_start`/`token_end` and `char_start`/`char_end` give the window's place in its document.

## Splits

Split by document: a document is in one split only.

| Split | Documents | Gold spans | Windows | O tokens | B tokens | I tokens |
|---|---:|---:|---:|---:|---:|---:|
| train | 49 | 9,306 | 1,388 | 10,411,381 | 14,854 | 933,542 |
| validation | 8 | 1,290 | 137 | 990,718 | 2,078 | 129,234 |

Token counts are over all windows, so overlapping windows count a token more than once.

Documents by genre:

| Genre | Train | Validation |
|---|---:|---:|
| liturgy | 15 | 1 |
| prayer | 26 | 5 |
| sadhana | 2 | 1 |
| songs | 6 | 1 |

## Data source and preparation

| Source (`source_batch`) | Documents: train / validation |
|---|---|
| `reviewed_opf` (BDRC books annotated in OpenPecha) | 16 / 2 |
| `bdrc`, `bdrc_annotator_reviewed` (BDRC volumes) | 13 / 3 |
| `lotsawa_combined` (Lotsawa House) | 20 / 3 |

The 41 classified books come from the 762-book release [BDRC/yigchung_detection_v2](https://huggingface.co/datasets/BDRC/yigchung_detection_v2) and four further BDRC books that are not in it. The 469 short Lotsawa House books are joined into 26 documents. Steps, in order, with the script for each (all in `yigchung/src/` of the [layer repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/yigchung)):

1. Download the text and spans: `fetch_yigchung_hub.py`.
2. Combine the Lotsawa books into documents of about 30,000 characters, within each split: `merge_yigchung_lotsawa.py`.
3. Classify each book by genre with Gemini: `classify_yigchung_genre.py`. Books of the genre "other" are dropped: `build_yigchung_genre_datasets.py`.
4. Clean the spans (annotator spans where a review exists, hand corrections and additions, spans end before an unmatched `)`, head marks never inside a span): `build_yigchung_genre_datasets.py`.
5. Tokenize with `jhu-clsp/mmBERT-base`, cut windows, and label tokens as BIO: `build_yigchung_genre_bio.py`.

## Annotation

Spans were annotated through OpenPecha. For 5 of the 67 documents (including the 10 benchmark documents) (about 28% of the spans) the spans come from an annotator review. Beyond that, the cleaning in step 4 was applied on top of the raw annotations.


## Citation

BDRC Yigchung classified layer detection dataset (Bo-Yigchung-Classified-Dataset), Buddhist Digital Resource Center, 2026.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

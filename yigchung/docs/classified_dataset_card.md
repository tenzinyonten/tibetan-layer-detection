---
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

Tokenized windows of Tibetan books of four genres (liturgy, prayer, sadhana, songs), labelled for Yigchung, the small-print interlinear note. Books were classified into genres with Gemini; books of the genre "other" are excluded. It trains [BDRC/Bo-Yigchung-Classified-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Classified-Detection).

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
| test | 10 | 1,806 | 229 | 1,669,729 | 2,856 | 195,129 |

Token counts are over all windows, so overlapping windows count a token more than once.

Documents by genre:

| Genre | Train | Validation | Test |
|---|---:|---:|---:|
| liturgy | 15 | 1 | 4 |
| prayer | 26 | 5 | 4 |
| sadhana | 2 | 1 | 1 |
| songs | 6 | 1 | 1 |

## Data source and preparation

| Source (`source_batch`) | Documents: train / validation / test |
|---|---|
| `reviewed_opf` (BDRC books annotated in OpenPecha) | 16 / 2 / 5 |
| `bdrc`, `bdrc_annotator_reviewed` (BDRC volumes) | 13 / 3 / 2 |
| `lotsawa_combined` (Lotsawa House) | 20 / 3 / 3 |

The 41 classified books come from the 762-book release [BDRC/yigchung_detection_v2](https://huggingface.co/datasets/BDRC/yigchung_detection_v2) and four further BDRC books that are not in it. The 469 short Lotsawa House books are joined into 26 documents. Steps, in order, with the script for each (all in `yigchung/src/` of the [layer repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/yigchung)):

1. Download the text and spans: `fetch_hub.py`.
2. Combine the Lotsawa books into documents of about 30,000 characters, within each split: `combine_lotsawa.py`.
3. Classify each book by genre with Gemini: `run_classify.py`. Books of the genre "other" are dropped: `build_genre_datasets.py`.
4. Clean the spans (annotator spans where a review exists, hand corrections and additions, spans end before an unmatched `)`, head marks never inside a span): `build_genre_datasets.py`.
5. Tokenize with `jhu-clsp/mmBERT-base`, cut windows, and label tokens as BIO: `build_genre_bio.py`.

## Annotation

Spans were annotated through OpenPecha. For 5 of the 67 documents (about 28% of the spans) the spans come from an annotator review. Beyond that, the cleaning in step 4 was applied on top of the raw annotations.

## Known limits

- Songs: the validation and test sets have only 4 and 1 gold spans.
- The Lotsawa House documents are labelled prayer as an assumption (about 90% of those books are prayers by a rough title check).
- Genres are unevenly sized: sadhana has 4 documents in total.
- Two test documents hold 86% of the test spans.
- Data rights: the texts come from BDRC and Lotsawa House. Check each source's terms before reuse; the licence field is a placeholder.

## Citation

BDRC Yigchung classified layer detection dataset (Bo-Yigchung-Classified-Dataset), Buddhist Digital Resource Center, 2026.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

---
pretty_name: Bo-Author-Dataset
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- author
- bio
- mmbert
---


# Bo-Author-Dataset

Windowed BIO token-classification data for the **author** of a Tibetan work, from the first and last 3,000 characters of each work. It is the training data of [`BDRC/Bo-Author-Detection`](https://huggingface.co/BDRC/Bo-Author-Detection).

It is the author version of the joint title/author data: the same works and splits, with only the author labelled and the title counted as "other" (`O`). The title version is `BDRC/Bo-Title-Dataset`.

## Format

Tokenized with `jhu-clsp/mmBERT-base`. Labels (`label_map.json`): `O` = 0, `B-AUTHOR` = 1, `I-AUTHOR` = 2. Special tokens and padding are -100. A token gets B if a span starts in it, I for the following tokens of the span.

Each work is cropped to its first and last ~3,000 characters (edges moved to a line break; the whole work if 6,000 characters or fewer; a span is never cut). One sequence per work is `[CLS] head [SEP] tail [SEP]`; the label at the SEP is -100. Windows are up to 8,192 tokens (about 2,000 in practice).

Columns: `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `work_id`, `cid` (collection id), `source`, `region` (full/head/tail), `region_start`, `window_index`, `n_tokens_region`.

## Splits

One row per work, so rows and works are the same. Copies of the same work were removed before the split, and duplicate works never sit in different splits (a collection can have works in more than one split). Validation and test contain annotator spans only. The test split is the benchmark for the model.

| split | works |
|---|---|
| train | 32,074 |
| validation | 2,494 |
| test | 2,384 |

Label counts (tokens): `B-AUTHOR` and `I-AUTHOR` — train 32,074 and 439,826, validation 2,494 and 35,255, test 2,384 and 33,115. The test split has 5 fewer works than the 2,389-work split built before cleaning, because works whose author is a translator were dropped (step 2 below).

## Data source and preparation

**Source.** The text of each work is a segment of a BDRC volume. The `source` column says where the text came from:

| source | train | validation | test | text origin |
|---|---|---|---|---|
| tei | 6,105 | 537 | 484 | BDRC e-text (TEI), human-typed |
| google_books | 3,810 | 377 | 349 | Google Books text |
| google_vision | 2,994 | 304 | 225 | Google Vision OCR |
| paddleocr_v2 | 5,075 | 637 | 640 | PaddleOCR |
| ocr_qwen_v1 | 3,206 | 308 | 341 | OCR (`ocr_qwen_v1` pipeline) |
| ocrv1-ws-ldv1 | 2,012 | 207 | 226 | OCR (`ocrv1-ws-ldv1` pipeline) |
| paddleocr_v1 | 1,061 | 124 | 119 | PaddleOCR (first version) |
| v3_auto | 7,811 | 0 | 0 | PaddleOCR text from the BDRC char-level outlines release v3; spans found automatically |
| **all** | **32,074** | **2,494** | **2,384** | |

**Preparation**, in order (scripts in `title_author/src/` of [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection)):

1. Take the annotator title and author spans (character offsets) for each work. For the `v3_auto` works, the spans come from the BDRC char-level outlines release v3 and are found automatically by string matching: `build_ta_v3_addon.py`. These are in train only.
2. Drop works whose gold author contains a translator word (`ལོ་ཙཱ`, `ལོ་ཙྪ`): `clean_ta_translators.py`.
3. Remove the `_` syllable-separator artifact that some sources have from the text, and shift the spans: `clean_ta_underscores.py`.
4. Crop each work to its first and last ~3,000 characters, edges moved to a line break; keep the whole work if it is 6,000 characters or fewer: `build_ta_gold_dataset.py`.
5. Remove copies of the same work (same typed title and author, same first 300 characters, or a near-identical text start; the higher-ranked source is kept) and split about 80/10/10 by collection group (see Splits).
6. Tokenize with `jhu-clsp/mmBERT-base` and label with the token-start rule (B on the token where a span starts, I on the following tokens).
7. For this author-only version: relabel the title `O` (not masked), keeping the same rows and splits as the joint data.

## Annotation

Two kinds of spans:
- **Annotator spans** (all of validation and test, and 24,263 of the train works): one title and one author span per work, marked by human annotators on the text above. The code does not edit them. Overlapping title/author spans: the title wins.
- **Automatic spans** (7,811 train works, `source = v3_auto`): found by string matching against the BDRC char-level outlines release v3, about 91-94% right. They are never in validation or test.

## Known limits

- One author per work; second authors and other mentions are not marked.
- Span edges vary between annotators, and the gold span may or may not include the cue word (`མཛད་པ་པོ།`) and honorifics.
- Duplicate detection looks at the first 1,500 characters only, so some copies can slip through and some different works can be wrongly merged.
- The text comes from BDRC volumes. Data-rights questions are still open, so check the terms before sharing any text derived from this dataset.

## Citation

```bibtex
@misc{bo_author_dataset,
  title  = {Bo-Author-Dataset: Tibetan author token classification data},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Author-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

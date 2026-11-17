---
pretty_name: Bo-Title-Dataset
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- title
- bio
- mmbert
license: other
---


# Bo-Title-Dataset

Windowed BIO token-classification data for the **title** of a Tibetan work, from the first and last 3,000 characters of each work. It is the training data of [`BDRC/Bo-Title-Detection`](https://huggingface.co/BDRC/Bo-Title-Detection) (revision `092bcd55`, the last commit that changed the data files).

It is derived from the joint title/author data `Yontenn/title-author-training` (revision `1c3226b3`) with the same rows, windows and splits. Only `labels` changed: the author is relabelled `O` (not masked). The author version is `BDRC/Bo-Author-Dataset`.

## Format

Tokenized with `jhu-clsp/mmBERT-base`. Labels (`label_map.json`): `O` = 0, `B-TITLE` = 1, `I-TITLE` = 2. Special tokens and padding are -100. A token gets B if a span starts in it, I for the following tokens of the span.

Each work is cropped to its first and last ~3,000 characters (edges moved to a line break; the whole work if 6,000 characters or fewer; a span is never cut). One sequence per work is `[CLS] head [SEP] tail [SEP]`; the label at the SEP is -100. Windows are up to 8,192 tokens (about 2,000 in practice).

Columns: `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `work_id`, `cid` (collection id), `source`, `region` (full/head/tail), `region_start`, `window_index`, `n_tokens_region`.

## Splits

One row per work, so rows and works are the same. Copies of the same work were removed before the split, and duplicate works never sit in different splits (a collection can have works in more than one split). Validation and test contain annotator spans only. The test split is the benchmark for the model.

| split | works |
|---|---|
| train | 32,074 |
| validation | 2,494 |
| test | 2,384 |

Label counts (tokens): `B-TITLE` and `I-TITLE` — train 32,074 and 1,197,216, validation 2,494 and 92,323, test 2,384 and 85,505. The test split has 5 fewer works than the 2,389-work split built before cleaning, because works whose author is a translator were dropped (step 2 below).

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

1. Take the annotator title and author spans (character offsets) for each work. For the `v3_auto` works, the spans come from the BDRC char-level outlines release v3 and are found automatically by string matching: `ta_build_v3_addon.py`. These are in train only.
2. Drop works whose gold author contains a translator word (`ལོ་ཙཱ`, `ལོ་ཙྪ`): `ta_drop_translator.py`.
3. Remove the `_` syllable-separator artifact that some sources have from the text, and shift the spans: `ta_strip_underscore.py`.
4. Crop each work to its first and last ~3,000 characters, edges moved to a line break; keep the whole work if it is 6,000 characters or fewer: `ta_build_gold_dataset.py`.
5. Remove copies of the same work (same typed title and author, same first 300 characters, or a near-identical text start; the higher-ranked source is kept) and split about 80/10/10 by collection group (see Splits).
6. Tokenize with `jhu-clsp/mmBERT-base` and label with the token-start rule (B on the token where a span starts, I on the following tokens).
7. For this title-only version: relabel the author `O` (not masked), keeping the same rows and splits as the joint data.

## Annotation

Two kinds of spans:
- **Annotator spans** (all of validation and test, and 24,263 of the train works): one title and one author span per work, marked by human annotators on the text above. The code does not edit them. Overlapping title/author spans: the title wins.
- **Automatic spans** (7,811 train works, `source = v3_auto`): found by string matching against the BDRC char-level outlines release v3, about 91-94% right. They are never in validation or test.

## Known limits

- One title and one author per work; other mentions and second authors are not marked.
- Span edges vary between annotators (for example whether the closing shad or `བཞུགས་སོ` is included).
- Duplicate detection looks at the first 1,500 characters only, so some copies can slip through and some different works can be wrongly merged.
- The text comes from BDRC volumes. Data-rights questions are still open, so check the terms before sharing any text derived from this dataset.

## Citation

```bibtex
@misc{bo_title_dataset,
  title  = {Bo-Title-Dataset: Tibetan title token classification data},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Title-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

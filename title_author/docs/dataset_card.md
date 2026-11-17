---
license: other
language: [bo]
task_categories: [token-classification]
pretty_name: Tibetan title / author (BIO)
---
# Tibetan title / author, tokenized (BIO)

Token-classification data for finding the **title** and the **author** of each work in Tibetan volumes, for fine-tuning `jhu-clsp/mmBERT-base`. One model learns both.

**Labels** (`label_map.json`): `O=0, B-TITLE=1, I-TITLE=2, B-AUTHOR=3, I-AUTHOR=4`. Special tokens and padding are `-100`. A token gets B if a span starts in it, I for the following tokens of the span.

**Input:** each work is cropped to its first and last ~3,000 characters (edges moved to a line break; whole work if 6,000 characters or fewer; a span is never cut). Windows are up to 8,192 tokens (in practice about 2,000). Tokenizer: `jhu-clsp/mmBERT-base`.

**Columns:** `input_ids, attention_mask, labels, token_start, token_end, char_start, char_end, work_id, cid (collection id), source, region (full/head/tail), region_start, window_index, n_tokens_region`.

**Split (this version):** one row per work (rows and works are the same). Copies of the same work (same typed title and author, same first 300 characters, or a near-identical text start) were reduced to one before the split; the higher-ranked source is kept: tei, google_books, google_vision, paddleocr_v2, ocrv1-ws-ldv1, ocr_qwen_v1, paddleocr_v1, then the longer text. The split is about 80/10/10, **stratified on source x book type x length**, spread over collection families; duplicate works never sit in different splits (a collection can have works in more than one split). Validation and test contain **annotator spans only**. Train also has 7,811 works with automatically found spans (`v3_auto`, from the BDRC char-level outlines release v3, about 91-94% right).

**Cleaning applied before tokenizing:** works whose author span contains a translator word (`ལོ་ཙཱ`, `ལོ་ཙྪ`) were dropped (149 works), and the `_` syllable-separator artifact was removed from the text. Checked on the Hub data on 2026-11-17: no `_` and no translator word in the validation and test works.

## Counts

Checked against the Hub data on 2026-11-17.

| split | works |
|---|---|
| train | 32,074 |
| validation | 2,494 |
| test | 2,384 |

| source | train | validation | test | text origin |
|---|---|---|---|---|
| tei | 6,105 | 537 | 484 | BDRC e-text (TEI), human-typed |
| google_books | 3,810 | 377 | 349 | Google Books text |
| google_vision | 2,994 | 304 | 225 | Google Vision OCR |
| paddleocr_v2 | 5,075 | 637 | 640 | PaddleOCR |
| ocr_qwen_v1 | 3,206 | 308 | 341 | OCR (`ocr_qwen_v1` pipeline) |
| ocrv1-ws-ldv1 | 2,012 | 207 | 226 | OCR (`ocrv1-ws-ldv1` pipeline) |
| paddleocr_v1 | 1,061 | 124 | 119 | PaddleOCR (first version) |
| v3_auto | 7,811 | 0 | 0 | PaddleOCR text from the BDRC outlines v3; spans found automatically |

Label counts (tokens): train B-TITLE 32,074, I-TITLE 1,197,216, B-AUTHOR 32,074, I-AUTHOR 439,826; validation 2,494 / 92,323 / 2,494 / 35,255; test 2,384 / 85,505 / 2,384 / 33,115.

**Known limits:** one title and one author per work (other mentions are unlabeled); span edges vary between annotators; duplicate detection looks at the first 1,500 characters only; overlapping title/author: title wins.

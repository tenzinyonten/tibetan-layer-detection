---
license: other
language: [bo]
task_categories: [token-classification]
pretty_name: Tibetan title / author (BIO)
---
# formatting-title-author-v1 (private)

Token-classification data for finding the **title** and the **author** of each work in Tibetan volumes, for fine-tuning `jhu-clsp/mmBERT-base`. One model learns both.

**Labels** (`label_map.json`): `O=0, B-TITLE=1, I-TITLE=2, B-AUTHOR=3, I-AUTHOR=4`. Special tokens and padding are `-100`. A token gets B if a span starts in it, I for the following tokens of the span.

**Input:** each work is cropped to its first and last ~3,000 characters (edges moved to a line break; whole work if 6,000 characters or fewer; a span is never cut). Windows are up to 8,192 tokens (in practice about 2,000). Tokenizer: `jhu-clsp/mmBERT-base`.

**Columns:** `input_ids, attention_mask, labels, token_start, token_end, char_start, char_end, work_id, cid (collection id), source, region (full/head/tail), region_start, window_index, n_tokens_region`.

**Split (this version):** all 34,774 annotated works were pooled; copies of the same work (same typed title+author, same first 300 characters, or a near-identical text start) were reduced to one (30,573 unique works; the higher-ranked source is kept: tei, google_books, google_vision, paddleocr_v2, ocrv1-ws-ldv1, ocr_qwen_v1, paddleocr_v1, then the longer text). The split is about 80/10/10, **stratified on source x book type x length**, spread over collection families; duplicate works never sit in different splits (a collection can have works in more than one split). Validation and test contain **annotator spans only**.

**Annotator works by source**

| source | train | validation | test | total |
|---|---|---|---|---|
| tei | 6,320 | 554 | 509 | 7,383 |
| google_books | 4,000 | 388 | 370 | 4,758 |
| google_vision | 3,087 | 354 | 229 | 3,670 |
| paddleocr_v2 | 5,378 | 658 | 660 | 6,696 |
| ocr_qwen_v1 | 3,407 | 336 | 355 | 4,098 |
| ocrv1-ws-ldv1 | 2,149 | 221 | 238 | 2,608 |
| paddleocr_v1 | 1,112 | 125 | 123 | 1,360 |

**Known limits:** one title and one author per work (other mentions are unlabeled); span edges vary between annotators; duplicate detection looks at the first 1,500 characters only; overlapping title/author: title wins.

## Counts

| split | works | windows |
|---|---|---|
| train | 25,453 | 36,207 |
| validation | 2,636 | 3,629 |
| test | 2,484 | 3,564 |

Train label counts: B-TITLE 25,353, I-TITLE 1,007,185, B-AUTHOR 24,928, I-AUTHOR 360,694.

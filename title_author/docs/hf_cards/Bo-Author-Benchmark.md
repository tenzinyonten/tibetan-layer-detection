---
license: mit
pretty_name: Bo-Author-Benchmark
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- author
- bio
- mmbert
- benchmark
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
- config_name: books
  data_files:
  - split: test
    path: books/test-*
---

# Bo-Author-Benchmark

The public test set for finding the **author** of a Tibetan work: 2,384 works, each cropped to its first and last 3,000 characters. Score a model on it and compare with others on the same works. The training and validation data are not published.

It is the author version of the joint title/author test data: the same works, with only the author labelled and the other one counted as "other" (`O`). The title version is `BDRC/Bo-Title-Benchmark`.

## Format

Tokenized with `jhu-clsp/mmBERT-base`. Labels (`label_map.json`): `O` = 0, `B-AUTHOR` = 1, `I-AUTHOR` = 2. Special tokens and padding are -100. A token gets B if a span starts in it, I for the following tokens of the span.

Each work is cropped to its first and last ~3,000 characters (edges moved to a line break; the whole work if 6,000 characters or fewer; a span is never cut). One sequence per work is `[CLS] head [SEP] tail [SEP]`; the label at the SEP is -100. Windows are up to 8,192 tokens (about 2,000 in practice).

Columns: `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `work_id`, `cid` (collection id), `source`, `region` (full/head/tail), `region_start`, `window_index`, `n_tokens_region`.

## The test set

One split, `test`. One row per work, so rows and works are the same. Copies of the same work were removed before the split, and no work in the test set is a copy of a training work. Every span is from a human annotator.

Label counts (tokens): `B-AUTHOR` 2,384 and `I-AUTHOR` 33,115.

| source | works | text origin |
|---|---|---|
| tei | 484 | BDRC e-text (TEI), human-typed |
| google_books | 349 | Google Books text |
| google_vision | 225 | Google Vision OCR |
| paddleocr_v2 | 640 | PaddleOCR |
| ocr_qwen_v1 | 341 | OCR (`ocr_qwen_v1` pipeline) |
| ocrv1-ws-ldv1 | 226 | OCR (`ocrv1-ws-ldv1` pipeline) |
| paddleocr_v1 | 119 | PaddleOCR (first version) |
| **all** | **2,384** | |

## Books config (plain text)

The `default` config holds token ids for `jhu-clsp/mmBERT-base`. To score a model with a different tokenizer, or an LLM, use the `books` config: one row per work with the exact text and the gold author span.

| Column | Meaning |
|---|---|
| `work_id` | the work (same ids as in the `default` config) |
| `source` | where the text came from (tei, google_books, google_vision, paddleocr_v1, paddleocr_v2, ocr_qwen_v1, ocrv1-ws-ldv1) |
| `whole_work` | true if the work is 6,000 characters or fewer and kept whole |
| `work_chars` | length of the full work in characters |
| `text_head` | the first ~3,000 characters of the work (the whole work when `whole_work` is true) |
| `text_tail` | the last ~3,000 characters (empty when `whole_work` is true) |
| `spans` | the gold author span: a list of `{region, start, end}`, where `region` is `head` or `tail` and `start`/`end` are character offsets inside that text (`end` exclusive, so the span is `text_head[start:end]` or `text_tail[start:end]`) |

The span is read back from the published window labels, because the models are scored in token space: its edges sit on `jhu-clsp/mmBERT-base` token boundaries and can differ by a character or two from the annotators' original offsets. Tokenizing `text_head`, a separator and `text_tail` with `jhu-clsp/mmBERT-base` and labelling them with the span gives back the window rows exactly: token ids and labels.

```python
from datasets import load_dataset
works = load_dataset("BDRC/Bo-Author-Benchmark", "books", split="test")
```

## Scoring

Span precision, recall and F1 at IoU >= 0.5, with Viterbi decoding. Overlapping title and author spans: the title wins. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/title_author). Score the test set once, after you have chosen your model on your own validation data.

## Citation

```bibtex
@misc{bo_author_benchmark,
  title  = {Bo-Author-Benchmark: Tibetan author test set},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Author-Benchmark}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

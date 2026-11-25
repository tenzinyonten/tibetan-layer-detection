---
license: mit
pretty_name: Bo-Chapter-Benchmark
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- chapter
- layer-detection
- bio
- mmbert
- openpecha
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

# Bo-Chapter-Benchmark

The public test set for the Chapter layer (ལེའུ་, chapter and section headings) of Tibetan Buddhist books: 36 books, 940 windows, 258 Chapter spans. Score a model on it and compare with others on the same books. The training and validation data are not published.

## Format

| item | value |
|---|---|
| tokenizer | `jhu-clsp/mmBERT-base` (fast, with offsets) |
| labels | `O` = 0, `B-CHAPTER` = 1, `I-CHAPTER` = 2; CLS, SEP and padding = -100 |
| columns | `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct` |

A window holds 8,192 tokens (8,190 content tokens plus CLS and SEP) and starts every 5,120 tokens, so consecutive windows overlap by 3,070 content tokens. Tokens in the overlap are labelled in both windows. `token_start` and `token_end` give the window's token range in the book, and `char_start` and `char_end` its character range. `source_batch` is `old` or `new` (the two annotation batches).

## The test set

One split, `test`, made of whole books (a book is never split across train and test).

| split | books | old / new books | windows | Chapter spans | spans per window |
|---|---|---|---|---|---|
| test | 36 | 16 / 20 | 940 | 258 | 0.274 |

## Books config (plain text)

The `default` config holds token ids for `jhu-clsp/mmBERT-base`. To score a model with a different tokenizer, or an LLM, use the `books` config: one row per book with the exact text and the gold spans.

| Column | Meaning |
|---|---|
| `pecha_id` | the book |
| `source_batch` | `old` or `new` (the two annotation batches) |
| `text` | the full text of the book |
| `doc_chars` | length of the text in characters |
| `spans` | the gold spans: a list of `{id, start, end}` |

`start` is the first character of the span and `end` is one past its last character, so the span is `text[start:end]`. These are the same offsets as `char_start` and `char_end` in the window rows. Tokenizing `text` with `jhu-clsp/mmBERT-base` and labelling it with the spans gives back the window rows exactly (token ids and labels).

```python
from datasets import load_dataset
books = load_dataset("BDRC/Bo-Chapter-Benchmark", "books", split="test")
```

## Scoring

Span precision, recall and F1 at IoU >= 0.5, greedy one-to-one matching, with character offsets and an inclusive end. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/chapter). Score the test set once, after you have chosen your model on your own validation data.

## Data source

Source texts are BDRC books in OpenPecha `.opf` format; the Chapter spans come from the `Chapter` layer of those books, after the span cleaning described in the GitHub repository.

## Citation

```bibtex
@misc{bo_chapter_benchmark,
  title  = {Bo-Chapter-Benchmark: Tibetan chapter-heading test set},
  author = {Yonten, Tenzin},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Chapter-Benchmark}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

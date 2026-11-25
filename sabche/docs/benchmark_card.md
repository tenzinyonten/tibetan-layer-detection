---
license: mit
pretty_name: Bo-Sabche-Benchmark
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- sabche
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

# Bo-Sabche-Benchmark

The public test set for finding **Sabche** (ས་བཅད་, outline heading) spans in Tibetan Buddhist texts: 29 books, 963 windows, 4,079 spans. Score a model on it and compare with others on the same books. The training and validation data are not published.

A Sabche span is the heading phrase itself (for example a numbered outline line), stopping just before the closing shad, not the section under it.

## Format

| item | value |
|---|---|
| Tokenizer | `jhu-clsp/mmBERT-base` (fast, offsets) |
| Window / stride | 8,192 tokens (8,190 content + CLS/SEP) / 5,120 between window starts |
| Labels | `O`=0, `B-SABCHE`=1, `I-SABCHE`=2; CLS/SEP/pad = -100 |
| Columns | `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct` |
| Offsets | `char_start`/`char_end` are start-inclusive, end-exclusive (`text[start:end]`). The scoring code in the GitHub repository uses an inclusive end instead, so convert with `end - 1`. |

Consecutive windows overlap by 3,070 content tokens; tokens in the overlap are labelled in both windows.

## The test set

One split, `test`, made of whole books. Books are grouped before the split, so that no book sharing a long heading verbatim with another book, or belonging to the same source outline group, crosses a split.

| Books | Windows | Spans |
|---:|---:|---:|
| 29 | 963 | 4,079 |

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
books = load_dataset("BDRC/Bo-Sabche-Benchmark", "books", split="test")
```

## Scoring

Span precision, recall and F1 at IoU >= 0.5, greedy one-to-one matching, with Viterbi decoding and inclusive offsets. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/sabche). Score the test set once, after you have chosen your model on your own validation data.

## Data source

Headings were annotated through OpenPecha over two annotation batches, told apart by `pecha_id` prefix and carried as `source_batch`. The spans are the annotators' spans after the cleaning described in the GitHub repository.

## Citation

```bibtex
@misc{bo_sabche_benchmark,
  title  = {Bo-Sabche-Benchmark: Tibetan outline-heading (sabche) test set},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Sabche-Benchmark}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

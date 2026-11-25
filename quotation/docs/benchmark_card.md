---
license: mit
pretty_name: Bo-Quotation-Benchmark
tags:
- tibetan
- quotation
- bioe
- mmbert
- openpecha
- benchmark
language:
- bo
task_categories:
- token-classification
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: test.jsonl
- config_name: reviewed_test
  data_files:
  - split: test
    path: reviewed/test.jsonl
- config_name: windowed_w8192_s4579_bioe
  data_files:
  - split: test
    path: windowed_w8192_s4579_bioe/test/*.parquet
- config_name: windowed_w8192_s4579_bioe_reviewed
  data_files:
  - split: test
    path: windowed_w8192_s4579_bioe_reviewed/test/*.parquet
---

# Bo-Quotation-Benchmark

The public test set for finding **quotation spans** in classical Tibetan commentaries: 19 books, 2,337 human-reviewed quotation spans (2,398 after a second, span-by-span review). Score a model on it and compare with others on the same books. The training and validation data are not published.

Character offsets are **inclusive on both ends**: the span is `text[start:end+1]`.

## Configs

All configs have one split, `test`, made of the same 19 books.

| Config | What it holds |
|---|---|
| `default` | one row per book: text and the original gold spans (2,337) |
| `reviewed_test` | the same books with the reviewed gold spans (2,398), see "Reviewed gold" |
| `windowed_w8192_s4579_bioe` | 657 windows of 8,192 tokens, BIOE labels, original gold |
| `windowed_w8192_s4579_bioe_reviewed` | the same windows, labels from the reviewed gold |

**Book-level rows** (`test.jsonl`, `reviewed/test.jsonl`):

| Column | Meaning |
|---|---|
| `book_id` | the book |
| `title` | the book's title |
| `text` | the full text |
| `doc_chars` | length of the text in characters |
| `batch` | source batch: old or new |
| `source_layer` | the OpenPecha layer the spans came from: `Quotation` or `Citation` |
| `density`, `density_bucket` | share of the text inside quotations, and its band |
| `spans` | the quotation spans (below) |

Each span has `start` and `end` (inclusive), `cls` (the quote type: `TEXT`, `PERSON`, `MISC`, `FORMULA` or `CHAINED`), `closed` (whether the span ends with an explicit closing marker) and `seen_in_train` (whether the same text also occurs in the training books).

**Windowed rows**: `book_id`, `window_index`, `input_ids`, `attention_mask`, `labels` and `offset_mapping`. Labels are `O`=0, `B-QUOTE`=1, `I-QUOTE`=2, `E-QUOTE`=3. All quote types collapse to one `QUOTE` label. Tokenizer `jhu-clsp/mmBERT-base`, window 8,192 tokens, step 4,579. Each overlapped token is labelled in only one window (the first); elsewhere it is `-100`. Joining each book's owned tokens in `window_index` order rebuilds the book exactly, so stitching for inference or scoring is safe.

## The test set

| Books | Spans | TEXT | PERSON | MISC | Density | Old batch |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 2,337 | 57.1% | 15.4% | 18.8% | 7.50% | 38.9% |

The books were chosen at random by book, stratified by quote type, density band and source batch. The quote type of each span is not hand-labelled: it is set by rule from the words that introduce the span.

**The same quotation can appear in different books**, because commentaries quote the same scriptures. Every span carries a `seen_in_train` flag. Report two scores: overall, and `seen_in_train == false` only. The gap shows how much a model relies on memorised quotes.

## Reviewed gold

All 19 books were checked by human reviewers, span by span. Each original span was accepted as is, had its edges corrected, or was removed, and reviewers added quotations the first annotators missed.

| | count |
|---|---|
| spans accepted unchanged | 2281 |
| spans with corrected edges | 106 |
| spans added | 82 |
| spans removed | 27 |
| quotations before review (merged) | 2337 |
| quotations after review (merged) | 2398 |
| identical before and after | 2211 |

Same conventions as the original set. Each reviewed span also has `review` (`accepted` / `edited` / `added`) and `cls_source` (where its marker class came from; added spans have no original class). In the two windowed configs, input ids, offsets and the ownership mask are identical; only `labels` differ, so a score on the two shows how much the gold change moved it. Per-book changes: `reviewed/review_report.md`; raw numbers: `reviewed/stats.json`.

## Loading

```python
from datasets import load_dataset
books = load_dataset("BDRC/Bo-Quotation-Benchmark", split="test")
reviewed = load_dataset("BDRC/Bo-Quotation-Benchmark", "reviewed_test", split="test")
```

For the windowed configs load the parquet files directly:

```python
windows = load_dataset("parquet", data_files={
    "test": "hf://datasets/BDRC/Bo-Quotation-Benchmark/windowed_w8192_s4579_bioe/test/*.parquet"})["test"]
```

## Scoring

Span precision, recall and F1 at IoU >= 0.5. Score the test set once, after you have chosen your model on your own validation data. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/quotation).

## Citation

```bibtex
@misc{tibetan_quotation_benchmark,
  title  = {Bo-Quotation-Benchmark: Tibetan quotation test set},
  author = {karma689},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Quotation-Benchmark}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

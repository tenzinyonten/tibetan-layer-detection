---
license: mit
pretty_name: Bo-Yigchung-Classified-Benchmark
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- layer-detection
- yigchung
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

# Bo-Yigchung-Classified-Benchmark

The public test set for Yigchung (the small-print interlinear note) in Tibetan books of four genres (liturgy, prayer, sadhana, songs): 10 documents, 229 windows, 1,806 gold spans. Score a model on it and compare with others on the same documents. The training and validation data are not published.

## Format

- One row per window, for `jhu-clsp/mmBERT-base`. 3-class BIO labels: 0 `O`, 1 `B-YIGCHUNG`, 2 `I-YIGCHUNG`; `-100` marks special tokens and tokens already labelled by an earlier window.
- Columns: `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `category`, `window_index`, `n_tokens_doc`, `coverage_pct`.
- Windows of 8,192 tokens (8,190 content tokens), stride 5,120. `token_start`/`token_end` and `char_start`/`char_end` give the window's place in its document.

## The test set

One split, `test`, made of whole documents.

| Documents | Gold spans | Windows | O tokens | B tokens | I tokens |
|---|---:|---:|---:|---:|---:|
| 10 | 1,806 | 229 | 1,669,729 | 2,856 | 195,129 |

Token counts are over all windows, so overlapping windows count a token more than once.

| Genre | Documents |
|---|---:|
| liturgy | 4 |
| prayer | 4 |
| sadhana | 1 |
| songs | 1 |

| Source (`source_batch`) | Documents |
|---|---:|
| `reviewed_opf` (BDRC books annotated in OpenPecha) | 5 |
| `bdrc`, `bdrc_annotator_reviewed` (BDRC volumes) | 2 |
| `lotsawa_combined` (Lotsawa House) | 3 |

## Books config (plain text)

The `default` config holds token ids for `jhu-clsp/mmBERT-base`. To score a model with a different tokenizer, or an LLM, use the `books` config: one row per document with the exact text and the gold spans.

| Column | Meaning |
|---|---|
| `pecha_id` | the document (`LOTSAWA_COMBINED_…` are joined Lotsawa House books) |
| `source_batch` | where the text came from |
| `category` | the genre |
| `text` | the full text of the document |
| `doc_chars` | length of the text in characters |
| `spans` | the gold Yigchung spans: a list of `{id, start, end}` |

Offsets are character offsets, inclusive on both ends: the span is `text[start:end + 1]`. Tokenizing `text` with `jhu-clsp/mmBERT-base` and labelling it with the spans gives back the window rows exactly: token ids and labels.

```python
from datasets import load_dataset
docs = load_dataset("BDRC/Bo-Yigchung-Classified-Benchmark", "books", split="test")
```

## Scoring

Span precision, recall and F1 at IoU >= 0.5, one-to-one matching. Windows overlap, so average the logits of overlapping windows per token, then Viterbi-decode the whole document once. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/yigchung). Score the test set once, after you have chosen your model on your own validation data.

## Citation

BDRC Yigchung classified layer detection benchmark (Bo-Yigchung-Classified-Benchmark), Buddhist Digital Resource Center, 2026.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

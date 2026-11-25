---
license: mit
pretty_name: Bo-Tsawa-Benchmark
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- tsawa
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

# Bo-Tsawa-Benchmark

The public test set for **Tsawa** (རྩ་བ་, root text) detection in Tibetan commentaries: 28 books, 1,273 windows, 6,373 spans. Score a model on it and compare with others on the same books. The training and validation data are not published.

## Format

Tokenized with `jhu-clsp/mmBERT-base`. Windows of 8,192 tokens, stride 5,120. Labels: `O` = 0, `B-TSAWA` = 1, `I-TSAWA` = 2, `-100` = special tokens, padding and masked tokens. A token gets B if a span starts in it, I for the following tokens of the span.

Columns: `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct`, `token_start`, `token_end`, `char_start`, `char_end`, `input_ids`, `attention_mask`, `labels`.

Span ends follow the shad-inside convention: a span includes its closing shad. Repeated verses in the classical commentaries are masked (-100).

## The test set

One split, `test`.

| part | books | windows | spans | source |
|---|---:|---:|---:|---|
| BDRC commentaries | 22 | 639 | 1,976 | BDRC volumes, OpenPecha `.opf` format, annotated by people |
| Classical commentaries | 6 | 634 | 4,397 | WeBuddhist `*-rails` repositories; labels from the root-verse links |
| **Total** | **28** | **1,273** | **6,373** | |

The `source_batch` column tells the parts apart: `old` and `new` are BDRC books, `classical` is a WeBuddhist commentary. The six classical books are one commentary on each of six root texts: the Vinaya sutra, the Pramāṇavārttika, the Madhyamakāvatāra, the Abhidharmakośa, the Bodhicaryāvatāra and the Mahāyānasūtrālaṃkāra.

## Books config (plain text)

The `default` config holds token ids for `jhu-clsp/mmBERT-base`. To score a model with a different tokenizer, or an LLM, use the `books` config: one row per book with the exact text and the gold spans.

| Column | Meaning |
|---|---|
| `pecha_id` | the book (`WB…` ids are the classical commentaries) |
| `source_batch` | `old`, `new` or `classical` |
| `text` | the full text of the book |
| `doc_chars` | length of the text in characters |
| `spans` | the gold Tsawa spans: a list of `{id, start, end}` |
| `ignore_spans` | classical books only: repeated verses that are masked (-100) in the windows and not scored. A list of `{start, end, reason}`. |

`start` is the first character of the span and `end` is one past its last character, so the span is `text[start:end]`.

For the 22 BDRC books (`old` and `new`) the spans are read back from the published window labels, because Tsawa is scored in token space. Their edges therefore sit on mmBERT token boundaries and can differ by a character or two from the annotators' original offsets. The 6 classical books carry the original character spans. Tokenizing `text` with `jhu-clsp/mmBERT-base` and labelling it with these spans (and masking the ignore spans) gives back the window rows exactly: token ids and labels.

```python
from datasets import load_dataset
books = load_dataset("BDRC/Bo-Tsawa-Benchmark", "books", split="test")
```

## Scoring

Span precision, recall and F1 at IoU ≥ 0.5 in token space, greedy one-to-one matching, Viterbi decoding (only legal B/I transitions). Do not decode with argmax. Overlapping windows count a span in each window. Score the test set once, after you have chosen your model on your own validation data.

## Citation

```bibtex
@misc{bo_tsawa_benchmark,
  title  = {Bo-Tsawa-Benchmark: Tibetan Tsawa test set},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Tsawa-Benchmark}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/). The classical commentaries come from the public [WeBuddhist](https://github.com/WeBuddhist) repositories.

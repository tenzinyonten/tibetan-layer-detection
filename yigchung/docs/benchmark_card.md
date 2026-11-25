---
license: mit
pretty_name: Bo-Yigchung-Benchmark
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
- config_name: windowed_w8192_s4914
  data_files:
  - split: test
    path: windowed_w8192_s4914/test-*
---

# Bo-Yigchung-Benchmark

The public test set for Yigchung, the small-print interlinear note written under or beside the main line, in Classical Tibetan books: 77 books, 4,059 spans, 1,189 windows. Score a model on it and compare with others on the same books. The training and validation data are not published.

## Format

- Config `default`: one row per book. Columns: `book_id`, `split`, `source`, `text`, `spans` (list of `{id, start, end}`).
- Offsets are character offsets, inclusive on both ends: `text[start:end + 1]`.
- Config `windowed_w8192_s4914`: tokenized windows for `jhu-clsp/mmBERT-base`, one row per window. 8,192 tokens per window (2 special tokens and 8,190 content tokens), step 4,914 (overlap 3,276), last window of a book end-aligned. Columns: `book_id`, `split`, `window_index`, `kind`, `token_start`, `token_end`, `char_start`, `char_end`, `own_start`, `own_end`, `input_ids`, `attention_mask`, `labels`, `source`. Labels are 0 `O`, 1 `B-YIGCHUNG`, 2 `I-YIGCHUNG`; `-100` marks the two special tokens of each window. Every content token of every window carries its label, including tokens that an earlier window also covers; `own_start` and `own_end` give the token range of the window.
- The text has no ༼ ༽ bracket marks.

## The test set

One split, `test`, made of whole books (a book is never split across train and test).

| Books | Spans | Windows | Text characters |
|---:|---:|---:|---:|
| 77 | 4,059 | 1,189 | 8,470,389 |

| Source | Books | Spans | What it is |
|---|---:|---:|---|
| reviewed_opf | 24 | 848 | Reviewed OpenPecha books |
| lotsawa | 47 | 61 | Short texts from Lotsawa House |
| bdrc | 5 | 1,570 | BDRC volumes |
| nalanda | 1 | 1,580 | Nalanda Vinaya volume |

## Scoring

Span precision, recall and F1 at IoU >= 0.5, one-to-one matching, with Viterbi decoding of the whole book. The scoring code is in the [GitHub repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/yigchung). Score the test set once, after you have chosen your model on your own validation data.

## Annotation

Spans were annotated through OpenPecha. The published spans are the original annotations; the later annotator review of the five BDRC books is not applied to this dataset.

## Citation

BDRC Yigchung layer detection benchmark (Bo-Yigchung-Benchmark), Buddhist Digital Resource Center, 2026.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

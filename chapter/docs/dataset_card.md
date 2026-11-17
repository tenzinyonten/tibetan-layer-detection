---
pretty_name: Bo-Chapter-Dataset
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
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Bo-Chapter-Dataset

Binary token classification data for the Chapter layer (ལེའུ་, chapter and section headings) of
Tibetan Buddhist books, in BIO labels over 8,192-token windows. It trains
[`BDRC/Bo-Chapter-Detection`](https://huggingface.co/BDRC/Bo-Chapter-Detection). It has 378 books and
11,070 windows.

## Format

| item | value |
|---|---|
| tokenizer | `jhu-clsp/mmBERT-base` (fast, with offsets) |
| labels | `O` = 0, `B-CHAPTER` = 1, `I-CHAPTER` = 2; CLS, SEP and padding = -100 |
| label rule | token-start rule (`label_tokens` in `common/build_tsawa_dataset.py`) |
| columns | `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct` |

**Window geometry.** A window holds 8,192 tokens (8,190 content tokens plus CLS and SEP) and starts
every 5,120 tokens, so consecutive windows overlap by 3,070 content tokens. Here "stride" means the
step between window starts; the Hugging Face tokenizer argument `stride` means the overlap, so the
equivalent value there is 3,070. Tokens in the overlap are labelled in both windows.

**Offsets.** `token_start` and `token_end` give the window's token range in the book, and
`char_start` and `char_end` its character range. `source_batch` is `old` or `new` (the two annotation
batches). Scoring code in the GitHub repository uses character offsets with an inclusive end.

## Splits

Split by book (one book in one split only), 83 / 8.5 / 8.5 by window count.

| split | books | old / new books | windows | Chapter spans | spans per window | old-batch windows |
|---|---|---|---|---|---|---|
| train | 308 | 114 / 194 | 9,165 | 2,509 | 0.274 | 49.3% |
| validation | 34 | 13 / 21 | 965 | 275 | 0.285 | 49.2% |
| test | 36 | 16 / 20 | 940 | 258 | 0.274 | 49.3% |

Train tokens: 74,840,009 `O`, 3,688 `B-CHAPTER`, 129,338 `I-CHAPTER` (windows overlap, so a token
is counted once per window it appears in). Chapter is about 0.18% of tokens. About a third of all
spans are shorter than 30 characters.

## Data source and preparation

Source texts are BDRC books in OpenPecha `.opf` format, from two annotation batches. 394 books
have a Chapter layer; books with none are not in the dataset.

| batch | books with a Chapter layer |
|---|---|
| old | 159 |
| new | 235 |
| total | 394 (3,252 raw spans) |

Steps, in order:

1. **Clean the spans** (`chapter/src/clean_chapter_spans.py`). Cut edges inside a syllable are walked to the nearest boundary within 3 characters; spans stop before the trailing shad. Spans separated only by shad, tsheg or spaces (never a newline) are merged: 5 merges, since each line is its own heading. Short spans are kept: 306 raw spans are under 15 characters, such as `ལེའུ་བརྒྱད་པ`. Result: 3,042 spans in 378 books.
2. **Exclude books** (same script, verdicts in `chapter_book_verdicts.csv`). 16 books are left out: one with unfixable edges, and 15 old-batch books whose offsets drift partway through (the first spans are right, later ones are cut mid-sentence; a constant shift does not recover them). A heading-shape check finds the drifted books.
3. **Split by book** (`chapter/src/prepare_chapter_split.py`). A greedy multi-objective assignment, best of 14 seeds, balances window share, Chapter spans per window, share of short spans and old-batch share across the three splits. Hard constraints: one book in one split, at least 30 books in validation and in test.
4. **Tokenize and label** (`chapter/src/build_chapter_dataset.py`). Windows of 8,192 tokens, stride 5,120, token-start labelling rule. No deduplication of text between books.

## Annotation

Chapter spans come from the `Chapter` layer of the OpenPecha books, in the old and the new batch.
On top of the raw annotations, only the cleaning in steps 1 and 2 was applied; no span was
relabelled by hand.

## Known limits

- Most books have very few Chapter spans (216 of 378 kept books have one or two), and some may be under-annotated.
- The test split has 258 spans in 36 books, so scores are noisy.
- Text shared between different books is not grouped, so the same passage can appear in more than one split.
- Chapter spans are sparse (0.18% of tokens), so class weights are needed in training.
- Data rights: the source texts and annotations come from BDRC and OpenPecha. Reuse of the source texts is subject to BDRC's terms.

## Citation

```bibtex
@misc{bo_chapter_dataset,
  title  = {Bo-Chapter-Dataset: Tibetan chapter-heading token classification data},
  author = {Yonten, Tenzin},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Chapter-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

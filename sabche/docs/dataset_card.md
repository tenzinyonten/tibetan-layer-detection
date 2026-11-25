---
license: mit
pretty_name: Bo-Sabche-Dataset
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
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# Bo-Sabche-Dataset

Training data for finding **Sabche** (ས་བཅད་, outline heading) spans in Tibetan Buddhist texts.
It is the training data of [`BDRC/Bo-Sabche-Detection`](https://huggingface.co/BDRC/Bo-Sabche-Detection). The test books are published separately as [`BDRC/Bo-Sabche-Benchmark`](https://huggingface.co/datasets/BDRC/Bo-Sabche-Benchmark).

A Sabche span is the heading phrase itself (for example a numbered outline line), stopping just
before the closing shad, not the section under it.

## Format

| item | value |
|---|---|
| Tokenizer | `jhu-clsp/mmBERT-base` (fast, offsets) |
| Window / stride | 8,192 tokens (8,190 content + CLS/SEP) / 5,120 between window starts |
| Labels | `O`=0, `B-SABCHE`=1, `I-SABCHE`=2; CLS/SEP/pad = -100 |
| Columns | `input_ids`, `attention_mask`, `labels`, `token_start`, `token_end`, `char_start`, `char_end`, `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct` |
| Offsets | `char_start`/`char_end` are start-inclusive, end-exclusive (`text[start:end]`). The scoring code in the GitHub repository uses an inclusive end instead, so convert with `end - 1`. |
| Metric (for training) | IoU >= 0.5, greedy one-to-one, inclusive offsets |

Consecutive windows overlap by 3,070 content tokens; tokens in the overlap are labelled in both
windows (unlike Quotation's dataset, which owns each token to one window only).

## Splits

Splits are **book-level**, grouped so that no book sharing a long heading verbatim with another
book, or belonging to the same source outline group, crosses a split.

| Split | Books | Windows | Spans |
|---|---:|---:|---:|
| Train | 261 | 8,886 (82.6%) | 35,122 |
| Validation | 33 | 911 (8.5%) | 3,091 |

## Data source and preparation

**Source.** Two annotation batches, told apart by `pecha_id` prefix and carried as `source_batch`:
an older batch (`P`-prefixed, 266 books, 18,845 raw spans) and a newer batch (`I`-prefixed, 273
books, 24,378 raw spans) with denser, more consistent heading annotation.

**Cleaning** (`sabche/src/clean_sabche_spans.py`), in order:

1. **Re-align drifted offsets.** 30 old-batch books had spans shifted from the text by a
   piecewise-constant offset. Found and corrected with a Viterbi alignment over length-matched
   ordinal-heading anchors (4,173 spans moved).
2. **Snap off-by-one edges.** Spans within 3 characters of a syllable boundary were walked to it
   (1,711 spans across both batches); the rest were left as is.
3. **Drop unverifiable spans.** 227 re-aligned spans that were still not heading-like after the
   shift were dropped (their text is labeled `O`).
4. **Exclude 10 books**: offsets that could not be recovered (3), no heading-like spans (2), or
   outline/table-of-contents annotated but body headings not (5, titled-heading recall < 0.2).

42,292 spans are active after cleaning (18,456 old batch, 23,836 new batch). No spans were merged:
reading samples confirmed that adjacent one-per-line outline headings are genuinely separate
spans, not one heading split across lines.

**Split** (`sabche/src/prepare_sabche_split.py`): books are grouped before splitting, by the
project's shared document-split groups plus a Sabche-specific link (two books sharing at least
10% of a book's long-span characters, or at least 10 shared spans outright). Groups are then
placed to hit 83/8.5/8.5 by window count, keeping every book that was already frozen into
validation or test by the shared split.

**Tokenization and labelling** (`sabche/src/build_sabche_dataset.py`): 8,192-token sliding
windows, stride 5,120, `jhu-clsp/mmBERT-base` tokenizer, token-start labelling rule.

## Annotation

Headings were annotated through OpenPecha with support from the Tsadra Foundation, over the two
batches described above. No independent review pass has been run on top of the original
annotations (unlike the project's Quotation and Chapter layers); the cleaning steps above correct
offset and formatting problems in the existing annotations, not their coverage.


## Citation

```bibtex
@misc{bo_sabche_dataset,
  title  = {Bo-Sabche-Dataset: Tibetan outline-heading (sabche) token classification data},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Sabche-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

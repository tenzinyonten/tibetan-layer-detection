---
license: mit
pretty_name: Bo-Yigchung-Dataset
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- layer-detection
- yigchung
dataset_info:
- config_name: default
  features:
  - name: book_id
    dtype: string
  - name: split
    dtype: string
  - name: source
    dtype: string
  - name: text
    dtype: string
  - name: spans
    list:
    - name: id
      dtype: string
    - name: start
      dtype: int32
    - name: end
      dtype: int32
  splits:
  - name: train
    num_bytes: 179889913
    num_examples: 601
  - name: validation
    num_bytes: 22042668
    num_examples: 84
  dataset_size: 201932581
- config_name: windowed_w8192_s4914
  features:
  - name: book_id
    dtype: string
  - name: split
    dtype: string
  - name: window_index
    dtype: int32
  - name: kind
    dtype: string
  - name: token_start
    dtype: int32
  - name: token_end
    dtype: int32
  - name: char_start
    dtype: int32
  - name: char_end
    dtype: int32
  - name: own_start
    dtype: int32
  - name: own_end
    dtype: int32
  - name: input_ids
    list: int32
  - name: attention_mask
    list: int8
  - name: labels
    list: int32
  - name: source
    dtype: string
  splits:
  - name: train
    num_bytes: 608462624
    num_examples: 8574
  - name: validation
    num_bytes: 73683141
    num_examples: 1041
  dataset_size: 682145765
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
- config_name: windowed_w8192_s4914
  data_files:
  - split: train
    path: windowed_w8192_s4914/train-*
  - split: validation
    path: windowed_w8192_s4914/validation-*
---

# Bo-Yigchung-Dataset

685 Classical Tibetan books with character-offset spans marking Yigchung, the small-print interlinear note written under or beside the main line. It is the data of [BDRC/Bo-Yigchung-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Detection); see that card for how the model was trained on it. The 77 test books are published separately as [BDRC/Bo-Yigchung-Benchmark](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Benchmark).

## Format

- Config `default`: one row per book. Columns: `book_id`, `split`, `source`, `text`, `spans` (list of `{id, start, end}`).
- Offsets are character offsets, inclusive on both ends: `text[start:end + 1]`.
- Config `windowed_w8192_s4914`: tokenized windows for `jhu-clsp/mmBERT-base`, one row per window. 8,192 tokens per window (2 special tokens and 8,190 content tokens), step 4,914 (overlap 3,276), last window of a book end-aligned. Columns: `book_id`, `split`, `window_index`, `kind`, `token_start`, `token_end`, `char_start`, `char_end`, `own_start`, `own_end`, `input_ids`, `attention_mask`, `labels`, `source`. Labels are 0 `O`, 1 `B-YIGCHUNG`, 2 `I-YIGCHUNG`; `-100` marks the two special tokens of each window. Every content token of every window carries its label, including tokens that an earlier window also covers; `own_start` and `own_end` give the token range of the window.
- The text has no ༼ ༽ bracket marks.

## Splits

Book-level: a book is in one split only.

| Split | Books | Spans | Windows | Text characters |
|---|---:|---:|---:|---:|
| train | 601 | 22,388 | 8,574 | 60,977,652 |
| validation | 84 | 2,637 | 1,041 | 7,456,195 |
| total | 685 | 25,025 | 9,615 | 68,433,847 |

Books by source:

| Source | Train | Validation | Total |
|---|---:|---:|---:|
| reviewed_opf | 180 | 24 | 204 |
| lotsawa | 374 | 48 | 422 |
| bdrc | 35 | 4 | 39 |
| nalanda | 12 | 8 | 20 |

## Data source and preparation

| Source | Books | What it is |
|---|---:|---|
| reviewed_opf | 204 | Reviewed OpenPecha books |
| lotsawa | 422 | Short texts from Lotsawa House |
| nalanda | 20 | Nalanda Vinaya volumes |
| bdrc | 39 | BDRC volumes |

Steps: combine the four sources, remove duplicate books and passages from the Lotsawa, BDRC and reviewed_opf train sets only, split by book, tokenize with `jhu-clsp/mmBERT-base` and cut windows. The scripts that did this are not in the [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection) repository.

## Annotation

Spans were annotated through OpenPecha. The `reviewed_opf` books come from the reviewed OpenPecha set; the other sources keep their own annotations. The published spans are the original annotations: the later annotator review of five BDRC books is not applied to this dataset.


## Citation

BDRC Yigchung layer detection dataset (Bo-Yigchung-Dataset), Buddhist Digital Resource Center, 2026.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

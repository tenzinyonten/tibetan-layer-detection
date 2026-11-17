---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Chapter-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- chapter
---

# Bo-Chapter-Detection

Token classifier that finds **Chapter** (ལེའུ་, chapter and section heading) spans in Tibetan
Buddhist texts. Fine-tuned from `jhu-clsp/mmBERT-base`. One model per layer: this model detects
Chapter only.

A Chapter span is the heading line that opens a division of a book. It names the division and is
not the text under it. Headings include numbered chapter lines such as `ལེའུ་བཅུ་བཞི་པ`, titles of
separate works inside a collected volume, plain section titles, and front matter such as `དཀར་ཆག`
(table of contents) or `དཔེ་སྐྲུན་གསལ་བཤད` (publisher's note). Spans are short (median 44
characters) and sparse: 216 of the 378 books have only one or two.

## Labels

`O` = 0, `B-CHAPTER` = 1, `I-CHAPTER` = 2 (BIO).

## Training data

[`BDRC/Bo-Chapter-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Chapter-Dataset): 8,192-token
windows, stride 5,120, mmBERT tokenizer. Split by book.

| split | books | windows | Chapter spans |
|---|---|---|---|
| train | 308 | 9,165 | 2,509 |
| validation | 34 | 965 | 275 |
| test | 36 | 940 | 258 |

Chapter is about 0.18% of tokens. Spans were cleaned before tokenizing and 16 books with drifting
offsets were left out; see the dataset card.

## Training

| item | value |
|---|---|
| base model | `jhu-clsp/mmBERT-base` |
| epochs | 8 planned |
| learning rate | 1e-5 |
| batch size, gradient accumulation | 8, 1 |
| weight decay | 0.01 |
| gradient clip | 0.3 |
| warmup | 549 steps |
| precision | bf16 |
| seed | 42 |
| loss, class weights | cross-entropy, square-root inverse frequency (O 1.0, B 142.5, I 24.1) |
| early stopping | on validation IoU 0.5 F1, evaluated every 286 steps (4 times per epoch) |
| best checkpoint | highest validation IoU 0.5 F1 (`load_best_model_at_end`) |
| input length | 8,192 tokens |

```bash
python common/train_layer.py --label-name CHAPTER --weight-scheme sqrt_inv \
    --epochs 8 --evals-per-epoch 4 --output-dir /workspace/runs/chapter_bio_sqrt
```

The test split was not scored during training.

## Results

Metric: span F1 at IoU 0.5 in character offsets, greedy one-to-one matching, Viterbi decoding with
a break penalty of 4.0. Scores are whole-book: each book is decoded once and duplicates from
overlapping windows are removed. Machine-readable test metrics are in `training/test_metrics.json`.

| split | F1 | precision | recall |
|---|---|---|---|
| test, 36 books (258 gold spans) | 0.829 | 0.768 | 0.899 |
| test, old batch (16 books) | 0.863 | 0.853 | 0.873 |
| test, new batch (20 books) | 0.800 | 0.705 | 0.924 |

On the test split the model predicts 302 spans for the 258 gold spans. The test split is small, so
scores are noisy.

## Usage

The repo includes the tokenizer files, copied unchanged from `jhu-clsp/mmBERT-base`, so it loads
on its own.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("BDRC/Bo-Chapter-Detection")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Chapter-Detection")
```

Decode with Viterbi rather than argmax: only legal BIO transitions (I may not follow O), and a
penalty for every span exit. A break penalty of 4.0 was used for the test scores. Long texts need
8,192-token windows with stride 5,120. The full pipeline is `detect/` in
[tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection).

## Limits

- The model over-proposes: it predicts about 17% more spans than the gold has, so precision (0.768) is lower than recall (0.899). On the new batch precision is 0.705.
- Most training books have only one or two Chapter spans, and some may be under-annotated, so some predictions counted as false positives may be real headings.
- About a third of the spans are under 30 characters (bare chapter numbers and short stubs), so a small offset error can fall below the IoU 0.5 threshold.
- The labels come from two annotation batches (old and new) that score differently (F1 0.863 and 0.800), so expect scores to depend on the source of the text.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

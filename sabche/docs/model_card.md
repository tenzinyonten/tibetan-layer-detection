---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Sabche-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- sabche
---

# Bo-Sabche-Detection

Token classifier that finds **Sabche** (ས་བཅད་, outline heading) spans in Tibetan
Buddhist texts. Fine-tuned from `jhu-clsp/mmBERT-base`. One model per layer:
this model detects Sabche only.

A Sabche span is the heading phrase itself (e.g. `གཉིས་པ་༼ངེས་ཚིག་༽ནི`,
or a numbered outline line), stopping just before the closing shad. It is not
the section under the heading.

## Labels

`O` = 0, `B-SABCHE` = 1, `I-SABCHE` = 2 (BIO).

## Training data

[`BDRC/Bo-Sabche-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Sabche-Dataset)
(83/8.5/8.5 split by window count): 8,192-token windows, stride 5,120, mmBERT tokenizer.
261 / 33 / 29 books and 8,886 / 911 / 963 windows for train / validation / test. Spans were
cleaned before tokenizing (drifted offsets re-aligned in 30 old-batch books, off-by-one
edges snapped, 10 unusable books excluded). See the dataset card.

## Training

| | |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 8 |
| Learning rate | 1e-5 |
| Batch size / gradient accumulation | 8 / 1 (effective 8) |
| Weight decay | 0.01 |
| Gradient clip | 0.3 |
| Warmup | 6% (532 of ~8,888 steps) |
| Precision | bf16 |
| Seed | 42 |
| Loss / class weights | weighted token cross-entropy; inverse-frequency BIO weights from this train set (O 1, B-SABCHE 1298, I-SABCHE 25) |
| Early stopping | patience 3, evaluated 4 times per epoch |
| Best checkpoint | highest validation IoU@0.5 F1 |
| Input length | 8,192 tokens, stride 5,120 |

```bash
python common/train_layer.py --label-name SABCHE --scheme bio --epochs 8 --output-dir runs/sabche
```

## Results

Metric: span F1 at IoU >= 0.5 in character offsets, greedy one-to-one matching, Viterbi decoding
with a break penalty of 4.0 (same as the tsawa and chapter models). Scores are whole-book: each
book is decoded once and duplicates from overlapping windows are removed. Machine-readable test
metrics are in `training/test_metrics.json`.

| test split | F1 | precision | recall |
|---|---|---|---|
| all 29 books (4,079 gold spans) | 0.962 | 0.954 | 0.970 |
| old batch (11 books) | 0.958 | 0.940 | 0.977 |
| new batch (18 books) | 0.964 | 0.964 | 0.965 |

The model predicts 4,144 spans for the 4,079 gold spans.

These numbers are the raw Viterbi decode, before the per-layer post-processing rules applied at
inference time (see Limits).

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("BDRC/Bo-Sabche-Detection")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Sabche-Detection")
```

Decode with Viterbi rather than argmax: only legal BIO transitions (I may not
follow O), and a penalty for every span exit, which discourages splitting one
heading into pieces. A break penalty of 4.0 was used for the test scores. Long texts need
8,192-token windows with stride 5,120. The full pipeline, including the post-processing rules
below, is `detect/` in [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection).

## Limits

- Production inference (`detect/layer_config.py`) turns off the default tsheg-extend and
  small-gap-merge rules for this layer (they cost true positives on the Sabche test set) and
  turns on a shad rule that drops short spans unless a shad follows. That shad rule was tuned on
  this same test split and has not yet been re-checked on validation.
- Some training books are only partly annotated: titled-heading recall is between 0.2 and 0.5 in
  21 of the kept books, so some real headings are labeled `O`. Bare `Nth-པ་ནི` headings with no
  title are inconsistently annotated (about 29% of the time).
- 30 old-batch books had drifted span offsets, re-aligned with an anchor heuristic; 227 spans in
  those books could not be verified after re-alignment and were dropped, so their true headings
  are labeled `O` in training.
- 14.2% of test spans' raw text also appears in train, but most of that is stock heading phrasing
  (numbered templates); real document-level leakage is about 6.8%, concentrated in one group of
  books (a shared lamrim outline). The split was kept as is rather than reworked.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

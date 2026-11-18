---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Yigchung-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- yigchung
---

# Bo-Yigchung-Detection

Token classifier for Yigchung, the small-print interlinear note written under or beside the main line of a Tibetan pecha. It reads a flat text string and returns character spans. Fine-tune of [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (`ModernBertForTokenClassification`, 22 layers, 8,192 positions). Weights are the best validation checkpoint (step 2412, epoch 2.25).

## Labels

| id | label |
|---:|---|
| 0 | `O` |
| 1 | `B-YIGCHUNG` |
| 2 | `I-YIGCHUNG` |

## Training data

[BDRC/Bo-Yigchung-Dataset](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Dataset), config `windowed_w8192_s4914`. Book-level split.

| Split | Books | Spans | Windows |
|---|---:|---:|---:|
| train | 601 | 22,388 | 8,574 |
| validation | 84 | 2,637 | 1,041 |
| test | 77 | 4,059 | 1,189 |

The model was not trained on exactly this text. It was trained on an earlier version of the same data (`karma689/yigchung_detection`, tag `v2.0`, no longer available), which had 8,585 train and 1,043 validation windows against 8,574 and 1,041 here; the test split has 1,189 windows in both. In that version the BDRC test books still carried ༼ ༽ bracket marks around the notes; the published text has none. Whether its training books carried them too, and what the 11 extra train windows contain, cannot be checked. Scores below are measured on the published text.

## Training

| Setting | Value |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 15 planned; early-stopped at step 6700 (epoch 6.24) |
| Learning rate | 1e-5 |
| Batch size, accumulation | 2 per device, 4 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Warmup | 20% of a 10-epoch reference |
| Precision | bf16 |
| Seed | 42 |
| Loss, class weights | weighted token cross-entropy; O 0.07, B 5.0, I 5.0; `-100` ignored |
| Early stopping | on validation span F1 |
| Best checkpoint | highest validation span F1 (IoU >= 0.5, in-loop break penalty 2.0): step 2412 |
| Input length | 8,192 tokens (8,190 content), step 4,914 |

## Results

Scored on the 77 test books of [BDRC/Bo-Yigchung-Dataset](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Dataset) (published text, no bracket marks) with `yigchung/src/score_yigchung_benchmark.py` in [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection). Metric: greedy one-to-one matching of predicted and gold character spans at IoU >= 0.5, predictions trimmed of edge whitespace, micro-averaged over spans.

Two gold versions: the published gold ("published", 4,059 spans), and the same books with annotator corrections ("corrected", 3,793 spans: annotator review of the 5 BDRC books, annotator patterns and a head-mark rule on the others; built by `yigchung/src/build_yigchung_benchmark_after.py`).

| Gold | Gold spans | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| published | 4,059 | 0.518 | 0.432 | 0.471 |
| corrected | 3,793 | 0.519 | 0.463 | 0.489 |

By source, on the published gold:

| Source | Books | Gold spans | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| bdrc | 5 | 1,570 | 0.480 | 0.500 | 0.490 |
| nalanda | 1 | 1,580 | 0.475 | 0.318 | 0.381 |
| reviewed_opf | 24 | 848 | 0.662 | 0.483 | 0.559 |
| lotsawa | 47 | 61 | 0.818 | 0.885 | 0.850 |

Caveats: the test set is dominated by two sources (nalanda and bdrc hold 78% of the gold spans), so the micro F1 follows those six books. The Lotsawa score rests on 61 spans.

## Usage

Cut windows of 8,192 tokens (8,190 content tokens) with step 4,914, take each token's logits from the first window that covers it, then Viterbi-decode the whole text once with break penalty 8 (the value chosen on validation). Leaving a span costs the break penalty and `O` to `I` is not allowed. Spans are inclusive: `text[start:end + 1]`. The `detect/` folder of the GitHub repository does this end to end.

The tokenizer is the one of the base model, `jhu-clsp/mmBERT-base`. Load it from there: the tokenizer file stored in this repo was written by a newer version of `transformers` and does not load with version 4.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Yigchung-Detection")
```

## Limits

- The model sees the text only; Yigchung is a typographic layer and many spans are not lexically obvious.
- Boundary errors of a few syllables fail the IoU 0.5 match.
- Scores differ much by source: 0.85 on the Lotsawa books, 0.38 on the one Nalanda volume, 0.49 on the five BDRC volumes.
- The training text may differ from the published text (see Training data).

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

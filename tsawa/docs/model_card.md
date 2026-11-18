---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Tsawa-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- tsawa
---

# Bo-Tsawa-Detection

Token classifier that finds **Tsawa** (རྩ་བ་, root text) spans in Tibetan Buddhist commentaries. Fine-tuned from `jhu-clsp/mmBERT-base`. One model per layer: this model detects Tsawa only. Its data is [`BDRC/Bo-Tsawa-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Tsawa-Dataset).

A commentary lifts a verse or a piece of prose out of the root text, quotes it, and then explains it. That quoted piece is the Tsawa. Quotations from other works look almost the same but are not Tsawa, and telling the two apart is the hard part.

## Labels

`O` = 0, `B-TSAWA` = 1, `I-TSAWA` = 2 (BIO). Special tokens, padding and masked tokens are -100. A span includes its closing shad.

## Training data

[`BDRC/Bo-Tsawa-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Tsawa-Dataset). 8,192-token windows, stride 5,120, mmBERT tokenizer.

| split | books | windows | spans |
|---|---:|---:|---:|
| train | 92 (53 existing, 26 re-annotated, 13 classical commentaries) | 4,850 | 16,716 |
| validation | 18 | 456 | 1,582 |
| test (benchmark) | 22 | 639 | 1,976 |
| classical_validation | 6 | 642 | 3,652 |
| classical_test | 6 | 634 | 4,397 |

The 13 + 6 + 6 classical commentaries (25 books, 15,876 spans) come from public WeBuddhist repositories. They are not hand-annotated: the Tsawa labels come from the WeBuddhist root-verse links (each root verse a commentary links to is labelled Tsawa), so they are only as good as those links. The other books are BDRC texts with human-annotated spans. See the dataset card for the per-book table.

The model was selected on `validation` only.

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 15 planned; stopped early (best checkpoint at epoch 9.45) |
| Learning rate | 1e-5 |
| Batch size | 8 (no gradient accumulation) |
| Weight decay | 0.01 |
| Gradient clip | 0.3 |
| Warmup | 6% of steps |
| Precision | bf16, gradient checkpointing |
| Seed | 42 |
| Loss | token cross-entropy, inverse-frequency class weights (O 1.0, B 1389.2, I 10.9) |
| Early stopping | patience 3, evaluated 4 times per epoch |
| Best checkpoint | chosen on validation IoU 0.5 F1 (in-training break penalty 5.0) |
| Input length | 8,192 tokens per window |


Command: `common/train_layer.py --label-name TSAWA --dataset Yontenn/formatting-tsawa-v10 --scheme bio --weight-scheme inv --break-penalty 5.0 --epochs 15 --patience 3 --evals-per-epoch 4 --lr 1e-5 --batch-size 8 --seed 42`

## Results

Metric: span precision, recall and F1 at IoU >= 0.5 in token space, greedy one-to-one matching, Viterbi decoding with break penalty 4.0 (the test split was scored with this value; the best checkpoint was picked with 5.0 on validation).

| split | F1 | precision | recall |
|---|---|---|---|
| test, 22 benchmark books (3,180 gold spans, counted per window) | 0.842 | 0.811 | 0.876 |
| test, with the quotation filter (see Usage) | 0.890 | 0.916 | 0.865 |
| validation (v11 labels) | 0.789 | 0.726 | 0.863 |
| classical_test (6 commentaries) | 0.959 | 0.941 | 0.978 |

Caveats:
- The 22 test books hold 1,976 distinct Tsawa spans. Test windows overlap, so a span can sit in two windows; the scorer counts it in each, which gives 3,180 gold spans (3,148 span starts plus 32 windows that begin inside a span). The scores are therefore per window, not per whole book.
- `classical_test` shares root verses with training commentaries (about 99% of its verses also occur in a train commentary). It measures recall of seen root text, not generalisation. Judge the model on the 22-book test.
- Argmax decoding is far worse (test F1 0.49). Use Viterbi.

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("BDRC/Bo-Tsawa-Detection")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Tsawa-Detection")
```

Decode with Viterbi (only legal BIO transitions, break penalty 4.0). Long texts need 8,192-token windows with stride 5,120.

Optional quotation filter: run `BDRC/Bo-Quotation-Detection` on the same text and drop a predicted Tsawa span when at least 50% of it lies inside predicted quotation spans. This raised test F1 from 0.842 to 0.890 (false positives 648 to 252, about 35 true spans lost). Only tested on this test set.

## Limits

- Over-predicts a little: 3,434 predicted spans for 3,180 gold spans (both counted per window) at break penalty 4.0.
- Quotations from other works are the main error source; the filter above helps but is not part of the model.
- The classical training data is labelled from the WeBuddhist root-verse links, not by hand, so its labels carry any errors in those links.
- Interlinear commentaries annotated a word at a time are outside the task (5 such books were set aside from training).
- Stricter overlap is weaker: F1 0.814 at IoU 0.9, 0.637 at exact match.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

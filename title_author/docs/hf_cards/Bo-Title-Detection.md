---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Title-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- title
---


# Bo-Title-Detection

Token classifier that finds the **title** of a Tibetan work. Fine-tuned from `jhu-clsp/mmBERT-base`. One model per field: this model detects the title only. Its pair is `BDRC/Bo-Author-Detection`.

It reads the first 3,000 and the last 3,000 characters of a work (a work of 6,000 characters or fewer is read whole) and marks the title span. One span per work is kept: the first one.

## Labels

`O` = 0, `B-TITLE` = 1, `I-TITLE` = 2 (BIO). The author in the text is labelled `O`, not masked.

## Training data

Trained on [`BDRC/Bo-Title-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Title-Dataset).

| Split | Works |
|---|---|
| Train | 32,074 (24,263 labelled by annotators, 7,811 labelled automatically) |
| Validation | 2,494 (annotators) |
| Test | 2,384 (annotators) |

Each example is one work: the first 3,000 and the last 3,000 characters of its text, or the whole text if it is 6,000 characters or fewer. Only the title is labelled; the author counts as "other" (`O`). The automatic labels are used in training only. See the dataset card for the sources and how the data was prepared.

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 8 (all ran) |
| Learning rate | 1e-5 |
| Batch size | 8 (no gradient accumulation) |
| Weight decay | 0.01 |
| Gradient clip | 0.3 |
| Warmup | 6% of steps |
| Precision | bf16 |
| Seed | 42 |
| Loss | token cross-entropy, class weights `sqrt_inv` |
| Early stopping | patience 3 epochs, evaluated twice per epoch |
| Best checkpoint | chosen on validation IoU 0.5 F1 |
| Input length | one sequence per work, `[CLS] head [SEP] tail [SEP]`, up to 8,192 tokens |

Command: `common/train_layer.py --label-name TITLE --scheme bio --dataset Yontenn/title-only-training --weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test`

## Results

Metric: per-work span precision, recall and F1 at IoU >= 0.5 against the gold span in the same text block, one predicted span per work (the first), Viterbi decoding with break penalty 6 (chosen on validation; the test split was scored once). Test set: 2,384 works (the full test split).

| test split | F1 | precision | recall |
|---|---|---|---|
| all 2,384 works | 0.845 | 0.848 | 0.842 |

On the same works the joint model `Yontenn/mmbert-title-author-run2` scores F1 0.835, and the Gemini skill (gemini-3.8-flash, GEPA prompt, rule layer) 0.787. The recall difference to the joint model is +0.010 (95% range -0.002 to +0.021), so it is not clearly different.

## Usage

The weights are in the `best/` subfolder of the repo. The repo has no tokenizer files; use the base tokenizer.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Title-Detection", subfolder="best")
```

Decode with Viterbi rather than argmax (only legal BIO transitions, penalty 6 for a span exit), then keep the first span. The full pipeline, with the cropping and decoding, is `detect/` and `title_author/src/ta_predict.py` in [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection).

## Limits

- One title per work; other mentions of the title are unlabelled, and gold span edges vary between annotators, so IoU 0.5 has a ceiling.
- The model is overconfident: a token probability is not an accuracy. Review-flag cutoffs are in `title_author/README.md`.
- Trained on single works, not whole books. On whole books use `detect/`.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

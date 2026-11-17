---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Author-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- author
---


# Bo-Author-Detection

Token classifier that finds the **author** of a Tibetan work. Fine-tuned from `jhu-clsp/mmBERT-base`. One model per field: this model detects the author only. Its pair is `BDRC/Bo-Title-Detection`.

It reads the first 3,000 and the last 3,000 characters of a work (a work of 6,000 characters or fewer is read whole) and marks the author span. One span per work is kept: the last one.

## Labels

`O` = 0, `B-AUTHOR` = 1, `I-AUTHOR` = 2 (BIO). The title in the text is labelled `O`, not masked.

## Training data

[`BDRC/Bo-Author-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Author-Dataset) (revision `a9609000`, the last commit that changed the data files). It has the same rows, windows and splits as the joint data `Yontenn/title-author-training` (revision `1c3226b3`); only the labels changed. 32,074 / 2,494 / 2,384 works for train / validation / test, one sequence per work (`[CLS] head [SEP] tail [SEP]`). The train split includes 7,811 works with automatically found spans (BDRC v3 outlines); validation and test are annotator spans only. See the dataset card.

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 8 planned; early stopping ended the run at epoch 5.5 |
| Learning rate | 1e-5 |
| Batch size | 8 (no gradient accumulation) |
| Weight decay | 0.01 |
| Gradient clip | 0.3 |
| Warmup | 6% of steps |
| Precision | bf16 |
| Seed | 42 |
| Loss | token cross-entropy, class weights `sqrt_inv` |
| Early stopping | patience 3 epochs, evaluated twice per epoch |
| Best checkpoint | step 10,020 (epoch 2.5), chosen on validation IoU 0.5 F1 |
| Input length | one sequence per work, `[CLS] head [SEP] tail [SEP]`, up to 8,192 tokens |

Command: `common/train_layer.py --label-name AUTHOR --scheme bio --dataset Yontenn/author-only-training --weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test`

## Results

Metric: per-work span precision, recall and F1 at IoU >= 0.5 against the gold span in the same text block, one predicted span per work (the last), Viterbi decoding with break penalty 6 (chosen on validation; the test split was scored once). Test set: 2,384 works (the full test split).

| test split | F1 | precision | recall |
|---|---|---|---|
| all 2,384 works | 0.738 | 0.742 | 0.734 |

On the same works the joint model `Yontenn/mmbert-title-author-run2` scores F1 0.727, and the Gemini skill (gemini-3.8-flash, GEPA prompt, rule layer) 0.756. The recall difference to the joint model is +0.023 (95% range +0.009 to +0.038).

About 11% of gold authors differ from the author a reader would name (stanza writer, compiler, copyist, requester), so every number has a ceiling.

## Usage

The weights are in the `best/` subfolder of the repo. The repo has no tokenizer files; use the base tokenizer.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Author-Detection", subfolder="best")
```

Decode with Viterbi rather than argmax (only legal BIO transitions, penalty 6 for a span exit), then keep the last span. The full pipeline, with the cropping and decoding, is `detect/` and `title_author/src/ta_predict.py` in [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection).

## Limits

- One author per work; second authors and other mentions are unlabelled. The gold span may or may not include the cue word (`མཛད་པ་པོ།`) and honorifics, so IoU 0.5 is partly a label-convention question.
- The model is overconfident: a token probability is not an accuracy. Review-flag cutoffs are in `title_author/README.md`.
- Trained on single works. On whole OpenPecha books the author is on the title page and "last span" is the wrong rule; see the `--author-span` setting in `detect/`.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

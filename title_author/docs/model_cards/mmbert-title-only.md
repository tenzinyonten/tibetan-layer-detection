---
license: other
language: [bo]
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-base
datasets: [Yontenn/title-only-training]
---
# mmbert-title-only

Finds the **title** of a Tibetan work. One of the two single-entity title/author models (the other is `mmbert-author-only`). Fine-tuned from `jhu-clsp/mmBERT-base`.

**Weights** are in the `best/` subfolder of the repo (load with `subfolder="best"` or point to `.../best`). `results.json` sits beside it.

**Labels:** `O=0, B=1, I=2`. The author in the text is labelled `O`, not masked.

**Input:** one sequence per work, `[CLS] first 3,000 characters [SEP] last 3,000 characters [SEP]`. A work of 6,000 characters or fewer is used whole. Tibetan text only.

**Decoding:** Viterbi with break penalty 6 (chosen on validation), then one span per work: the first span.

## Training

| | |
|---|---|
| Data | [Yontenn/title-only-training](https://huggingface.co/datasets/Yontenn/title-only-training), the title relabelling of [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) |
| Command | `common/train_layer.py --label-name TITLE --scheme bio --dataset Yontenn/title-only-training --weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test` |
| Learning rate, batch size, seed | 1e-5, 8, 42 (trainer defaults) |
| Epochs | all 8 ran; best checkpoint chosen on validation |
| Version | v1, trained 2026-11-16 |

## Results

Cleaned test set, 2,384 works. Share of works whose predicted title has IoU ≥ 0.5 with the gold span in the same block. Penalty chosen on validation, test scored once.

| | Title |
|---|---|
| This model | 0.842 |
| Joint model `mmbert-title-author-run2` | 0.833 |

Paired difference to the joint model: +0.010, 95% range −0.002 to +0.021 (not clearly different).

## Limits

- One title per work. Other mentions of the title are unlabelled.
- Gold span edges vary between annotators, so IoU 0.5 has a ceiling.
- Trained on works cropped to head and tail. Not tested on whole books; use `detect/` post-processing there.
- Overconfident: do not read the token probability as accuracy. Review-flag cutoffs are in `title_author/README.md`.

Code and evaluation: https://github.com/tenzinyonten/tibetan-layer-detection (`title_author/`).

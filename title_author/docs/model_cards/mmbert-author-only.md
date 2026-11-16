---
license: other
language: [bo]
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-base
datasets: [Yontenn/author-only-training]
---
# mmbert-author-only

Finds the **author** of a Tibetan work. One of the two single-entity title/author models (the other is `mmbert-title-only`). Fine-tuned from `jhu-clsp/mmBERT-base`.

**Weights** are in the `best/` subfolder of the repo (load with `subfolder="best"` or point to `.../best`). `results.json` sits beside it.

**Labels:** `O=0, B=1, I=2`. The title in the text is labelled `O`, not masked.

**Input:** one sequence per work, `[CLS] first 3,000 characters [SEP] last 3,000 characters [SEP]`. A work of 6,000 characters or fewer is used whole. Tibetan text only.

**Decoding:** Viterbi with break penalty 6 (chosen on validation), then one span per work: the last span.

## Training

| | |
|---|---|
| Data | [Yontenn/author-only-training](https://huggingface.co/datasets/Yontenn/author-only-training), the author relabelling of [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) |
| Command | `common/train_layer.py --label-name AUTHOR --scheme bio --dataset Yontenn/author-only-training --weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test` |
| Learning rate, batch size, seed | 1e-5, 8, 42 (trainer defaults) |
| Epochs | early stopping at epoch 5.5; best checkpoint at step 10,020 |
| Version | v1, trained 2026-11-16 |

## Results

Cleaned test set, 2,384 works. Share of works whose predicted author has IoU ≥ 0.5 with the gold span in the same block. Penalty chosen on validation, test scored once.

| | Author |
|---|---|
| This model | 0.734 |
| Joint model `mmbert-title-author-run2` | 0.711 |
| Skill (gemini-3.8-flash, GEPA prompt, rule layer) | 0.756 |

Paired difference to the joint model: +0.023, 95% range +0.009 to +0.038.

## Limits

- About 11% of gold authors differ from the author a reader would name (stanza writer, compiler, copyist, requester), so every number has a ceiling.
- The gold span is the one typed by the annotator; the cue word and honorifics are not always inside it.
- Trained on single works. On whole OpenPecha books the author sits on the title page, and "last span" is the wrong rule there; see the `--author-span` setting in `detect/`.
- Overconfident: do not read the token probability as accuracy.

Code and evaluation: https://github.com/tenzinyonten/tibetan-layer-detection (`title_author/`).

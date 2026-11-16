---
license: other
language: [bo]
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-base
datasets: [Yontenn/title-author-training]
---
# mmbert-title-author-run2

One model that finds both the **title** and the **author** of a Tibetan work (5 labels). Fine-tuned from `jhu-clsp/mmBERT-base`. Superseded by the single-entity pair `mmbert-title-only` and `mmbert-author-only`; kept as an experiment.

**Weights** are in the `best/` subfolder of the repo.

**Labels:** `O=0, B-TITLE=1, I-TITLE=2, B-AUTHOR=3, I-AUTHOR=4`.

**Input and decoding:** as in the single-entity models, with Viterbi break penalty 12. Title = first span, author = last span.

## Training data: read this first

Run 2 was trained on an **earlier version** of `Yontenn/title-author-training`: 32,914 works (24,971 annotator works plus 7,943 auto-labelled v3 works), validation 2,595, test 2,443. The repo has since been overwritten with the cleaned, deduplicated version, so the data this model saw is no longer what the repo shows. The single-entity models use the current version.

Training: 8 epochs, class weights `sqrt_inv`, best checkpoint by the mean of title and author F1 on validation. Version: run 2, 2026-11-06.

## Results

| Test set | Title | Author |
|---|---|---|
| Cleaned, 2,384 works (share with IoU ≥ 0.5) | 0.833 | 0.711 |
| Earlier cleaned, 2,389 works (F1, `docs/run2_results.md`) | 0.828 | 0.744 |

The two rows use different test sets and different metrics; do not compare them.

Code and evaluation: https://github.com/tenzinyonten/tibetan-layer-detection (`title_author/`).

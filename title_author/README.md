# Title and author

Find the title and the author of a Tibetan work from the first and last 3,000 characters of its text (a text of 6,000 characters or fewer is used whole). Two systems do it: a fine-tuned mmBERT-base token classifier, and an LLM-based skill with a rule-based layer. This layer is not wired into `detect/` yet.

```
title_author/
├── src/                          builders, the rule extractor, Gemini batch and GEPA scripts, mmBERT scoring
├── skills/tibetan-title-author/  the skill: SKILL.md, scripts/ (ta.py is the command), references/ (rules, examples, second look)
├── skill_dev/                    benchmark, score and cut-coverage scripts for the skill
├── docs/                         overview, Gemini results, run 2 results, dataset card, handoff
├── data/gemini_prompts/          the prompts (GEPA final, hand-made, locked examples; older ones in archive/)
└── data/processed/, rule_runs/   small tables and rule-script runs (the gold datasets and exports are gitignored)
```

The gold datasets, parquet exports, Gemini batches and runs, and model predictions are on disk under `data/` and gitignored. They are several gigabytes.

## Models and data

| | |
|---|---|
| Training data, 5 labels (O, B/I-TITLE, B/I-AUTHOR) | [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) |
| Title-only and author-only views of the same data | `Yontenn/title-only-training`, `Yontenn/author-only-training` (private) |
| Text and span tables | [Yontenn/formatting-title-author](https://huggingface.co/datasets/Yontenn/formatting-title-author) |
| Trained model (joint, run 2) | `Yontenn/mmbert-title-author-run2` (private, weights in `best/`) |
| Trained title-only and author-only models | `Yontenn/mmbert-title-only`, `Yontenn/mmbert-author-only` (private, weights in `best/`) |

Training is `common/train_layer.py`. The joint model is `--label-name TITLE_AUTHOR --scheme multi`. A single-entity model is `--label-name TITLE --scheme bio --dataset Yontenn/title-only-training` (or `AUTHOR` with `Yontenn/author-only-training`), with `--weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test`. The title-only run used all 8 epochs; the author-only run stopped early at epoch 5.5, with its best checkpoint at step 10,020.

Decoding is Viterbi, then one span per work: title = first span, author = last span. The joint model uses break penalty 12; the single-entity models use 6 (chosen on validation). Scoring: `src/ta_mmbert_score.py` (joint) and `src/ta_mmbert_single_score.py` (single-entity).

## Run the skill

```
python3 title_author/skills/tibetan-title-author/scripts/ta.py BOOK.txt
```

It cuts the text, asks a model (Gemini or Claude, whichever key is set; without a key it gives a weak rule-based guess), repairs the answer, and sets `needs_review`. Details are in `skills/tibetan-title-author/SKILL.md` and `references/pipeline.md`.

## Results

Cleaned test set, 2,384 works, share of works with IoU ≥ 0.5 against the gold span in the same block. Decoder settings were chosen on validation; test was scored once.

| | Title | Author | Mean |
|---|---|---|---|
| mmBERT joint, run 2 (penalty 12) | 0.833 | 0.711 | 0.772 |
| mmBERT title-only + author-only (penalty 6) | 0.842 | 0.734 | 0.788 |
| Skill (gemini-3.8-flash, GEPA prompt, rule layer) | 0.786 | 0.756 | 0.771 |
| Title-only model for the title + skill for the author | 0.842 | 0.756 | 0.799 |

Paired 95% ranges: two single models minus joint, mean +0.017 (+0.007 to +0.026); author-only minus joint, author +0.023 (+0.009 to +0.038); title-only minus joint, title +0.010 (−0.002 to +0.021); mix minus the skill, mean +0.028 (+0.020 to +0.036); mix minus the two single models, mean +0.011 (+0.002 to +0.019).
Either one right: joint model or skill, title 0.881, author 0.837; title-only or skill, title 0.892; author-only or skill, author 0.838.

`docs/run2_results.md` reports run 2 on an earlier 2,389-work test set: title F1 0.828, author F1 0.744. About 11% of the gold authors differ from the author a reader would name (stanza writer, compiler, copyist, requester), so every number has a ceiling.

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

| | Hugging Face repo | Card |
|---|---|---|
| Final model, title | `BDRC/Bo-Title-Detection` (private, weights in `best/`) | [`docs/hf_cards/Bo-Title-Detection.md`](docs/hf_cards/Bo-Title-Detection.md) |
| Final model, author | `BDRC/Bo-Author-Detection` (private, weights in `best/`) | [`docs/hf_cards/Bo-Author-Detection.md`](docs/hf_cards/Bo-Author-Detection.md) |
| Dataset for the title model | `BDRC/Bo-Title-Dataset` (private) | [`docs/hf_cards/Bo-Title-Dataset.md`](docs/hf_cards/Bo-Title-Dataset.md) |
| Dataset for the author model | `BDRC/Bo-Author-Dataset` (private) | [`docs/hf_cards/Bo-Author-Dataset.md`](docs/hf_cards/Bo-Author-Dataset.md) |
| Joint data, 5 labels (O, B/I-TITLE, B/I-AUTHOR); both datasets above are relabelled copies of it | [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) | [`docs/dataset_card.md`](docs/dataset_card.md) |
| Text and span tables (an older version, with auto-labelled works) | [Yontenn/formatting-title-author](https://huggingface.co/datasets/Yontenn/formatting-title-author) | |
| Joint model, run 2 (an experiment, trained on data before the translator and `_` cleaning) | `Yontenn/mmbert-title-author-run2` (private, weights in `best/`) | [`docs/model_cards/mmbert-title-author-run2.md`](docs/model_cards/mmbert-title-author-run2.md) |

Training is `common/train_layer.py`. The joint model is `--label-name TITLE_AUTHOR --scheme multi`. A single-entity model is `--label-name TITLE --scheme bio --dataset BDRC/Bo-Title-Dataset` (or `AUTHOR` with `BDRC/Bo-Author-Dataset`), with `--weight-scheme sqrt_inv --evals-per-epoch 2 --epochs 8 --skip-test`. The title-only run used all 8 epochs; the author-only run stopped early at epoch 5.5, with its best checkpoint at step 10,020.

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

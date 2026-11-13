# Title and author layer: run 2 results

Model: `jhu-clsp/mmBERT-base`, one model with 5 labels (O, B/I-TITLE, B/I-AUTHOR).
Model on the Hub: `Yontenn/mmbert-title-author-run2` (private).

## Setup

- Input: one sequence per work, the first 3,000 and last 3,000 characters (works of 6,000 characters or fewer are used whole).
- Training data: 32,914 works (24,971 annotator works plus 7,943 v3 auto-labelled works). Validation (2,595) and test (2,443) are annotator-only.
- Training: 8 epochs, class weights `sqrt_inv`, best checkpoint chosen on validation mean of title F1 and author F1.
- Decoding: Viterbi with break penalty 12 (the trainer used 5; 12 was picked on validation).
- Metric: per-work span precision, recall and F1, a prediction counts as correct at IoU 0.5 or more.

## Results

### Test, all 2,443 works (penalty 12)

| | Precision | Recall | F1 |
|---|---|---|---|
| Title | 0.809 | 0.834 | 0.821 |
| Author | 0.742 | 0.737 | 0.739 |

### Test, cleaned (2,389 works)

Works where title and author overlap, or where a label is missing after the crop, are removed (54 works).

| | Precision | Recall | F1 |
|---|---|---|---|
| Title | 0.818 | 0.839 | 0.828 |
| Author | 0.751 | 0.737 | 0.744 |

### Validation

| | Precision | Recall | F1 |
|---|---|---|---|
| Title, all 2,595 works | 0.806 | 0.835 | 0.820 |
| Author, all 2,595 works | 0.791 | 0.799 | 0.795 |
| Title, cleaned (2,499 works) | 0.817 | 0.844 | 0.830 |
| Author, cleaned (2,499 works) | 0.807 | 0.800 | 0.803 |

### Compared with the baseline (test, all works)

| | Title P | Title R | Title F1 | Author P | Author R | Author F1 |
|---|---|---|---|---|---|---|
| Baseline | 0.674 | 0.835 | 0.746 | 0.643 | 0.767 | 0.699 |
| Run 2 (penalty 12) | 0.809 | 0.834 | 0.821 | 0.742 | 0.737 | 0.739 |

### Extra rows, not the headline metric (test, cleaned)

Looser overlap threshold:

| IoU threshold | Title P | Title R | Title F1 | Author P | Author R | Author F1 |
|---|---|---|---|---|---|---|
| 0.5 (headline) | 0.818 | 0.839 | 0.828 | 0.751 | 0.737 | 0.744 |
| 0.3 | 0.840 | 0.861 | 0.851 | 0.818 | 0.803 | 0.810 |
| 0.1 | 0.851 | 0.872 | 0.861 | 0.838 | 0.823 | 0.830 |

A prediction also counts if it has at least 90% of the same words as the gold span anywhere in the work:

| | Precision | Recall | F1 |
|---|---|---|---|
| Title | 0.844 | 0.866 | 0.855 |
| Author | 0.761 | 0.748 | 0.754 |

## Errors, most to least (validation, cleaned, 2,499 works)

### Author

| # | Error | Works | Share | Meaning |
|---|---|---|---|---|
| 1 | Wrong place | 238 | 9.5% | The model picked a different name than the gold |
| 2 | Too wide | 102 | 4.1% | The gold plus extra words (role word or particle) |
| 3 | Missed | 98 | 3.9% | Nothing predicted |
| 4 | Boundary off | 63 | 2.5% | The prediction is inside or shifted from the gold |
| 5 | Extra wrong span | 61 | 2.4% | Correct span found, plus a wrong one |

### Title

| # | Error | Works | Share | Meaning |
|---|---|---|---|---|
| 1 | Different text | 219 | 8.8% | Other text picked, 155 truly different and 64 similar words at another position |
| 2 | Same title, different position | 71 | 2.8% | Nearly the same words (90% or more) elsewhere in the work |
| 3 | Extra wrong span | 70 | 2.8% | Correct span found, plus a wrong one |
| 4 | Too wide | 60 | 2.4% | The gold plus the next line or extra words |
| 5 | Boundary off | 28 | 1.1% | The prediction is inside or shifted from the gold |
| 6 | Missed | 12 | 0.5% | Nothing predicted |

## Gemini prompt vs our model (test, 2,389 works, IoU 0.5)

Same works for both. Gemini used `title_author/data/gemini_prompts/final.md` with `gemini-3.8-flash` and default thinking. The prompt was written from annotation patterns found by Cursor and fixed after two rounds of failures on train and validation.

| | Title P | Title R | Title F1 | Author P | Author R | Author F1 |
|---|---|---|---|---|---|---|
| Our model (run 2) | 0.818 | 0.839 | 0.828 | 0.751 | 0.737 | 0.744 |
| Gemini prompt | 0.780 | 0.780 | 0.780 | 0.759 | 0.758 | 0.759 |
| Gemini 3.5 flash-lite, `final.md` (cleaned test) | 0.708 | 0.659 | 0.683 | 0.658 | 0.620 | 0.639 |
| Gemini 3.5 flash-lite, GEPA run-1 prompt (cleaned test) | 0.723 | 0.677 | 0.699 | 0.656 | 0.624 | 0.640 |
| Gemini 3.5 flash-lite, GEPA run-2 prompt `gepa_v2.md` (cleaned test) | 0.760 | 0.686 | 0.721 | 0.657 | 0.606 | 0.630 |
| **Gemini 3.8 flash, `gepa_v2.md`, low thinking, batch (cleaned test)** | 0.849 | 0.788 | 0.817 | 0.730 | 0.688 | 0.708 |
| Rule-based script | 0.688 | 0.688 | 0.688 | 0.375 | 0.375 | 0.375 |

Rule-based script (`title_author/src/ta_rule_extract.py`): plain Python, no model and no Gemini. Weights are fitted on train only (`ta_rule_fit.py` for title, `ta_rule_rank.py` for author). It was scored on the cleaned test set (2,384 works, no translator works, no underscores), so the works differ slightly from the 2,389 above; precision equals recall because it always returns one span. On validation (2,494 works) it scores title 0.619, author 0.411. Its author ceiling is 0.75: for about a quarter of works none of its candidates matches the gold span.

GEPA (automatic prompt rewriting) was run twice with a small budget and gave no reliable gain, so it is not in this table.

## Gemini prompt tuning with GEPA

GEPA (DSPy) rewrites a prompt from the mistakes it reads and keeps a rewrite only if it scores higher on a validation sample. Gemini does both the task and the rewriting. All Gemini rows above that say "3.5 flash-lite" use `gemini-3.5-flash-lite`, one work per call, the first and last 3,000 characters, scored on the cleaned test set (2,384 works). The older row "Gemini prompt" used `gemini-3.8-flash` on the earlier test set (2,389 works), so the rows are not directly comparable.

| Run | Setup | Score on the works it tuned on | On works it never saw |
|---|---|---|---|
| 1 | 40 train, 60 validation works, start `final.md` | best rewrite 0.733 vs 0.717 | on 42 new works: tie with `final.md` |
| 1, on test | the best rewrite (GEPA prompt) vs `final.md`, all 2,384 test works | | mean F1 0.669 vs 0.661 (title +1.6 points, author +0.1) |
| 2 | 300 train, 500 validation works, start from the run-1 prompt, budget 4,000 | 71.1 vs 66.8 | held-out check below |
| 3 | 600 train, 800 validation works spread over the sources, 12 mistakes read per rewrite, 4 works per call with a 2,000-character crop, budget 1,200 | no rewrite beat the start prompt, so it was returned unchanged (53.1 vs 53.9 on the same prompt is scoring noise) | not needed |

Full cleaned test set (2,384 works), `gemini-3.5-flash-lite`, mean of title and author F1: `final.md` 0.661, GEPA run-1 prompt 0.669, GEPA run-2 prompt (`gepa_v2.md`) 0.675. The run-2 prompt is better on title (0.721 against 0.683 for `final.md`) and slightly worse on author (0.630 against 0.639).

`gemini-3.8-flash` with `gepa_v2.md`, `thinkingLevel: low` and the Batch API, on the whole cleaned test set (2,384 works): title F1 0.817, author F1 0.708, mean 0.762. On the same works `gemini-3.5-flash-lite` with the same prompt scores 0.738 mean share-correct against 0.646 for flash-lite; the mean difference is +9.2 points (95% range +8.1 to +10.4). Cost of the full run: 9.09M input, 0.20M output and 0.02M thinking tokens, about $3.83 at batch prices.

Held-out check of the run-2 prompt against the run-1 prompt: 1,898 validation works that run 2 never used, both answered. Title 0.683 vs 0.663, author 0.621 vs 0.654, mean 0.652 vs 0.659. The mean difference is -0.6 points (95% range -1.9 to +0.7), so the two prompts tie. The run-2 prompt is about 40% as long (911 tokens against 2,186), which makes each call about 25% cheaper.

Token cost on this data (measured on 10 works): the run-2 prompt sends about 3,600 input tokens and gets about 76 output tokens per call, with no thinking tokens. Putting 4 works in one call saves only about 18% of the input once the prompt is short, and it lowered accuracy by about 2 points on 200 works, so it was used only for search in run 3.

## What was tried and did not help

- Forcing at least one span per work, or keeping one span only: author F1 0.795 down to 0.780 and 0.782 on validation. Keeping the most confident span gives about +0.006 on title and nothing on author.
- Post-processing author spans (strip the trailing particle, the leading role word, or both): author F1 on test 0.744 down to 0.740, 0.732 and 0.729.
- Break penalty sweep on validation (mean of title F1 and author F1): 1 gives 0.749, 5 gives 0.802, 8 gives 0.805, 12 gives 0.808, 20 gives 0.807, 50 gives 0.789.

## Data notes

- The builder gives each token one label. Where the author span overlaps the title span, the title wins and the author is trimmed or dropped (351 works have this note).
- Long works are cropped to the first and last 3,000 characters. A span in the middle is cut out, so some works end up with no title or no author label (109 training works have no title label and 599 have no author label).
- A both-labels version of the data exists at `title_author/data/processed/gold_dataset_all_strat_single_v3_both` (train 32,213, validation 2,499, test 2,389). It is now the data in `Yontenn/title-author-training` (private) and, as raw text, in `Yontenn/formatting-title-author` (public). Run 2 was trained on the earlier version (32,914 / 2,595 / 2,443 works); no model has been trained on this one yet.

## Open questions for the annotation side

1. When a work names several people (author, translator, compiler, teacher), which one is the author label?
2. Should the author span be the name only, or the name with its role words and the "by" particle?
3. When a work contains several texts, which title is the label: the first, or the main one?

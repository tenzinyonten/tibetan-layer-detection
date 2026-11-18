# Sabche zero-shot baselines

Zero-shot LLM span detection for the Sabche (section-heading) layer, scored against
the frozen split (`sabche/data/processed/sabche_split_frozen.csv`) using the
cleaned spans (`sabche_spans_clean.csv`). No training and no examples: each model gets
the prompt in `sabche/docs/prompts/gemini_sabche_anchors.md` on 16,000-character
windows with 2,000 characters of overlap, and returns anchor text that is located back
in the book.

Metric: span-level IoU >= 0.5, micro-averaged over all gold spans in the split
(scored by `common/zeroshot/run_layer_zeroshot.py --layer sabche --locate-only`).

## Test split (29 books, 4,079 gold spans)

| Model | F1 | Precision | Recall | Gold found |
|---|---|---|---|---|
| Claude Sonnet 5 (Message Batches API) | **0.681** | 0.613 | 0.766 | 3,124 / 4,079 |
| Gemini (script default `gemini-3.1-flash-lite`; the log does not record the model) | 0.668 | 0.600 | 0.754 | 3,075 / 4,079 |

The 1.3-point gap comes from a single run per model, so treat the two as roughly equal.

Claude was run with `common/zeroshot/run_claude_batch.py` in three batches
(179 + 180 + 179 windows, about $5 per 180 windows at batch pricing).

## Validation split (Gemini only, incomplete)

The Gemini validation run stopped after 22 of 33 books: F1 0.549, precision 0.466,
recall 0.669 (1,769 / 2,645 gold spans in those books). The 22 books are not a random
sample, so this number is not comparable to the test score. Claude has not been run
on validation.

## Per-book test F1

| Book | Gold spans | Gemini | Claude |
|---|---|---|---|
| I07240379 | 118 | 0.813 | 0.867 |
| I0FCFA88F | 248 | 0.824 | 0.833 |
| I319DAFF7 | 15 | 0.933 | 0.875 |
| I36A7A668 | 88 | 0.495 | 0.556 |
| I3F4A91F5 | 367 | 0.875 | 0.591 |
| I52248444 | 236 | 0.816 | 0.774 |
| I575514A8 | 77 | 0.632 | 0.637 |
| I7D476A8B | 55 | 0.545 | 0.543 |
| I9AEEF96A | 22 | 0.930 | 0.894 |
| I9B6A4525 | 196 | 0.982 | 0.982 |
| I9D9C7AC9 | 42 | 0.215 | 0.612 |
| IC05A6BE0 | 8 | 0.024 | 0.093 |
| IC6F06BCD | 122 | 0.691 | 0.725 |
| ICDC84458 | 181 | 0.834 | 0.819 |
| IDB6093E9 | 154 | 0.596 | 0.665 |
| IE5895799 | 227 | 0.743 | 0.736 |
| IF3ACC3E1 | 45 | 0.815 | 0.684 |
| IFA88A536 | 254 | 0.891 | 0.613 |
| P000013 | 44 | 0.623 | 0.763 |
| P000027 | 52 | 0.606 | 0.776 |
| P000037 | 161 | 0.482 | 0.488 |
| P000067 | 382 | 0.610 | 0.696 |
| P000083 | 225 | 0.417 | 0.587 |
| P000118 | 42 | 0.660 | 0.735 |
| P000144 | 107 | 0.618 | 0.645 |
| P000164 | 313 | 0.621 | 0.785 |
| P000242 | 261 | 0.585 | 0.650 |
| P000247 | 8 | 0.280 | 0.314 |
| P000269 | 29 | 0.140 | 0.099 |

## Files

- `zeroshot/claude_sonnet5_test_summary.json`: per-book and micro scores, Claude.
- `zeroshot/gemini_test_summary.json`: per-book and micro scores, Gemini.
- `zeroshot/gemini_val_partial_summary.json`: the 22 scored validation books, Gemini.

The raw per-window model replies are under `scratch/` (gitignored) and are not tracked.

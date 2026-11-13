# Sabche single-char / single-syllable post-processing

Read-only experiment. Saved predictions only — no re-inference.

**Predictions:** `~/sabche-layer-detection/results/mmbert-sabche-v1/test/spans` (29 test books, `mmbert-sabche-v1`).  
**Gold:** `sabche_gold.csv`. **Text:** `data/raw_opf`.  
**Score:** IoU ≥ 0.5, greedy one-to-one, micro over books.  
**Script / numbers:** `scratch/sabche_single_span/run.py`, `scratch/sabche_single_span/results.json`.

Not validated on the val set (needs GPU to produce val spans).

## What was tested

Short predicted headings — a single character, or a single syllable (one run between tsheg/shad/whitespace) — against three treatments:

| family | rule |
|---|---|
| **Baseline** | saved Viterbi spans, no extra cleanup |
| **Delete** | drop the short prediction |
| **Extend to syllable** | grow the span to the enclosing syllable (tsheg/shad/whitespace bounds) |
| **Extend to shad** | grow the span to the next `།` / `༎` / newline (and back to the previous) |

Single-syllable includes single-char. Predictions are inclusive `(start, end)` in this experiment.

## How many short predictions

4,144 predicted spans on test. 0 zero-syllable predictions.

| kind | n | TP | FP |
|---|---:|---:|---:|
| single-character | 10 | 0 | 10 |
| single-syllable (incl. single-char) | 36 | 9 | 27 |
| single-syllable, longer than one char | 26 | 9 | 17 |

Almost all of these are false positives. The 9 true-positive single-syllable spans are real (very short) headings.

## Results

Headline comparison:

| approach | F1 | vs baseline |
|---|---:|---|
| baseline | **0.9619** | — |
| delete single-syllable | **0.9640** | +0.0021 |
| extend to syllable | **0.9619** | no effect |
| extend to shad (single-syllable) | **0.9633** | +0.0013 |

Full table from `results.json`:

| approach | F1 | P | R | TP | pred | changed |
|---|---:|---:|---:|---:|---:|---:|
| 1 baseline | 0.9619 | 0.9544 | 0.9696 | 3955 | 4144 | 0 |
| 2a delete single-char | 0.9631 | 0.9567 | 0.9696 | 3955 | 4134 | 0 |
| 2b delete single-syllable | 0.9640 | 0.9606 | 0.9674 | 3946 | 4108 | 0 |
| 3a extend single-char → its syllable | 0.9619 | 0.9544 | 0.9696 | 3955 | 4144 | 10 |
| 3b extend single-syllable → full syllable | 0.9619 | 0.9544 | 0.9696 | 3955 | 4144 | 16 |
| 3c extend single-char → next shad/newline | 0.9624 | 0.9551 | 0.9698 | 3956 | 4142 | 10 |
| 3d extend single-syllable → next shad/newline | 0.9633 | 0.9563 | 0.9703 | 3958 | 4139 | 25 |

Baseline reproduces the published test F1 (~0.962).

## Reading

- **Extend-to-syllable does nothing.** Growing a 1-character or 1-syllable stub to the rest of that syllable is still far too short to reach IoU 0.5 against a real heading, so the match set does not change.
- **Delete helps slightly** (0.9619 → 0.9640) because 27 of the 36 single-syllable preds are FP. It also drops the **9 real short headings**, so recall falls (0.9696 → 0.9674). Delete-single-char only is safer (all 10 are FP; F1 0.9631) and is already covered in spirit by the pipeline’s “drop spans under 5 characters” rule.
- **Extend-to-shad** recovers a few TPs (+3 at the single-syllable setting) but the net is still ~0.1 F1, and it is the same *kind* of rule as `extend_lone_fragments`, which already lost on the full Sabche test (0.9642 → 0.9630) because leftover fragments become long false positives.

All deltas are tiny (~0.2 F1 at most). Baseline is already near-best.

## Conclusion

**Do not add a Sabche-specific single-char/single-syllable extend or delete rule.** The production setup (tsheg-extend and gap-merge off, `repair_fragments` on, drop spans under 5 characters) is enough.

This was **test-only**. The same table needs a val-span dump (GPU) before anyone should treat delete-single-syllable as a real improvement. Without that check, it is not shipping.

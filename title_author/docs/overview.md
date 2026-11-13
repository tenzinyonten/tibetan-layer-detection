# Title and author extraction for Tibetan works: overview

## Goal
Given the first and last ~3,000 characters of a Tibetan work, return the **title** and the **author** as exact text spans (with region head/tail). Scoring: IoU ≥ 0.5, same region, one title and one author per work, precision/recall/F1.

## The four pieces
| Piece | What it is | Status |
|---|---|---|
| Trained model | mmBERT-base token classifier (run 2) | title F1 0.828, author 0.744 (older 2,389-work test) |
| Rule-based script | Plain Python extractor (`src/ta_rule_extract.py`), no LLM | title 0.688, author 0.375 on the cleaned test set |
| LLM prompt | Gemini with a hand-made prompt (`final.md`, 12 examples), then GEPA rewrites | see below |
| Skill module | `skills/tibetan-title-author/scripts/title_author_skill.py`: LLM call + exact-substring check + rule layer | works offline (selftest passes); benchmarked with Gemini only |

## Benchmark
Cleaned test set: 2,384 works. CSV and HF dataset are published. Gold labels are noisy (about 11.5% of the LLM's authors are a different person than the label).

## How the rule-based method is used with GEPA (the manager's request)
GEPA (DSPy `dspy.GEPA`) rewrites the prompt by reading the model's mistakes. We added the rule-based method in three places in `src/ta_gepa3.py`:
1. **Typed feedback.** The mistake text GEPA's rewriter reads says *what kind* of error it was (different place, requester cue, missing role word, too long, too short), using the rule script's knowledge.
2. **Locked block.** The 12 examples are appended to the input field description so the rewriter cannot delete them (earlier runs deleted them, which cost ~6.7 author points on the large model).
3. **Hybrid step.** Before scoring, the rule layer adds a touching role word and falls back to the rule script if the LLM gave no usable author.

## Results (honest summary)
- Prompt rewriting with GEPA helped the small model slightly (mean +1.4) but hurt the large one (author −9.4) because it deleted the examples.
- Run 4 (examples locked, rule feedback on): 0.710 → 0.737 on its own 300 validation works. **On 300 unseen test works it tied** the other prompts (mean 0.660 vs 0.657 for `final.md`; every difference within ±3–4 points noise). Title fell, author rose.
- The rule layer (role word + fallback) added +5–7 author points on saved flash-lite answers (validation).
- Best measured prompt on the large model: `final.md` (author 0.759, 0.762 with the rule layer).
- Cost of the unseen check: $0.377 (about ₹34).

## Open problem: what "author" means
`སྦྱར་བ` and similar verbs are used by stanza writers, compilers, scribes and proofreaders, not only the author. The prompts treat the verb as a stop word and say "if it is the only signature, take it". `final_combined.md` (now in `archive/`) added "take the composer of the whole work, not a stanza writer, compiler, copyist or proofreader"; on 300 test works with flash-lite it tied the others (mean 0.652 vs 0.657 for `final.md`).

## Files
- Prompts: `data/gemini_prompts/` (`gepa_final_full.md` final, `gepa_final.md` rules only, `final.md` hand-made baseline, `examples_locked.md`); older ones in `data/gemini_prompts/archive/`
- Results: `docs/gemini_results.md`, `docs/run2_results.md`
- Skill: `skill/`
- Code: `src/`

## Final GEPA run and test (2026-11-11)
- Setup: gemini-3.8-flash, low thinking, 300 validation works (75 DSPy examples of 4 works), 600 train works, budget 300 metric calls, examples locked, rule feedback and rule layer (with the new repair step) on. Cost about Rs 370 (4.6M input, 118k output tokens at standard price).
- 5 rewrites proposed, 3 accepted: validation 0.713 -> 0.745 -> 0.758 (best, saved as `gepa_final.md`) and a third accepted candidate that scored 0.738 on the full validation set.
- Unseen test, first 300 works, rule layer applied: title 0.837, author 0.790, mean 0.813. `final.md` (default thinking): 0.830 / 0.797 / 0.813. Difference 0.000 (95% range -0.025 to +0.025): a tie. Against `final_noexamples` (0.777): +0.037 (clear). Against `gepa_v2` (0.793): +0.020 (not clear).
- Batch cost of the final prompt: about $0.95-0.99 per 300 works.

## Repair step (new)
`repair_span` in `src/ta_rule_extract.py` finds a model answer again on the page when it is not an exact substring because a space or newline changed (7% of author answers, 4.6% of title answers). Validation, saved flash-lite answers: author 0.688 -> 0.708 and 0.707 -> 0.731, title +1.1 and +2.0 points over the earlier rule layer.

## BDRC catalog test (free, 40 validation works)
The public BDRC API resolves our work IDs, but the record describes the whole book. 26 of 40 IDs gave a creator; the catalog author matched the label in 4 of 18 wrong-author works and 2 of 8 right-author works (about 25%). Labels are sometimes in Chinese or a different name form. Not a reliable signal for the author of a single text.

## Fix pass (offline only)
`skills/tibetan-title-author/scripts/title_author_fix.py`: for flagged answers (author repaired/fallback/missing, requester cue, 2+ signature verbs; about 32% of works) the LLM picks among the rule script's top 5 candidates with context. Perfect-picker ceiling: +4.6 to +5.2 author points; rule top-1 alone is worse than the LLM answer. Trial with gemini-3.5-flash-lite as the picker on all 627 flagged validation works (free tier, `src/ta_fix_trial.py`): author on the flagged works 0.563 -> 0.584, 24 fixed and 11 broken (net +13), overall author 0.708 -> 0.715 (+0.7 points). The first 300 works looked better (+3.7 on flagged) but the remaining 327 added net 0, so the true effect is small. Not tried with 3.8 or on the final prompt's answers.


## Update 2026-11-12
- Same 2,100 test works: mmBERT title 0.832 / author 0.723; Gemini 3.8-flash (final GEPA prompt + rule layer) title 0.794 / author 0.762. Either right: 0.882 / 0.844. Details: `gemini_results.md`.
- Old gepa_v2 rows 0.817 / 0.708 are F1 with nulls; share correct is 0.788 / 0.688.
- Skill (`skill/`): added efficiency wrappers (cache, retry, meter, parallel), local HTTP server, benchmark, `skill_score.py` (accuracy minus token penalty), `needs_review` flag. Default prompt is `gepa_final_full.md` (about 8,000 tokens). A shorter GEPA prompt was tried and turned out to be cut off (no Title/Author sections), so that result is withdrawn; see `gemini_results.md`.
- Tool calling, retrieval, voting, two-model judge: each at most about 1 point on author. BDRC matches the label about 25%.

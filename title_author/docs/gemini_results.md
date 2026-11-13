# Gemini results on the cleaned test set (all 2,384 works)

Two models, one prompt (`gepa_v2.md`), same works. Everything here is scored the same way: a span is correct at IoU 0.5 or more in the same region; one title and one author per work; precision, recall and F1 per field.

## 1. Setup

| Item | `gemini-3.5-flash-lite` | `gemini-3.8-flash` |
|---|---|---|
| Prompt | `title_author/data/gemini_prompts/archive/gepa_v2.md` (about 900 tokens) | same |
| Input per work | first 3,000 and last 3,000 characters (whole work if 6,000 or fewer), one work per call | same |
| Thinking | none (0 tokens used) | `thinkingLevel: "low"` (about 10 tokens per call; default is about 1,300) |
| Mode | regular calls, temperature 0, JSON output | Batch API (half price), four chunks of about 600 works |
| Test works | all 2,384 cleaned test works (`parquet_export_strat_both_notrans_nous`) | same |

The prompt is the output of GEPA run 2: `final.md` (Cursor) → `gepa_lite_candidate.md` (GEPA run 1) → `gepa_v2.md` (GEPA run 2). GEPA run 3 found nothing better.

## 2. Results

| Model and prompt | Title P | Title R | Title F1 | Author P | Author R | Author F1 | Mean F1 |
|---|---|---|---|---|---|---|---|
| **`gemini-3.8-flash`, `gepa_v2.md`, low thinking** (F1; share correct: title 0.788, author 0.688, mean 0.738) | 0.849 | 0.788 | **0.817** | 0.730 | 0.688 | **0.708** | **0.762** |
| `gemini-3.5-flash-lite`, `gepa_v2.md` | 0.760 | 0.686 | **0.721** | 0.657 | 0.606 | **0.630** | **0.675** |
| `gemini-3.5-flash-lite`, GEPA run-1 prompt | 0.723 | 0.677 | 0.699 | 0.656 | 0.624 | 0.640 | 0.669 |
| `gemini-3.5-flash-lite`, `final.md` (Cursor) | 0.708 | 0.659 | 0.683 | 0.658 | 0.620 | 0.639 | 0.661 |

### What the outcomes were (works out of 2,384)

| | Correct | Wrong boundaries (IoU < 0.5) | No answer (`null`) | Wrong region | Not an exact substring |
|---|---|---|---|---|---|
| 3.8-flash, title | 1,878 | 282 | 173 | 48 | 3 |
| 3.8-flash, author | 1,641 | 538 | 135 | 53 | 17 |
| 3.5-flash-lite, title | 1,635 | 335 | 234 | 67 | 113 |
| 3.5-flash-lite, author | 1,445 | 502 | 184 | 66 | 187 |

## 3. Same works, side by side (share of works answered correctly)

| Run | Title | Author | Mean |
|---|---|---|---|
| `gemini-3.8-flash`, `gepa_v2.md`, low thinking | **0.788** | **0.688** | **0.738** |
| `gemini-3.5-flash-lite`, `gepa_v2.md` | 0.686 | 0.606 | 0.646 |
| `gemini-3.5-flash-lite`, `final.md` | 0.659 | 0.620 | 0.640 |

- 3.8-flash against 3.5-flash-lite (same prompt): **+9.2 points** on the mean, 95% range +8.1 to +10.4. The range excludes zero, so the bigger model is clearly better.
- 3.5-flash-lite with `gepa_v2.md` against `final.md`: about +0.6 mean; title is up and author is down. That is small.

### Earlier 3.8 run, scored on the cleaned test set (no new calls)

The earlier `gemini-3.8-flash` run with `final.md` and default thinking answered all 2,389 old test works. Scoring those same answers on the 2,340 works that are unchanged in the cleaned test set, next to the new run on the same works:

| Same 2,340 works, `gemini-3.8-flash` | Title | Author | Mean |
|---|---|---|---|
| `final.md`, default thinking (earlier run) | 0.781 | **0.759** | **0.770** |
| `gepa_v2.md`, low thinking (new run) | 0.788 | 0.692 | 0.740 |

The earlier run is 6.7 points better on author and 0.7 points worse on title; the mean difference is +3.0 points (95% range +1.8 to +4.1). The works are the same, so the cleaned test set is not the cause. Two things changed between the runs, the prompt (role words kept or dropped in the author) and the thinking level, and these results cannot separate them.

### Prompt or thinking? A controlled test on the same works

To separate the two changes between the earlier run and the new run, `gepa_v2.md` was also run with default thinking on a random 1,200 of the cleaned test works (batch). Scores on the 1,178 works that are in all three runs with unchanged text:

| `gemini-3.8-flash`, same 1,178 works | Title | Author | Mean |
|---|---|---|---|
| `final.md`, default thinking | 0.789 | **0.764** | **0.776** |
| `gepa_v2.md`, default thinking | 0.795 | 0.670 | 0.732 |
| `gepa_v2.md`, low thinking | 0.806 | 0.685 | 0.745 |

- Prompt effect (same default thinking): `final.md` is +9.4 points on author (95% range +6.8 to +12.0); title is a tie.
- Thinking effect (same prompt): default thinking is 1.5 points lower on author than low thinking (not significant), so low thinking loses nothing.
- Conclusion: the author drop with `gepa_v2.md` comes from its prompt, most likely the rule that excludes role words from the author span, and not from the thinking setting.
- Cost of this control run: 4.60M input, 0.10M output and 3.37M thinking tokens (about 2,810 thinking tokens per call at the default setting), about $8.23.

### Which part of `final.md` matters? The 12 examples

`final.md` with only the 12 examples removed (`final_noexamples.md`: all rules, statistics and extra rules kept; 2,394 tokens against 6,022) was run on the first 600 cleaned test works with `gemini-3.8-flash`, low thinking, batch. Scores on the 597 works that are in all three runs with unchanged text:

| `gemini-3.8-flash`, same 597 works | Prompt tokens | Title | Author | Mean |
|---|---|---|---|---|
| `final.md` (everything), default thinking | 6,022 | 0.792 | **0.757** | **0.775** |
| `final.md` without the 12 examples, low thinking | 2,394 | 0.782 | 0.690 | 0.736 |
| `gepa_v2.md`, low thinking | 911 | 0.782 | 0.688 | 0.735 |

- Removing only the examples costs **6.7 points on author** (95% range +3.5 to +10.1); title is a tie.
- Without the examples, the rules alone score the same as `gepa_v2.md` (author 0.690 against 0.688). So the role-word rule is not the cause of the author drop; the 12 examples are what the big model needs.
- Thinking differs between the first row (default) and the others (low), but the earlier test on `gepa_v2.md` showed default thinking is not better (about 1.5 points lower on author), so this does not explain the gap.
- Cost of this test: 3.26M input, 0.05M output and 0.02M thinking tokens, about $1.35 (about ₹121) for 600 works.

## 4. Cost

| Run | Tokens | Cost |
|---|---|---|
| 3.8-flash, batch, low thinking | 9.09M input, 0.20M output, 0.02M thinking | about **$3.83, about ₹345** (batch: $0.375 per million input, $1.875 per million output) |
| 3.5-flash-lite, regular | about 3,600 input and about 80 output tokens per call, 2,384 calls | about **$3.0, about ₹270** (standard: $0.30 per million input, $2.50 per million output; estimated from the measured per-call tokens) |

At default thinking, 3.8-flash used about 2,810 thinking tokens per call (measured on 1,200 works), which cost about $0.0069 per call in batch mode, against about $0.0016 at low thinking.

## 5. Compared with the other rows (different models or test sets, so compare with care)

| Row | Test set | Title F1 | Author F1 |
|---|---|---|---|
| Our trained model (run 2) | older, 2,389 works | 0.828 | 0.744 |
| `gemini-3.8-flash`, `final.md`, default thinking (earlier run) | older, 2,389 works | 0.780 | 0.759 |
| **`gemini-3.8-flash`, `gepa_v2.md`, low thinking** | **cleaned, 2,384 works** | **0.817** | **0.708** |
| `gemini-3.5-flash-lite`, `gepa_v2.md` | cleaned, 2,384 works | 0.721 | 0.630 |
| Rule-based script | cleaned, 2,384 works | 0.688 | 0.375 |

## 6. Caveats

- Title with 3.8-flash (0.817) is close to our trained model (0.828). Author is behind it (0.708 against 0.744) and behind the earlier 3.8 run with `final.md` (0.759).
- `gepa_v2.md` drops role words from the author span, which fits the lower author score. `final.md` keeps them. A copy of `gepa_v2.md` with the role-word rule restored has not been tested.
- The earlier 3.8 rows used the older test set and default thinking, so they are not like-for-like with the rows in section 2.
- The 3.5-flash-lite run had a few failed calls at first (rate limits on the free keys, then a missing credit) that were all redone; the final run has 0 failed calls.
- Thinking `low` was checked on only 2 works before the full run.

## 7. Where my comments go

- [ ] Which row is the Gemini number to report: 3.8-flash (0.762 mean) or 3.5-flash-lite (0.675)?
- [ ] Keep the 12 examples fixed (locked) in the next GEPA run, since they are what the big model needs?
- [ ] Restore the role-word rule in `gepa_v2.md` and test it?
- [ ] Anything else to add before this goes to the manager?

## 8. Where the files are

| What | Where |
|---|---|
| Prompts | `title_author/data/gemini_prompts/` |
| 3.8-flash answers and scores | `title_author/data/gemini_runs/gepa_v2_gemini-3.8-flash_test_low_thinking/` |
| 3.5-flash-lite answers and scores | `title_author/data/gemini_runs/gepa_v2_test_0_rows0-99999/` (and `final_test_0_rows0-99999`, `gepa_lite_candidate_test_0_rows0-99999`) |
| Scripts | `title_author/src/ta_gemini_eval.py` (regular calls), `ta_gemini_batch.py` and `ta_batch_driver.sh` (batch), `ta_compare_runs.py` (comparison) |

## Update 2026-11-11: final GEPA run, unseen test, repair step
See `overview.md` for the details. Short version:
- GEPA run (3.8-flash, low thinking, 300 validation works, budget 300 calls, rule feedback on, examples locked): validation 0.713 -> 0.758. Prompt: `data/gemini_prompts/gepa_final_full.md`.
- First 300 test works, rule layer applied, 3.8-flash: `gepa_final_full` (low thinking) title 0.837, author 0.790, mean 0.813; `final.md` (default thinking) 0.830 / 0.797 / 0.813; `final_noexamples` 0.810 / 0.743 / 0.777; `gepa_v2` 0.830 / 0.757 / 0.793. The final GEPA prompt ties the hand-made one.
- flash-lite, same 300 works, raw: `gepa_v4_full` 0.657 / 0.663 / 0.660; `final_combined` 0.667 / 0.637 / 0.652; `final.md` 0.670 / 0.643 / 0.657; `gepa_lite_candidate` 0.703 / 0.657 / 0.680. All tie within noise.
- New repair step: +2 author and +1 to +2 title points on saved validation answers.
- BDRC catalog lookup (40 works): catalog author matched the label about 25% of the time; not used.
- Full 2,384-work test of `gepa_final_full` (8 chunks of 300): numbers to be added when the last chunk finishes.


## Update 2026-11-12: same-works comparison, skill, token-penalty GEPA

Note on old rows: the 0.817 / 0.708 values of `gepa_v2` are F1 (precision differs from recall because some answers were null), which is higher than the share of works answered correctly (title 0.788, author 0.688, mean 0.738). From here on all tables use share correct (every work answered, so P = R = F1).

### Same 2,100 test works (share correct, IoU >= 0.5)

| System | Title | Author | Mean |
|---|---|---|---|
| mmBERT (trained, `Yontenn/mmbert-title-author-run2`) | 0.832 | 0.723 | 0.778 |
| Gemini 3.8-flash, `gepa_final_full`, low thinking, raw answers | not split out here | not split out here | 0.777 |
| Gemini 3.8-flash, hand-made `final.md`, raw answers | not split out here | not split out here | 0.774 |
| Gemini 3.8-flash, `gepa_final_full` with the rule layer | 0.794 | 0.762 | 0.778 |
| Gemini 3.8-flash, `final.md` with the rule layer | not split out here | not split out here | 0.778 |

Either model right: title 0.882, author 0.844 (upper bound for a combiner; no combiner built yet). The last 284 test works were not run with Gemini.

### Author-fix ideas tested (all free, validation)

| Idea | Result |
|---|---|
| Tool calling (find text, BDRC lookup), 30 flagged works, flash-lite | about 2 more correct of 30 |
| Retrieval of similar train works (K = 3) | no gain |
| Voting (5 runs); two models + judge | no gain |
| Candidate picker (fix pass), 627 flagged works | author +0.7 |
| BDRC catalog author | matches label about 25% |

### GEPA with a token penalty (flash-lite, free tier): CORRECTION, the result is invalid

The run picked a prompt that was cut off. Candidates 1 and 2 of 3 end in the middle of a sentence and have no Title Extraction Rules and no Author Extraction Rules section; only the starting prompt (candidate 0) was complete. The token penalty (validation score minus 0.004 x tokens/1000) preferred the cut-off candidate because it scored the same 0.662 on 80 works with fewer tokens.
Everything said earlier in this file or in the skill about a "31% shorter prompt that ties" described the cut-off prompt and is withdrawn. Its re-evaluation scores (new 65.5 / 65.5; old 66.1 / 64.4 / 63.3, flash-lite, failed batches dropped) show only that a prompt WITHOUT the title and author rules scored about like the full one on that model; they say nothing about a complete shorter prompt. Not run on gemini-3.8-flash.
The cut-off files are kept only on the local machine (`TRUNCATED_*`), not in this repo. Fixes: the skill refuses a prompt that lacks its required sections; `ta_gepa3.py` with `EFFICIENCY=1` rejects candidates that miss a heading of the starting prompt or end inside a code span, and uses the token penalty only as a tie-break between complete prompts.

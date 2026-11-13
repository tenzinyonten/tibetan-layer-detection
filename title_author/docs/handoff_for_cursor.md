# Handoff: Tibetan title and author extraction (backup briefing)

You are taking over a project that was worked on with Claude Code. Read this whole file first. Everything lives under `/Users/tenzinyonten/layer_detection/title_author/` (call it `ROOT` below). Nothing is committed or pushed to git, and that is on purpose (see "Rules").

## 1. The task
Given the first and last ~3,000 characters of a Tibetan work (the whole work if it has 6,000 characters or fewer), return the **title** and the **author** as exact substrings of the text, with the region each came from (`head` or `tail`).

Scoring: a prediction is correct when its character span has IoU >= 0.5 with the gold span in the same region. One title and one author per work. Title = first exact match in the region, author = last exact match. P/R/F1 (all equal when every work is answered).

The manager's asks:
1. Create a rule-based script and improve it.
2. Integrate the rule-based method into GEPA (prompt optimization) so it helps improve the prompt.
3. Use DSPy (yes: `dspy.GEPA`, DSPy 3.4.1, environment `~/dspy-env`).
4. Research "skills for title and author" (done as a packaged module any LLM can call; an Anthropic-style `SKILL.md` was not built).
5. Provide a benchmark (CSV and cleaned data on Hugging Face) and the Gemini + GEPA results.

## 2. Where things are

| What | Path (under ROOT) |
|---|---|
| Rule-based extractor (title + author, plain Python) | `src/ta_rule_extract.py` (+ `ta_rule_fit.py`, `ta_rule_rank.py`, `ta_rule_weights.json`, `ta_rule_title_weights.json`) |
| `repair_span()` (new): finds a span again when the model changed a space/newline | in `src/ta_rule_extract.py` |
| GEPA script (DSPy 3) | `src/ta_gepa3.py` (run with `~/dspy-env/bin/python`) |
| Regular Gemini evaluator (key rotation, resume) | `src/ta_gemini_eval.py` |
| Gemini Batch API (submit/status/collect/combine) | `src/ta_gemini_batch.py` |
| Chunked sequential batch driver | `src/ta_batch_driver.sh` |
| Paired comparison of two runs (bootstrap 95% range) | `src/ta_compare_runs.py` |
| Skill module | `skills/tibetan-title-author/scripts/title_author_skill.py`, `skills/tibetan-title-author/references/tool_schema.json`, the skill docs |
| Prompts | `data/gemini_prompts/` |
| Run outputs | `data/gemini_runs/` (each folder has `partial_raw.json` and `results.json`), `data/gemini_batches/`, `data/rule_runs/` |
| Data (parquet, gold) | `data/processed/parquet_export_strat_both_notrans_nous` (the one to use; train/validation/test) |
| Benchmark CSVs | `data/exports/benchmark*.csv` and `.zip` (the user's files, do not delete) |
| Docs | `docs/overview.md`, `docs/gemini_results.md`, `docs/run2_results.md`, `docs/dataset_card.md` |
| Trained model | on the Hugging Face Hub: `Yontenn/mmbert-title-author-run2` (private). Dataset: `Yontenn/title-author-training`. `src/ta_predict.py` runs it (GPU box). |

### Prompts (`data/gemini_prompts/`)
- `gepa_final_full.md`: **the final prompt** (GEPA-rewritten rules + the 12 locked examples). The full-test run uses it.
- `gepa_final.md`: only GEPA's rewritten rules (best candidate of the last run).
- `final.md`: the hand-made baseline from Cursor (rules + 12 examples). Skill default for now.
- `examples_locked.md`: the 12 examples, identical to the examples section of `final.md`. GEPA must not change them (passed as `LOCK_FILE`).
- `archive/`: older prompts (`gepa_v2`, `gepa_lite_candidate`, `gepa_v4`, `gepa_v4_full`, `final_combined*`, `final_authorrule`, `final_noexamples`). Docs still point to some of these by their old path.

## 3. What was built

**Rule-based layer (used around any LLM answer, in `skills/tibetan-title-author/scripts/title_author_skill.py` -> `apply_rules` and in `ta_gepa3.py` -> `postprocess`):**
1. Repair: an answer that is not an exact substring is found again ignoring spaces/newlines.
2. Role word: a role word touching the name (e.g. `དགེ་སློང`) is added to the author (in 76% of train works the gold includes it).
3. Fallback: with no usable author, use the rule script's author.

**GEPA integration (`src/ta_gepa3.py`):** typed feedback from `explain()` (different place, requester cue, missing role word, too long/short, plus gold context), `LOCK_FILE` keeps the examples fixed, `HYBRID=1` runs the rule layer before scoring. GEPA calls the model live (not the Batch API). `BATCH=4` means 4 works per request.

**Skill (`skill/`):** `extract(head, tail, whole_work, call_llm, prompt, hybrid)`. `call_llm(system_prompt, user_text) -> str` is any function that calls an LLM; `gemini_call()` is included. The skill is not yet pip-installable (it reads `../src` and `../data`).

## 4. Results (gemini-3.8-flash unless noted)

First 300 test works (rows 0:300 of the cleaned test set, 2,384 works), rule layer applied:

| Prompt | Title | Author | Mean |
|---|---|---|---|
| `gepa_final_full`, low thinking | 0.837 | 0.790 | 0.813 |
| `final.md`, default thinking | 0.830 | 0.797 | 0.813 |
| `final_noexamples`, low thinking | 0.810 | 0.743 | 0.777 |
| `gepa_v2`, low thinking | 0.830 | 0.757 | 0.793 |

- GEPA's final prompt ties the hand-made `final.md` (difference 0.000, range -0.025 to +0.025) and beats the rules-only prompts. GEPA did not beat the hand-made prompt.
- Last GEPA run: 0.713 -> 0.758 on its 300 validation works (5 proposals, 3 accepted; the saved one is iteration 4). Not proof on its own; the unseen test is the real number.
- Rule layer on saved flash-lite validation answers: author 0.621 -> 0.708 (and 0.655 -> 0.731 on another run); the repair step alone adds about +2 author, +1 to +2 title.
- Older table (older 2,389-work test, no rule layer): trained mmBERT title 0.828, author 0.744; Gemini 3.8 with `final.md` title 0.780, author 0.759. Not comparable with the 300-work table (that slice is easier).
- Flash-lite (no thinking), 300 works: about 0.65-0.66 mean for all prompts, which tie within noise.
- The full 2,384-work test of `gepa_final_full` was running in chunks of 300 (`data/gemini_runs/batch_gepa_final_full_gemini-3.8-flash_test_all_think-low_rows*`). To merge: `python title_author/src/ta_gemini_batch.py combine NAME TAG1 TAG2 ...`. If it is not merged yet, merge and score it, then add the numbers to the docs.

## 5. How to run things

```
# offline checks (no key, no cost)
python3 title_author/skills/tibetan-title-author/scripts/run_tests.py
python3 src/ta_rule_extract.py score validation
~/dspy-env/bin/python src/ta_gepa3.py demo gepa_v2_validation_heldout 3

# compare two saved runs (free)
TA_PQ=title_author/data/processed/parquet_export_strat_both_notrans_nous ALLOW_TEST=1 \
  python3 src/ta_compare_runs.py test RUN_A_FOLDER RUN_B_FOLDER

# paid: one 300-work chunk through the Batch API (key comes from the environment)
ROWS=0:300 THINK_LEVEL=low GEMINI_API_KEY=... TA_PQ=<parquet folder> ALLOW_TEST=1 \
  python3 src/ta_gemini_batch.py submit PROMPT.md test gemini-3.8-flash 0
```
`TA_PQ` must be the path from the repo root of the parquet folder. Run from `/Users/tenzinyonten/layer_detection`.

## 6. Prices and limits (as measured)
- gemini-3.8-flash: standard $0.75 in / $3.75 out per 1M tokens (batch half). `thinkingLevel: "low"` gives about 10 thinking tokens per call and scored as well as the default (about 2,800).
- gemini-3.5-flash-lite: $0.30 / $2.50 (batch half), no thinking.
- Measured: 300 works with the final prompt via batch about $0.97 (about Rs 88 at Rs 90 per dollar). A GEPA run (300 metric calls, 300 validation works, live calls) cost about Rs 370.
- Free API keys cannot create batches or call 3.8-flash. The batch quota allows about one 300-work chunk of the long prompt at a time. GEPA needs `THREADS=3` (8 threads hit the per-minute token limit).

## 7. Rules the user set (follow them)
- **Money:** the user is on very tight credit. State the cost in rupees (about Rs 90 per dollar) BEFORE any paid call and wait for a go. No exploratory paid probes. Run ONE chunk, read the real bill, then decide. Do not let a background loop auto-submit more chunks when a cost cap matters. Estimate from token counts, not from small probes (a 10-work probe under-counted thinking tokens by 2x once).
- **Keys:** never write an API key into a file, doc, memory or commit. Keys only as a one-command environment variable or in the user's own terminal. About ten keys were pasted in chat earlier; the user should revoke them.
- **No git commit or push**, and no "Co-Authored-By" or "Generated with" lines anywhere.
- Raw data in `~/Downloads` is read-only. New files go under `title_author/`. HF uploads only when asked. README edits: numbers only. Do not delete the user's `data/exports` files without asking.
- Never open the test split for tuning. GEPA uses train and validation only; the test split is for final scores (`ALLOW_TEST=1`).

## 8. Open items
1. Merge and score the full 2,384-work run of `gepa_final_full`; write the numbers into `docs/gemini_results.md`, `docs/overview.md` and the skill docs.
2. Skill: set `DEFAULT_PROMPT` to `data/gemini_prompts/gepa_final_full.md`; update the skill docs (repair step, GEPA results, archived prompt paths); optionally package it (pyproject, bundle the rule script, weights and prompt) and add an Anthropic-style `SKILL.md`.
3. Fix doc paths that still point to prompts moved into `data/gemini_prompts/archive/`.
4. Score the trained mmBERT on the cleaned test (needs the user's Hugging Face login; run `src/ta_predict.py` on a GPU box or the Mac) for a three-way comparison with Gemini.
5. Optional: GEPA ablation without rule feedback (`RULE_FEEDBACK=0`, about Rs 370) to measure the rule feedback's own effect; test a Claude or OpenAI `call_llm` on 20-30 works.

### Status 2026-11-12 (supersedes items above where it differs)
- Item 2 done: skill default is `gepa_final_full.md` (env `TA_PROMPT` overrides). The token-penalty GEPA prompt `gepa_eff_full.md` was cut off mid-sentence and is kept only locally as `TRUNCATED_*` files (not in this repo); the skill now refuses prompts without the Title/Author sections.
- Item 4 done: mmBERT scored on validation and test (`data/mmbert_preds/`); on 2,100 test works it beats Gemini on title and loses on author.
- Item 1 partly done: Gemini ran 2,100 of 2,384 test works; the last 284 are not run.
- New: `skill/efficiency.py`, `server.py`, `bench.py`, `skill_score.py`, `run_tests.py`; `src/ta_gepa3.py` has `EFFICIENCY=1` and `EXCLUDE_N`; `src/ta_check_keys.py`. Fixed after Cursor's review: whole-work crop, thread-safe meter, cache key with model + thinking level, `parse_answer`, no `sys.exit` in `gemini_call`, `postprocess` shared with `skill_score`. Next: mmBERT combiner on validation (free); return `author_start` and fix-pass candidates; restructure to SKILL.md + scripts/ + references/; a complete shorter prompt (Batch API, cost stated first).

## 9. Known weak spots
- About 11.5% of the model's authors are a different person than the label (stanza writer, compiler, copyist, requester, or noisy labels). Some are label noise; no prompt rule fixes those.
- GEPA's final prompt says "never return null", which forces a guess when the page has no author.
- The rule script alone is weak on author (0.375 on the cleaned test), so it is a fallback and a fixer, not a main extractor.

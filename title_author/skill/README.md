# Title and author skill for Tibetan works

A small reusable module that finds the **title** and the **author** of a Tibetan work using **any LLM**, with a rule-based layer around it.

## What is in this folder

| File | What it is |
|---|---|
| `title_author_skill.py` | The module. `extract(head, tail, call_llm=...)` is the main function. |
| `title_author_fix.py` | Experimental second pass: for shaky author answers, the LLM picks among the rule script's top 5 candidates. Offline-tested only. |
| `tool_schema.json` | The same ability described as a function/tool (name, inputs, outputs), for LLMs that support function calling. |
| `tool_handler.py` | Tool-calling wrapper: `handle_tool_call(name, args, call_llm)` plus ready tool definitions for Claude, OpenAI and Gemini formats. Self-test: `python3 title_author/skill/tool_handler.py selftest`. |
| `efficiency.py` | Wrappers: `cached`, `retrying`, `metered`, `extract_many` (parallel). Selftest: `python3 efficiency.py selftest`. |
| `server.py` | Local HTTP API: `POST /extract`, `GET /tool`, `/stats`, `/health`. Rules mode is free. |
| `bench.py` | Benchmark with the same scorer as every other result: `python3 bench.py validation 300 rules` (free). Paid modes need a key and are cached. |
| `skill_score.py` | One score for a candidate skill: accuracy (using the skill's own `postprocess`) minus a penalty for prompt tokens and tool calls. Free, reads saved answers. The token penalty may only choose between COMPLETE prompts. |
| `run_tests.py` | Runs every offline selftest (no key, no network). |
| `SKILL.md` | The same skill in Anthropic Agent Skills format (name, description, when to use, inputs, how to run, limits). Not tested inside Claude's skill loader. |
| `README.md` | This file. |

The prompt and the rule-based extractor live elsewhere in the repo and are used from here: `title_author/data/gemini_prompts/gepa_final_full.md` (default; env `TA_PROMPT` overrides; the skill refuses a prompt that lacks its Title/Author sections or the examples) and `title_author/src/ta_rule_extract.py`.

## How it works

1. **Input.** The first 3,000 and last 3,000 characters of the work as `[HEAD]` and `[TAIL]` blocks (the whole work if it has 6,000 characters or fewer).
2. **LLM step.** Your LLM gets the prompt plus the text and returns JSON: `title`, `title_region`, `author`, `author_region`. Both must be exact substrings of the text.
3. **Rule-based layer.**
   - Repair: an answer that is not an exact substring (the model changed a space or newline) is found again on the page.
   - Role word: a role word such as `དགེ་སློང` that touches the name on its left is added to the author.
   - Fallback: if the LLM gave no usable author, the rule-based extractor (plain Python, no LLM) supplies one.
4. **Optional fix pass** (`title_author_fix.py`): for answers the rules flag as shaky, the LLM chooses among the top 5 candidate names with context.

## How to use it with any LLM

You pass one function, `call_llm(system_prompt, user_text) -> str`. A Gemini version is included:

```python
from title_author_skill import extract, gemini_call
result = extract(head, tail, whole_work=False, call_llm=gemini_call("gemini-3.8-flash"))
# {"title": ..., "title_region": "head", "author": ..., "author_region": "tail", "notes": [...]}
```

For another LLM, write the function yourself (send the system prompt and the text, return the reply as a string). With no function at all, `extract` returns the rule-based answer. Only Gemini has been benchmarked.

## Quick checks

```
python title_author_skill.py selftest                 # offline, no key, no cost
python title_author_fix.py selftest                   # offline, no key, no cost
python title_author_skill.py rule-only work.txt       # rule-based answer for one text file
GEMINI_API_KEY=... python title_author_skill.py extract work.txt gemini-3.8-flash
```

The key is read only from the environment. Never put it in a file.

## Measured results (cleaned test set)

First 300 works of the cleaned test set (2,384 works), `gemini-3.8-flash`, rule layer applied:

| Prompt | Title | Author | Mean |
|---|---|---|---|
| `gepa_final_full.md` (previous default), low thinking | 0.837 | 0.790 | 0.813 |
| `final.md` (hand-made), default thinking | 0.830 | 0.797 | 0.813 |
| `final_noexamples`, low thinking | 0.810 | 0.743 | 0.777 |
| `gepa_v2`, low thinking | 0.830 | 0.757 | 0.793 |

The default prompt ties the hand-made one and is clearly better than the rules-only prompts. Full-test (2,384 works) numbers: see `title_author/docs/gemini_results.md`.

Effect of the rule layer on saved `gemini-3.5-flash-lite` answers (validation, not test):

| Answers | Author before | Author after |
|---|---|---|
| `gepa_v2` prompt (1,926 works) | 0.621 | 0.708 |
| `gepa_lite_candidate` prompt (1,922 works) | 0.655 | 0.731 |

The repair step alone adds about +2 author and +1 to +2 title points on top of the role-word fix and the fallback.

Rule-based extractor alone (full cleaned test): title 0.688, author 0.375.

## Fix pass (experimental)

- Flags (free): author repaired / from the fallback / missing, a requester cue after the author, or two or more signature verbs in the region. About 32% of works are flagged; the flagged ones are right 56-60% of the time, the others 78-79%.
- Perfect-picker ceiling among the top 5 candidates: +4.6 to +5.2 author points overall. Blindly taking the rule script's top candidate is worse than the LLM's answer (0.34 vs 0.56 on flagged works), so the gain depends on the LLM picking well.
- Prompt about 1,160 tokens, about Rs 0.05 per flagged work with the Batch API.
- Trial with `gemini-3.5-flash-lite` as the picker on ALL 627 flagged validation works (saved flash-lite first-pass answers from the `gepa_v2` prompt; free-tier keys; `src/ta_fix_trial.py`): author correct on the flagged works 0.563 -> 0.584. It changed 91 answers: 24 fixed, 11 broken (net +13). Overall author 0.708 -> 0.715 (+0.7 points). A small gain, borderline significant (24 vs 11). Not yet tried with `gemini-3.8-flash` or on the final prompt's answers, where fewer works are flagged.

## Efficiency

- Rules only, 300 validation works: title 0.640, author 0.357, 0.007 s per work, cost 0.
- Every result carries `needs_review` and `review_reasons` (free checks; flagged works are right 56-60% of the time, others 78-79%; validation).
- `cached` makes a repeat free; `retrying` survives 429/5xx; `metered` reports calls, cache hits, seconds and tokens; `extract_many` runs works in parallel.
- Per-work LLM cost with the GEPA prompt on `gemini-3.8-flash` is about Rs 0.29 with the Batch API (300 works Rs 86-89). Live (non-batch) cost and time per call: not measured yet.

## Limits

- Author is the weak part. Labels are inconsistent (a stanza writer or a compiler versus the book's author), and about 11.5% of authors are a different person than the labelled one.
- A BDRC catalog lookup was tested on 40 works and is not reliable: the catalog author matched the label in about 25% of cases (it describes the whole book, not each text).
- The 12 examples matter for the large model: removing them cost about 6.7 points on author.
- The default prompt `gepa_final_full.md` costs about 8,000 input tokens per call. A shorter GEPA prompt (`TRUNCATED_gepa_eff_full.md` in the archive) was cut off and its result is withdrawn; a complete shorter prompt has not been found.
- Not yet tried with a Claude or OpenAI model; not yet pip-installable (it reads `../src` and `../data`); the mmBERT model is not part of the skill yet.

## Details

Full results, costs and the GEPA experiments: `title_author/docs/gemini_results.md`, `title_author/docs/run2_results.md`, `title_author/docs/overview.md`.

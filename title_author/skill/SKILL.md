---
name: tibetan-title-author
description: Find the title and the author of a Tibetan work (book, article, manuscript) from its first and last parts. Use when you have Tibetan text and need the exact title and author spans, with which part (head or tail) each came from.
---

# Tibetan title and author extraction

Returns the **title** and the **author** of one Tibetan work as exact substrings of the text you give it, plus the block (`head` or `tail`) each came from.

## When to use

- You have the text of a Tibetan work and need its title and author.
- You need exact text spans (not a translation or a paraphrase).

## Inputs

- `text_head`: the first 3,000 characters of the work, or the whole work if it has 6,000 characters or fewer.
- `text_tail`: the last 3,000 characters. Leave empty for a short work.
- `whole_work`: true if `text_head` is the whole work.

## How to run

```python
from title_author_skill import extract, gemini_call      # any LLM works: pass your own call_llm(system_prompt, user_text) -> str
result = extract(head, tail, whole_work=False, call_llm=gemini_call("gemini-3.8-flash"))
# {"title": ..., "title_region": "head", "author": ..., "author_region": "tail", "notes": [...]}
```

Without an LLM function, `extract` returns the rule-based answer (no network).
To expose it to a model as a tool, use `tool_handler.py` (`handle_tool_call`, plus schemas for Claude, OpenAI and Gemini formats).

## Efficiency (what costs what)

| Mode | Title | Author | Time per work | Cost per work |
|---|---|---|---|---|
| Rules only (`call_llm=None`) | 0.640 | 0.357 | 0.007 s | 0 (300 validation works) |
| GEPA prompt + `gemini-3.8-flash` (low thinking) + rule layer | 0.794 | 0.762 | not measured | about Rs 0.29 (Batch API, 300 works Rs 86-89) |

- Default prompt is about 5,500 input tokens per call (31% shorter than the previous 8,000; same accuracy within noise on flash-lite, 300 fresh validation works: 65.5 new vs 64.6 old average; not yet measured on gemini-3.8-flash), read from disk once per process. Input is capped at 3,000 characters per block.
- `efficiency.py`: `cached` (repeat = free, no network), `retrying` (429/5xx backoff), `metered` (calls, cache hits, seconds, tokens), `extract_many` (parallel, order kept).
- Every result has `needs_review` (+ `review_reasons`): free checks that doubt the author. About 32% of works are flagged; flagged ones were right 56-60%, others 78-79% (validation). Route flagged works to a person or a second pass.
- `server.py`: local HTTP API (`POST /extract`, `GET /tool`, `/stats`). `bench.py`: same scorer as all other results. `run_tests.py`: all offline selftests.

## What it does

1. The LLM reads the prompt (`data/gemini_prompts/gepa_eff_full.md`, about 5,500 tokens; the previous default `gepa_final_full.md`, about 8,000 tokens, is the fallback; override with env `TA_PROMPT`) and the text and answers JSON.
2. A rule-based layer repairs answers that are not exact substrings, restores a role word at the start of the author, and falls back to the rule script when the LLM gives no usable author.

## Output rules

- `title` and `author` are exact substrings of the input, or null.
- The author is the composer of THIS work, not the translator, stanza writer, compiler, copyist, proofreader or the person who requested it.

## Limits

- Benchmarked on Gemini only (3.8-flash: title 0.794, author 0.762 on 2,100 test works; see README.md).
- About 11% of authors are a different person than the dataset label, so author accuracy has a ceiling.
- The experimental fix pass (`title_author_fix.py`) adds under 1 point.

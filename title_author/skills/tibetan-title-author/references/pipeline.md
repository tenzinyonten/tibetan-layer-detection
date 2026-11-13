# How the skill works

One book or whole text goes in (as a file), one JSON answer comes out. You never supply a title or an author. The model decides; plain scripts cut the input, check the answer and flag doubt.

## The pipeline

```
text of one work
 1. guard    ta.py          under 20% Tibetan characters -> refuse
 2. cut      ta.py,         6,000 characters or fewer -> whole work as [HEAD]
             title_author_skill.normalize_input
                            longer -> first 3,000 as [HEAD] + last 3,000 as [TAIL]
 3. draft    a language model reads rules.md + examples.md + the text
             -> title, title_region, author, author_region
 4. check    title_author_skill.postprocess (no model)
             - title and author must be exact substrings; a space or newline that differs is found again
             - a role word that touches the author's name is put back (for example དགེ་སློང)
             - no usable author -> the rule script's author (ta_rule_extract.py)
 5. flag     title_author_fix.flags (no model)
             needs_review when: the author was repaired or came from the fallback or is missing,
             a requester cue follows the name (བསྐུལ, ངོར, ཞུས, གསོལ་བ, བཀའ་བསྒོ), or two or more signature verbs sit in the region
 5b. second   only with --second-look and only when needs_review: one more call re-checks the author (references/second_look.md);
             only an exact substring can replace it; result in "second_look": changed / kept / no reply
 6. answer   title, title_region, author, author_region, notes, needs_review, review_reasons, second_look
```

## Who does step 3 (the draft)

| Mode | How | Command |
|---|---|---|
| Agent | The host agent runs `ta.py excerpt` (only the cut text, never the whole book), reads `rules.md`, decides, then runs `ta.py verify` on its own answer (steps 4-5). | `ta.py excerpt BOOK.txt`, then `ta.py verify` |
| API model | `ta.py llm` sends `rules.md` + `examples.md` (joined with a blank line) as the system prompt and the text as the user message. `llm_for(model)` picks the adapter: `claude-*` -> `anthropic_call`, `gemini-*` -> `gemini_call`. Any other provider: pass your own `call_llm(system, user) -> str`. | `ta.py llm WORK.txt MODEL` |
| Rules only | No model. The rule script ranks name candidates. Free and instant, weak on the author. | `ta.py rules WORK.txt` |

`ta.py verify` runs only steps 4-5 on a draft that already exists (the agent's own answer). It is not needed with `llm`, which runs all steps. It never compares against labels. `ta.py excerpt` prints the cut text only (step 2). `--chars N` changes the block size (default 3,000; 2,000 is untested).

## Files

| File | Role |
|---|---|
| `scripts/ta.py` | The command: `llm`, `rules`, `excerpt`, `verify`, `selftest`; guard and cut |
| `scripts/title_author_skill.py` | `extract()`, `postprocess()`, `normalize_input()`, `parse_answer()`, adapters (`gemini_call`, `anthropic_call`, `llm_for`); refuses a prompt that lacks its Title/Author sections or the examples |
| `scripts/ta_rule_extract.py` + two `*_weights.json` | The rule script: candidates, repair of spans, role-word list |
| `scripts/title_author_fix.py` | The `needs_review` flags and the name candidates used by the second look; an experimental candidate picker (not used by `ta.py`) |
| `scripts/tool_handler.py` + `references/tool_schema.json` | The skill as a tool for Claude, OpenAI or Gemini |
| `scripts/efficiency.py` | `cached`, `retrying`, `metered`, `extract_many` (thread-safe) |
| `scripts/server.py` | Local HTTP service: `POST /extract`, `/tool`, `/stats`, `/health` |
| `scripts/run_tests.py` | All offline self-tests (no key, no network) |
| `references/rules.md`, `references/examples.md` | The prompt: GEPA-rewritten rules, then 12 worked examples |
| `references/second_look.md` | The prompt for the second look at a doubted author |

## Extending

- A new model provider: write `call_llm(system, user) -> str` (set `call.local.usage = (in, out)` and `call.tag` if you want metering and a correct cache key) and pass it to `extract()`.
- Many works: `efficiency.extract_many(works, call_llm, workers=4)`, or `server.py`.
- Change the rules: edit `references/rules.md`. A prompt missing the Title or Author section, or the examples, is refused.

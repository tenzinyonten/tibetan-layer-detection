---
name: tibetan-title-author
description: Finds the title and the author of a Tibetan work (book, article, manuscript, colophon page) as exact substrings of its text, and flags when the author needs a human check. TRIGGER whenever the user gives Tibetan text or a file of it and asks for the title, author, composer, colophon, "who wrote this", "what is it called", or to label the works in a collection. SKIP translation, summaries, transliteration or Wylie conversion, word meanings, catalog (BDRC) lookups and non-Tibetan text. If the user mentions a Tibetan work and wants to know what it is or who made it, use this skill.
---

# Tibetan title and author

Input: the text of a book or any whole text, as a file. You do NOT need a title or an author.
Output: the **title** and the **author**, each an exact substring of the text, with the block (`head` = start, `tail` = end) it came from.
The skill reads only the first 3,000 and the last 3,000 characters (a text of 6,000 characters or fewer is used whole), because the title is at the start and the author's colophon is usually at the end.

## Quick Start

```bash
python3 scripts/ta.py llm book.txt gemini-3.8-flash     # or claude-sonnet-5-5; needs GEMINI_API_KEY / ANTHROPIC_API_KEY; costs money
```

```json
{
 "title": "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་",
 "title_region": "head",
 "author": "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན",
 "author_region": "tail",
 "notes": ["added role word 'དགེ་སློང'"],
 "needs_review": false,
 "review_reasons": []
}
```

The command cuts the text, lets the model read the excerpt with the rules, then the scripts make both strings exact substrings, put back a role word that touches the name, and flag doubt.

## If you are an agent (no API key needed)

1. `python3 scripts/ta.py excerpt book.txt` prints only the `[HEAD]` and `[TAIL]`. Read that, never the whole book.
2. Read `references/rules.md`. Decide the title and the author yourself, copied exactly from the excerpt.
3. `python3 scripts/ta.py verify book.txt --title "<title>" --author "<author>"` repairs spacing, restores the role word, falls back to the rule script if your author cannot be found, and sets `needs_review`. Answer with its JSON.
4. If `needs_review` is true, re-read the colophon or tell the user the author is uncertain.

## Quick Reference

| Task | Command |
|---|---|
| Book in, answer out (API model, costs money) | `python3 scripts/ta.py llm book.txt MODEL` |
| Free first guess, no model (author right about 36%) | `python3 scripts/ta.py rules book.txt` |
| Only the text the model would read | `python3 scripts/ta.py excerpt book.txt` |
| Check and repair a draft that already exists | `python3 scripts/ta.py verify book.txt --title "..." --author "..."` |
| Block size other than 3,000 (2,000 is untested) | add `--chars 2000` |
| Local HTTP service | `python3 scripts/server.py 8765` |
| Cache, retry, metering, parallel runs in Python | `scripts/efficiency.py` (read its docstring) |
| Tool definition for an LLM host (Claude, OpenAI, Gemini) | `scripts/tool_handler.py`, `references/tool_schema.json` |
| Offline self-tests | `python3 scripts/run_tests.py` |

The skill is model-neutral: any agent can follow the steps itself, and the scripts call any model through one function (`call_llm(system, user) -> str`). Accuracy was measured with Gemini only; the Claude adapter is untested.

## IMPORTANT

- **Never read or send the whole book.** Use `excerpt` or let `llm` cut it.
- **Never reword.** Title and author must be copied exactly from the text. Do not translate, normalize spacing or fix spelling; `verify` repairs only newline and space differences.
- **The author is the composer of THIS work.** Not the translator, a stanza or verse writer, the compiler, the copyist, the proofreader, or the person who asked for the text (བསྐུལ, ངོར, ཞུས, གསོལ་བ).
- **Keep the role word** (for example `དགེ་སློང`) when it touches the name; the labels include it about three times in four.
- **`needs_review: true` means doubt.** Re-read the colophon or tell the user the author is uncertain. Flagged works were right 56-60% of the time, the others 78-79%.
- **Only Tibetan text.** `ta.py` refuses input with under 20% Tibetan characters.

## Next Steps

- Rules for the title and the author: `references/rules.md` (read it before deciding).
- Twelve worked examples: `references/examples.md` (long; read only when a case is unclear).
- How the pieces fit (the pipeline, who produces the draft, which file does what): `references/pipeline.md` (read it to debug, extend, or explain the skill).

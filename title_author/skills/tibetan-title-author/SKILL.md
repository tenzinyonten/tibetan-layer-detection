---
name: tibetan-title-author
description: Finds the title and the author of a Tibetan work (book, article, manuscript, colophon page) as exact substrings of its text, and flags when the author needs a human check. TRIGGER whenever the user gives Tibetan text or a file of it and asks for the title, author, composer, colophon, "who wrote this", "what is it called", or to label the works in a collection. SKIP translation, summaries, transliteration or Wylie conversion, word meanings, catalog (BDRC) lookups and non-Tibetan text. If the user mentions a Tibetan work and wants to know what it is or who made it, use this skill.
---

# Tibetan title and author

Returns the **title** and the **author** of ONE work, each an exact substring of the text, with the block (`head` = start, `tail` = end) it came from. A work of 6,000 characters or fewer is used whole; a longer one is cut to its first and last 3,000 characters (the scripts do this).

## Quick Start

Put the text in a file, decide the title and author yourself (rules in `references/rules.md`), then check them:

```bash
python3 scripts/ta.py verify work.txt --title "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་" --author "གྲགས་པ་རྒྱལ་མཚན"
```

```json
{
 "title": "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་",
 "title_region": "head",
 "author": "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན",
 "author_region": "head",
 "notes": ["added role word 'དགེ་སློང'"],
 "needs_review": false,
 "review_reasons": []
}
```

`verify` made both strings exact substrings, put back the role word that touches the name, and checked for reasons to doubt the author. Answer with this JSON.

## Quick Reference

The skill is model-neutral: any agent can follow the steps itself, and the scripts call any model through one function (`call_llm(system, user) -> str`). Accuracy was measured with Gemini only; the Claude adapter is untested.

| Task | Command |
|---|---|
| Free first guess (weak: author right about 36%) | `python3 scripts/ta.py rules work.txt` |
| Check and repair my answer | `python3 scripts/ta.py verify work.txt --title "..." --author "..."` |
| Whole pipeline with an API model, many works (costs money) | `python3 scripts/ta.py llm work.txt gemini-3.8-flash` (needs `GEMINI_API_KEY`) or `... claude-sonnet-5-5` (needs `ANTHROPIC_API_KEY`) |
| Local HTTP service | `python3 scripts/server.py 8765` |
| Cache, retry, metering, parallel runs in Python | `scripts/efficiency.py` (read its docstring) |
| Tool definition for an LLM host (Claude, OpenAI, Gemini) | `scripts/tool_handler.py`, `references/tool_schema.json` |
| Offline self-tests | `python3 scripts/run_tests.py` |

## IMPORTANT

- **Never reword.** Title and author must be copied exactly from the text. Do not translate, normalize spacing or fix spelling; `verify` repairs only newline and space differences.
- **The author is the composer of THIS work.** Not the translator, a stanza or verse writer, the compiler, the copyist, the proofreader, or the person who asked for the text (བསྐུལ, ངོར, ཞུས, གསོལ་བ).
- **Keep the role word** (for example `དགེ་སློང`) when it touches the name; the labels include it about three times in four.
- **`needs_review: true` means doubt.** Re-read the colophon or tell the user the author is uncertain. Flagged works were right 56-60% of the time, the others 78-79%.
- **Only Tibetan text.** `ta.py` refuses input with under 20% Tibetan characters.

## Next Steps

- Rules for the title and the author: `references/rules.md` (read it before deciding).
- Twelve worked examples: `references/examples.md` (long; read only when a case is unclear).

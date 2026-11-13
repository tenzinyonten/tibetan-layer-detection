---
name: tibetan-title-author
description: Find the title and the author of a Tibetan work (book, article, manuscript) from its first and last parts. Use when you have Tibetan text and need the exact title and author spans, with which part (head or tail) each came from. Not for translation or catalog lookups.
---

# Tibetan title and author extraction

Call the tool `extract_title_author` (definition in `tool_schema.json`). It returns the **title** and the **author** as exact substrings of the text, and the block (`head` or `tail`) each came from.

## Inputs
- `text_head`: the first 3,000 characters, or the whole work if it has 6,000 characters or fewer.
- `text_tail`: the last 3,000 characters. Leave empty for a short work.
- `whole_work`: true if `text_head` is the whole work.

## Result
`title`, `title_region`, `author`, `author_region`, `notes`, and `needs_review` (+ `review_reasons`).
- `needs_review` true: free checks doubt the author (author repaired, rule-script fallback, a requester cue after the name, two or more signature verbs). Flagged works were right 56-60% of the time, the others 78-79%. Show those to a person or verify them.
- The author is the composer of THIS work, not a translator, stanza writer, compiler, copyist, proofreader or the person who asked for it.

## Limits
- About 11% of authors are a different person than the dataset label, so author accuracy has a ceiling.
- Benchmarked on Gemini only. Numbers, cost and efficiency notes: `README.md`.
- Run `python3 run_tests.py` for the offline self-tests.

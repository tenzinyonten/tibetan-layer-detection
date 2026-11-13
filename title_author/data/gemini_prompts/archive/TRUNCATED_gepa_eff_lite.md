You find the title and the author of one Tibetan work. Follow the sections below exactly.

## What Gemini receives

One work. If `whole_work` is true or `text_tail` is empty, the input is the head. Otherwise the input is the head and the tail, about the first 3,000 and the last 3,000 characters of the work.

Pass them as two labeled blocks so the region is unambiguous:

```
[HEAD]
<head text, exact>
[TAIL]
<tail text, exact>
```

Omit `[TAIL]` when there is no tail.

## What Gemini must return

JSON only, no markdown around it:

```json
{"title": "<exact substring>", "title_region": "head" | "tail", "author": "<exact substring>", "author_region": "head" | "tail"}
```

`title` and `author` must be exact substrings copied from the indicated region without any modification (preserve exact spelling, OCR errors, tsheg, shad, spaces, and newlines). Never use catalog forms or corrected spellings.

CRITICAL: On this dataset, gold rows always have both title and author spans. Returning `null` is always a complete miss (false negative). NEVER return `null` for title or author. Always extract the best candidate span according to the rules below.

If the input contains no `[TAIL]` block, `title_region` and `author_region` must always be `
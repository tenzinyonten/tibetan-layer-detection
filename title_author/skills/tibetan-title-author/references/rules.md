You find the title and the author of one Tibetan work. Follow the sections below exactly.

## What you receive

One work. If `whole_work` is true or `text_tail` is empty, the input is the head. Otherwise the input is the head and the tail, about the first 3,000 and the last 3,000 characters of the work.

Pass them as two labeled blocks so the region is unambiguous:

```
[HEAD]
<head text, exact>
[TAIL]
<tail text, exact>
```

Omit `[TAIL]` when there is no tail.

## What you must return

JSON only, no markdown around it:

```json
{"title": "<exact substring>", "title_region": "head" | "tail", "author": "<exact substring>", "author_region": "head" | "tail"}
```

`title` and `author` must be exact substrings copied from the indicated region without any modification (preserve exact spelling, OCR errors, tsheg, shad, spaces, and newlines). Never use catalog forms or corrected spellings.

CRITICAL: On this dataset, gold rows always have both title and author spans. Returning `null` is always a complete miss (false negative). NEVER return `null` for title or author. Always extract the best candidate span according to the rules below.

If the input contains no `[TAIL]` block, `title_region` and `author_region` must always be `"head"`.

---

## Title Extraction Rules

1. Opening Title:
   - Find the title at the opening in `[HEAD]`. Skip opening initial marks (`༄༅། །`, `༈`, `༁ྃཿ`), homages (`ན་མོ...`, `ཨོཾ་སྭ་སྟི`), seed syllables (`ཧཱུྃ`, `ཧྲཱིཿ`), Sanskrit lines, and `བོད་སྐད་དུ`. Take the Tibetan title that follows.
   - If there is a top label, margin title, or short header line immediately preceding the title line (e.g., `སྒོ་ཆ་སྒྲུབ\n...`), include both lines as part of the opening title block.
   - If the first line is already a complete title ending with `བཞུགས་` or `བཞུགས། །`, take that line. Do not replace it with a later variant unless the first line was only a short 1–2 word section tag.
   - Always stop before title formulas: `ཞེས་བྱ་བ་བཞུགས་སོ`, `ཅེས་བྱ་བ་བཞུགས་སོ`, `ཞེས་བྱ་བ`, `ཅེས་བྱ་བ`, and `བཞུགས་སོ` (these closing formula words stay outside the title span; trailing punctuation belonging to the formula stays outside).

2. No Explicit Heading:
   - If the text opens directly with verses, prayers, or mantras without any formal heading (e.g. starting `རཾ་ཡཾ་ཁཾ། ཨོཾ་ཨཱཿཧཱུྃ། ...` or `ཕྱོགས་བཅུའི་རྒྱལ་བ་...`), take the opening verse line as the title. Never return `null`.

3. Colophon Title:
   - Only if there is no title or verse opening at the start, check the closing colophon for `... ཞེས་བྱ་བ་འདི་ནི་` or `ཞེས་པ་འདི་ནི་` and extract the entire naming phrase before the next clause.

---

## Author Extraction Rules

1. Location & Colophon Selection:
   - For long works with a `[TAIL]`, look in `[TAIL]`. For short works with only `[HEAD]`, look near the end (last quarter) or the opening line.
   - If multiple author colophons appear in sequence (e.g., composer colophon followed by a final prayer/dedication signature like `... ཞེས་ཨ་རཱ་གས་སོ།།` or `༧གདོང་རིག་དྷརྨ་ཤྲཱིས་སོ།`), take the very last author signature at the end of the text.
   - Treasure Texts (Terma): Padmasambhava (`པདྨ་འབྱུང་གནས`) is often cited in the narrative text, but the author is the treasure revealer (Tertön) who retrieved or transcribed the text (`གདན་དྲངས་`, `གཏེར་ནས་བཏོན་`, e.g., `ཨོ་རྒྱན་རྩ་གསུམ་གླིང་པ་ཆོས་ཀྱི་རྒྱ་མཚོ`). Take the Tertön.
   - Scribes vs. Authors: Take the actual composer of the work (`མཛད`, `སྦྱར`, `བྲིས`, `བཀོད`, `སྨྲས`). Do not take patrons (`བསྐུལ་བ`), copyists/proofreaders (`དག་ཞུས་`, `དག་པར་ཕབ་པ`), or carvers unless the scribe is the sole named contributor.

2. Author Span Boundaries:
   - Agentive Particles: Always stop BEFORE the agentive particle (`ཀྱིས`, `གྱིས`, `གིས`, `ཡིས`). For vowel-final names with an attached agentive `ས` (e.g., `མི་ཕམ་པས་` -> `མི་ཕམ་པ`; `ཨ་རཱ་གས་` -> `ཨ་རཱ་ག`; `རྒྱ་མཚོས་` -> `རྒྱ་མཚོ`), strip the attached agentive `ས`, but keep nominal syllables ending in `པ` or intrinsic root `ས` (like `ཤེས`, `གྲོས`, `གནས`).
   - Leading Honorifics and Symbols: Omit a leading `༧` at the start of an author's name.
   - Descriptive Epithets vs. Names: Do NOT include long descriptive clauses, origin titles, or epithets preceding the name (e.g., exclude `ཆོས་སྨྲ་བའི་བཙུན་པ་`, `པ་ཤཱཿའི་བཙུན་པ་`, `རྒྱལ་ཁམས་པ`, `བོད་ལྗོངས་ཐུབ་བསྟན་...`). Extract only the personal name (e.g., `སངས་རྒྱས་ཕུན་ཚོགས`, `བློ་བཟང་དཔལ་ལྡན་ཡེ་ཤེས`, `དཔལ་ལྡན་ཤེས་རབ`). Simple standard titles directly attached like `བནྡེ་`, `དགེ་སློང` may be kept if fused, but long epithet phrases must be excluded.
   - Life Dates: If life dates appear in brackets right after an opening author name, include the name as written in `[HEAD]`.
   - Exact substring: Preserve all OCR quirks, line breaks, and internal spaces within the chosen name.
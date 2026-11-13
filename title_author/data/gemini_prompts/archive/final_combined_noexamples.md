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
{"title": "<exact substring or null>", "title_region": "head" | "tail" | null, "author": "<exact substring or null>", "author_region": "head" | "tail" | null}
```

`title` and `author` are copied from the named region with no change of spelling, tsheg, shad, space, or newline. They are not a catalogue form and not a repaired OCR reading. `null` means that field is absent from the page. Gold is `parquet_export_strat_both_notrans` (lotsawa authors already dropped). On this train and validation set a `null` is always a miss, because every gold row has both spans. Use `null` only when the text really has no such span.

If the substring occurs more than once in the named region, resolve it like this when you score: the title is the first exact match, the author is the last exact match. You may later change the Gemini output to include `title_start` / `title_end` / `author_start` / `author_end` (end exclusive, inside the named region) if that raises validation F1. The slice `region_text[start:end]` must still equal the returned string.

## Scoring

One title span and one author span per work. A prediction is correct when its character span and the gold span have intersection-over-union of at least 0.5. IoU is `|A ∩ B| / |A ∪ B|` on the character offsets in the same region. A span in the wrong region does not match.

For each of title and author, over the works in the split:

- true positive: gold span exists and the predicted span has IoU ≥ 0.5
- false positive: a span was predicted and it is not a true positive
- false negative: a gold span exists and the prediction is not a true positive

Precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = the harmonic mean. Report title and author separately. Also report the mean of the two F1 scores; use that mean to choose the final prompt.

A wrong string is both a false positive and a false negative. `null` against a gold span is a false negative. A one-character tsheg or shad on a normal span usually stays above IoU 0.5. A role word (`ཤཱཀྱའི་དགེ་སློང་` is about 16 characters) or a formula (`ཞེས་བྱ་བ་བཞུགས་སོ`) on a short span often falls below 0.5. Author spans have median length 18. Title spans have median length 50.

## Starting point for the Gemini prompt

The first Gemini prompt is the task statement above, the output JSON, and the two blocks below (annotation rules and the 12 examples). Run that prompt first. Its validation F1 is the “before” number.

### Annotation rules

One title and one author. Copy the page as it is, including a wrong syllable, a space, a newline, and `༧`.

Title:

- Take the work’s name from the opening. Skip `༄༅། །`, a homage (`ན་མོ`, `ཨོཾ་སྭསྟི`), a seed syllable, a `བརྡ་ཡིག` line, and Sanskrit. After `བོད་སྐད་དུ`, take the Tibetan title.
- Short title followed by the full title of the same work: take the full title.
- Several texts in the work: take the first text’s title.
- The same title printed again in the colophon: take the opening copy.
- No opening title, and the close says `X་ཞེས་བྱ་བ་འདི་ནི`: take that `X`, including when it is in the tail.
- Stop before `ཞེས་བྱ་བ`, `ཅེས་བྱ་བ`, and `བཞུགས་སོ`. A single closing shad or tsheg may sit just inside or just outside; the formula words stay outside.
- The heading may contain `མཛད་པའི་` and still be the title. The author is a separate span.

Author:

- Take the person who made this text, the name governed by `མཛད`, `སྦྱར`, `བྲིས`, or `བཀོད`. Long work: look in the tail. Short work: look in the last quarter.
- If the opening line is already `title, name, མཛད་པ་བཞུགས་སོ`, that name is the author.
- Author (`མཛད`) and translator (`བསྒྱུར`, `ལོ་ཙཱ་བ`) both named: take the author, never the lotsawa. Gold rows whose author string is `ལོ་ཙཱ` / `ལོ་ཙྪ` were dropped. The only person is a treasure revealer and the verb on that name is `བསྒྱུར`: take that person.
- The only signature is `བྲིས་པ` or `སྦྱར་བ`: take that signer. A master who appears only inside the title is not the author.
- Include the role in the same name phrase (`ཤཱཀྱའི་དགེ་སློང`, `དགེ་སློང`, `སློབ་དཔོན`, `བཙུན་པ`, `བནྡེ`, `རྗེ་བཙུན`, `མཁན་པོ`, `པཎྜི་ཏ`, and the same kind of word). When the word touches the name, include it.
- Stop before the agentive: `ཀྱིས`, `གྱིས`, `གིས`, `ཡིས`, and the `ས` glued to a vowel-final name (`རྡོ་རྗེས་`, `བཟང་པོས་`, `རྒྱ་མཚོས་`). Keep a `ས` that belongs to the name (`ཤེས`, `གྲོས`, `གནས`).
- Stop before `མཛད་པ`, `སྦྱར་བ`, `བྲིས་པ`, `བཀོད་པ`, and before `འི་ཞབས` / `འི་ཞལ`.
- The name printed twice: take the colophon copy, unless the opening line already supplied it.
- Keep a newline or a space that falls inside the name.
- `སྒྲུབ་པ` and `སྒྲུབ་པ་པོ` do not mark the author.

Measured on the 24,263 annotator training works in `_notrans` (138 lotsawa rows dropped from the old 24,401), and stable on the 2,494 validation works: the title is in the head in 96.8%; on a long work the author is in the tail in 89.8%; on a whole work the author is in the last quarter in 93.1%. A role prefix is inside the author span in 15.0%, and when the role word is adjacent it is inside in 76%. The agentive particle is outside (inside in 0.06%; `ཀྱིས`-type particle immediately after the span in 4.7%; vowel-final name followed by agentive `ས` in 41.7% of all authors). `བཞུགས་སོ` is inside the title in 0.7%. The title string occurs in both head and tail in 3.8%, and 96.8% of those marks are the head copy. `བསྒྱུར` is the verb on the marked name in 1.8% of works that mention both `མཛད` and `བསྒྱུར`. Newlines sit inside 21.6% of titles and 8.9% of authors.

## Additional rules (round 1). They take priority over the rules above where they conflict.

Author:
- The author span is exactly the name. Stop at the end of the name. Do not carry on into the next word: the ablative `ནས` (in `རྣལ་འབྱོར་པ་ནཱ་ག་རྫུ་ནས` the span ends before `ནས`), the pronoun `ཁོས`, a page mark (`༄༅།`), a blank line, or a following clause.
- A role word that touches the name stays inside the span (`རྣལ་འབྱོར་པ་`, `རིག་པ་འཛིན་པ་`). Do not drop it.
- In a colophon, the author is the person who wrote or composed the text: the name just before the verb `སྦྱར`, `བྲིས`, `བཀོད`, `མཛད` or `བགྱིས`, or the name after `ཡི་གེ་པ་ནི་` ("the writer is"). Take that person, not a teacher, a patron, a lineage master or any other person named nearby. This holds when the OCR spelling is broken: copy the garbled spelling exactly as it is.
- A name followed by life dates in brackets, for example `(༡༨༦༥་་་༡༩༢༩)`, on the first lines is the author. Take that copy, in the [HEAD], with its closing shad if the shad follows the name directly.

Title:
- If the closing note says `ཞེས་པ་འདི་ནི་` followed by a long phrase that names the text (it may end in `ཚིགས་བཅད`, `བསྟོད་པ` or `གསོལ་འདེབས`), take the whole phrase, from its first word to its last word before the next clause. Do not shorten it to its last part.
- If the first line of the text is already a complete title and ends with `བཞུགས་` or `བཞུགས།`, take that first line. Do not replace it with a longer variant that appears later. Use the "full title" rule only when the first line is a short heading of one or two words.
- If the work opens with a verse or a prayer and has no heading, take the first line of the verse (for example `ཕྱོགས་བཅུའི་རྒྱལ་བ་ སྲས་བཅས།`), not a heading that appears later.

## Additional rules (round 2). They take priority over the rules above where they conflict.

Region:
- If the input has no `[TAIL]` block, never return `"tail"` for `title_region` or `author_region`. Everything is in `[HEAD]`.

Author:
- Several people can carry the verb `སྦྱར`, `མཛད`, `བྲིས` or `བཀོད`. The author is the person who made the whole work. Do not take someone who only wrote a stanza, a verse or a prayer inside or at the end of the text, who only compiled or collected it, or who only copied, proofread (`དག་པར་ཕབ་པ`), edited or printed it. When both kinds of name appear, take the composer of the whole work, usually named in the closing colophon. If the only name is the scribe or the compiler, take that name.
- A `༧` at the very start of the author span is left out of the span. A `༧` that falls inside the span, after a role word and a space, stays (example 11).
- A name given with an alternate name (`[Name A] འམ། མིང་གཞན་ [Name B] ཞེས་...`): take only one name, `[Name B]`. Leave out `མིང་གཞན་` and `ཞེས་`.

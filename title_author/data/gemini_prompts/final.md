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

### Twelve examples

`[…]` is a cut, not part of the text. `\n` in the gold JSON is a real newline inside the span.

**1. Opening title, formula outside, author at the end, particle outside, verb `སྦྱར`.**
`IE00EGS1017151_VE3KG259_1_tei#7`, tei, whole work.

```
[HEAD]
༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།།
[…]
ཚེས་བཅུ་ལ་སོགས་པ་དུས་ཀྱི་མཆོད་པའི་ཡི་གེ་རྣལ་འབྱོར་གྱི་དབང་ཕྱུག་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།
```

```json
{"title": "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་", "title_region": "head", "author": "གྲགས་པ་རྒྱལ་མཚན་", "author_region": "head"}
```

**2. Role inside the author span, agentive `ས` outside.**
`IE1KG17189_VE3KG163_1_tei#9`, tei, whole work.

```
[HEAD]
༈ །རྗེ་བཙུན་སྒྲོལ་མ་དཀར་མོ་ལ་བསྟོད་པ་ཚེ་དང་ཡེ་ཤེས་སྤེལ་བ་རིན་པོ་ཆེའི་ཕྲེང་བ་ཞེས་བྱ་བ།
[…]
ཤཱཀྱའི་དགེ་སློང་དགེ་འདུན་གྲུབ་པ་དཔལ་བཟང་པོས་
```

```json
{"title": "རྗེ་བཙུན་སྒྲོལ་མ་དཀར་མོ་ལ་བསྟོད་པ་ཚེ་དང་ཡེ་ཤེས་སྤེལ་བ་རིན་པོ་ཆེའི་ཕྲེང་བ་", "title_region": "head", "author": "ཤཱཀྱའི་དགེ་སློང་དགེ་འདུན་གྲུབ་པ་དཔལ་བཟང་པོ", "author_region": "head"}
```

**3. Composer marked, translator not marked. The title line also contains the author’s name and `མཛད་པ`; the author span is only the name in the colophon. `མཁན་པོ` stays outside the author span.**
`IE1KG17189_VE3KG161_1_tei#2`, tei, whole work.

```
[HEAD]
༈ །སྒྲོལ་མའི་བྱད་འགྲོལ་གྱི་ཡི་གེ་སློབ་དཔོན་པདྨ་ནག་པོས་མཛད་པ་བཞུགས་སོ།
[…]
ངག་གི་བྱད་འགྲོལ་བ་རྒྱ་གར་གྱི་མཁན་པོ་པདྨ་ནག་པོས་མཛད་པ། ལོ་ཙཱ་བ་དགེ་སློང་བ་རིས་བསྒྱུར་བའོ།
```

```json
{"title": "སྒྲོལ་མའི་བྱད་འགྲོལ་གྱི་ཡི་གེ་སློབ་དཔོན་པདྨ་ནག་པོས་མཛད་པ", "title_region": "head", "author": "པདྨ་ནག་པོ", "author_region": "head"}
```

**4. Sanskrit line, then the Tibetan title. Role inside the author. Genitive and translator outside.**
`IE1KG17189_VE3KG164_1_tei#4`, tei, long work.

```
[HEAD]
༄༅། །རྒྱ་གར་སྐད་དུ། དེ་བཱི་ཏཱ་རཱ་ཨེ་ཀ་བིངྴ་ཏི་སྟོ་ཏྲ་བི་ཤུདྡྷེ་ཙུཎྜ་མ་ཎི་ནཱ་མ། བོད་སྐད་དུ། ལྷ་མོ་སྒྲོལ་མ་ཉི་ཤུ་རྩ་གཅིག་ལ་བསྟོད་པ་རྣམ་དག་གཙུག་གི་ནོར་བུ་ཞེས་བྱ་བ།
[TAIL]
ལྷ་མོ་སྒྲོལ་མ་ཉི་ཤུ་རྩ་གཅིག་ལ་བསྟོད་པ་རྣམ་དག་གཙུག་གི་ནོར་བུ་ཞེས་བྱ་བ། སློབ་དཔོན་ཆེན་པོ་སཱུརྻ་གུཔྟའི་ཞབས་ཀྱིས་མཛད་པ་རྫོགས་སོ། །རྒྱ་གར་གྱི་མཁན་པོ་དཱི་པཾ་ཀ་ར་ཤྲཱི་ཛྙཱ་ན་དང་། བོད་ཀྱི
```

```json
{"title": "ལྷ་མོ་སྒྲོལ་མ་ཉི་ཤུ་རྩ་གཅིག་ལ་བསྟོད་པ་རྣམ་དག་གཙུག་གི་ནོར་བུ་", "title_region": "head", "author": "སློབ་དཔོན་ཆེན་པོ་སཱུརྻ་གུཔྟ", "author_region": "tail"}
```

**5. The person inside the title is not the author. The author is named on the opening line of a long work.**
`IE1PD100944_VE3KG585_1_tei#0`, tei. `བདེ་གཤེགས` is inside the title. The span is `སློབ་དཔོན་ཡེ་ཤེས་རྒྱལ་མཚན་`.

```
[HEAD]
༄༅། །དམ་པ་བདེ་གཤེགས་ཀྱིས་མཛད་པའི་ཐེག་དགུ་སྤྱི་བཅིང་གི་འགྲེལ་བཤད་ཉི་མའི་འོད་ཟེར་ཞེས་བྱ་བ་སློབ་དཔོན་ཡེ་ཤེས་རྒྱལ་མཚན་གྱིས་མཛད་པ་བཞུགས་སོ། །
```

```json
{"title": "དམ་པ་བདེ་གཤེགས་ཀྱིས་མཛད་པའི་ཐེག་དགུ་སྤྱི་བཅིང་གི་འགྲེལ་བཤད་ཉི་མའི་འོད་ཟེར", "title_region": "head", "author": "སློབ་དཔོན་ཡེ་ཤེས་རྒྱལ་མཚན་", "author_region": "head"}
```

**6. Same title in the head and the tail: mark the head. The author is the colophon signer, with no `མཛད` on the name.**
`IE00EGS1016703_VE00EGS1016703_004_1_tei#0`, tei, long work.

```
[HEAD]
༄༅། །དཔལ་གསང་བ་འདུས་པ་སྤྱན་རས་གཟིགས་དབང་ཕྱུག་གི་སྒྲུབ་ཐབས་སྐལ་ལྡན་ཤིང་རྟ་ཞེས་བྱ་བ་བཞུགས་སོ།།
[TAIL]
ཞེས་དཔལ་གསང་བ་འདུས་པ་སྤྱན་རས་གཟིགས་དབང་ཕྱུག་གི་སྒྲུབ་ཐབས་སྐལ་ལྡན་ཤིང་རྟ་ཞེས་བྱ་བ་འདི་ནི།
[…]
བནྡེ་ཚུལ་ཁྲིམས་རིན་ཆེན་དུ་འབོད་པས་
```

```json
{"title": "དཔལ་གསང་བ་འདུས་པ་སྤྱན་རས་གཟིགས་དབང་ཕྱུག་གི་སྒྲུབ་ཐབས་སྐལ་ལྡན་ཤིང་རྟ་", "title_region": "head", "author": "བནྡེ་ཚུལ་ཁྲིམས་རིན་ཆེན", "author_region": "tail"}
```

**7. Two texts. Mark the first title. Author in the tail, particle outside, verb `བཀོད`.**
`IE1KG4884_VE1KG4884_008_1_tei#1`, tei. The later text `ཤེས་རབ་རང་བྱུང་གི་སྒྲོན་མ་སྐུ་ལྔའི་སྙིང་པོ` is not the title.

```
[HEAD]
༄༅། །ཟབ་དོན་གནད་གྱི་ལྡེ་མིག་བཞུགས་སོ༔
[…]
ཤེས་རབ་རང་བྱུང་གི་སྒྲོན་མ་སྐུ་ལྔའི་སྙིང་པོ་བཞུགས་སོ༔
[TAIL]
ཟབ་དོན་གནད་ཀྱི་ལྡེ་མིག་ཅེས་བྱ་བ༔ དྲི་མེད་འོད་ཟེར་གྱིས་བཀོད་པ་རྫོགས་སོ༔
```

```json
{"title": "ཟབ་དོན་གནད་གྱི་ལྡེ་མིག", "title_region": "head", "author": "དྲི་མེད་འོད་ཟེར", "author_region": "tail"}
```

**8. A short title line is not the span. The full title after the homage is the span.**
`IE00EGS1016899_VE3KG270_1_tei#1`, tei, long work. The author span in this row includes the final agentive `ས`; that is uncommon, and the gold is still this slice.

```
[HEAD]
ཉེར་མཁོ་བསྡུས་པ་ལུང་ཆོས་རྒྱ་མཚོའི་སྙིང་པོ་ཞེས་བྱ་བ་བཞུགས་སོ།། ན་མོ་ཨཱརྻ་མཻ་ཏྲི་ནཱ་ཐཱ་ཡ། མངོན་པར་རྟོགས་པའི་རྒྱན་འགྲེལ་པ་དང་བཅས་པའི་སྤྱིའི་དོན་ཉེར་མཁོ་བསྡུས་པ་ལུང་ཆོས་རྒྱ་མཚོའི་སྙིང་པོ་ཞེས་བྱ་བ།
[TAIL]
དཔལ་ཤཱཀྱ་མཆོག་ལྡན་དྲི་མེད་ལེགས་པའི་བློས།
```

```json
{"title": "མངོན་པར་རྟོགས་པའི་རྒྱན་འགྲེལ་པ་དང་བཅས་པའི་སྤྱིའི་དོན་ཉེར་མཁོ་བསྡུས་པ་ལུང་ཆོས་རྒྱ་མཚོའི་སྙིང་པོ་", "title_region": "head", "author": "ཤཱཀྱ་མཆོག་ལྡན་དྲི་མེད་ལེགས་པའི་བློས", "author_region": "tail"}
```

**9. No opening title. Title and author are both in the tail, in the closing sentence. Verb `བྲིས`.**
`IE00EGS1016703_VE00EGS1016703_006_1_tei#20`, tei, long work.

```
[TAIL]
དཔལ་ས་སྐྱ་པ་ངག་དབང་ཀུན་དགའ་བློ་གྲོས་བསྟན་པའི་རྒྱལ་མཚན་དཔལ་བཟང་པོ་པར་འབུལ་རྒྱུའི་ཞུ་འཕྲིན་གཞན་གསོས་བུ་མོའི་གདངས་སྙན་ཞེས་བྱ་བ་འདི་ནི་དོན་གཉེར་གྱི་ངོར། བནྡེ་ཚུལ་ཁྲིམས་རིན་ཆེན་གྱིས་བྲིས་པའོ།།
```

```json
{"title": "དཔལ་ས་སྐྱ་པ་ངག་དབང་ཀུན་དགའ་བློ་གྲོས་བསྟན་པའི་རྒྱལ་མཚན་དཔལ་བཟང་པོ་པར་འབུལ་རྒྱུའི་ཞུ་འཕྲིན་གཞན་གསོས་བུ་མོའི་གདངས་སྙན", "title_region": "tail", "author": "བནྡེ་ཚུལ་ཁྲིམས་རིན་ཆེན", "author_region": "tail"}
```

**10. A newline inside the author name, and a closing shad inside the title. The next line is not the title.**
`IE1KG14500_VE1ER1262_1_tei#0`, tei, whole work.

```
[HEAD]
བསྟན་བཅོས་ལེགས་པར་བསྡེབས་པའི་ཚུལ་བཤད་པ།
བཞི་བ་དོན་དེ་ལེགས་པར་བསྡེབས་པའི་ཚུལ་ལ་བཞི།
[…]
ཐོས་ཆུང་ཚུལ་ཁྲིམས་
ཕུན་ཚོགས་ཞེས་པ་བདག་གིས་བསྡེབས་ཤིང་བརྩམས་པ་རྫོགས་སོ
```

```json
{"title": "བསྟན་བཅོས་ལེགས་པར་བསྡེབས་པའི་ཚུལ་བཤད་པ།", "title_region": "head", "author": "ཐོས་ཆུང་ཚུལ་ཁྲིམས་\nཕུན་ཚོགས", "author_region": "head"}
```

**11. OCR. Copy the space, the `༧`, and the broken spelling. Leave the broken formula `བཞུག་སོ` outside the title.**
`W00KG0555_I1KG110531_85c399_paddleocr_v1#1`, paddleocr_v1, long work.

```
[HEAD]
༄༅། །མན་ངག་བདུད་ཀྱི་གཅོད་ཡུལ་ཟབ་དོན་ཐུགས་སྙིང་ཞེས་བྱ་བ་བཞུག་སོ།།
[TAIL]
ཤཱཀྱའི་དགེ་སློང་ ༧འཇམ་དབྱངས་མགོན་པོས་ཉ་བར་སྦྱར་པའོ།
```

```json
{"title": "མན་ངག་བདུད་ཀྱི་གཅོད་ཡུལ་ཟབ་དོན་ཐུགས་སྙིང", "title_region": "head", "author": "ཤཱཀྱའི་དགེ་སློང་ ༧འཇམ་དབྱངས་མགོན་པོ", "author_region": "tail"}
```

**12. Opening attribution on a long work: title, then the author, then `ས་མཛད་པ་བཞུགས་སོ`.**
`IE1KG17189_VE3KG168_1_tei#6`, tei.

```
[HEAD]
༼ཉ༽༄༅། །རྗེ་བཙུན་བཅོམ་ལྡན་འདས་མ་ཀུ་རུ་ཀུལླེའི་དབང་གི་སྦྱིན་སྲེག་གི་ཆོ་ག་གསལ་བ། ཚར་ཆེན་ཆོས་ཀྱི་རྒྱལ་པོས་མཛད་པ་བཞུགས་སོ།།
```

```json
{"title": "རྗེ་བཙུན་བཅོམ་ལྡན་འདས་མ་ཀུ་རུ་ཀུལླེའི་དབང་གི་སྦྱིན་སྲེག་གི་ཆོ་ག་གསལ་བ།", "title_region": "head", "author": "ཚར་ཆེན་ཆོས་ཀྱི་རྒྱལ་པོ", "author_region": "head"}
```

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

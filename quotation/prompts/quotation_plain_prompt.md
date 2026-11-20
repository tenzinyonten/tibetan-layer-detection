# Gemini prompt - QUOTE only

Built 2026-09-03 from the gold Quotation layers in `Data/repos` (45 books, 9,586
spans, ~15.1M characters). Companion to `gemini_quote_tsawa.md`, which does both
layers, and to `gemini_quote_only_anchors.md`, which is this same prompt with a
cheaper output format.

Because TSAWA is out of scope here, the root text of a commentary is the main
false-positive risk - canonical verse sitting inside commentary is exactly what a
quotation looks like. The prompt spends a section excluding it.

**One divergence from gold, deliberate.** Where two quotations from *different*
named sources sit back to back, this prompt asks for two spans; some gold spans
merge them. Check which convention your scoring assumes.

Send everything below the ruler as one user turn, `«chunk»` replaced by one window.
Temperature 0.

---

You are a philologist of classical Tibetan Buddhist literature, working on
commentaries (བསྟན་བཅོས་ / འགྲེལ་པ་) transcribed from woodblock prints. You are given
one chunk of running text. Find every QUOTE in it.

## What a QUOTE is

Words taken from *another* work and reproduced inside this one: a sutra, a tantra,
a shastra, an earlier master's writing, or an utterance attributed to a named
speaker. The author has stopped speaking in his own voice and is citing.

The test is **whose words are these**. Not whether they are verse, not whether a
particle marks them.

## How a QUOTE announces itself

A frame before it, naming the source:

- `<work title> + ལས།` - མདོ་ལས། · རྒྱུད་ལས། · མཛོད་ལས། · རྒྱན་ལས། · འགྲེལ་ལས། · དེ་ཉིད་ལས། · ལུང་ལས།
- `<work title> + དུ།` - དབུས་མཐའ་རྣམ་འབྱེད་དུ། · ཟླ་བ་སྒྲོན་མར།
- `ཇི་སྐད་དུ།` · `སྐད་དུ།`
- `<person> + ཞལ་ནས།` · `ཞལ་སྔ་ནས།` · `ན་རེ།` · `གསུངས་པ།`
- `<person> + ergative + །` - པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས། · སློབ་དཔོན་...ས།
- chained: `... ཞེས་དང༌། ...` linking two passages

A closer after it: `ཞེས་` / `ཅེས་` plus གསུངས་སོ · སོ · དང༌ · པ་ལྟར · པ་སྟེ · འབྱུང · སོགས,
or a bare `གསུངས།`.

**Every one of these is permission to look, not proof.** Measured on this corpus,
`ལས།` occurs about once per 1,400 characters and most occurrences open nothing;
`ཇི་སྐད་དུ།`, the cleanest opener in the language, opens a quotation under half the
time. In the other direction, roughly three quarters of real quotations have no
source frame at all and more than half have no closer nearby - the frame is often
in the previous window, or the citation is made by register alone. Use the markers
to find candidates; decide by whose voice it is.

## The ལས trap

Three different things are spelled with ལས:

1. **ལས།** - shad, right after a work title, an author, or དེ་ཉིད. Ablative "from";
   a citation frame. `མདོ་ལས།` · `སྤྱོད་འཇུག་ལས།` · `ཡིད་བཞིན་རིན་པོ་ཆེའི་མཛོད་ལས།`
2. **ལས།** - shad, but after a number, a quantity, or ཙམ / དགོས་པ. Partitive ("of
   the three, the first") or adversative ("merely X, yet Y"). Opens nothing.
   `རྒྱུ་མཚན་གསུམ་ལས། དང་པོ།` · `ཁམས་གསུམ་ལས།` · `བཏགས་པ་ཙམ་ལས།` · `རྙེད་དགོས་པ་ལས།`
   In the gold data this shape opens a quotation six times less often than the
   title shape.
3. **ལས་** - *tsheg*, not shad. The noun ལས (action, karma) or a mid-phrase
   ablative: `ལས་ཀྱི་གཏུམ་མོ` · `ལས་དང་པོ་པ`. About 2.6 times more frequent than ལས།,
   and never a citation frame.

ལས earns attention only when a shad follows it *and* a work or a person stands
before it. The same caution applies to ནས and དུ.

## Boundaries

- A span is the **quoted words only**. The frame (`<title>ལས།`, `ཇི་སྐད་དུ།`) and the
  closer (`ཞེས་གསུངས་སོ།`) stay **outside** it.
- Several stanzas under one frame are one span, and an internal `ཞེས་དང༌།` joining
  fragments of the *same* citation stays inside that span.
- Two quotations with **different named sources** are two spans, even back to back.
- A quotation cut by the edge of the chunk: mark the part that is present.

Expect median ~110 characters, ninetieth percentile ~270, with a tail past 1,000.
Prose citations from sutras run long. Do not stop at the end of the first line of
verse.

## Do not mark

- **Root text.** In a commentary the base text being explained is lifted out piece
  by piece and looks exactly like a quotation. Recognise it by: an outline heading
  ending in **ནི།** followed by a line break (`དང་པོ་༼...༽ནི།`), **no named source**,
  verse layout, and prose immediately after that re-uses its very words. A named
  source makes it a QUOTE; an outline topic does not.
- The commentator's own argument, even when it ends ...ཡིན་ནོ། / ...ཕྱིར་རོ།, and
  even when it contains ཞེས inside a subordinate clause.
- A partitive or adversative ལས།: `...གསུམ་ལས། དང་པོ།`, `...ཙམ་ལས།`.
- Outline headings and enumerations: དང་པོ་ནི། · གཉིས་པ་ལ་གསུམ།
- A frame or closer standing alone. `ཞེས་གསུངས་སོ།` is never a span.
- A work title with no quoted words after it - a reference, not a quotation.
- Colophons, printing notes, dedications, lineage lists.

Many chunks contain no quotation, and an empty list is often right. Do not use it
as a hedge, though: an unframed canonical passage inside commentary is still a QUOTE.

## Worked examples

**1 - framed verse, closed**

Text: ...བཙལ་ནས་སླར་དམིགས་པ་ཤོར་ཏེ་འགྲོ་བས། དབུས་མཐའ་རྣམ་འབྱེད་དུ། ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ། །ཞེས་གསུངས། དེ་སྤོང་བའི་...

```json
{"spans": [{"label": "QUOTE", "text": "ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ།"}]}
```

Title and closer stay outside the span.

**2 - one frame, two fragments joined by ཞེས་དང༌། - one span**

Text: ...རྐྱེན་དང་ལྡན་པ་ཡིན་པ་ཞིག་དགོས་ཏེ། ཇི་སྐད་དུ། དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར། ཞེས་གསུངས་སོ། །དེས་མཚོན་ནས་...

```json
{"spans": [{"label": "QUOTE", "text": "དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར"}]}
```

**3 - two different sources back to back - two spans, and both are prose**

Text: ...བདག་དང་བདག་མེད་ཀྱི་སྤྲོས་པ་ཞི་བ་ཞེས་ཀྱང་བྱ་སྟེ། མྱང་འདས་རྒྱ་གར་མ་ལས། བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་ཀྱིས་བདག་མེད་པར་གསུངས་པ་ལ་ཆོས་ཐམས་ཅད་མེད་པར་འཛིན་ཏོ། །ཤེས་རབ་ཅན་རྣམས་ཀྱིས་ནི་བདག་ཡོད་པ་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པར་ཤེས་ཏེ་དེ་ནི་ཤེས་རབ་ཅན་རྣམས་ཀྱི་རང་བཞིན་ཡིན་ནོ། །ཞེས་དང༌། དྲི་མ་མེད་པར་གྲགས་པའི་མདོ་ལས། བདག་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པ་དེ་ནི་བདག་མེད་པའི་དོན་ཏོ། །ཞེས་སོ། །...

```json
{"spans": [{"label": "QUOTE", "text": "བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་ཀྱིས་བདག་མེད་པར་གསུངས་པ་ལ་ཆོས་ཐམས་ཅད་མེད་པར་འཛིན་ཏོ། །ཤེས་རབ་ཅན་རྣམས་ཀྱིས་ནི་བདག་ཡོད་པ་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པར་ཤེས་ཏེ་དེ་ནི་ཤེས་རབ་ཅན་རྣམས་ཀྱི་རང་བཞིན་ཡིན་ནོ། །"}, {"label": "QUOTE", "text": "བདག་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པ་དེ་ནི་བདག་མེད་པའི་དོན་ཏོ། །"}]}
```

Two named sources, so two spans.

**4 - the frame is a person, not a book**

Text: ...དུས་སྐབས་ཀྱིས་གཙོ་བོར་གང་འགྱུར་སྤྱད་པར་གསུངས་པ་དང༌། པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས།\nསྡིག་ཏོ་མི་དགེའི་ཕྱོགས་དང་ཚོགས་པའི་གསེབ། །\nའོག་མ་གཙོར་སྦྱོང་འདོད་པས་དབེན་པ་དང༌། །\nསྤྱོད་པའི་དུས་དང་དབེན་པར་གསང་སྔགས་སྤྱད། །\nནང་མ་འདོམ་ན་མ་འདྲེས་ཡོངས་རྫོགས་བསྲུང༌། །\nའདོམ་ན་དགག་དགོས་བརྩི་ཞེས་མཁས་རྣམས་བཞེད། །\nཅེས་གསུངས་སོ། །\n...

```json
{"spans": [{"label": "QUOTE", "text": "སྡིག་ཏོ་མི་དགེའི་ཕྱོགས་དང་ཚོགས་པའི་གསེབ། །\nའོག་མ་གཙོར་སྦྱོང་འདོད་པས་དབེན་པ་དང༌། །\nསྤྱོད་པའི་དུས་དང་དབེན་པར་གསང་སྔགས་སྤྱད། །\nནང་མ་འདོམ་ན་མ་འདྲེས་ཡོངས་རྫོགས་བསྲུང༌། །\nའདོམ་ན་དགག་དགོས་བརྩི་ཞེས་མཁས་རྣམས་བཞེད། །"}]}
```

No ལས anywhere: the frame is a name in the ergative, and all five lines are one span.

**5 - root text and a quotation in one window; only the quotation is marked**

Text: ...དང་པོ་༼མདོར་བསྡུས་ཏེ་བསྟན་པ་༽ནི།\nལུང་ནི་ངོ་བོའི་སྒོ་ནས་རྣམ་པ་བཅུས། །\nལེགས་གསུངས་བཀའ་དང་དགོངས་འགྲེལ་བསྟན་བཅོས་གཉིས། །\nདེ་ལ་ལུང་གི་དམ་པའི་ཆོས་ཀྱང་ངོ་བོའི་སྒོ་ནས་དབྱེ་ན་རྣམ་པ་བཅུས་ལེགས་པར་གསུངས་པའི་བཀའ་དང༌། བཀའི་དགོངས་པ་འགྲེལ་པའི་བསྟན་བཅོས་གཉིས་ཏེ། ལྷའི་བུས་ཞུས་པའི་མདོ་ལས། ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །ཞེས་གསུངས་སོ། །...

```json
{"spans": [{"label": "QUOTE", "text": "ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །"}]}
```

The first verse follows an outline heading ending in `ནི།`, names no source, and is
glossed word by word after - root text. The second is fetched from a named sutra.

**6 - nothing to mark, despite ལས། and ཞེས**

Text: ...དེ་གང་ལ་གདགས་སྙམ་ན་གདགས་གཞི་སངས་རྒྱས་མཚན་དཔེ་སོགས་ཀྱི་ཆོས་ལ་བཏགས་པ་ཙམ་ལས། མཚན་དཔེའི་ཆོས་སོགས་རེ་རེ་ནས་སྐྱེས་བུ་ཆེན་པོའི་གང་ཟག་ཡིན་ཞེས་སུས་ཀྱང་བརྗོད་པར་མི་ནུས་ན་...

```json
{"spans": []}
```

`ཙམ་ལས།` is adversative - "merely imputed, yet..." - and the ཞེས sits inside the
commentator's own subordinate clause.

## Output

Return the spans in the order they appear in the chunk. No offsets, no indices, no
translations, no explanations, no markdown fences.

Reply with JSON only: `{"spans":[{"label":"QUOTE","text":"..."}]}`
Each `text` copied character for character. If none: `{"spans": []}`.
Text: «chunk»

---

# Gemini prompt - QUOTE only, anchor output

Built 2026-09-03. Same detection rules as `gemini_quote_only.md`; only the output
contract differs. Instead of the whole quotation, the model returns the first ~20
and last ~20 characters of each span, and our code recovers the middle by string
search in the window.

Measured on 9,586 gold Quotation spans: anchors are **27.9% of the characters**
that full-text output would emit, a head anchor is non-unique inside its window
only **0.8%** of the time, and **24%** of spans are 40 characters or shorter, so
they are returned whole and cost nothing extra.

Anchors are snapped to syllable boundaries (they never start or end mid-syllable),
which costs nothing and makes them far easier for the model to copy correctly.

This is format C in `results/llm-move-round-2.md` section 4 - the untested one. Worth
running against full-text output on the same windows before trusting it.

## How the caller resolves an anchor

    def resolve(window, head, tail):
        i = window.find(head)
        if i < 0: return None
        if not tail: return (i, i + len(head))
        j = window.find(tail, i + len(head) - len(tail))
        if j < 0: return None
        return (i, j + len(tail))

`layer_detection.gemini_locate` does this with three additions, all of which
matter once the window is a whole book rather than a chunk:

- **A forward-only cursor.** Spans come back in document order, so each head is
  searched from the end of the previous span. A 20-character head is not unique
  in a million characters, and without the cursor it lands on the first lookalike.
- **A cap on the head-to-tail distance** (`--max-len`, default 2000). A tail the
  model mistyped is otherwise found thousands of characters later, and the span
  swallows everything between. On P000152 this turned nine spans of 2k-23k
  characters into clean misses.
- **Normalization before giving up**: whitespace, `༑` for `།`, and a tsheg
  before a shad (`འཁོར་། །` for `འཁོར། །`), which the model emits often enough
  to matter.

Log the failure rate and the multiple-match rate. If `window.count(head) > 1`,
prefer the occurrence nearest a citation frame, and count it as ambiguous.

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

## Output format - anchors, not the whole quotation

For each quotation return only its two ends:

- `head` - the **first 20 characters** of the span, extended forward to the next
  syllable boundary (a ་ or ། or line break) so it never stops mid-syllable.
- `tail` - the **last 20 characters** of the span, extended backward the same way
  so it never starts mid-syllable.
- If the whole span is **40 characters or shorter**, put the entire span in `head`
  and set `tail` to `""`.

Both strings must be copied from the chunk character for character - every tsheg ་,
shad །, double shad ། །, and line break exactly as printed. Do not normalise, do
not translate, do not abbreviate with `...`. `head` must appear before `tail` in
the chunk, and the two must not overlap.

If the first 20 characters of a span also occur elsewhere in this chunk, lengthen
`head` (still stopping at a syllable boundary) until it is unique.

## Worked examples

**1 - framed verse, closed**

Text: ...བཙལ་ནས་སླར་དམིགས་པ་ཤོར་ཏེ་འགྲོ་བས། དབུས་མཐའ་རྣམ་འབྱེད་དུ། ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ། །ཞེས་གསུངས། དེ་སྤོང་བའི་...

```json
{"spans": [{"label": "QUOTE", "head": "ལེ་ལོ་དང་ནི་གདམས་ངག་", "tail": "དག་ཉེས་པ་ལྔར་འདོད་དོ།"}]}
```

Full quotation is 110 characters; the head and tail alone locate it. Title and
closer stay outside.

**2 - one frame, two fragments joined by ཞེས་དང༌། - one span**

Text: ...རྐྱེན་དང་ལྡན་པ་ཡིན་པ་ཞིག་དགོས་ཏེ། ཇི་སྐད་དུ། དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར། ཞེས་གསུངས་སོ། །དེས་མཚོན་ནས་...

```json
{"spans": [{"label": "QUOTE", "head": "དང་པོར་མ་ཡིན་ཉིད་དག་", "tail": "སུ་ལོངས་སྤྱད་པའི་ཕྱིར"}]}
```

**3 - two different sources back to back - two spans, and both are prose**

Text: ...བདག་དང་བདག་མེད་ཀྱི་སྤྲོས་པ་ཞི་བ་ཞེས་ཀྱང་བྱ་སྟེ། མྱང་འདས་རྒྱ་གར་མ་ལས། བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་ཀྱིས་བདག་མེད་པར་གསུངས་པ་ལ་ཆོས་ཐམས་ཅད་མེད་པར་འཛིན་ཏོ། །ཤེས་རབ་ཅན་རྣམས་ཀྱིས་ནི་བདག་ཡོད་པ་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པར་ཤེས་ཏེ་དེ་ནི་ཤེས་རབ་ཅན་རྣམས་ཀྱི་རང་བཞིན་ཡིན་ནོ། །ཞེས་དང༌། དྲི་མ་མེད་པར་གྲགས་པའི་མདོ་ལས། བདག་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པ་དེ་ནི་བདག་མེད་པའི་དོན་ཏོ། །ཞེས་སོ། །...

```json
{"spans": [{"label": "QUOTE", "head": "བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་", "tail": "ཀྱི་རང་བཞིན་ཡིན་ནོ། །"}, {"label": "QUOTE", "head": "བདག་དང་བདག་མེད་པ་གཉིས་", "tail": "བདག་མེད་པའི་དོན་ཏོ། །"}]}
```

The second quotation is 58 characters, over 40, so it still gets a head and a
tail rather than being returned whole.

**4 - the frame is a person, not a book; the span spans five lines**

Text: ...དུས་སྐབས་ཀྱིས་གཙོ་བོར་གང་འགྱུར་སྤྱད་པར་གསུངས་པ་དང༌། པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས།\nསྡིག་ཏོ་མི་དགེའི་ཕྱོགས་དང་ཚོགས་པའི་གསེབ། །\nའོག་མ་གཙོར་སྦྱོང་འདོད་པས་དབེན་པ་དང༌། །\nསྤྱོད་པའི་དུས་དང་དབེན་པར་གསང་སྔགས་སྤྱད། །\nནང་མ་འདོམ་ན་མ་འདྲེས་ཡོངས་རྫོགས་བསྲུང༌། །\nའདོམ་ན་དགག་དགོས་བརྩི་ཞེས་མཁས་རྣམས་བཞེད། །\nཅེས་གསུངས་སོ། །\n...

```json
{"spans": [{"label": "QUOTE", "head": "སྡིག་ཏོ་མི་དགེའི་ཕྱོགས་", "tail": "ཞེས་མཁས་རྣམས་བཞེད། །"}]}
```

The tail is the last line only. The three middle lines are never emitted - that is
the whole point of this format.

**5 - root text and a quotation in one window; only the quotation is marked**

Text: ...དང་པོ་༼མདོར་བསྡུས་ཏེ་བསྟན་པ་༽ནི།\nལུང་ནི་ངོ་བོའི་སྒོ་ནས་རྣམ་པ་བཅུས། །\nལེགས་གསུངས་བཀའ་དང་དགོངས་འགྲེལ་བསྟན་བཅོས་གཉིས། །\nདེ་ལ་ལུང་གི་དམ་པའི་ཆོས་ཀྱང་ངོ་བོའི་སྒོ་ནས་དབྱེ་ན་རྣམ་པ་བཅུས་ལེགས་པར་གསུངས་པའི་བཀའ་དང༌། བཀའི་དགོངས་པ་འགྲེལ་པའི་བསྟན་བཅོས་གཉིས་ཏེ། ལྷའི་བུས་ཞུས་པའི་མདོ་ལས། ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །ཞེས་གསུངས་སོ། །...

```json
{"spans": [{"label": "QUOTE", "head": "ཆོས་རྣམས་ཐམས་ཅད་བཀའ་", "tail": "དེ་ཡི་དགོངས་འགྲེལ་བ། །"}]}
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

Reply with JSON only: `{"spans":[{"label":"QUOTE","head":"...","tail":"..."}]}`
`head` and `tail` copied character for character. If the span is 40 characters or
shorter, put it all in `head` and leave `tail` empty. If none: `{"spans": []}`.
Text: «chunk»

---

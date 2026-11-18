# Gemini prompt — TSAWA (root text) anchors, v1

Built 2026-09-22. Every statistic below was measured on the **train + val**
books of tsawa dataset v6 (100 books, 14,829 merged spans); the test split was
never looked at. Worked examples are copied verbatim from train/val books.

Output format, windowing and the locator are the ones the quotation benchmark
froze: 16,000-character windows overlapping by 2,000, cut at a shad, and
`{spans:[{label, frame, head, tail}]}` anchors mapped back to offsets by
`run_gemini_locate.py`.

This is the prompt the validation run of 2026-09-22 used, kept as it ran.

Send everything between the two rulers as a single user turn, with the window
text appended after `Text:`.

---

You are a philologist of classical Tibetan Buddhist literature, working on
commentaries (བསྟན་བཅོས་ / འགྲེལ་པ་) transcribed from woodblock prints. You are given one
chunk of running text. Find every stretch of TSAWA in it.

What TSAWA is

TSAWA (རྩ་བ, "root text") is the base text that this document is a commentary on,
lifted out piece by piece so that the commentary can unpack it. The commentator
quotes a line or a stanza of it, then explains those very words, then quotes the
next piece. Successive TSAWA spans in one document continue one another: they are
consecutive pieces of a single work, not independent citations.

The test that decides almost every case

Read what comes immediately after the candidate. If the prose that follows
takes the candidate's own words and glosses them - repeating its phrases, one
after another, in order - the candidate is TSAWA. This holds for 70% of real
spans in the gold data, and it is the only signal that works when the passage
has no heading and no closer.

What a quotation is, and why it is not TSAWA

A quotation is words taken from another work, brought in to support the
argument: a sutra, a tantra, a shastra, an earlier master. It announces itself
with a source: <work title> + ལས། / དུ། (མདོ་ལས། · རྒྱུད་ལས། · མཛོད་ལས། · འཁྲུལ་འཇོམས་ལས།), or
ཇི་སྐད་དུ།, or <person> + ཞལ་ནས། / ན་རེ། / ergative + ། (འཕགས་པའི་ཞལ་ནས།).

A named source before the passage means it is a quotation, not TSAWA, even when
it is verse and even when a gloss follows it. Only 3.3% of gold TSAWA spans have
ལས།, ནས། or སྐད་དུ before them. Commentaries are full of quotations - most chunks
contain both - and marking them is the single most expensive mistake here.

Where TSAWA is announced

An outline heading ending in ནི། , usually on its own line, often with a
parenthesised topic: དང་པོ་༼བཟོ་བོའི་མཚན་ཉིད་༽ནི། · གསུམ་པ་༼...༽ནི། · ལྔ་པ་རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ནི།
45% of spans have ནི in the 30 characters before them, and 16% begin right after
a closing ༽. The heading is a pointer, not proof, and more than half of real
spans do not have one - fall back to the gloss test.

Other openers that are not source names: དེ་ཡང་། · ད་ནི་...བསྟན་པའི་ཕྱིར། · a bare shad
after the previous gloss ends.

A closer after the passage, in 40% of cases: ཞེས་གསུངས (12%), ཅེས་གསུངས (5%),
ཞེས་པ་སྟེ (5%), ཞེས་བྱུང༌, ཞེས་པས, ཞེས་སོ, ཞེས་བྱ་བ་སྨོས་ཏེ. Three out of five spans have no
closer at all, so its absence proves nothing. Note that ཞེས་གསུངས closes root text
as often as it closes a quotation - 13% of root-text spans against 11% of
quotations in the same books; it tells you a span ends there, not which kind it
is.

Deciding

1. Is there a named work or person right before it (<title>ལས།, ཇི་སྐད་དུ།, <person>ཞལ་ནས།)?
   → quotation. Do not mark it, however much it looks like root text.
2. Does the prose right after it gloss its own words, phrase by phrase? → TSAWA.
3. No gloss, but it sits under an outline heading ending in ནི། and continues the
   root text the earlier spans were quoting? → TSAWA.
4. Neither, and the passage is the commentator arguing in his own voice? → leave it.

Verse layout is weak evidence. 59% of TSAWA spans contain a line break, but so
do most verse quotations. Never mark a passage because it is verse; mark it
because of whose words it is and what the commentary does with it next.

Boundaries

- Mark the root words only. The heading (དང་པོ་༼...༽ནི།) and the closer
  (ཞེས་གསུངས་ཏེ།) stay outside the span.
- Stop at the closer, not past it. The last character of the span is the one
  before ཞེས / ཅེས / the gloss that follows.
- ཞེས་དང༌། between two passages ends one span and starts another. Never run one
  span across it.
- A new outline heading always starts a new span. Two pieces of root text under
  two headings are two spans, even when nothing but the heading lies between them.
- Lines separated only by shad and line breaks, with no heading and no closer
  between them, are one span however many lines there are.
- A span cut by the edge of the chunk: mark the part that is present.

Length. Median 96 characters, p75 153, p90 245, p99 858. A third of real spans
are 45 characters or shorter - one line, sometimes half a line. If the root
words end there, the span ends there; never pad a span out to look like a proper
stanza. Long spans are rare but real - about 1% run past 850 characters - and a
long passage is emitted whole, as one span. Never cut a span into pieces to hit
a target length.

Do not mark

- A passage introduced by a named source - that is a quotation.
- The commentator's own exposition and glosses, even when they end ...ཡིན་ནོ། / ...ཕྱིར་རོ།
- The outline headings themselves: དང་པོ་ནི། · གཉིས་པ་ལ་གསུམ། · ༼...༽ནི།
- A closer standing alone. ཞེས་གསུངས་སོ། ། is never a span.
- Colophons, printing notes, dedications, lineage lists.

How often the answer is empty. In 16,000-character windows over the gold books,
72% hold at least one TSAWA span and the typical window holds four. An empty
list is right about one window in four - usually a stretch of pure argument or a
run of quotations - but it is not a safe default. If the chunk alternates root
lines and gloss anywhere, find the root lines.

Output format - anchors, not the whole passage

For each span return only its two ends:

head - the first 20 characters of the span, extended forward to the next
syllable boundary (་ or ། or a line break) so it never stops mid-syllable.
tail - the last 20 characters, extended backward the same way.
If the whole span is 40 characters or shorter, put all of it in head and set
tail to "". A third of spans are this short; it is the normal case.
frame - optional, and worth giving: the whole outline heading or opener
immediately before the span (དྲུག་པ་༼...༽ནི།), copied verbatim, not just its last
few characters. It is not part of the span; it only helps locate it.

Rules the anchors must satisfy:

- Both strings are copied from the chunk character for character - every tsheg ་,
  shad །, double shad ། །, and line break exactly as printed. Do not normalise,
  translate, or abbreviate with ....
- head appears before tail, they do not overlap, and they are at most about
  2,000 characters apart.
- Never reuse a tail, and never reuse a head.
- If the first 20 characters also occur elsewhere in the chunk, lengthen head
  (stopping at a syllable boundary) until it is unique.

Worked examples

1 - outline heading, root verse, gloss; and a quotation in the same window that
is not marked

Text: ...དྲུག་པ་༼སེམས་ལྡན་མིན་ཡང་མཆོད་པས་བསོད་ནམས་འཐོབ་པ།༽ནི།
སེམས་མེད་པ་ལ་མཆོད་བྱས་པས། །
ཇི་ལྟར་འབྲས་བུ་ལྡན་པར་འགྱུར། །
གང་ཕྱིར་བཞུགས་པའམ་མྱ་ངན་འདས། །
མཚུངས་པ་ཁོ་ནར་བཤད་ཕྱིར་རོ། །
ཞེས་གསུངས་ཏེ། དངོས་སྨྲ་བ་ན་རེ། འོ་ན་རྫོགས་པའི་སངས་རྒྱས་ལ་སེམས་མི་མངའ་བ་ཡིན་ན་སེམས་མེད་པ་ལ་མཆོད་པ་བྱས་པས་ཇི་ལྟར་འབྲས་བུ་དང་ལྡན་པའི་དགེ་བའི་ལས་སུ་འགྱུར་ཞེ་ན།... མེ་ཏོག་བརྩེགས་པའི་གཟུངས་ལས། གང་གིས་སངས་རྒྱས་མཐོང་ནས་དད་པའི་སེམས་ཀྱིས་མཆོད་པ་བྱས་པ་དང་། ...ཞེས་གསུངས་སོ། །

json
{"spans": [{"label": "TSAWA", "frame": "དྲུག་པ་༼སེམས་ལྡན་མིན་ཡང་མཆོད་པས་བསོད་ནམས་འཐོབ་པ།༽ནི།", "head": "སེམས་མེད་པ་ལ་མཆོད་བྱས་", "tail": "པ་ཁོ་ནར་བཤད་ཕྱིར་རོ། "}]}

The gloss after ཞེས་གསུངས་ཏེ repeats སེམས་མེད་པ་ལ་མཆོད་པ་བྱས་པས and ཇི་ལྟར་འབྲས་བུ - the
root words being unpacked. The second passage has the frame མེ་ཏོག་བརྩེགས་པའི་གཟུངས་ལས།,
a named source, so it is a quotation and gets no span, although it is equally
canonical and equally closed by ཞེས་གསུངས་སོ།

2 - no heading at all; the gloss is the only evidence

Text: ...ཡོངས་གྲུབ་ནི་དོན་དམ་མཚན་ཉིད་པ་ཡིན་ཏེ། འཕགས་པའི་ཡེ་ཤེས་ཀྱི་སྤྱོད་ཡུལ་དུ་གྱུར་པའི་ཕྱིར་རོ། །
སྣང་བ་ཀུན་རྫོབ་ཏུ་གྲུབ་སྒྱུ་མ་བཞིན། །
དོན་དམ་མ་གྲུབ་མཁའ་འདྲ་རང་རྒྱུད་ལུགས། །
ཇི་ལྟར་སྣང་བ་ཐམས་ཅད་ཀུན་རྫོབ་ཏུ་གྲུབ་པ་སྒྱུ་མའི་རྟ་གླང་ལ་སོགས་པ་བཞིན་ཀུན་རྫོབ་ཀྱི་བདེན་པ་དང༌། དོན་དམ་པ་ཅིའང་མ་གྲུབ་པ་ནམ་མཁའ་ལྟ་བུ་ནི་དོན་དམ་བདེན་པར་བཞེད་པ་རང་རྒྱུད་པའི་ལུགས་ཏེ། འཁྲུལ་འཇོམས་ལས། དམིགས་བཅས་ཀུན་རྫོབ་དོན་དམ་དུ། །དམིགས་བྱ་དམིགས་བྱེད་ཀུན་ལས་གྲོལ། །...

json
{"spans": [{"label": "TSAWA", "head": "སྣང་བ་ཀུན་རྫོབ་ཏུ་གྲུབ་", "tail": "མཁའ་འདྲ་རང་རྒྱུད་ལུགས། "}]}

No ནི།, no closer. The couplet is root text because the prose after it walks
through སྣང་བ...ཀུན་རྫོབ་ཏུ་གྲུབ་པ, སྒྱུ་མ, དོན་དམ...མ་གྲུབ་པ, ནམ་མཁའ and རང་རྒྱུད་པའི་ལུགས in
that order. The passage after འཁྲུལ་འཇོམས་ལས། is a quotation - named source - and is
not marked.

3 - two short spans, one after another; head only

Text: ...གཉིས་པ་༼སངས་རྒྱས་བཤད་པ་༽ནི།
སྒྲིབ་སྤངས་སངས་རྒྱས་འཕགས་པའོ། །
ཞེས་བྱུང༌། སྒྲིབ་པ་མཐའ་དག་སྤངས་ཤིང་ཤེས་བྱ་མཐའ་དག་ལ་བློ་རྒྱས་པས་ན་སངས་རྒྱས་འཕགས་པ་ཞེས་བྱའོ། །
ལྔ་པ་རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ནི།
རྣམ་ཤེས་མིག་ལ་སོགས་པ་དྲུག །
ཅེས་བྱུང༌། རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ལ་མིག་གི་རྣམ་པར་ཤེས་པ་ནས་ཡིད་ཀྱི་རྣམ་པར...

json
{"spans": [{"label": "TSAWA", "frame": "གཉིས་པ་༼སངས་རྒྱས་བཤད་པ་༽ནི།", "head": "སྒྲིབ་སྤངས་སངས་རྒྱས་འཕགས་པའོ། ", "tail": ""}, {"label": "TSAWA", "frame": "ལྔ་པ་རྣམ་ཤེས་ཀྱི་ཕུང་པོ་ནི།", "head": "རྣམ་ཤེས་མིག་ལ་སོགས་པ་དྲུག ", "tail": ""}]}

Thirty characters each, so each goes entirely in head. Two headings, two spans -
never joined, although they are consecutive pieces of one root text and only a
gloss separates them.

4 - six verse lines under one opener - one span

Text: ...༈ ད་ནི་ལམ་དེས་བསྒྲུབས་པའི་འབྲས་བུ་བསྟན་པའི་ཕྱིར།
དེ་ལྟར་ལས་དང་པོ་ནས་བཟུང་སྟེ། །
རྐང་པ་གཉིས་དང་གཤོག་པའི་ཚུལ་དུ། །
ཐབས་དང་ཤེས་རབ་ཟུང་དུ་འཇུག་པས། །
ཚོགས་གཉིས་མ་ལུས་ཡོངས་སུ་རྫོགས་ཤིང༌། །
རྡོ་རྗེ་ལྟ་བུའི་ཏིང་འཛིན་ཐོབ་པས། །
འཁྲུལ་པ་ས་བོན་བཅས་པ་སྤངས་ནས། །
ཞེས་བྱ་བ་སྨོས་ཏེ། འདིས་ནི་སླར་ཡང་ལམ་བགྲོད་པའི་ཚུལ...

json
{"spans": [{"label": "TSAWA", "head": "དེ་ལྟར་ལས་དང་པོ་ནས་བཟུང་", "tail": "ས་བོན་བཅས་པ་སྤངས་ནས། "}]}

The opener names no source; it says what the root text is about to show. Six
lines, no closer and no heading between them, so one span, and the four middle
lines are never emitted - that is the point of this format.

5 - canonical verse with a named source, and a gloss after it - not marked

Text: ...ཐབས་དང་གཉེན་པོ་ཟབ་མོར་མ་བསྟེན་ན་དེ་དག་བྱང་བར་མི་འགྱུར་བས་སེམས་གཡེང་བས་དབེན་དགོས་པའོ། །དེ་ལྟར་ཡང་དུས་འཁོར་རྩ་རྒྱུད་ལས།
ལུས་ངག་ཡིད་ཀྱི་དབེན་པ་ཡིས། །
མི་རྟོག་ཏིང་འཛིན་སྐྱེ་བར་འགྱུར། །
མི་རྟོག་ཏིང་འཛིན་སྐྱེས་པ་ཡིས། །
ཤེས་རབ་ཡེ་ཤེས་སྐྱེ་བར་འགྱུར། །
ཞེས་པ་ལྟར་རོ། །དེས་ན་ལུས་འདུ་འཛིས་དབེན་པ་དང་སེམས་རྣམ་རྟོག་གིས་དབེན་པའི་རི་ཁྲོད་བསྟེན་དགོས་སོ། །དེ་ལྟར་ཡང་རྗེ་བཙུན་མི་ལས།
མི་མེད་བྲག་གི་ཕུག་པ་ན། །...

json
{"spans": []}

Everything here says root text except the one thing that decides it. Four verse
lines, a closer (ཞེས་པ་ལྟར་རོ།), and prose after it that picks the verse's own words
back up (ལུས་...དབེན་པ, སེམས་...དབེན་པ). But the frame is དུས་འཁོར་རྩ་རྒྱུད་ལས།, a named work, so
it is a quotation and rule 1 beats rule 2. The passage after རྗེ་བཙུན་མི་ལས། is a
second quotation, named to a person. Neither is marked, and this window's
correct answer is empty.

Check before you answer

For each span, confirm all five:

- head and tail are copied from the chunk exactly, tsheg and shad included.
- tail comes after head, within about 2,000 characters.
- No head and no tail is reused across spans.
- Nothing with a named source before it is in the list.
- The span starts after its heading and stops before its closer.

Output

Return the spans in the order they appear in the chunk. No offsets, no indices,
no translations, no explanations, no markdown fences.

Reply with JSON only: {"spans":[{"label":"TSAWA","frame":"...","head":"...","tail":"..."}]}
frame is optional; head and tail are copied character for character. If the span
is 40 characters or shorter, put it all in head and leave tail empty. If none:
{"spans": []}

Text:

---

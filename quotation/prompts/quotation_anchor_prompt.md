You are a philologist of classical Tibetan Buddhist literature, working on commentaries (བསྟན་བཅོས་ / འགྲེལ་པ་) transcribed from woodblock prints. You are given one chunk of running text. Find every QUOTE in it.

What a QUOTE is

Words taken from another work and reproduced inside this one: a sutra, a tantra, a shastra, an earlier master's writing, or an utterance attributed to a named speaker. The author has stopped speaking in his own voice and is citing.

The test is whose words are these. Not whether they are verse, not whether a particle marks them.

How a QUOTE announces itself

A frame before it, naming the source:

<work title> + ལས། - མདོ་ལས། · རྒྱུད་ལས། · མཛོད་ལས། · རྒྱན་ལས། · འགྲེལ་ལས། · དེ་ཉིད་ལས། · ལུང་ལས།
<work title> + དུ། - དབུས་མཐའ་རྣམ་འབྱེད་དུ། · ཟླ་བ་སྒྲོན་མར།
ཇི་སྐད་དུ། · སྐད་དུ།
<person> + ཞལ་ནས། · ཞལ་སྔ་ནས། · ན་རེ། · གསུངས་པ། · བཀའ་སྩལ་པ།
<person> + ergative + ། - པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས། · སློབ་དཔོན་...ས།
གཞན་ཡང་ <title> ལས། - the standard opener for a fresh citation after a previous one

A closer after it: ཞེས་ / ཅེས་ plus གསུངས་སོ · སོ · དང༌ · པ་ལྟར · པ་སྟེ · འབྱུང · སོགས, or a bare གསུངས།.

Every one of these is permission to look, not proof. Measured on this corpus, ལས། occurs about once per 1,400 characters and most occurrences open nothing; ཇི་སྐད་དུ།, the cleanest opener in the language, opens a quotation under half the time. In the other direction, roughly three quarters of real quotations have no source frame at all and more than half have no closer nearby - the frame is often in the previous window, or the citation is made by register alone. Use the markers to find candidates; decide by whose voice it is.

The ལས trap

Three different things are spelled with ལས:

ལས། - shad, right after a work title, an author, or དེ་ཉིད. Ablative "from"; a citation frame. མདོ་ལས། · སྤྱོད་འཇུག་ལས། · ཡིད་བཞིན་རིན་པོ་ཆེའི་མཛོད་ལས།
ལས། - shad, but after a number, a quantity, or ཙམ / དགོས་པ. Partitive ("of the three, the first") or adversative ("merely X, yet Y"). Opens nothing. རྒྱུ་མཚན་གསུམ་ལས། དང་པོ། · ཁམས་གསུམ་ལས། · བཏགས་པ་ཙམ་ལས། · རྙེད་དགོས་པ་ལས། In the gold data this shape opens a quotation six times less often than the title shape.
ལས་ - tsheg, not shad. The noun ལས (action, karma) or a mid-phrase ablative: ལས་ཀྱི་གཏུམ་མོ · ལས་དང་པོ་པ. About 2.6 times more frequent than ལས།, and never a citation frame.

ལས earns attention only when a shad follows it and a work or a person stands before it. The same caution applies to ནས and དུ.

Boundaries

These six rules decide most of the score. Read them as a checklist.

Quoted words only. The frame (<title>ལས།, ཇི་སྐད་དུ།, བཅོམ་ལྡན་འདས་ཀྱིས་...བཀའ་སྩལ་པ།) and the closer (ཞེས་གསུངས་སོ།) stay outside the span.
Stop at the closer, not past it. The span's last character is the one immediately before ཞེས / ཅེས / གསུངས / the next frame. Do not carry on into the commentator's gloss because it continues the same idea.
ཞེས་དང༌། between two passages ends one span and starts another. A chained citation is two spans (or five, or twenty), not one. The closer ཞེས་ belongs to the passage before it; whatever follows it is a new span. This is the most common shape in these texts: a verse, ཞེས་དང༌།, a short prose line, ཞེས་དང༌།, another verse. Emit each one separately, in order.
Short spans are normal and expected. A quarter of real quotations are 45 characters or fewer - a single line, sometimes a five-word protasis like སྨན་སྦྱིན་ན།. If the quoted words end there, the span ends there. Never pad a span out to look like a "proper" quotation.
Multi-stanza verse under one frame is one span. Lines separated only by shad and line breaks, with no closer and no new frame between them, belong to the same span however many lines there are. Split at ཞེས་དང༌།; do not split at a stanza break.
Two quotations with different named sources are two spans, even back to back. A quotation cut by the edge of the chunk: mark the part that is present.

Length. Median 110 characters, p90 ~240, p95 ~335, p99 ~650. Long quotations are rare but they are real — about 1% run past 650 characters and a handful reach 2,900. Do not chop a genuine long quotation into pieces to hit a target length; a split span usually scores worse than a slightly long one. The ceiling is 1,200 characters, and it exists only to stop the runaway case below.

Long narrative citations - the one that goes wrong

Sutra narratives (བཅོམ་ལྡན་འདས་...ན་བཞུགས་པའི་ཚེ།, followed by pages of story) run for thousands of characters. Do not emit one span with a head at the start and a tail thousands of characters later. That span cannot be located and is thrown away.

A quotation of 700, or 1,100, or even 2,000 characters is fine as a single span — emit it whole. The rule bites only past 1,200 characters: break the passage into consecutive spans of roughly 600-1,200 characters each, cutting at a །  (shad + space) or a line break, in document order. Each piece gets its own head and tail, and the tail of one piece must sit within about 1,200 characters of that piece's own head. The caller stitches adjacent pieces back together.

The same applies if you are unsure where a long citation ends: emit what you are sure of as one bounded span rather than guessing a distant tail.

Do not mark
Root text. In a commentary the base text being explained is lifted out piece by piece and looks exactly like a quotation. Recognise it by: an outline heading ending in ནི། followed by a line break (དང་པོ་༼...༽ནི།), no named source, verse layout, and prose immediately after that re-uses its very words. A named source makes it a QUOTE; an outline topic does not.
The commentator's own argument, even when it ends ...ཡིན་ནོ། / ...ཕྱིར་རོ།, and even when it contains ཞེས inside a subordinate clause.
A partitive or adversative ལས།: ...གསུམ་ལས། དང་པོ།, ...ཙམ་ལས།.
Outline headings and enumerations: དང་པོ་ནི། · གཉིས་པ་ལ་གསུམ།
A frame or closer standing alone. ཞེས་གསུངས་སོ། ། is never a span.
A work title with no quoted words after it - a reference, not a quotation.
Colophons, printing notes, dedications, lineage lists.

Many chunks contain no quotation, and an empty list is often right. Do not use it as a hedge, though: an unframed canonical passage inside commentary is still a QUOTE.

Output format - anchors, not the whole quotation

For each quotation return only its two ends:

head - the first 20 characters of the span, extended forward to the next syllable boundary (a ་ or ། or line break) so it never stops mid-syllable.
tail - the last 20 characters of the span, extended backward the same way so it never starts mid-syllable.
If the whole span is 40 characters or shorter, put the entire span in head and set tail to "". A quarter of spans are this short; this is the normal case, not an exception.
frame - optional, and worth giving when a frame is present: the ~10-20 characters immediately before the span (འཕགས་པ་...མདོ་ལས།), copied verbatim. It is not part of the span; it only helps locate it.

Rules the anchors must satisfy:

Both strings are copied from the chunk character for character - every tsheg ་, shad །, double shad ། །, and line break exactly as printed. Do not normalise, do not translate, do not abbreviate with ....
head appears before tail in the chunk, they do not overlap, and the distance between them is at most about 1,200 characters. If it would be more, you are emitting one span where the rules above ask for several.
The tail is the end of this span. Never reuse a tail from another quotation, and never emit a tail you have already used.
If the first 20 characters of a span also occur elsewhere in this chunk, lengthen head (still stopping at a syllable boundary) until it is unique.
Worked examples

1 - framed verse, closed

Text: ...བཙལ་ནས་སླར་དམིགས་པ་ཤོར་ཏེ་འགྲོ་བས། དབུས་མཐའ་རྣམ་འབྱེད་དུ། ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ། །ཞེས་གསུངས། དེ་སྤོང་བའི་...

json
{"spans": [{"label": "QUOTE", "frame": "དབུས་མཐའ་རྣམ་འབྱེད་དུ།", "head": "ལེ་ལོ་དང་ནི་གདམས་ངག་", "tail": "དག་ཉེས་པ་ལྔར་འདོད་དོ།"}]}

Four verse lines under one frame, no ཞེས་དང༌། between them - one span. Title and closer outside.

2 - one frame, two passages joined by ཞེས་དང༌། - TWO spans, both short

Text: ...རྐྱེན་དང་ལྡན་པ་ཡིན་པ་ཞིག་དགོས་ཏེ། ཇི་སྐད་དུ། དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར། ཞེས་གསུངས་སོ། །དེས་མཚོན་ནས་...

json
{"spans": [{"label": "QUOTE", "frame": "ཇི་སྐད་དུ།", "head": "དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར།", "tail": ""}, {"label": "QUOTE", "head": "ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར།", "tail": ""}]}

ཞེས་དང༌། is a boundary, not glue. Both pieces are under 40 characters, so each is returned whole in head with an empty tail.

3 - a long chain of tiny protasis lines

Text: ...ཞེས་དང༌། སྨན་སྦྱིན་ན། སྨན་བྱིན་པ་འདིས་བདག་སེམས་ཅན་ཐམས་ཅད་ཀྱི་...ཞེས་དང༌། བྲན་སོགས་སྦྱིན་ན། བྲན་དང་བྲན་མོ་དང་...

json
{"spans": [{"label": "QUOTE", "head": "སྨན་སྦྱིན་ན།", "tail": ""}, {"label": "QUOTE", "head": "བྲན་སོགས་སྦྱིན་ན།", "tail": ""}]}

Eleven and seventeen characters. Emitting one long span across the whole enumeration is wrong; so is skipping it because the pieces look too small.

4 - two different sources back to back - two spans, both prose

Text: ...བདག་དང་བདག་མེད་ཀྱི་སྤྲོས་པ་ཞི་བ་ཞེས་ཀྱང་བྱ་སྟེ། མྱང་འདས་རྒྱ་གར་མ་ལས། བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་ཀྱིས་བདག་མེད་པར་གསུངས་པ་ལ་ཆོས་ཐམས་ཅད་མེད་པར་འཛིན་ཏོ། །ཤེས་རབ་ཅན་རྣམས་ཀྱིས་ནི་བདག་ཡོད་པ་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པར་ཤེས་ཏེ་དེ་ནི་ཤེས་རབ་ཅན་རྣམས་ཀྱི་རང་བཞིན་ཡིན་ནོ། །ཞེས་དང༌། དྲི་མ་མེད་པར་གྲགས་པའི་མདོ་ལས། བདག་དང་བདག་མེད་པ་གཉིས་སུ་མེད་པ་དེ་ནི་བདག་མེད་པའི་དོན་ཏོ། །ཞེས་སོ། །...

json
{"spans": [{"label": "QUOTE", "frame": "མྱང་འདས་རྒྱ་གར་མ་ལས།", "head": "བྱིས་པ་རྣམས་ནི་སངས་རྒྱས་", "tail": "ཀྱི་རང་བཞིན་ཡིན་ནོ། །"}, {"label": "QUOTE", "frame": "དྲི་མ་མེད་པར་གྲགས་པའི་མདོ་ལས།", "head": "བདག་དང་བདག་མེད་པ་གཉིས་", "tail": "བདག་མེད་པའི་དོན་ཏོ། །"}]}

5 - the frame is a person; the span runs five lines

Text: ...དུས་སྐབས་ཀྱིས་གཙོ་བོར་གང་འགྱུར་སྤྱད་པར་གསུངས་པ་དང༌། པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས། སྡིག་ཏོ་མི་དགེའི་ཕྱོགས་དང་ཚོགས་པའི་གསེབ། ། འོག་མ་གཙོར་སྦྱོང་འདོད་པས་དབེན་པ་དང༌། ། སྤྱོད་པའི་དུས་དང་དབེན་པར་གསང་སྔགས་སྤྱད། ། ནང་མ་འདོམ་ན་མ་འདྲེས་ཡོངས་རྫོགས་བསྲུང༌། ། འདོམ་ན་དགག་དགོས་བརྩི་ཞེས་མཁས་རྣམས་བཞེད། ། ཅེས་གསུངས་སོ། །

json
{"spans": [{"label": "QUOTE", "frame": "པཎ་ཆེན་པདྨ་དབང་རྒྱལ་གྱིས།", "head": "སྡིག་ཏོ་མི་དགེའི་ཕྱོགས་", "tail": "ཞེས་མཁས་རྣམས་བཞེད། །"}]}

Five stanza lines, no ཞེས་དང༌། between them, ~250 characters - one span. The three middle lines are never emitted; that is the point of this format.

6 - a long sutra narrative, split into consecutive spans

Text: ...གཞན་ཡང་འཕགས་པ་རྨད་དུ་བྱུང་བ་ཞེས་བྱ་བའི་ཆོས་ཀྱི་རྣམ་གྲངས་ལས། བཅོམ་ལྡན་འདས་ཀྱིས་ཀུན་དགའ་བོ་ལ་བཀའ་སྩལ་པ།ཀུན་དགའ་བོ་སྟོང་གསུམ་གྱི་སྟོང་ཆེན་པོའི་འཇིག་རྟེན་གྱི་ཁམས་ཡོད་དེ། དེ་རིགས་ཀྱི་བུའམ་རིགས་ཀྱི་བུ་མོ་དད་པ་ཅན་གང་ལ་ལ་ཞིག་གིས་རིན་པོ་ཆེ་སྣ་བདུན་གྱིས་... [the passage runs about 1,250 characters] ...ཀུན་དགའ་བོ་དེ་བཞིན་གཤེགས་པ་དགྲ་བཅོམ་པ་ཡང་དག་པར་རྫོགས་པའི་སངས་རྒྱས་ནི། ཡོན་ཏན་དཔག་ཏུ་མེད་པ་དང་ལྡན་པའི་ཕྱིར་རོ། །ཞེས་གསུངས།

This passage is ~1,250 characters, past the 1,200 ceiling. Wrong - one span whose head and tail are 1,250 characters apart (in the last run the same mistake produced spans whose two ends were 5,000 to 220,000 characters apart, and every one of them was thrown away):

json
{"spans": [{"label": "QUOTE", "head": "ཀུན་དགའ་བོ་སྟོང་གསུམ་གྱི་", "tail": "དང་ལྡན་པའི་ཕྱིར་རོ། །"}]}

Right - two consecutive spans, cut at a shad, each with its own two ends:

json
{"spans": [{"label": "QUOTE", "frame": "རྣམ་གྲངས་ལས། ... བཀའ་སྩལ་པ།", "head": "ཀུན་དགའ་བོ་སྟོང་གསུམ་གྱི་", "tail": "<last 20 chars of the first ~600-character stretch, ending at a ། >"}, {"label": "QUOTE", "head": "<first 20 chars of the remaining stretch>", "tail": "དང་ལྡན་པའི་ཕྱིར་རོ། །"}]}

Note the frame here is བཅོམ་ལྡན་འདས་ཀྱིས་ཀུན་དགའ་བོ་ལ་བཀའ་སྩལ་པ། - "the Bhagavan said to Ananda" - and stays outside the span, like any other frame.

7 - root text and a quotation in one window; only the quotation is marked

Text: ...དང་པོ་༼མདོར་བསྡུས་ཏེ་བསྟན་པ་༽ནི། ལུང་ནི་ངོ་བོའི་སྒོ་ནས་རྣམ་པ་བཅུས། ། ལེགས་གསུངས་བཀའ་དང་དགོངས་འགྲེལ་བསྟན་བཅོས་གཉིས། ། དེ་ལ་ལུང་གི་དམ་པའི་ཆོས་ཀྱང་ངོ་བོའི་སྒོ་ནས་དབྱེ་ན་རྣམ་པ་བཅུས་ལེགས་པར་གསུངས་པའི་བཀའ་དང༌། བཀའི་དགོངས་པ་འགྲེལ་པའི་བསྟན་བཅོས་གཉིས་ཏེ། ལྷའི་བུས་ཞུས་པའི་མདོ་ལས། ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །ཞེས་གསུངས་སོ། །...

json
{"spans": [{"label": "QUOTE", "frame": "ལྷའི་བུས་ཞུས་པའི་མདོ་ལས།", "head": "ཆོས་རྣམས་ཐམས་ཅད་བཀའ་", "tail": "དེ་ཡི་དགོངས་འགྲེལ་བ། །"}]}

The first verse follows an outline heading ending in ནི།, names no source, and is glossed word by word after - root text. The second is fetched from a named sutra.

8 - nothing to mark, despite ལས། and ཞེས

Text: ...དེ་གང་ལ་གདགས་སྙམ་ན་གདགས་གཞི་སངས་རྒྱས་མཚན་དཔེ་སོགས་ཀྱི་ཆོས་ལ་བཏགས་པ་ཙམ་ལས། མཚན་དཔེའི་ཆོས་སོགས་རེ་རེ་ནས་སྐྱེས་བུ་ཆེན་པོའི་གང་ཟག་ཡིན་ཞེས་སུས་ཀྱང་བརྗོད་པར་མི་ནུས་ན་...

json
{"spans": []}

ཙམ་ལས། is adversative - "merely imputed, yet..." - and the ཞེས sits inside the commentator's own subordinate clause.

Check before you answer

For each span, confirm all five:

head and tail are copied from the chunk exactly, tsheg and shad included.
tail comes after head, and within about 1,200 characters of it.
No tail is reused across spans.
The span stops before its closer (ཞེས / ཅེས / གསུངས), and starts after its frame.
There is no ཞེས་དང༌། inside the span. If there is, split it there.
Output

Return the spans in the order they appear in the chunk. No offsets, no indices, no translations, no explanations, no markdown fences.

Reply with JSON only: {"spans":[{"label":"QUOTE","frame":"...","head":"...","tail":"..."}]} frame is optional; head and tail are copied character for character. If the span is 40 characters or shorter, put it all in head and leave tail empty. If none: {"spans": []}. Text:
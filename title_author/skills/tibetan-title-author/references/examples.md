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

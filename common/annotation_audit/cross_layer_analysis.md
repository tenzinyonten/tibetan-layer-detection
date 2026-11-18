# Cross-layer analysis: tsawa, sabche, chapter

Read-only analysis of `data/raw_opf` and the frozen split / cleaned span files. Numbers only. Script: `analyze_cross_layer.py`; detail CSVs: `commentary_no_tsawa_verse.csv`, `cross_layer_conflicts.csv`, `tsawa_offset_shift_test.csv`.

## Analysis 1: books with a Commentary layer and no Tsawa layer

- Books with `Commentary.yml`: 273. With `Tsawa.yml`: 212. Commentary and no Tsawa: **150**. Both: 123. Tsawa and no Commentary: 89.
- Metre test: same as `scan_missed_verses.py`, a run is 4+ consecutive shad-delimited clauses with the same 7, 9, 11 or 13 syllable count. No tsawa exists in these books, so no text is masked. Runs overlapping a Quotation or Citation span are counted separately (`in Quotation`). `in commentary verse` counts runs that overlap a Commentary span flagged `isverse` (the annotator already marked that text as verse).
- Books by unlabelled runs (not in Quotation/Citation): at least 1: 138; at least 20: 61; at least 50: 33; at least 100: 22; none: 12. Total runs 7,707; runs inside Quotation/Citation 2,820.
- Reference rates, unlabelled runs per 100,000 characters (median over books): these 150 commentary-no-tsawa books **12.1**; 177 books with neither a Commentary nor a Tsawa layer 28.8; tsawa books (runs outside their tsawa spans) 5.8.

Top 30 by unlabelled runs:

| pecha_id | text_chars | n_commentary | commentary isverse spans | unlabelled runs | in Quotation | in commentary verse | run chars | runs per 100k chars |
|---|---|---|---|---|---|---|---|---|
| I52F33C02 | 279669 | 383 | 0 | 505 | 0 | 0 | 259560 | 180.6 |
| I6F70ED49 | 711926 | 1454 | 1007 | 458 | 0 | 414 | 159761 | 64.3 |
| I57BB7B28 | 583830 | 384 | 0 | 432 | 0 | 0 | 79200 | 74.0 |
| IACBA255A | 311043 | 187 | 0 | 426 | 0 | 0 | 250082 | 137.0 |
| I2342D649 | 453046 | 171 | 0 | 399 | 0 | 0 | 62198 | 88.1 |
| I7A79CC95 | 739853 | 29 | 0 | 333 | 0 | 0 | 101295 | 45.0 |
| I3D257F64 | 160319 | 301 | 0 | 317 | 0 | 0 | 123890 | 197.7 |
| I79F57A79 | 160644 | 301 | 0 | 317 | 0 | 0 | 124143 | 197.3 |
| I518A01A8 | 290789 | 188 | 0 | 281 | 1 | 0 | 141713 | 96.6 |
| IE1B4BBB9 | 92715 | 402 | 5 | 280 | 0 | 2 | 48613 | 302.0 |
| ID13E3867 | 483768 | 752 | 5 | 270 | 109 | 3 | 33581 | 55.8 |
| I4EE8C636 | 407949 | 140 | 0 | 259 | 0 | 0 | 40006 | 63.5 |
| IDAE0EA08 | 279689 | 122 | 0 | 185 | 0 | 0 | 57887 | 66.1 |
| IFAAADEA1 | 94614 | 129 | 0 | 164 | 0 | 0 | 70231 | 173.3 |
| IB4B89F9C | 425538 | 312 | 0 | 147 | 0 | 0 | 24007 | 34.5 |
| I7C0B99BB | 119011 | 53 | 0 | 146 | 0 | 0 | 103843 | 122.7 |
| IBCF6B0BA | 149555 | 160 | 0 | 146 | 36 | 0 | 21807 | 97.6 |
| I55D7A55C | 30274 | 166 | 0 | 125 | 0 | 0 | 20103 | 412.9 |
| I99DF06AA | 475299 | 327 | 0 | 122 | 134 | 0 | 22097 | 25.7 |
| IC8A4519C | 146968 | 112 | 0 | 105 | 0 | 0 | 118293 | 71.4 |
| I27F687A4 | 87045 | 6 | 0 | 100 | 0 | 0 | 31961 | 114.9 |
| IC421942A | 63180 | 16 | 0 | 100 | 0 | 0 | 43975 | 158.3 |
| IDB6093E9 | 389471 | 518 | 64 | 90 | 19 | 51 | 25659 | 23.1 |
| IB2BA1922 | 790885 | 444 | 7 | 85 | 40 | 5 | 15159 | 10.7 |
| IF4F5D041 | 48607 | 4 | 0 | 78 | 1 | 0 | 38416 | 160.5 |
| IC7A5FC73 | 357663 | 495 | 7 | 77 | 24 | 2 | 12605 | 21.5 |
| I29CFF9F3 | 281600 | 176 | 2 | 75 | 23 | 2 | 12222 | 26.6 |
| IEB5D4BEB | 289482 | 25 | 0 | 74 | 0 | 0 | 27915 | 25.6 |
| I9658BC87 | 559981 | 97 | 0 | 68 | 4 | 0 | 15954 | 12.1 |
| IFFA911F5 | 156768 | 387 | 230 | 66 | 9 | 62 | 19647 | 42.1 |


All books are in `commentary_no_tsawa_verse.csv`.

## Analysis 2: test sets across the three layers

### Test-book overlap (frozen splits)

| layer | books in dataset | train | val | test |
|---|---|---|---|---|
| tsawa | 124 | 84 | 18 | 22 |
| sabche | 323 | 261 | 33 | 29 |
| chapter | 378 | 308 | 34 | 36 |


| test sets | shared test books |
|---|---|
| tsawa&sabche | 21 |
| tsawa&chapter | 16 |
| sabche&chapter | 16 |
| all three | 15 |


Split of each layer's test books in the other layers:

| test books of | their split in | test | val | train | not in dataset |
|---|---|---|---|---|---|
| tsawa | sabche | 21 | 0 | 0 | 1 |
| tsawa | chapter | 16 | 0 | 0 | 6 |
| sabche | tsawa | 21 | 0 | 2 | 6 |
| sabche | chapter | 16 | 0 | 6 | 7 |
| chapter | tsawa | 16 | 0 | 0 | 20 |
| chapter | sabche | 16 | 1 | 10 | 9 |


### Per-layer benchmark statistics (cleaned spans, the files the datasets are built from)

| layer | set | books | spans | median_spans_per_book | old_books | new_books | old_spans | new_spans |
|---|---|---|---|---|---|---|---|---|
| tsawa | test | 22 | 2734 | 75.0 | 10 | 12 | 1203 | 1531 |
| tsawa | all | 124 | 17999 | 82.0 | 56 | 68 | 6528 | 11471 |
| sabche | test | 29 | 4079 | 118.0 | 11 | 18 | 1624 | 2455 |
| sabche | all | 323 | 42292 | 64.0 | 134 | 189 | 18456 | 23836 |
| chapter | test | 36 | 258 | 3.5 | 16 | 20 | 126 | 132 |
| chapter | all | 378 | 3042 | 2.0 | 143 | 235 | 1427 | 1615 |


Span length in characters:

| layer | set | len_p5 | len_p25 | len_median | len_p75 | len_p95 | len_max | short_under5 |
|---|---|---|---|---|---|---|---|---|
| tsawa | test | 4 | 18 | 77 | 155 | 369 | 1863 | 176 |
| tsawa | all | 4 | 25 | 103 | 153 | 353 | 14049 | 903 |
| sabche | test | 12 | 22 | 34 | 69 | 229 | 3171 | 0 |
| sabche | all | 14 | 26 | 39 | 91 | 254 | 5845 | 11 |
| chapter | test | 10 | 19 | 51 | 74 | 122 | 188 | 0 |
| chapter | all | 10 | 24 | 44 | 72 | 123 | 212 | 61 |


Old:new batch ratio by books: tsawa test 10:12, all 56:68; sabche test 11:18, all 134:189; chapter test 16:20, all 143:235.

### Disagreements in books that are test books of two or more layers (cleaned spans)

Books that are test in at least two layers: 23 (I0FCFA88F, I319DAFF7, I3F4A91F5, I52248444, I575514A8, I9AEEF96A, I9B6A4525, I9D9C7AC9, IC05A6BE0, IC6F06BCD, ICDC84458, IE5895799, IF3ACC3E1, P000013, P000027, P000056, P000067, P000083, P000118, P000144, P000164, P000242, P000269).
Overlapping span pairs between the layers' cleaned spans in those books: 388. By pair and kind: tsawa/sabche a_in_b: 18; tsawa/sabche b_in_a: 187; tsawa/sabche partial: 183.

Per book (all rows are in the shared-books comparison above; counts by kind):

| pecha_id | pairs | tsawa_inside_sabche | sabche_inside_tsawa | partial | sabche_verdict | sabche_realigned |
|---|---|---|---|---|---|---|
| I9B6A4525 | 1 | 0 | 1 | 0 | keep | False |
| P000027 | 43 | 0 | 15 | 28 | realigned | True |
| P000083 | 143 | 0 | 65 | 78 | realigned | True |
| P000144 | 54 | 4 | 26 | 24 | realigned | True |
| P000242 | 147 | 14 | 80 | 53 | realigned | True |


For context, the same cleaned-span comparison over every book present in both layers' datasets: tsawa/sabche 1109 overlapping pairs in 32 books; tsawa/chapter 0 overlapping pairs in 0 books; sabche/chapter 2 overlapping pairs in 2 books.

### Where the cleaned tsawa and sabche spans disagree

Books present in both the tsawa and sabche datasets, split by whether the Sabche cleaning step realigned the book (`sabche_book_verdicts.csv`, `realigned=True`: the raw Sabche offsets were shifted, 2-4 offset segments per book):

| Sabche realigned | books | books with a tsawa/sabche overlap | overlapping pairs |
|---|---|---|---|
| True | 23 | 19 | 910 |
| False | 175 | 13 | 199 |


The tsawa spans in the realigned books were not shifted. Test of whether they carry a similar offset: shift every tsawa span by k characters (k from -60 to 60) and measure the share of clauses inside the spans with a 7/9/11/13 syllable metre; `best shift` is the k with the highest share. Books where the best shift gains at least 5 points over shift 0:

| pecha_id | split | tsawa_spans | metre_pct_shift0 | best_shift | metre_pct_best | sabche_realigned | sabche_pairs_overlapping |
|---|---|---|---|---|---|---|---|
| P000117 | train | 45 | 77.4 | 20 | 87.7 | - | - |
| P000153 | train | 102 | 49.3 | 53 | 75.4 | True | 71 |
| P000083 | test | 190 | 64.6 | 57 | 81.0 | True | 143 |
| P000193 | train | 12 | 41.2 | 25 | 53.2 | True | 2 |
| P000225 | train | 31 | 50.6 | 42 | 62.1 | True | 22 |
| P000219 | train | 27 | 44.8 | 37 | 55.6 | True | 23 |
| P000175 | val | 280 | 49.3 | 36 | 58.6 | True | 243 |
| P000195 | train | 216 | 71.2 | -28 | 79.1 | True | 1 |
| P000185 | train | 56 | 17.5 | 29 | 23.2 | True | 11 |
| P000242 | test | 171 | 57.7 | 41 | 63.4 | True | 147 |
| I19A08A51 | val | 18 | 1.9 | -27 | 34.8 | False | 0 |
| I5134B437 | train | 207 | 12.5 | 27 | 38.7 | False | 0 |
| I9CB71958 | train | 138 | 21.1 | 1 | 38.8 | False | 132 |
| I4FDF07E7 | val | 43 | 74.1 | -37 | 91.4 | False | 0 |
| I23323023 | train | 195 | 30.2 | -30 | 41.4 | False | 1 |
| IB8D599AA | train | 307 | 25.5 | -32 | 35.9 | False | 0 |


Of the 16 books above, 9 are Sabche-realigned books (the tsawa dataset has 17 realigned books in total). The other 7 are not realigned and are listed for completeness; `IF3ACC3E1` (747 short word-fragment spans, 0.6% metrical at shift 0) is not meaningful for this test.
Test books that are Sabche-realigned books: `P000027` (51 spans, shift 0: 80.5%, best shift +30: 84.4%); `P000083` (190 spans, shift 0: 64.6%, best shift +57: 81.0%); `P000144` (113 spans, shift 0: 61.5%, best shift +60: 66.2%); `P000242` (171 spans, shift 0: 57.7%, best shift +41: 63.4%). Together 525 of the 2734 tsawa test spans (19%) are in realigned books. For `P000027` and `P000144` the gain is under 5 points (3.9 and 4.7), so the metre test is weak evidence for them; a book with several offset segments has no single shift and the search is capped at 60, so the numbers show that an offset is likely present, not what it is. The stronger evidence is the overlap table above: in the 23 realigned books the cleaned tsawa and sabche spans overlap in 19 books (910 pairs), against 13 of 175 books (199 pairs) elsewhere.
Example, `P000027` (test): the cleaned Sabche heading at `[18207:18445]` starts at a line start (it was shifted +30 from the raw `[18177:18415]`); the tsawa span `[18414:18570]` starts with the last words of that same heading line (`…འཁོར་བའི་ཚུལ༽ནི།`) and runs into the verse that begins on the next line.

## Analysis 3: conflicts between layers on the same text

Raw layer files (`layers/v001/`), all 539 books. Overlap = any shared character. `same`: identical span; `a in b` / `b in a`: one contains the other; `partial`: they cross.

| layer_a | layer_b | span_pairs | books |
|---|---|---|---|
| Chapter | Author | 3 | 3 |
| Chapter | BookTitle | 17 | 16 |
| Sabche | Chapter | 1 | 1 |
| Tsawa | Sabche | 152 | 6 |


By kind:

| layer_a | layer_b | kind | pairs | books |
|---|---|---|---|---|
| Chapter | Author | same | 3 | 3 |
| Chapter | BookTitle | a_in_b | 7 | 7 |
| Chapter | BookTitle | same | 10 | 9 |
| Sabche | Chapter | same | 1 | 1 |
| Tsawa | Sabche | a_in_b | 147 | 3 |
| Tsawa | Sabche | same | 5 | 4 |


### Sabche vs Chapter: 1 overlapping pairs in 1 books

| pecha_id | a_start | a_end | b_start | b_end | overlap_chars | kind | split_tsawa | split_sabche | split_chapter |
|---|---|---|---|---|---|---|---|---|---|
| I069801F1 | 41735 | 41791 | 41735 | 41791 | 56 | same | - | train | train |


### Chapter vs BookTitle: 17 overlapping pairs in 16 books

| pecha_id | a_start | a_end | b_start | b_end | overlap_chars | kind | split_tsawa | split_sabche | split_chapter |
|---|---|---|---|---|---|---|---|---|---|
| I24EDBDA0 | 0 | 51 | 0 | 51 | 51 | same | - | train | train |
| I29CFF9F3 | 0 | 103 | 0 | 107 | 103 | a_in_b | - | train | train |
| I4A6AC1AA | 80 | 121 | 80 | 121 | 41 | same | - | train | train |
| I4ABDA91A | 5 | 153 | 0 | 153 | 148 | a_in_b | - | train | val |
| I57BB7B28 | 149 | 236 | 149 | 236 | 87 | same | - | train | train |
| I57BB7B28 | 238 | 331 | 238 | 331 | 93 | same | - | train | train |
| I6D4C317A | 0 | 34 | 0 | 34 | 34 | same | - | - | train |
| I8006C360 | 0 | 67 | 0 | 67 | 67 | same | - | train | train |
| I8995895C | 0 | 41 | 0 | 45 | 41 | a_in_b | - | - | train |
| I9658BC87 | 126 | 199 | 126 | 199 | 73 | same | - | train | train |
| IA83BAC6F | 107 | 170 | 107 | 170 | 63 | same | - | - | val |
| IC555D0EB | 75 | 188 | 75 | 188 | 113 | same | - | train | train |
| ID5714205 | 5 | 88 | 0 | 88 | 83 | a_in_b | - | train | train |
| IE1B4BBB9 | 0 | 2 | 0 | 92 | 2 | a_in_b | - | train | train |
| IE3ACE254 | 5 | 153 | 0 | 153 | 148 | a_in_b | - | - | val |
| IE9C4806D | 0 | 28 | 0 | 29 | 28 | a_in_b | - | train | test |
| IEA9B747E | 75 | 147 | 75 | 147 | 72 | same | - | train | train |


### Tsawa vs Sabche: 152 overlapping pairs in 6 books

Per book:

| pecha_id | pairs | same | tsawa_inside_sabche | sabche_inside_tsawa | partial | split_tsawa | split_sabche |
|---|---|---|---|---|---|---|---|
| I23323023 | 1 | 1 | 0 | 0 | 0 | train | train |
| I2EAAF38A | 2 | 2 | 0 | 0 | 0 | val | val |
| I881A57E8 | 13 | 0 | 13 | 0 | 0 | train | train |
| I9B6A4525 | 1 | 1 | 0 | 0 | 0 | test | test |
| I9CB71958 | 132 | 1 | 131 | 0 | 0 | train | train |
| IDAD44BA2 | 3 | 0 | 3 | 0 | 0 | train | train |


### Layers overlapping BookTitle or Author: books and pairs

| layer_a | layer_b | pairs | books |
|---|---|---|---|
| Chapter | Author | 3 | 3 |
| Chapter | BookTitle | 17 | 16 |


Full list of every conflict row: `cross_layer_conflicts.csv`.

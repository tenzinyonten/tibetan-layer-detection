# Where tsawa may be missing: unlabelled verse by book

Scan of all 212 books with a Tsawa layer (`scan_missed_verses.py`; per-book numbers in `missed_tsawa_by_book.csv`). A **run** is 4 or more consecutive shad-delimited clauses that all have the same 7, 9, 11 or 13 syllable count (an isometric stanza) and do not touch any tsawa span. Runs that overlap a Quotation or Citation span are counted separately and left out of the main numbers, because verse quoted from another work is not tsawa.

## What this does and does not show

- Totals: 20,107 runs in 212 books (median 46 per book; 8 books have none). By split: dropped 8,888, test 2,027, train 8,140, val 1,052.
- I read 9 random runs from 3 of the top books (`P000218`, `P000275`, `I2FCD4B1D`): all 9 are real verse, not noise. Several are introduced by a source marker (`བཤེས་སྤྲིངས་སུ།` 'in the Letter to a Friend', `སྡོམ་འབྱུང་ལས།` 'from the Sdom 'byung'), so they are quoted verse.
- So a run is verse that nobody labelled as tsawa. It may be (a) missed tsawa, (b) a quotation from another text that was not marked Quotation, or (c) the author's own verse. The annotator decides which. Use the counts to choose where to look first, not as a count of errors.
- To read a book's runs in context: `python common/annotation_audit/build_book_check.py <book id>` writes an HTML page with the tsawa spans highlighted and every run underlined in blue (the page uses the looser definition, any mix of the four metres, so it shows slightly more runs).

## 1. Dropped books: add tsawa from scratch (44 books with at least 20 unlabelled runs)

These books have fewer than 10 tsawa spans, so they were left out of training. Most have hundreds of verse stanzas and almost no tsawa, so they are the clearest case for adding annotations.

| book | split | tsawa spans | tsawa chars | unlabelled verse runs | runs inside Quotation/Citation (not counted) | chars in all runs | run chars per tsawa char | book chars |
|---|---|---|---|---|---|---|---|---|
| P000275 | dropped | 3 | 428 | 622 | 0 | 220,225 | 514.5 | 580,644 |
| P000054 | dropped | 3 | 565 | 607 | 91 | 158,474 | 280.5 | 689,458 |
| P000115 | dropped | 1 | 1,688 | 311 | 4 | 127,815 | 75.7 | 717,305 |
| P000089 | dropped | 7 | 991 | 304 | 353 | 177,086 | 178.7 | 620,370 |
| P000098 | dropped | 4 | 620 | 303 | 191 | 103,061 | 166.2 | 629,352 |
| I2FCD4B1D | dropped | 2 | 343 | 289 | 134 | 141,404 | 412.3 | 674,216 |
| P000111 | dropped | 1 | 150 | 231 | 27 | 39,905 | 266.0 | 411,816 |
| I0B81CD66 | dropped | 1 | 115 | 170 | 50 | 41,939 | 364.7 | 347,591 |
| P000204 | dropped | 7 | 1,052 | 168 | 2 | 145,761 | 138.6 | 314,590 |
| P000085 | dropped | 3 | 392 | 164 | 161 | 58,954 | 150.4 | 752,850 |
| P000101 | dropped | 5 | 1,677 | 141 | 162 | 67,489 | 40.2 | 684,161 |
| IBF8C5BB6 | dropped | 7 | 528 | 120 | 30 | 40,365 | 76.4 | 256,079 |
| IB47565DE | dropped | 1 | 104 | 119 | 50 | 37,871 | 364.1 | 275,566 |
| P000188 | dropped | 3 | 385 | 106 | 51 | 58,557 | 152.1 | 216,193 |
| P000226 | dropped | 2 | 266 | 99 | 7 | 15,752 | 59.2 | 386,581 |
| I97B90DC0 | dropped | 2 | 85 | 98 | 92 | 77,414 | 910.8 | 301,570 |
| P000114 | dropped | 6 | 567 | 97 | 11 | 22,547 | 39.8 | 207,201 |
| IFA88A536 | dropped | 6 | 807 | 78 | 1 | 29,145 | 36.1 | 173,506 |
| IF85C9484 | dropped | 2 | 79 | 70 | 176 | 59,526 | 753.5 | 420,489 |
| P000066 | dropped | 9 | 1,309 | 66 | 14 | 17,784 | 13.6 | 47,162 |
| P000176 | dropped | 1 | 151 | 62 | 27 | 34,160 | 226.2 | 262,016 |
| IE2421BEA | dropped | 10 | 3,728 | 60 | 52 | 25,575 | 6.9 | 581,305 |
| P000178 | dropped | 1 | 62 | 60 | 1 | 10,072 | 162.5 | 47,274 |
| IF4C4BF01 | dropped | 1 | 118 | 56 | 124 | 33,683 | 285.4 | 385,966 |
| IE166DA22 | dropped | 4 | 128 | 54 | 0 | 8,969 | 70.1 | 21,584 |
| I51B9FE5F | dropped | 1 | 145 | 52 | 232 | 57,398 | 395.8 | 256,065 |
| P000230 | dropped | 4 | 1,408 | 51 | 36 | 20,978 | 14.9 | 361,653 |
| I849C345E | dropped | 3 | 791 | 46 | 21 | 48,450 | 61.3 | 234,862 |
| P000180 | dropped | 1 | 112 | 45 | 32 | 20,832 | 186.0 | 307,762 |
| P000217 | dropped | 2 | 426 | 43 | 44 | 30,022 | 70.5 | 54,308 |


## 2. Training, validation and test books where tsawa is a small share of the verse (9 books)

Criteria: at least 50 unlabelled runs and run characters at least 5 times the tsawa characters. Verse left unlabelled here teaches the model that verse is not tsawa (the `audit_before_v6` under-labelling concern). **The validation and test books matter most, since they also lower the measured score:** `IC6F06BCD` (test), `I4FD99A33` (val), `P000269` (test).

| book | split | tsawa spans | tsawa chars | unlabelled verse runs | runs inside Quotation/Citation (not counted) | chars in all runs | run chars per tsawa char | book chars |
|---|---|---|---|---|---|---|---|---|
| P000218 | train | 22 | 3,497 | 670 | 0 | 150,716 | 43.1 | 847,725 |
| P000219 | train | 27 | 4,085 | 297 | 1 | 65,790 | 16.1 | 335,434 |
| IC6F06BCD | test | 14 | 2,950 | 150 | 28 | 95,705 | 32.4 | 272,537 |
| P000074 | train | 35 | 4,230 | 108 | 185 | 59,130 | 14.0 | 663,463 |
| IA2F1ACFA | train | 66 | 8,744 | 92 | 164 | 46,182 | 5.3 | 284,423 |
| IDDA7F69E | train | 27 | 4,382 | 87 | 97 | 50,905 | 11.6 | 533,162 |
| P000193 | train | 12 | 1,534 | 63 | 61 | 18,833 | 12.3 | 176,415 |
| I4FD99A33 | val | 79 | 4,267 | 50 | 54 | 60,260 | 14.1 | 293,101 |
| P000269 | test | 50 | 7,821 | 50 | 171 | 41,605 | 5.3 | 314,487 |


## 3. Probably fine

8 training/validation/test books have at least 50 unlabelled runs but their tsawa already covers more than half as much text as the unlabelled verse (ratio under 2), for example `P000078` (ratio 0.9). Their unlabelled verse is likely quotation or the author's own stanzas next to a well-annotated root text. Listed in the CSV (`ratio` column).


## 4. Verse that no layer covers

Same runs as above, but a run is dropped if it shares any character with **any** layer in the book (every `layers/v001/*.yml`: Commentary, Sabche, Chapter, Yigchung, Footnote, Quotation, Citation, Author, BookTitle and so on). Tsawa spans were already excluded. What is left is verse that no annotation touches.

- Runs: **9,448 excluding Quotation/Citation only (previous version) -> 6,544 excluding every layer** (20,107 before any exclusion). Characters in those runs: 1,836,045.
- Books with no such run: 129 of 212 (was 32). In 97 books every previously counted run lies inside some layer.
- Almost all of the drop is Commentary. Of the 9,448 previous runs, the number that overlap each layer (a run can overlap several): Chapter 15, Commentary 2572, Sabche 335, Yigchung 108, Footnote 4. In books with a Commentary layer the count goes 2,589 -> 1; in books without one 6,859 -> 6,543.

| split | all runs outside tsawa | excluding Quotation/Citation (version above) | excluding every layer |
|---|---|---|---|
| dropped | 8,888 | 5,264 | 3,546 |
| test | 2,027 | 452 | 178 |
| train | 8,140 | 3,222 | 2,482 |
| val | 1,052 | 510 | 338 |



| books with at least N runs | excluding Quotation/Citation | excluding every layer |
|---|---|---|
| >= 1 | 180 | 83 |
| >= 20 | 81 | 45 |
| >= 50 | 47 | 30 |
| >= 100 | 24 | 17 |


Top 25 books by verse runs that no layer covers:

| book | split | tsawa spans | has Commentary | runs outside Quotation/Citation | runs outside every layer | chars | book chars |
|---|---|---|---|---|---|---|---|
| P000218 | train | 22 | False | 670 | 670 | 150,716 | 847,725 |
| P000054 | dropped | 3 | False | 607 | 606 | 141,734 | 689,458 |
| P000275 | dropped | 3 | False | 622 | 582 | 212,165 | 580,644 |
| P000078 | train | 377 | False | 374 | 371 | 68,639 | 842,034 |
| P000115 | dropped | 1 | False | 311 | 311 | 126,289 | 717,305 |
| P000089 | dropped | 7 | False | 304 | 304 | 108,839 | 620,370 |
| P000098 | dropped | 4 | False | 303 | 299 | 65,895 | 629,352 |
| P000219 | train | 27 | False | 297 | 284 | 63,510 | 335,434 |
| P000151 | train | 184 | False | 231 | 206 | 35,952 | 507,551 |
| P000199 | train | 188 | False | 227 | 196 | 32,113 | 702,627 |
| P000204 | dropped | 7 | False | 168 | 164 | 140,881 | 314,590 |
| P000111 | dropped | 1 | False | 231 | 163 | 24,747 | 411,816 |
| P000085 | dropped | 3 | False | 164 | 158 | 27,256 | 752,850 |
| P000101 | dropped | 5 | False | 141 | 132 | 24,126 | 684,161 |
| P000161 | val | 337 | False | 131 | 126 | 34,199 | 537,718 |
| P000068 | train | 103 | False | 131 | 112 | 18,474 | 560,179 |
| P000188 | dropped | 3 | False | 106 | 106 | 46,254 | 216,193 |
| P000074 | train | 35 | False | 108 | 98 | 14,516 | 663,463 |
| P000114 | dropped | 6 | False | 97 | 97 | 20,769 | 207,201 |
| P000226 | dropped | 2 | False | 99 | 96 | 14,007 | 386,581 |
| P000028 | val | 103 | False | 86 | 86 | 13,687 | 546,785 |
| P000179 | train | 170 | False | 104 | 83 | 15,624 | 350,528 |
| P000066 | dropped | 9 | False | 66 | 66 | 14,966 | 47,162 |
| P000225 | train | 31 | False | 66 | 66 | 11,437 | 619,695 |
| P000164 | test | 183 | False | 66 | 66 | 18,072 | 467,097 |

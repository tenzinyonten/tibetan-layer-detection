# Predicted test spans with no Tibetan character

A span counts as non-Tibetan when its text has no character in U+0F00-U+0FFF. Window level counts a span once per window it appears in (neighbouring windows overlap, as in the reported scores); unique spans count each (book, start, end) once.

| layer | total predictions (window level) | non-Tibetan | % | unique spans | non-Tibetan  | %  | what they are (unique spans) |
|---|---|---|---|---|---|---|---|
| tsawa | 4656 | 44 | 0.95 | 3511 | 30 | 0.85 | whitespace only 30 |
| sabche | 5979 | 9 | 0.15 | 4269 | 9 | 0.21 | Latin letters 9 |
| chapter | 406 | 0 | 0.00 | 305 | 0 | 0.00 | none |
| quotation | 2456 | 0 | 0.00 | 2456 | 0 | 0.00 | none |
| yigchung | n/a | n/a |  | n/a | n/a |  | no saved test predictions |
| total (4 layers) | 13497 | 53 | 0.39 | 10541 | 39 | 0.37 |  |


Examples:

- tsawa: `I3F4A91F5` 88469:88470 '⏎'; `P000164` 152707:152708 ' '
- sabche: `I07240379` 3021:3030 'Return to'; `IC6F06BCD` 2752:2783 'Return to the table of contents' (whole-book predictions from the sabche repo: 4144 spans)
- chapter: none
- quotation: none (quotation_eval mmBERT spans on the 19-book test set)

Yigchung: the Yigchung model's test windows (`karma689/layer_detection`, config `windowed_w8192_s4916`) have no saved predictions on this machine; producing them means running the model, which was not done here.

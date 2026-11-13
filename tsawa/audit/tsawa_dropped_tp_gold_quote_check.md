# Dropped true-positive tsawa spans vs the gold Quotation layer

Filter: drop a predicted tsawa span when predicted quotations cover at least 50% of it (window level, patched test gold). The filter takes the true-positive count from 2930 to 2840 (net 90 lost). The same windows hold 90 baseline true-positive predictions that the rule drops (68 distinct spans); the net figure is lower because a dropped duplicate can leave its gold span matched by another prediction.

For each dropped span, the part covered by the predicted quotation is checked against the gold Quotation + Citation layer (and Quotation alone).

| gold quotation at that spot | quotation + citation | quotation only |
|---|---:|---:|
| real gold quotation (>= 50% of the overlap lies in a gold quotation) | 16 | 0 |
| partly (0 < share < 50%) | 4 | 4 |
| no gold quotation at all (false quotation prediction) | 70 | 86 |

Spans whose own text overlaps any gold quotation at all: 20 of 90 (quotation + citation), 4 (quotation only).

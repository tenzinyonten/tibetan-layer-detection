# Claude Sonnet 5, zero-shot, Chapter test split

One Message Batches request per 16,000-character window (536 windows, 36 test books), prompt in
`chapter/docs/prompts/gemini_chapter_anchors.md`, adaptive thinking at low effort, JSON-schema
output. The prompt was written from train and validation books only, and the test split was
scored once, with no validation run and no prompt change after seeing test.

- `test/spans/<book>.json`: predicted spans as inclusive character offsets
- `test/summary.json`: per-book scores from the runner
- `test/run_info.json`: batch id, tokens, cost, and the scores from the shared scorer

Test F1 0.302 (precision 0.196, recall 0.651): 856 predictions for 258 gold spans, and 605 of
the 688 false positives overlap a Sabche outline-heading span.

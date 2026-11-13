# Gemini 3.1 Flash Lite, zero-shot, Chapter test split

One request per 16,000-character window (536 windows, 36 test books), scored once. The prompt is
`chapter/docs/prompts/gemini_chapter_anchors.md` at commit `dc5e686`, which includes the density
instruction. That instruction was written after looking at Claude's test errors, so this prompt is
not independent of the test set, and it is not the prompt Claude's test run used.

- `test/spans/<book>.json`: predicted spans as inclusive character offsets
- `test/summary.json`: per-book scores from the runner
- `test/run_info.json`: prompt version, settings, tokens and the scores from the shared scorer

Test F1 0.540 (precision 0.400, recall 0.829): 535 predictions for 258 gold spans, and 156 of the 321
false positives overlap a Sabche outline-heading span.

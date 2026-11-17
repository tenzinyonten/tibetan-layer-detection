# Chapter: pipeline details

Details for each step of the Chapter pipeline: thresholds, split ratios, flags and edge cases. The short version (the table and the Quick start) is in [../README.md](../README.md). All commands run from the repository root.

## 1. Clean the spans

Reads `data/shared/tsawa_audit.csv` and the `layers/v001/Chapter.yml` of each book; writes `chapter/data/processed/chapter_spans_clean.csv` and `chapter_book_verdicts.csv`. No `Chapter.yml` is rewritten. It snaps cut edges inside a syllable to the nearest boundary within 3 characters (the end stops before the trailing shad), trims the overlaps this creates, and merges spans separated only by shad, tsheg or spaces (never a newline). A book is `exclude` when more than 30% of its spans still have an edge mid-syllable, or when it has at least 3 spans and under 50% are heading-shaped (old-batch books whose offsets drift partway through). No flags.

## 2. Make the split

A greedy multi-objective assignment, best of 14 seeds, gives a book-level split of 83 / 8.5 / 8.5 by window count. It balances window share, Chapter spans per window, the share of short spans (under 30 characters) and the old-batch share. Hard constraints: one book in one split, at least 30 books in validation and in test. With `--keep-v3`, books already in the tsawa `data/shared/split_v3_frozen.csv` keep their split. Writes `chapter_split_frozen.csv` and a report at `scratch/chapter/splits/chapter_split_report.json` (gitignored). Other flag: `--out PATH`.

## 3. Build the dataset

Uses the mmBERT tokenizer, 8,192-token windows and stride 5,120. Reuses `label_tokens`, `sliding_windows` and `pack_window` from `common/build_tsawa_dataset.py`. Nothing is re-cleaned here. Writes `chapter/data/processed/chapter_dataset/` (gitignored) and `chapter_dataset_stats.json`. Flags: `--audit-csv`, `--sidecar`, `--split-file`, `--tokenizer`, `--max-length 8192`, `--stride 5120`, `--out-dir`, `--stats-json`, and `--limit N` for debugging.

## 4. Push the dataset

Pushes the local `DatasetDict` to the Hub, then uploads `docs/dataset_card.md` as the repo `README.md` and a `label_map.json`. It does not re-tokenize. Needs a Hugging Face login with write access. Other flags: `--local-dir`, `--card`.

## 5. Train

Run on Vast.ai (see `../setup_vast.sh`). The weight scheme is `sqrt_inv` because Chapter is about 0.18% of tokens. Training can also read the local `chapter_dataset` folder instead of the Hub copy.

## 6. Upload the model

Upload the trained model and `docs/model_card.md` (as the model README) to `BDRC/Bo-Chapter-Detection`. This is done by hand; no script in this folder does it.

## 7. Evaluate

Score on **validation** first and look at errors only there. The test split is frozen and was not scored during training; score it once.

`eval_viterbi_iou.py`: `--dump-spans` writes per-window gold and predicted spans as JSONL (keys such as `viterbi_bp4.0`). Other flags: `--sweep`, `--batch-size`, `--device`, `--limit`, `--out`. The metric is span F1 at IoU 0.5 in character offsets, greedy one-to-one matching, Viterbi decoding with break penalty 4.0, whole-book scores. The model card names `training/test_metrics.json` in the model repo as the machine-readable test metrics.

The decoding rules used at inference (a `༼` stub under about 10 characters extends to just before the next `༽`; no tsheg extension, no small-gap merge) are in `detect/layer_config.py`.

## Optional tools

### check_chapter_leakage.py

A read-only report using verbatim shingle matching: how many long (at least 40 characters) Chapter spans in validation and test also appear verbatim in a train book, and how many book pairs share text. It prints to the terminal and has no flags. It is informational only: text shared between books is not treated as leakage in this project.

### analyze_chapter_errors.py

**Use on validation only, never test.**

Light, no model. It maps predicted token spans back to characters, removes duplicates from overlapping windows, matches them to the cleaned gold at IoU 0.5, and prints per-book results, what kinds of gold spans are missed and what the false positives look like. It writes the missed and false-positive lists as CSV to `--out-dir` (default `scratch/chapter`; copies of the validation ones are in `results/`). Other flag: `--show` (default 25). Reading test errors and then changing the model would use up the frozen test split.

### test_chapter_cleanup.py and test_drop_quoted_tsawa.py

`test_chapter_cleanup.py`: tests the `༼...༽` stub-extend rule (`extend_to_closing_bracket` in `detect/decode.py`) and the Chapter `LayerConfig` flags. If `scratch/chapter/chapter_test_spans.jsonl` exists, it also checks that the rules reproduce F1 about 0.8448 on that dump and that the stubs in P000110 and P000275 extend to an exact gold match.

`test_drop_quoted_tsawa.py` tests `drop_quoted_tsawa` and `layers_for_run` in `detect/`: tsawa and quotation code, not the Chapter layer, although the file sits here. Run it after changing `detect/postprocess.py` or `detect/infer.py`.

## How the results were measured

The test split has 258 gold spans in 36 books; the model predicts 302 spans, Claude made 856 predictions and Gemini 535. The Gemini prompt was edited after looking at Claude's test errors, so its run is not independent of the test set. Training data: 308 / 34 / 36 books and 9,165 / 965 / 940 windows (train / validation / test), 378 books in all after 16 drifted books were excluded.

# Tsawa (རྩ་བ, root text)

Finds the **root text** a commentary quotes: a commentary lifts a verse or a passage out of the root text, quotes it, and then explains it; that quoted piece is the Tsawa. One binary token classifier (labels `O`, `B-TSAWA`, `I-TSAWA`), fine-tuned from `jhu-clsp/mmBERT-base`. Quotations from other works look almost the same, and telling them apart is the hard part.

- **In:** the `Tsawa.yml` layers of the OpenPecha `.opf` repos (in `data/raw_opf/`, gitignored), the shared book list `data/shared/tsawa_audit.csv` and the frozen split `data/shared/split_v3_frozen.csv`, and 25 classical commentaries (outside the repo, in `~/tsawa_commentary_data/`).
- **Out:** a dataset and a model on Hugging Face (see below).
- **In the whole system:** `detect/` runs this model on a text (the `tsawa` entry in [layer_config.py](../detect/layer_config.py)). The data scripts live here in `src/`; training and scoring use the shared scripts in [common/](../common/).

## Folder layout

```
tsawa/
├── src/             scripts: audit, clean, split, build, push (sub-folders: analysis, checks, eval)
├── audit/           offset drift and quotation-filter studies, 124 re-annotation flags
├── tests/           four standalone claim checks (not pytest)
├── data/processed/  span tables, split files, classical commentaries, reports
└── docs/            model card, dataset card, zero-shot prompt
```

## Hugging Face

| | Repo | Visibility |
|---|---|---|
| Model | [BDRC/Bo-Tsawa-Detection](https://huggingface.co/BDRC/Bo-Tsawa-Detection) ([card](docs/model_card.md)) | public |
| Dataset | [BDRC/Bo-Tsawa-Dataset](https://huggingface.co/datasets/BDRC/Bo-Tsawa-Dataset) ([card](docs/dataset_card.md)) | private |

Visibility read from the Hub on 2026-11-18; a 404 on the dataset means you need access from BDRC. Each card equals the README on the Hub. The `push_*` scripts still default to the old `Yontenn/formatting-tsawa-v10` and `-v11` names (the dataset in the training command of the model card); pass `--repo-id` to publish elsewhere. Do not write to `BDRC/*` without the owner's go-ahead.

## Pipeline

```mermaid
flowchart TD
  subgraph PREP["Prepare data"]
    direction TB
    S1["Audit repos<br/><i>audit_tsawa_data.py</i>"] --> S2["Snap boundaries<br/><i>snap_tsawa_boundaries.py</i>"]
    S2 --> S3["Resolve overlaps<br/><i>resolve_overlaps.py</i>"]
    S3 --> S4["Merge spans<br/><i>merge_tsawa_spans.py</i>"]
    S4 --> F1(["Merged spans"])
    F1 --> S5["Make split<br/><i>prepare_v6.py</i>"]
    S5 --> S6["Build dataset<br/><i>build_v10_dataset.py</i>"]
    S7["Add classical<br/><i>convert_classical_v3.py</i>"] --> S6
    S6 --> S8["Patch validation<br/><i>patch_validation.py</i>"]
    S8 --> S9["Push dataset<br/><i>push_v11_dataset.py</i>"]
  end
  subgraph TRAIN["Train"]
    direction LR
    T1["Train model<br/><i>common/train_layer.py</i>"] --> T2["Upload model<br/><i>Hugging Face CLI</i>"]
  end
  subgraph EVAL["Evaluate"]
    direction LR
    E1["Evaluate<br/><i>common/eval_viterbi_iou.py</i>"] -.-> E2["Rescore masked<br/><i>rescore_masked.py</i>"]
  end
  PREP --> TRAIN
  TRAIN --> EVAL
```

Squares are steps with the script that runs them; rounded boxes are files; dashed arrows are optional steps. "Add classical" stands for three scripts (convert, split, collapse; see Main pipeline). The cards are in [docs/](docs/).

## Main pipeline

| Script | What it does | When to run |
|---|---|---|
| [audit_tsawa_data.py](src/audit_tsawa_data.py) | Audits the Tsawa layer of every cloned `.opf` repo. | First, after fetching the repos. |
| [snap_tsawa_boundaries.py](src/snap_tsawa_boundaries.py) | Moves dirty span edges to the nearest syllable boundary. | After the audit. |
| [resolve_overlaps.py](src/resolve_overlaps.py) | Fixes overlapping span pairs with three review rules. | After snapping. |
| [merge_tsawa_spans.py](src/merge_tsawa_spans.py) | Merges consecutive spans separated only by shad or space. | After resolving overlaps. |
| [snap_quotation_boundaries.py](src/snap_quotation_boundaries.py) | Snaps Quotation and Citation spans for ignore lists. | Before the split. |
| [prepare_v6.py](src/prepare_v6.py) | Builds the frozen book split and the ignore-span list. | Only to rebuild the split. |
| [convert_classical_v3.py](src/convert_classical_v3.py) | Converts the 25 classical commentaries to span tables. | Once, before the dataset. |
| [make_classical_split.py](src/make_classical_split.py) | Holds out whole commentaries per root text. | After the conversion. |
| [collapse_classical_blank_lines.py](src/collapse_classical_blank_lines.py) | Collapses blank-line runs and remaps spans. | After the classical split. |
| [build_v10_dataset.py](src/build_v10_dataset.py) | Tokenizes everything into 8,192-token BIO windows. | After spans and split exist. |
| [patch_validation.py](src/patch_validation.py) | Fixes labels of 6 validation books; writes v11. | After building v10. |
| [push_v11_dataset.py](src/push_v11_dataset.py) | Uploads the v11 dataset and a short card. | After patching. |
| [train_layer.py](../common/train_layer.py) | Fine-tunes mmBERT for the Tsawa layer. | On a GPU machine. |
| [eval_viterbi_iou.py](../common/eval_viterbi_iou.py) | Scores a model: Viterbi decoding, span F1 at IoU 0.5. | After training; validation first. |

`train_layer.py` and `eval_viterbi_iou.py` are shared scripts in [common/](../common/). The steps up to the frozen split already produced the span tables committed in `data/processed/`, so for the current model you can start at the build step.

## How to run

All commands run from the repository root. **Never decode with argmax (test F1 0.49); use Viterbi. Select models on validation only.**

```bash
# 1. Fetch the .opf repos (also data/ids/newdata.txt)
python common/fetch_opf_repos.py --ids-file data/ids/olddata.txt
# 2. Audit, then clean the spans
python tsawa/src/audit_tsawa_data.py --raw-dir data/raw_opf --manifest data/raw_opf/_manifest.csv --out-csv data/shared/tsawa_audit.csv
python tsawa/src/snap_tsawa_boundaries.py --audit-csv data/shared/tsawa_audit.csv --raw-opf-dir data/raw_opf --out tsawa/data/processed/tsawa_spans_snapped.csv
python tsawa/src/resolve_overlaps.py --sidecar tsawa/data/processed/tsawa_spans_snapped.csv --raw-opf-dir data/raw_opf
python tsawa/src/merge_tsawa_spans.py
python tsawa/src/snap_quotation_boundaries.py
# 3. Frozen split and ignore list (rebuild only if needed)
python tsawa/src/prepare_v6.py
# 4. Classical commentaries
python tsawa/src/convert_classical_v3.py && python tsawa/src/make_classical_split.py && python tsawa/src/collapse_classical_blank_lines.py
# 5. Build the dataset and patch the validation labels (see Known issues: both need extra files)
python tsawa/src/build_v10_dataset.py && python tsawa/src/patch_validation.py
# 6. Push the dataset (private unless --public)
python tsawa/src/push_v11_dataset.py --repo-id <repo>
# 7. Train on a GPU machine (command from the model card)
python common/train_layer.py --label-name TSAWA --dataset Yontenn/formatting-tsawa-v10 --scheme bio --weight-scheme inv --break-penalty 5.0 --epochs 15 --patience 3 --evals-per-epoch 4 --lr 1e-5 --batch-size 8 --seed 42
# 8. Upload the model and docs/model_card.md (as README.md) with the Hugging Face CLI
# 9. Score the 22-book test split once
python common/eval_viterbi_iou.py --model BDRC/Bo-Tsawa-Detection --dataset <dataset> --split test --break-penalty 4.0 --dump-spans test_spans.jsonl
```

## Optional tools

- **Review overlaps**: lists overlapping span pairs for review; stops if the count is not 18. `python tsawa/src/review_overlaps.py --sidecar tsawa/data/processed/tsawa_spans_snapped.csv --raw-opf-dir data/raw_opf --out tsawa/data/processed/overlap_review.csv`
- **Review snapping**: prints before/after text of moved spans. `python tsawa/src/review_snap_diffs.py --sidecar tsawa/data/processed/tsawa_spans_snapped.csv --raw-opf-dir data/raw_opf --n 15`
- **Spot-check labels**: scores sampled spans on metre and closing punctuation. `python tsawa/src/spot_check_tsawa.py --audit-csv data/shared/tsawa_audit.csv --raw-opf-dir data/raw_opf`
- **Rescore masked**: re-scores a `--dump-spans` file with doubtful gold (quotations) masked. `python tsawa/src/eval/rescore_masked.py --dump test_spans.jsonl`
- **Check double labels**: counts tsawa spans that are also quotations. `python tsawa/src/eval/check_double_labels.py`
- **Find label errors**: ranks gold spans a trained model finds unlikely. `python tsawa/src/checks/find_label_errors.py --model <model> --dataset <dataset> --spans tsawa/data/processed/tsawa_spans_resolved.csv --raw-opf data/raw_opf`
- **Analyze errors**: fragmentation, short spans, exact-match F1. `python tsawa/src/analysis/analyze_errors.py --model <model> --dataset <dataset>`
- **Quotation-filter study**: would dropping predicted spans that overlap quotations help? `python tsawa/audit/tsawa_quotation_filter_test.py` (needs saved predictions under `scratch/`)
- **Book catalog**: train, validation and test book list as xlsx. `python tsawa/src/build_tsawa_catalog.py` (see Known issues)

## Results

Test split, 22 benchmark books. Numbers are copied from the [model card](docs/model_card.md): span F1 at IoU 0.5 in token space, greedy one-to-one matching, Viterbi with break penalty 4.0.

| Split | F1 | P | R |
|---|---|---|---|
| Test, 22 benchmark books (3,180 gold spans) | 0.842 | 0.811 | 0.876 |
| Test, with the quotation filter | 0.890 | 0.916 | 0.865 |
| Validation (v11 labels) | 0.789 | 0.726 | 0.863 |
| classical_test (6 commentaries) | 0.959 | 0.941 | 0.978 |

The quotation filter drops a predicted Tsawa span when at least 50% of it lies inside spans predicted by `BDRC/Bo-Quotation-Detection`; it was tested only on this test set. `classical_test` shares root verses with training commentaries (about 99% of its verses also occur in a train commentary), so it measures recall of seen root text, not generalisation. Stricter overlap is weaker: F1 0.814 at IoU 0.9, 0.637 at exact match.

## Known issues

<details><summary>Show 6 known issues</summary>

All checked in the code on 2026-11-18.

- `build_v7` to `build_v10_dataset.py` and `patch_validation.py` read `tsawa/data/processed/tsawa_audit.csv`, which is not in the repo: the audit moved to `data/shared/tsawa_audit.csv`. Copy the file there before running them.
- Several scripts read files under `scratch/` that are not in the repo: `prepare_v6.py` (`scratch/tsawa/scripts/`), `build_v7` to `build_v10` (the patched 22-book test set `scratch/tsawa/analysis/formatting-tsawa-v6_test_patched_full22_final`), `convert_classical_v3.py` (`scratch/tsawa/analysis/end_boundary_check.py`), `rescore_masked.py` and the three `audit/` studies (saved predictions). The classical source docs are in `~/tsawa_commentary_data/`, also outside the repo.
- `tsawa_spans_merged_reannotated_normalized.csv` is committed and read by the builders, but no script in the repo produces it.
- `build_tsawa_catalog.py` has a hard-coded `ROOT = Path("/Users/tenzinyonten/layer_detection")`.
- The analysis and eval scripts default to old models and datasets (`Yontenn/mmbert-tsawa-io-v2`, `formatting-tsawa-v1`, `ds_v5`, `runs/v5_merged/best`); pass `--model` and `--dataset` for the current ones.
- Five interlinear books annotated a word at a time (`IDAD44BA2`, `I1637B774`, `I9FF2B59B`, `I9CB71958`, `IB522F095`; see `data/processed/tsawa_set_aside.csv`) were set aside from training; the layer is not defined for that commentary form. The four scripts in `tests/` are standalone scripts with argparse, not pytest tests.

</details>

## All files

<details><summary>Show the full file tree</summary>

```
tsawa/
├── README.md
├── src/
│   ├── audit_tsawa_data.py              # audit every .opf repo's Tsawa layer
│   ├── snap_tsawa_boundaries.py         # snap dirty span edges to syllable boundaries
│   ├── review_snap_diffs.py             # sample of before/after snapping text
│   ├── review_overlaps.py               # list overlapping span pairs (expects 18)
│   ├── resolve_overlaps.py              # apply three overlap rules
│   ├── merge_tsawa_spans.py             # merge spans separated only by shad or space
│   ├── snap_quotation_boundaries.py     # same snapping for Quotation and Citation
│   ├── spot_check_tsawa.py              # score sampled spans on metre and closers
│   ├── prepare_v6.py                    # frozen split split_v3 and ignore spans
│   ├── convert_classical_v3.py          # classical commentaries to span tables
│   ├── make_classical_split.py          # whole-commentary holdout per root text
│   ├── collapse_classical_blank_lines.py  # collapse blank lines, remap spans
│   ├── build_v10_dataset.py             # build the v10 dataset
│   ├── patch_validation.py              # fix 6 validation books, write v11
│   ├── push_v11_dataset.py              # push v11 to the Hub
│   ├── build_tsawa_catalog.py           # book catalog xlsx
│   ├── build_v7_dataset.py              # old dataset v7, no caller, superseded by v10
│   ├── build_v8_dataset.py              # old dataset v8, no caller, superseded by v10
│   ├── build_v9_dataset.py              # old dataset v9 (one classical_test), no caller
│   ├── push_v10_dataset.py              # push v10 to the Hub, no caller
│   ├── push_v8_dataset.py               # push v8 to the Hub, no caller
│   ├── push_tsawa_dataset.py            # push the first local dataset, old flow
│   ├── check_features.py                # f4/f5 feature check on v6, experiment, no caller
│   ├── count_trailing_particles.py      # trailing citation particles, no caller
│   ├── analysis/
│   │   ├── analyze_errors.py            # fragmentation, short spans, exact-match F1
│   │   ├── show_missed_spans.py         # missed short spans: CONTAINED, OVERLAP, SPLIT, MISSED
│   │   ├── compare_tsawa_citation.py    # Tsawa vs Citation surface form
│   │   ├── count_incipits.py            # incipit pointers, headings ending in ནི།
│   │   ├── show_segmentation.py         # HTML of old vs new batch segmentation
│   │   └── show_new_gaps.py             # HTML of gaps between new-batch spans
│   ├── checks/
│   │   ├── audit_before_v6.py           # four data checks before a build
│   │   └── find_label_errors.py         # rank gold spans by model P(TSAWA)
│   └── eval/
│       ├── rescore_masked.py            # re-score with doubtful gold masked
│       ├── eval_merged.py               # does merging adjacent spans close the recall gap
│       ├── check_false_positives.py     # false positives: on citation, near tsawa, commentary
│       ├── show_old_fp.py               # old-batch false positives (imports check_false_positives)
│       └── check_double_labels.py       # tsawa spans also labelled quotation
├── audit/
│   ├── tsawa_drift_train_val.py         # offset-drift scan of train and val books
│   ├── tsawa_offset_correction_test.py  # shift offsets in 5 test books, re-score v6 predictions
│   ├── tsawa_quotation_filter_test.py   # quotation-filter study
│   ├── tsawa_dropped_tp_gold_quote_check.py  # real spans the filter would drop
│   ├── build_quote_dropped_review.py    # review pack (csv, html) of those drops
│   ├── *.csv, *.md, *.html              # outputs of the five scripts above
│   └── needs_reannotation/              # 124 YAML flags (eval/test, eval/validation, train), no caller
├── tests/
│   ├── test_interlinear.py              # do a book's spans reconstruct continuous root text
│   ├── test_repetition_signal.py        # does repeated text mark tsawa
│   ├── test_retrieval_premise.py        # can tsawa be matched in other books instead of tagged
│   └── test_tsawa_claims.py             # four structural claims on the annotations
├── data/processed/
│   ├── tsawa_spans_*.csv                # snapped, resolved, merged and re-annotated spans
│   ├── quotation_spans_snapped.csv      # snapped Quotation and Citation spans
│   ├── split_v2_frozen.csv/.md, split_v3_frozen_excl_set_aside.csv  # split files
│   ├── v6_ignore_spans.csv, named_source_pairs.csv, dropped_spans.csv  # ignore and review lists
│   ├── tsawa_set_aside.csv              # 5 interlinear books set aside
│   ├── classical_v3/, classical_v3_collapsed/  # classical commentaries and their splits
│   └── tsawa_book_catalog.*, tsawa_reannotation_37.*, other reports  # catalogs and reports
└── docs/
    ├── model_card.md                    # also the Hugging Face README of the model
    ├── dataset_card.md                  # also the Hugging Face README of the dataset
    └── prompts/gemini_tsawa_anchors.md  # zero-shot prompt
```

Scripts run from the repo root. Built Arrow datasets (`data/processed/tsawa_dataset_v10`, `v11`) are gitignored.

</details>

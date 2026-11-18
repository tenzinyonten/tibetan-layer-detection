# Quotation

Finds **quotation spans** in classical Tibetan commentaries: a passage where one text quotes a scripture, a named teacher, or a stock phrase. One token classifier (labels `O`, `B`, `I`, `E`), fine-tuned from `jhu-clsp/mmBERT-base` on 8,192-token sliding windows. **The code of this layer is not in this repository.** It lives in [Webuddhist-tech/Tibetan-quotation-detection](https://github.com/Webuddhist-tech/Tibetan-quotation-detection) (scripts in `script/`, library in `src/layer_detection/`, training in `experiments/mmbert_quotation_v3/`). Every script path below is a path in that repository. This folder holds the Hugging Face cards and this guide.

- **In:** the OpenPecha books named in the Tsadra list, with their `Quotation` layers.
- **Out:** a dataset and a model on Hugging Face (see below).
- **In the whole system:** `detect/` runs this model on a text (the `quotation` entry in [layer_config.py](../detect/layer_config.py)).

## Folder layout

```
quotation/
├── README.md             this guide to the other repository
└── docs/
    ├── dataset_card.md   copy of the dataset README on the Hub
    └── model_card.md     copy of the model README on the Hub
```

## Hugging Face

| | Repo | Visibility |
|---|---|---|
| Model | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) ([card](docs/model_card.md)) | private |
| Dataset | [BDRC/Bo-Quotation-Dataset](https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset) ([card](docs/dataset_card.md)) | public |

Read from the Hub with the local login on 2026-11-18. A 404 on the model means you need access from the owner. The old dataset name `tibetan-quotation-detection` redirects, and the code of the other repository still uses it (`karma689/tibetan-quotation-detection`). Dataset configs: `default` (book level), `windowed_w8192_s4579`, `windowed_w8192_s4579_bioe`, `reviewed_test`, `windowed_w8192_s4579_bioe_reviewed`.

## Pipeline

![Pipeline flowchart](docs/pipeline.png)

<sub>Chart source: [pipeline.mmd](docs/pipeline.mmd)</sub>

Squares are steps with the script that runs them; rounded boxes are files; dashed arrows are optional steps. Dataset uploads use the `--push` flag of the data scripts.

## Main pipeline

All links go to the other repository.

| Script | What it does | When to run |
|---|---|---|
| [download_data.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/download_data.py) | Downloads the OpenPecha books of the Tsadra list. | First, on a new machine. |
| [review_quotations.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/review_quotations.py) | Keyboard review of books and flagged spans; exports gold. | After download; before the split. |
| [build_quotation_benchmark.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/build_quotation_benchmark.py) | Dedupes books and makes the train, validation, test split. | After the gold export. |
| [build_quotation_windows.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/build_quotation_windows.py) | Tokenizes books into 8,192-token windows with BIO labels. | After the split; optional. |
| [relabel_quotation_bioe.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/relabel_quotation_bioe.py) | Relabels the BIO windows to BIOE. | After windowing. |
| [setup_vast_train_env.sh](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/setup_vast_train_env.sh) | Sets up conda, torch and Flash Attention on Vast.ai. | Once per new GPU machine. |
| [train.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/experiments/mmbert_quotation_v3/train.py) | Fine-tunes mmBERT-base with a boundary-weighted loss. | On a GPU machine. |
| [eval_quotation_v3.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/eval_quotation_v3.py) | Sweeps the decode, scores IoU@0.5 span F1. | After training; validation first. |
| [upload_model.py](https://github.com/Webuddhist-tech/Tibetan-quotation-detection/blob/main/script/upload_model.py) | Uploads the checkpoint, metrics and card to the Hub. | After evaluation. |

`train.py` reads `experiments/mmbert_quotation_v3/config.yaml` (learning rate 1e-5, batch 2 with 4 accumulation, 15 epochs, bf16, class weights O 0.07, B 5.0, I 1.0, E 5.0, seed 42, best model by validation IoU@0.5 F1, early stopping 4 epochs).

## How to run

Run from the root of the other repository (`pip install -r requirements.txt && pip install -e .`). `Data/` is gitignored there.

```bash
# 1. Download the corpus
python script/download_data.py
# 2. Review the annotations, then export the usable set
python3 script/review_quotations.py
python3 script/review_quotations.py --export
# 3. Make the book-level split (add --push to upload)
python script/build_quotation_benchmark.py
# 4. Build the BIO windows (step 4579 despite the file name)
python script/build_quotation_windows.py --config script/quotation_w8192_s4751.yaml
# 5. Relabel to BIOE and push (config windowed_w8192_s4579_bioe, tag w2.0)
python script/relabel_quotation_bioe.py --push
# 6. Train on a 24 GB GPU (Vast.ai: bash script/setup_vast_train_env.sh first). Steps 3 to 5 can be skipped: train.py uses the published windows.
cd experiments/mmbert_quotation_v3
python train.py --prepare-only && python train.py --smoke && python train.py
cd ../..
# 7. Choose the decode on validation, then score test with it frozen
python script/eval_quotation_v3.py --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint experiments/mmbert_quotation_v3/outputs/final --split validation
python script/eval_quotation_v3.py --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint experiments/mmbert_quotation_v3/outputs/final --split test --tune-on validation
# 8. Upload the model
python script/upload_model.py --config experiments/mmbert_quotation_v3/config.yaml --repo <owner/name> --private
```

**`train.py` refuses a dataset revision that is not pinned.**

## Optional tools

All in the other repository, run from its root.

- **Reviewed gold test set**: turns the human review YAMLs of the 19 test books into a gold test set (tag `w2.1`). `python script/build_reviewed_benchmark.py` (add `--push`, with `HF_TOKEN` set, to upload)
- **Score on reviewed gold**: scores a checkpoint on the original and the reviewed gold in one pass. `python script/eval_quotation_reviewed.py --config experiments/mmbert_quotation_v3/config.yaml --checkpoint <folder or Hub id>`
- **DharmaBench QUDT**: scores on the external QUDT task (406 short documents). `python script/eval_dharmabench_qudt.py --config experiments/mmbert_quotation_v3/config.yaml --checkpoint experiments/mmbert_quotation_v3/outputs/checkpoint-6562`
- **QUDT bootstrap**: confidence intervals over per-document QUDT scores. `python script/qudt_bootstrap.py <run_dir> [<run_dir> ...]`
- **Window cache**: uploads or downloads the tokenized window cache of an experiment. `python script/upload_window_cache.py upload --config <config.yaml> --repo <owner/name> --private`
- **Gemini and Claude benchmark**: three-model LLM comparison on the 19 test books, needs `GEMINI_API_KEY` (and `ANTHROPIC_API_KEY` for Claude). `python script/run_gemini_benchmark.py {build-units,preflight,freeze,run,claude-batch,locate,score} --split test --source units`; `run --model gemini-3.1-flash-lite` and `claude-batch --action create` and `poll` are the model steps. Results go to `experiments/gemini_anchor_v2_test/results.md`.
- **One-book Gemini debugging**: `python script/gemini_extract.py --book P000152 --prompt docs/prompts/gemini_quote_only_anchors_v2.md --chunked --out quotes.json`, then `python script/check_gemini_quotes.py --book P000152 --quotes quotes.json`. `python script/gemini_eval.py --split test` is the older whole-split flow.
- **Book inspection**: `python script/quotation_inspector.py P000115`, `python script/make_book_viewer.py P000078 --layers Quotation --gaps`, `python script/extract_span.py P000151 validation '[304112, 308503]'`
- **Repo listing**: `python script/discover_openpecha_repos.py --since-days 30 --missing`, `python script/list_downloaded_repo_ids.py`
- **Annotation audit**: `python script/find_quotation_closers.py`, `python script/build_quotation_tags.py`, `python script/check_yigchung_in_quotation.py`
- **Window size check for the LLM flow**: `python script/verify_window_plan.py`

## Results

Encoder: test set (19 books), IoU@0.5, Viterbi decode chosen on validation and frozen, per window in tokens (from the [model card](docs/model_card.md)). LLM rows: character-level IoU@0.5 on the same 19 books, stitch on (from `docs/zero-shot.md` in the other repository); not the same basis, so do not compare the two blocks number for number.

| Model | F1 | P | R |
|---|---|---|---|
| Bo-Quotation-Detection (mmBERT-base, BIOE) | 0.853 | 0.831 | 0.876 |
| gemini-3.1-flash-lite, zero-shot | 0.696 | 0.595 | 0.838 |
| gemini-3.5-flash-lite, zero-shot | 0.649 | 0.530 | 0.839 |
| claude-sonnet-5 (250k units), zero-shot | 0.454 | 0.714 | 0.333 |

`gemini-3.8-flash` is also a benchmark choice (`run`); the source README says it is not scored yet.

## Known issues

<details><summary>Show 6 known issues</summary>

- `script/quotation_w8192_s4751.yaml` is named for step 4751 but sets `step: 4579` and `hf_stride: 3613` (its comments explain: the longest span needs 3613 tokens). The published config is `windowed_w8192_s4579`.
- The code and configs of the other repository still point to `karma689/tibetan-quotation-detection`; the published dataset is `BDRC/Bo-Quotation-Dataset` (the old name redirects).
- `script/build_quotation_tags.py` has a hard-coded root (`~/mnt/quotation_detection`) and the `script/analysis/qd_*.py` files read relative paths such as `qd/...`, so they run only in the author's layout.
- `script/build_windowed_dataset.py` and `script/rescore_v13_quotation.py` belong to the older seven-layer work.
- No script or doc in the other repository calls `rescore_v13_quotation.py`, `diagnose_quote_tsawa.py`, `gemini_phase_a.py`, `list_downloaded_repo_ids.py`, `verify_window_plan.py`, `quotation_inspector.py`, `make_book_viewer.py`, `extract_span.py`, `check_yigchung_in_quotation.py`, `qudt_bootstrap.py` or `discover_openpecha_repos.py`.
- The model repo is private; a 404 on it means you need access from the owner.

</details>

## All files

<details><summary>Show the full file tree</summary>

This folder:

```
quotation/
├── README.md
└── docs/
    ├── dataset_card.md   # copy of the dataset README on the Hub
    └── model_card.md     # copy of the model README on the Hub
```

The other repository (`Webuddhist-tech/Tibetan-quotation-detection`), scripts only:

```
script/
├── download_data.py               # download OpenPecha books of the Tsadra list
├── discover_openpecha_repos.py    # find recently pushed repos missing locally
├── list_downloaded_repo_ids.py    # list pecha ids on disk (no caller)
├── review_quotations.py           # keyboard review tool; --export writes the gold
├── build_quotation_benchmark.py   # dedupe, merge, book-level split, leakage audit
├── build_quotation_windows.py     # 8192-token windows, BIO labels
├── quotation_w8192_s4751.yaml     # config of the step above (sets step 4579)
├── relabel_quotation_bioe.py      # BIO -> BIOE windows
├── build_reviewed_benchmark.py    # reviewed gold test set (tag w2.1)
├── setup_vast_train_env.sh        # conda, torch, Flash Attention on Vast.ai
├── eval_quotation_v3.py           # decode sweep and IoU@0.5 scoring
├── eval_quotation_reviewed.py     # score on original and reviewed gold
├── eval_dharmabench_qudt.py       # external QUDT benchmark
├── qudt_bootstrap.py              # bootstrap intervals for QUDT runs
├── upload_model.py                # publish checkpoint, metrics, card
├── upload_window_cache.py         # upload/download tokenized window cache
├── run_gemini_benchmark.py        # Gemini and Claude benchmark, subcommands
├── gemini_extract.py              # one book through a Gemini prompt
├── check_gemini_quotes.py         # locate and score Gemini quotes
├── gemini_eval.py                 # older whole-split Gemini run
├── gemini_phase_a.py              # zero-shot window sizing (no caller)
├── verify_window_plan.py          # LLM window size check (no caller)
├── quotation_inspector.py         # HTML view of one book's spans (no caller)
├── make_book_viewer.py            # HTML view of annotation layers (no caller)
├── extract_span.py                # print a character span (no caller)
├── find_quotation_closers.py      # candidate missed closers, audit
├── build_quotation_tags.py        # per-span opener/closer tags, hard-coded root
├── check_yigchung_in_quotation.py # Yigchung inside Quotation, audit (no caller)
├── diagnose_quote_tsawa.py        # old seven-layer diagnostic (no caller)
├── rescore_v13_quotation.py       # old v1.3 rescoring (no caller)
├── build_windowed_dataset.py      # old seven-layer windowed set
├── windowed_w8192_s4916.yaml      # config of the step above
└── analysis/                      # qd_compare, qd_final_sweep, qd_meta, qd_mine, qd_newsent,
                                   # qd_perbook, qd_series, qd_top3, qd_validate: exploratory
experiments/
├── mmbert_quotation_v3/           # train.py, config.yaml, MODEL_CARD.md, QUDT outputs
├── gemini_anchor_v2_test/         # benchmark inputs, limits.yaml, results.md
└── gemini_flash_zeroshot/         # early zero-shot runs, config.yaml
src/layer_detection/               # library: windowing, BIOE, metrics, Gemini benchmark, uploads
tests/                             # pytest suite for the library
```

</details>

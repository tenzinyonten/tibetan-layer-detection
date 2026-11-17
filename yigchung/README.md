# Yigchung layer (ཡིག་ཆུང་)

Yigchung is the small-print interlinear note written under or beside the main line of a pecha. This folder trains and evaluates a **genre-specific** Yigchung model for four genres: liturgy, prayer, sadhana and songs. The general Yigchung model in production, `BDRC/Bo-Yigchung-Detection` (v2), was trained elsewhere (see [Models](#models)); its benchmark scores are here too.

Everything here starts from the public dataset [`BDRC/Bo-Yigchung-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Dataset) (762 books, 601 / 84 / 77) at revision `778c059b`.

```
yigchung/
├── src/
│   ├── paths.py                  # every input and output path; YIGCHUNG_DATA overrides the local folder
│   ├── fetch_hub.py              # download BDRC/Bo-Yigchung-Dataset (text + spans) at the pinned revision
│   ├── combine_lotsawa.py        # 469 short Lotsawa House books -> 26 documents of ~30k chars, within each HF split
│   ├── run_classify.py           # Gemini genre label per book (prompts in prompts/); needs GEMINI_KEYS
│   ├── build_genre_datasets.py   # genre books + corrections -> parquet, with and without Lotsawa
│   ├── build_genre_bio.py        # parquet -> BIO DatasetDict for mmBERT (8192 tokens, stride 5120)
│   ├── stratify_genre.py         # proposes a book-level split per genre (not applied yet)
│   ├── analyze_predictions.py    # model predictions -> per-span and per-book tables + an HTML viewer
│   ├── build_benchmark_after.py  # the 77-book benchmark with the annotator corrections applied
│   └── score_benchmark.py        # scores models on the benchmark, before and after the corrections
├── data/processed/               # small tracked tables (below)
├── prompts/                      # genre-classification prompts (v4, v5b, v6)
└── results/                      # results.json per run, per-book scores, benchmark scores
```

## Data

### Tracked tables (`data/processed/`)

| File | What it is |
|---|---|
| `yigchung_catalog.csv` | One row per book: genre (`category`), `genre_split`, `borderline`, label source, title and text snippets. The split comes from here. |
| `span_corrections.csv` | Hand fixes: spans dropped as "not Yigchung" and edited span edges. |
| `span_additions.csv` | Hand-added spans: section labels and `< >` bracket notes the gold had missed. |
| `headmark_fixes.csv` | Output log of the head-mark rule (below), one row per changed span. |
| `lotsawa_combined_manifest.csv` | Which Lotsawa book went into which combined document, and at what offset. |
| `genre_split_proposal.csv` | Output of `stratify_genre.py`. A proposal only: not used by the current models. |
| `genre_bio_with_lotsawa_stats.json` | Window, span and label counts of the BIO set the v3 model was trained on. |

### Local inputs (`data/local/`, gitignored)

| Path | Where it comes from |
|---|---|
| `hub/<split>.parquet` | `python yigchung/src/fetch_hub.py` |
| `lotsawa_combined.parquet` | `python yigchung/src/combine_lotsawa.py` |
| `annotator_spans/<book_id>.json` | Annotator-reviewed spans for a book that is in the HF set (`B-UT1ER2253_0001`). |
| `extra_books/<book_id>.json` | Four annotator-reviewed BDRC books that are not in the HF set (`B-UT1ER1130`, `1210`, `1222`, `1596`): text with brackets removed, plus the annotator spans mapped onto it. |
| `annotator_review/yaml/`, `annotator_review/zips/` | The annotators' layer files and the review OPF zips their offsets refer to. Only `build_benchmark_after.py` needs these. |

The annotator files are not public. Ask the layer owner for them.

### Preparation steps

`build_genre_datasets.py` applies these in order, on every rebuild:

1. Keep the books whose `category` in the catalog is not `other`, and give each its `genre_split`.
2. Where an annotator review exists, use the annotator's spans instead of the HF spans.
3. Apply `span_corrections.csv`: drops and edge edits.
4. End a span before an unmatched closing `)`. This is the annotators' convention: 605 of their 652 edits on 19 reviewed books.
5. Add the spans in `span_additions.csv`. The script stops if an added span overlaps an existing one.
6. Add the four extra books.
7. Head marks (༄༅ ༈ …) open the next text, so they are never part of a span. A mark at a span edge is trimmed. A mark after a text ending (shad, or ག / ཀ) inside a span splits it in two. A mark inside a sentence is left alone.
8. Write `with_lotsawa/` (genre books plus the 26 combined Lotsawa documents, labelled prayer) and `without_lotsawa/`, in the HF schema with spans `{id, start, end}` and **inclusive** ends. Extra columns: `category`, `borderline`, `label_note`.

`build_genre_bio.py` then tokenizes with `jhu-clsp/mmBERT-base` and cuts windows of 8192 tokens with a stride of 5120. Labels are 3-class BIO (`O`, `B-YIGCHUNG`, `I-YIGCHUNG`).

### Splits (final data, with Lotsawa)

| Split | Docs | Spans | Windows |
|---|---:|---:|---:|
| train | 49 | 9,306 | 1,388 |
| validation | 8 | 1,290 | 137 |
| test | 10 | 1,806 | 229 |

5 of the 67 documents carry annotator-reviewed spans: `B-UT1ER2253_0001` and the four extra books. Together they hold about 28% of the spans.

## Run it

```bash
export YIGCHUNG_DATA=/path/to/yigchung_local     # optional; default yigchung/data/local
python yigchung/src/fetch_hub.py
python yigchung/src/combine_lotsawa.py
python yigchung/src/build_genre_datasets.py
python yigchung/src/build_genre_bio.py with_lotsawa  # -> $YIGCHUNG_DATA/genre_training/bio_with_lotsawa

# train on Vast (see setup_vast.sh)
python common/train_layer.py --label-name YIGCHUNG --scheme bio \
    --dataset BDRC/Bo-Yigchung-Classified-Dataset --epochs 20 --lr 1e-5 \
    --break-penalty 5 --stitched-eval --skip-test \
    --run-name yigchung-genre-lotsawa-v3-swap --output-dir /workspace/runs/yigchung_genre_v3

# predict validation + test, then analyse on the Mac
python common/predict_layer_spans.py --label-name YIGCHUNG \
    --model /workspace/runs/yigchung_genre_v3/best \
    --dataset BDRC/Bo-Yigchung-Classified-Dataset --splits validation,test \
    --break-penalty 5 --out /workspace/runs/yigchung_genre_v3/predictions_validation_test.json
python yigchung/src/analyze_predictions.py predictions_validation_test.json with_lotsawa
```

The other training settings are the `train_layer.py` defaults: batch size 2, gradient accumulation 4, weight decay 0.01, gradient clip 0.3, warmup 6%, bf16, seed 42, inverse-frequency class weights, early stopping with patience 3, and the best checkpoint picked on validation span F1 (IoU ≥ 0.5, per window). The stitched validation score (whole documents, overlapping windows averaged) is computed once at the end and is the one reported below. All of these are recorded in `results/genre_v*/results.json`.

The `--dataset` above is the private BDRC copy of `bio_with_lotsawa`. A local path from `build_genre_bio.py` works too.

## Results

Metric: one-to-one span matching at IoU ≥ 0.5. Overlapping windows are averaged per token, then Viterbi decoding with break penalty 5.

| Run | Data | Epochs (best epoch) | Validation F1 | Test F1 |
|---|---|---|---:|---:|
| genre v1 | `yigchung-genre-bio-lotsawa` | 10 | 0.665 | not scored |
| genre v2 | `yigchung-genre-bio-lotsawa` | 20 (13, stopped at 16) | 0.737 | 0.768 |
| **genre v3** | `yigchung-genre-bio-lotsawa-v2` | 20 (17) | **0.737** | **0.786** (P 0.772, R 0.800) |

v3 against the general model `BDRC/Bo-Yigchung-Detection` (v2, from `karma689/mmbert-yigchung-v2`) on the same test documents:

| Test set | Spans | general (v2) | genre v3 |
|---|---:|---:|---:|
| 8 docs that neither model trained on | 1,663 | 0.62 | **0.81** |
| all 10 test docs | 1,806 | 0.62 | **0.79** |

Two test documents (`I518A01A8`, `B-UT1ER1399_0001`) were in the general model's training data, which is why the fair comparison leaves them out. Per-book scores are in `results/genre_v3/by_book.csv` and `results/previous_model_karma689_v2/`.

### Benchmark (77 books)

`results/benchmark/benchmark_scores.md` scores models on the `BDRC/Bo-Yigchung-Dataset` test split, against the original gold ("before") and the corrected gold ("after": annotator review for the 5 BDRC books, the annotator patterns and the head-mark rule elsewhere; 4,059 -> 3,793 spans).

| Model | Before | After |
|---|---:|---:|
| general, `BDRC/Bo-Yigchung-Detection` (v2) | 0.471 | 0.489 |
| genre v2 | 0.417 | 0.437 |

The genre model wins on liturgy (0.79 against 0.62) but scores 0.06 on the "other" genres, against 0.31 for the general model. **It is not a replacement for the general model.**

The model's source repo `karma689/mmbert-yigchung-v2` reports a BDRC test F1 of 0.980. Those scores were computed on a version of the text that still had ༼༽ brackets around the notes. On the published text the same 5 books score 0.49.

## Models

| | Hugging Face repo | Trained on |
|---|---|---|
| General, in production (v2) | [`BDRC/Bo-Yigchung-Detection`](https://huggingface.co/BDRC/Bo-Yigchung-Detection) (public), from `karma689/mmbert-yigchung-v2` | `karma689/yigchung_detection` rev `v2.0`, no longer available; the published data is [`BDRC/Bo-Yigchung-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Dataset) (public) |
| Classified (genre v3, this folder) | `BDRC/Bo-Yigchung-Classified-Detection` (private) | `BDRC/Bo-Yigchung-Classified-Dataset` (private) |

Cards: [`docs/model_card.md`](docs/model_card.md) and [`docs/dataset_card.md`](docs/dataset_card.md) (general pair, public); [`docs/classified_model_card.md`](docs/classified_model_card.md) and [`docs/classified_dataset_card.md`](docs/classified_dataset_card.md) (classified pair, private). Both pairs are listed in [`docs/huggingface.md`](docs/huggingface.md). The general model was trained on an earlier version of the data than the published one (window counts differ; see its card). It decodes with break penalty 8, stride 4914 and first-window stitching, set in `detect/layer_config.py`. The classified model is for the four genres only and decodes differently: stride 5120 instead of 4914, windows averaged instead of first-window-only, and break penalty 5 instead of 8.

## Known limits

- **Songs can't be measured.** The validation and test songs books have 4 and 1 gold spans.
- **The test set is small and uneven.** Two books hold 86% of the test spans. A bootstrap over test books puts v2 anywhere from 0.50 to 0.85.
- **Not all test gold is annotator-reviewed.** Four of the reviewed-OPF test books (`P000030`, `P000228`, `P000249`, `P000268`) are still waiting for annotator review.
- **The Lotsawa documents are labelled prayer as an assumption.** About 90% of Lotsawa books are prayers by a rough title check.

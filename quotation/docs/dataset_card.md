---
pretty_name: Bo-Quotation-Dataset
tags:
- tibetan
- quotation
- bio
- bioe
- mmbert
- openpecha
language:
- bo
task_categories:
- token-classification
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
- config_name: windowed_w8192_s4579
  data_files:
  - split: train
    path: windowed_w8192_s4579/train/*.parquet
  - split: validation
    path: windowed_w8192_s4579/validation/*.parquet
  - split: test
    path: windowed_w8192_s4579/test/*.parquet
- config_name: windowed_w8192_s4579_bioe
  data_files:
  - split: train
    path: windowed_w8192_s4579_bioe/train/*.parquet
  - split: validation
    path: windowed_w8192_s4579_bioe/validation/*.parquet
  - split: test
    path: windowed_w8192_s4579_bioe/test/*.parquet
- config_name: reviewed_test
  data_files:
  - split: test
    path: reviewed/test.jsonl
- config_name: windowed_w8192_s4579_bioe_reviewed
  data_files:
  - split: test
    path: windowed_w8192_s4579_bioe_reviewed/test/*.parquet
---

# Bo-Quotation-Dataset

Benchmark and training data for finding **quotation spans** in classical Tibetan commentaries. It is the training data of [`BDRC/Bo-Quotation-Detection`](https://huggingface.co/BDRC/Bo-Quotation-Detection).

**254 classical Tibetan books** with **23,095 human-reviewed quotation spans**, split by book into train / validation / test.

Character offsets are **inclusive on both ends**: the span is `text[start:end+1]`.

## Format

The data comes in two forms: one row per book, and windows of tokens for training mmBERT.

**Book-level rows** (`train.jsonl`, `validation.jsonl`, `test.jsonl`):

| Column | Meaning |
|---|---|
| `book_id` | the book |
| `title` | the book's title |
| `text` | the full text |
| `doc_chars` | length of the text in characters |
| `batch` | source batch: old or new |
| `source_layer` | the OpenPecha layer the spans came from: `Quotation` or `Citation` |
| `density`, `density_bucket` | share of the text inside quotations, and its band |
| `spans` | the quotation spans (below) |

Each span has `start` and `end` (character offsets, **inclusive** on both ends: `text[start:end+1]`), `cls` (the quote type: `TEXT`, `PERSON`, `MISC`, `FORMULA` or `CHAINED`), `closed` (whether the span ends with an explicit closing marker) and `seen_in_train` (whether the same text also occurs in train).

**Windowed rows** (the `windowed_*` configs): `book_id`, `window_index`, `input_ids`, `attention_mask`, `labels` and `offset_mapping`. Labels are `O`=0, `B-QUOTE`=1, `I-QUOTE`=2, and in the BIOE config also `E-QUOTE`=3. All quote types collapse to one `QUOTE` label. Window length, step and token ownership are described under "Windowed configs" below.

## Splits

Splits are **book-level**: a book's spans never appear in more than one split.

Books are **stratified by quote type** (`TEXT` / `PERSON` / `MISC`), so each split has the same mix of quote types. Density band, source batch, and overall quotation density are matched as well. `FORMULA` and `CHAINED` are too small to stratify on and come out slightly above target.

| Split | Books | Spans | TEXT | PERSON | MISC | Density | Old batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 210 | 18,753 | 57.0% | 15.8% | 18.9% | 7.40% | 40.0% |
| validation | 25 | 2,005 | 57.0% | 15.7% | 18.7% | 7.37% | 39.3% |
| test | 19 | 2,337 | 57.1% | 15.4% | 18.8% | 7.50% | 38.9% |

## Data source and preparation

**Source.**

Classical Buddhist commentaries digitized with support from the Tsadra Foundation and OpenPecha, annotated in two batches: an old batch (125 books, `Quotation` layer) and a new batch (195 books, `Citation` layer). Both layers mark the same thing — one text quoting a scripture, a named teacher, or a stock phrase — so they are merged into one label.

**Cleaning**, in order:

1. **Human review.** 398 books reviewed; 78 rejected whole for bad boundaries; 254 boundaries repaired in accepted books.
2. **Span merging.** The new batch often splits one quotation across lines. Pieces separated only by 0–2 spaces, newlines, or a shad (and not already closed) were joined: 28,892 → 23,476 spans.
3. **Dropping under-annotated books.** Quote markers appear at the same rate in every book (~1,200 per million characters), but some books had almost none of them annotated. Books with quotation density below 1% were dropped: 66 books, 22.5% of characters, only 1.6% of spans.

## Annotation

Quotation spans were marked by human annotators on classical Tibetan commentaries, in two batches: an older batch (the OpenPecha `Quotation` layer) and a newer one (the `Citation` layer). Both layers mark the same thing, so they are merged into one label.

Every book was then reviewed by humans: books with bad boundaries were rejected whole, and boundaries were repaired in accepted books (see the cleaning steps above). The test set was reviewed again, span by span (see "Reviewed test set"). The quote type of each span is not hand-labelled: it is set by rule from the words that introduce the span.

## Quote types

Each span gets a quote type from the words that introduce it:

| Type | Spans | Share | Introduced by |
|---|---:|---:|---|
| `TEXT` | 13,164 | 57.0% | ablative markers (ལས, ནས) naming a text |
| `MISC` | 4,354 | 18.9% | no marker found, or marker separated from the span |
| `PERSON` | 3,636 | 15.7% | ergative markers or speech verbs (ན་རེ, གསུངས) naming a speaker |
| `FORMULA` | 1,007 | 4.4% | set phrases (སྐད་དུ, ཇི་སྐད་དུ) |
| `CHAINED` | 934 | 4.0% | follows a previous closer (ཞེས་དང, ཅེས་དང) |

90.6% of spans end with an explicit closing marker.

## Leakage

- **No text is shared between splits** (the split is by book).
- **The same quotation can appear in different books.** Commentaries quote the same scriptures, so 724 of 4,342 held-out spans (16.7%) have text that also appears in train. These are kept, not removed, and every span carries a `seen_in_train` flag.

**Report two scores:** overall, and `seen_in_train == false` only. The gap shows how much the model relies on memorised quotes.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("BDRC/Bo-Quotation-Dataset")
```

## Windowed configs (for mmBERT training)

Pre-tokenized 8192-token windows built with `jhu-clsp/mmBERT-base`. **This is derived data** — rebuild it if the tokenizer, window length, or step changes, and never mix windows from different builds.

| Config | Labels |
|---|---|
| `windowed_w8192_s4579` | `O`=0, `B-QUOTE`=1, `I-QUOTE`=2 |
| `windowed_w8192_s4579_bioe` | same, plus `E-QUOTE`=3 on the last token of each span |

All quote types collapse to one `QUOTE` label. Load the parquet files directly (plain `load_dataset(repo, config)` picks the JSON builder from the default config):

```python
from datasets import load_dataset
cfg = "windowed_w8192_s4579"   # or windowed_w8192_s4579_bioe
base = f"hf://datasets/BDRC/Bo-Quotation-Dataset/{cfg}"
ds = load_dataset("parquet", data_files={
    s: f"{base}/{s}/*.parquet" for s in ["train", "validation", "test"]
})
```

The BIO config is also pinned at tag `w1.0`.

**Geometry.** Window 8192 tokens, step 4579, overlap 3613 (44%). In the name, `s4579` is the step; the HuggingFace tokenizer argument is `stride=3613`. The overlap equals the longest span (3613 tokens).

**Ownership.** Each overlapped token is labeled in only one window (the first); elsewhere it is `-100`. Joining each book's owned tokens in `window_index` order rebuilds the book exactly, so stitching for inference or scoring is safe.

**BIOE rule.** An `I` token becomes `E` when the next owned token is not `I`, checked across window seams. Mapping `E → I` gives back the BIO config exactly. The shortest span is 3 tokens, so no single-token tag is needed.

| Split | Windows | Spans |
|---|---:|---:|
| train | 6,195 | 18,753 |
| validation | 645 | 2,005 |
| test | 657 | 2,337 |

- 513 spans cross a window seam.
- Token balance (owned tokens): `O` 92.37%, `I` 7.50%, `B` 0.07%, `E` 0.07% (in BIO, `I` is 7.57%).
- Tokenizer sha256 (combined): `9818cf906b79a54b964992d2f76db685ae8b9b3824e28a8f4e4f16425d026f9d`

## Known limits

- Some small boundary errors remain in accepted books.
- Some real quotations are still unlabeled, mostly in books with 1–3% density.
- Density (7.4%) reflects well-annotated books, not typical Tibetan text.

## Citation

```bibtex
@misc{tibetan_quotation_benchmark,
  title  = {Tibetan Quotation Detection Benchmark Dataset},
  author = {karma689},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

<!-- reviewed-test:start -->
## Reviewed test set (`w2.1`)

All 19 test books were checked by human reviewers, span by span. Each
original span was accepted as is, had its edges corrected, or was removed, and
reviewers added quotations the first annotators missed. Train and validation are
unchanged.

| | count |
|---|---|
| spans accepted unchanged | 2281 |
| spans with corrected edges | 106 |
| spans added | 82 |
| spans removed | 27 |
| quotations before review (merged) | 2337 |
| quotations after review (merged) | 2398 |
| identical before and after | 2211 |

Same conventions as the original set: `end` is **inclusive**, pieces of one long
quotation are re-joined, `seen_in_train` uses the same rule. Each span also has
`review` (`accepted` / `edited` / `added`) and `cls_source` (where its marker class
came from — added spans have no original class).

Two new configs, both test-only:

* `reviewed_test` — book-level rows, `reviewed/test.jsonl`
* `windowed_w8192_s4579_bioe_reviewed` — the `windowed_w8192_s4579_bioe` test windows with reviewed labels.
  Input ids, offsets and the ownership mask are identical to `windowed_w8192_s4579_bioe`
  test at `w2.0`; only `labels` differ, so a score on the two tells you
  exactly how much the gold change moved it.

```python
from datasets import load_dataset
test = load_dataset("BDRC/Bo-Quotation-Dataset",
                    data_files={"test": "windowed_w8192_s4579_bioe_reviewed/test/*.parquet"},
                    revision="w2.1")["test"]
```

Per-book changes: `reviewed/review_report.md`; raw numbers: `reviewed/stats.json`.
<!-- reviewed-test:end -->

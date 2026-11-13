# layer_detection

One token-classification model per layer of a Tibetan pecha (OpenPecha `.opf`).
Shared training and fetching live in `common/`. Inference over a finished book,
loading the published per-layer models, is the `tibetan_layer_detection` package
(`tibetan-detect`). The spell checker is a separate repo:
[tenzinyonten/TiSpell](https://github.com/tenzinyonten/TiSpell).

| Layer | What it marks | Code and data | Results |
|---|---|---|---|
| Tsawa | root text (རྩ་བ) | `tsawa/` | `tsawa/audit/` |
| Sabche | outline headings | `sabche/` | `sabche/results/zeroshot/` |
| Chapter | chapter breaks | `chapter/` | `chapter/results/zeroshot/<model>/` |
| Title and author | title and author spans | `title_author/` (its own layout, not reorganized here) | — |

Book text paths and the frozen document split are shared, because more than one layer reads them: `data/shared/tsawa_audit.csv`, `data/shared/split_v3_frozen.csv`. Cloned `.opf` repos are in `data/raw_opf/` (gitignored). A loose OCR sample is in `data/ocr/`.

Datasets and models on Hugging Face:

## Hugging Face

| Layer | Dataset | Model |
|---|---|---|
| Tsawa | [Yontenn/formatting-tsawa-v6](https://huggingface.co/datasets/Yontenn/formatting-tsawa-v6) | [Yontenn/mmbert-tsawa-v6-nofeat](https://huggingface.co/Yontenn/mmbert-tsawa-v6-nofeat) |
| Sabche | [Yontenn/formatting-sabche-v1](https://huggingface.co/datasets/Yontenn/formatting-sabche-v1) | [Yontenn/mmbert-sabche-v1](https://huggingface.co/Yontenn/mmbert-sabche-v1) |
| Chapter | [Yontenn/formatting-chapter-v1](https://huggingface.co/datasets/Yontenn/formatting-chapter-v1) (private) | not trained yet |

If a page returns 404, the repo is private and needs access from its owner. Per-layer copies
with more detail are in `tsawa/docs/huggingface.md`, `sabche/docs/huggingface.md` and `chapter/docs/huggingface.md`.
The Sabche zero-shot evaluation (Gemini and Claude vs mmBERT) is also in the public repo
[`tenzinyonten/sabche-zeroshot-eval`](https://github.com/tenzinyonten/sabche-zeroshot-eval).

## Chapter layer (ལེའུ་)

Same layout as the other layers: everything for Chapter lives under `chapter/`.

```
chapter/
├── src/
│   ├── clean_chapter_spans.py      # snap edges, merge same-line fragments, exclude drifted books
│   ├── prepare_chapter_split.py    # frozen, stratified book-level split
│   ├── build_chapter_dataset.py    # BIO DatasetDict for mmBERT
│   ├── check_chapter_leakage.py    # read-only shingle-match leakage report
│   └── push_chapter_dataset.py     # push to Hugging Face (with the card)
├── data/processed/
│   ├── chapter_spans_clean.csv     # cleaned span sidecar
│   ├── chapter_book_verdicts.csv   # keep / exclude per book, with reason
│   ├── chapter_split_frozen.csv    # frozen split (test is frozen)
│   ├── chapter_dataset_stats.json
│   └── chapter_dataset/            # built Arrow dataset (gitignored)
└── docs/
    ├── dataset_card.md             # also the Hugging Face README
    └── huggingface.md
```

```bash
python chapter/src/clean_chapter_spans.py
python chapter/src/prepare_chapter_split.py --keep-v3
python chapter/src/build_chapter_dataset.py
python chapter/src/push_chapter_dataset.py --private
# train on Vast (see setup_vast.sh); sqrt_inv because Chapter is ~0.18% of tokens
python common/train_layer.py --label-name CHAPTER --weight-scheme sqrt_inv \
    --output-dir /workspace/runs/chapter_bio_sqrt
```

Numbers and cleaning details are in `chapter/docs/dataset_card.md`. Scripts here
import shared code from `common/` and read book text paths from
`data/shared/tsawa_audit.csv`; nothing reads from `scratch/` (the split script only writes a report there).

## OpenPecha `.opf` layout (what we fetch)

```
<PECHA_ID>.opf/
├── base/
│   └── v001.txt              # full plain text; layer spans are char offsets into this
└── layers/v001/
    ├── Tsawa.yml             # root-text spans (this project’s first target)
    ├── Author.yml
    ├── Chapter.yml
    └── …                    # Citation, Commentary, Sabche, Yigchung, …
```

A layer YAML is a map of annotation IDs → `{span: {start, end}, …}` plus
optional fields such as `isverse`.

## Data batches

| File | Role |
|------|------|
| `data/ids/newdata.txt` | Pecha IDs from the newer batch (273 lines as copied) |
| `data/ids/olddata.txt` | Pecha IDs from the older batch (266 lines as copied) |

These lists were copied from your Downloads folder so Phase 1 can run without
another copy step. **Presence on the list is not permission to train** — see
[Open questions before training](#open-questions-before-training).

Clones go to `data/raw_opf/<PECHA_ID>.opf/` (gitignored). A running log lives
at `data/raw_opf/_manifest.csv`.

**Layout note from the Phase 1 smoke clone:** GitHub repos are often a git
root that *contains* the Pecha tree one level down
(`data/raw_opf/I88CF073C.opf/I88CF073C.opf/{base,layers}/`), not
`base/` at the clone root. Phase 2 will resolve either layout.

## Setup

Python 3.10+. Phase 1 needs `git` on PATH (`gh` is used when available).

```bash
cd /path/to/layer_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # needed for Phases 2–3; Phase 1 is stdlib-only
```

## Pipeline (run in this order)

### Phase 1 — fetch `.opf` repos

```bash
# smoke-test a handful of IDs first
python common/fetch_opf_repos.py --ids-file data/ids/newdata.txt --limit 3

# full batches (separate invocations; manifest is merged)
python common/fetch_opf_repos.py --ids-file data/ids/newdata.txt
python common/fetch_opf_repos.py --ids-file data/ids/olddata.txt
```

Useful flags: `--dry-run`, `--batch NAME` (override inferred `new`/`old`),
`--org ORG` (repeatable; default `OpenPecha-Data` then `Webuddhist-tech`),
`--no-gh`, `--timeout SECONDS`, `--raw-dir`, `--manifest`.

Skip/continue on missing or private remotes; the process does not abort.

### Phase 2 — audit Tsawa layers

Scans every checkout in `data/raw_opf/`, resolves the nested
`<ID>.opf/<ID>.opf/` layout, parses `layers/v001/Tsawa.yml` against
`base/v001.txt`, and writes `data/shared/tsawa_audit.csv` plus an
old-vs-new summary (span counts, union coverage %, `isverse`, quality flags).

Spans are treated as **start inclusive, end exclusive**.

```bash
python tsawa/src/audit_tsawa_data.py --raw-dir data/raw_opf --manifest data/raw_opf/_manifest.csv
```

Useful flags: `--out-csv PATH`, `--limit N`.

### Phase 3 — build a tsawa BIO dataset (no training)

Uses Phase 2 `tsawa_audit.csv` as the source of truth (212 Tsawa repos).
By default labels come from `tsawa/data/processed/tsawa_spans_resolved.csv`
(snapped edges + overlap resolve; skip `dropped=True`). Use `--from-yaml`
to label raw `Tsawa.yml` instead. Cloned YAML is never rewritten.

Tokenizes with `jhu-clsp/mmBERT-base`, labels `O` / `B-TSAWA` / `I-TSAWA`,
and slices overlapping token windows.

Splits **by pecha**. Pass `--split-file` to use the frozen v2 assignment
(recommended); without it the builder falls back to its original
coverage-quartile split (seed 42), kept so older calls do not change.

```bash
# v2 — frozen, window-balanced split (current)
python common/build_tsawa_dataset.py --source combined \
    --split-file tsawa/data/processed/split_v2_frozen.csv \
    --out-dir tsawa/data/processed/tsawa_dataset_v2 \
    --dropped-csv tsawa/data/processed/dropped_spans.csv

# v1 — legacy coverage-quartile split
python common/build_tsawa_dataset.py --source combined
# rollback to pre-snap YAML offsets: --from-yaml
```

Useful flags: `--split-file`, `--max-length 8192`, `--stride 5120`,
`--seed 42`, `--tokenizer`, `--out-dir`, `--dropped-csv`, `--audit-csv`,
`--sidecar`.

### Document split v2 (frozen)

`tsawa/data/processed/split_v2_frozen.csv` + `split_v2_frozen.md`. Greedy
window-balanced assignment (seed 123) stratified on batch, `n_windows`,
tsawa density and short-span share, honouring 11 reprint must-link groups.
It replaced the v1 split, whose token positive density was 4.70 / 4.17 /
5.53 %; v2 is **4.76 / 4.77 / 4.42 %**. **The test split is frozen** — do
not use it for tuning or model selection.

Output: `tsawa/data/processed/tsawa_dataset/` (`save_to_disk`) plus
`dataset_card.md`. Aborts if the audit CSV drifted from the frozen Phase 2
counts (539 / 212 / 123 new / 89 old / 41 zero-length spans) or if the
resolved sidecar drifted from 21,155 / 6 dropped / 21,149 active.

## Hugging Face dataset

```bash
python tsawa/src/push_tsawa_dataset.py
# default: Yontenn/formatting-tsawa-v1
```

```python
from datasets import load_dataset
ds = load_dataset("Yontenn/formatting-tsawa-v1")
```

## Open questions before training

1. ~~Data-rights / which batch~~ — **cleared for combined** (new + old).
2. **Hyperparameters** for a single-label tsawa BIO fine-tune of mmBERT-base
   (lr, epochs, class imbalance: ~4–5.5% positive tokens).
3. Coverage is thin (~4.5% of base text in Tsawa repos). That matches the old
   joint model’s tsawa F1 of 0.038; a dedicated binary model is still the plan.

## Training

`common/train_layer.py` is the training loop (one label name per run). On a fresh Vast.ai box, `bash setup_vast.sh`, then for example `python common/train_layer.py --smoke-test --output-dir /workspace/runs/smoke`.

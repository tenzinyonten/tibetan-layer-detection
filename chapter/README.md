# Chapter (ལེའུ)

One binary token classifier for chapter spans.

## Hugging Face

- Model: [BDRC/Bo-Chapter-Detection](https://huggingface.co/BDRC/Bo-Chapter-Detection) (private), card in [`docs/model_card.md`](docs/model_card.md).
- Dataset: [BDRC/Bo-Chapter-Dataset](https://huggingface.co/datasets/BDRC/Bo-Chapter-Dataset) (private), card in [`docs/dataset_card.md`](docs/dataset_card.md).
- Details: [`docs/huggingface.md`](docs/huggingface.md).

`src/` reads book text paths from `data/shared/tsawa_audit.csv` and the frozen tsawa split from `data/shared/split_v3_frozen.csv`. Training is `common/train_layer.py --label-name CHAPTER`.

## Commands and file layout

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
    ├── dataset_card.md             # also the Hugging Face README of the dataset
    ├── model_card.md               # also the Hugging Face README of the model
    └── huggingface.md
```

```bash
python chapter/src/clean_chapter_spans.py
python chapter/src/prepare_chapter_split.py --keep-v3
python chapter/src/build_chapter_dataset.py
python chapter/src/push_chapter_dataset.py --private --repo-id BDRC/Bo-Chapter-Dataset
# train on Vast (see setup_vast.sh); sqrt_inv because Chapter is ~0.18% of tokens
python common/train_layer.py --label-name CHAPTER --weight-scheme sqrt_inv \
    --output-dir /workspace/runs/chapter_bio_sqrt
```

Numbers and cleaning details are in `chapter/docs/dataset_card.md`. Scripts here
import shared code from `common/` and read book text paths from
`data/shared/tsawa_audit.csv`; nothing reads from `scratch/` (the split script only writes a report there).

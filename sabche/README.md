# Sabche (outline headings)

One binary token classifier for sabche spans.

```
sabche/
├── src/              clean spans, freeze the split, build the dataset
├── data/processed/   span table, book verdicts, frozen split (the Arrow dataset is gitignored)
├── docs/             dataset card, Hugging Face notes, the short-span experiment
├── results/zeroshot/ Gemini and Claude summaries
└── tests/            none yet
```

`src/` reads book text paths from `data/shared/tsawa_audit.csv` and windowing from `common/build_tsawa_dataset.py`. Training is `common/train_layer.py --label-name SABCHE`.

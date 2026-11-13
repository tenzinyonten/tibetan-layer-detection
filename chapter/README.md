# Chapter (ལེའུ)

One binary token classifier for chapter spans.

```
chapter/
├── src/                 clean spans, freeze the split, build the dataset, error analysis
├── data/processed/      cleaned spans, book verdicts, frozen split (the Arrow dataset is gitignored)
├── docs/                dataset card and Hugging Face notes
├── results/             per-book validation tables
│   └── zeroshot/        one folder per model (claude-sonnet-5, gemini-3.1-flash-lite)
└── tests/
```

`src/` reads book text paths from `data/shared/tsawa_audit.csv` and the frozen tsawa split from `data/shared/split_v3_frozen.csv`. Training is `common/train_layer.py --label-name CHAPTER`.

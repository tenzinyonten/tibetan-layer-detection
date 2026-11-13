# Tsawa (རྩ་བ, root text)

One binary token classifier for the root-text layer, fine-tuned from `jhu-clsp/mmBERT-base`.

```
tsawa/
├── src/          builders (v6–v11), boundary snapping, checks, eval, analysis
├── audit/        drift, offset correction, quotation-filter study, re-annotation flags
├── data/processed/   span tables and frozen sidecars (built Arrow datasets are gitignored)
├── docs/         dataset card and Hugging Face notes
└── tests/
```

Shared inputs this layer does not own: `data/shared/tsawa_audit.csv` and `data/shared/split_v3_frozen.csv`.
The shared audit that also covers sabche and chapter stays in `common/annotation_audit/`.
Training uses `common/train_layer.py`. The book-level audit and the frozen split used to live in `data/processed/` and moved to `data/shared/` because chapter and sabche read them too.

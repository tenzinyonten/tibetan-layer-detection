# tsawa BIO dataset v6

- **Built:** 2026-09-21
- **Spans:** `tsawa/data/processed/tsawa_spans_merged.csv`
- **Split:** `data/shared/split_v3_frozen.csv` (seed 123, targets 76/12/12 by windows; test frozen)
- **Dropped:** pechas with < 10 merged tsawa spans
- **Must-link:** v6 root-text groups unchanged
- **Ignore (−100):** `tsawa/data/processed/v6_ignore_spans.csv`
- **Tokenizer:** jhu-clsp/mmBERT-base, 8192/5120, BIO; `features` float [seq_len, 5]

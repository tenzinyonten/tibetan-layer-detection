# Title and author

Find the title and the author of a Tibetan work from the opening and closing of the text. The write-up is `docs/overview.md`.

```
title_author/
├── src/                         builders, the rule extractor, Gemini and mmBERT scoring
├── skills/tibetan-title-author/ the skill (SKILL.md, scripts, references)
├── skill_dev/                   scoring and bench tools for the skill
├── skill/                       the earlier skill layout
├── docs/
└── data/gemini_prompts/         the prompts
```

The gold datasets, parquet exports, Gemini batches and model predictions are on disk under `data/` and are gitignored. They are several gigabytes.

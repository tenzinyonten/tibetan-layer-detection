# Layer folder template

Every layer follows this layout and this README, so a reader finds the same thing in the same place.

## Folder

```
<layer>/
├── README.md          this template
├── src/               scripts: build the dataset, train and score helpers
├── data/processed/    small tracked tables (large data is gitignored)
├── docs/
│   ├── dataset_card.md    the README of the Hugging Face dataset
│   ├── model_card.md      the README of the Hugging Face model
│   ├── huggingface.md     repo names and visibility
│   └── prompts/           LLM prompts, if the layer uses any
├── results/           score tables
└── tests/             if the layer has tests
```

Names on Hugging Face: models `BDRC/Bo-<Layer>-Detection`, datasets `BDRC/Bo-<Layer>-Dataset`. The card format is in `WORKPLAN.md` (local) and the cards in `docs/` are copies of the READMEs on the Hub.

## README

```
# <Layer> (<name in Tibetan>)

One sentence: what the layer marks.

## Hugging Face

| | Repo | Visibility |
|---|---|---|
| Dataset | BDRC/Bo-<Layer>-Dataset | |
| Model | BDRC/Bo-<Layer>-Detection | |

Cards: docs/dataset_card.md, docs/model_card.md.

## Data

Where the text comes from, how it is prepared (the scripts in src/, in order), and the split.

## Train

The exact command.

## Evaluate

How the model is scored, and the command.

## Results

One table: F1, precision and recall on the test split.
```

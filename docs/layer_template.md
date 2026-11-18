# Layer folder template

Every layer folder follows this layout and this README, so a reader finds the same thing in the same place. The reference is [chapter/README.md](../chapter/README.md).

## Folder

```
<layer>/
├── README.md          the README described below
├── src/               scripts: build the dataset, train and score helpers
├── data/processed/    small tracked tables (large data is gitignored)
├── docs/              model card, dataset card, prompts if the layer uses any
├── results/           score tables
└── tests/             if the layer has tests
```

Names on Hugging Face: models `BDRC/Bo-<Layer>-Detection`, datasets `BDRC/Bo-<Layer>-Dataset`. The cards in `docs/` are copies of the READMEs on the Hub. Training and scoring use the shared scripts in `common/`.

## README sections, in this order

1. **Title and one paragraph**: what the folder is, then **In** and **Out** bullets.
2. **Folder layout**: a short tree (about 10 lines, visible), one comment per folder.
3. **Hugging Face** (layer folders): a table Repo | Visibility with the model and the dataset (link and card). Read the visibility from the Hub.
4. **Pipeline**: one native `mermaid` block, `flowchart TD`, with subgraphs "Prepare data", "Train", "Evaluate". Square step boxes have two lines: the step name, then `<br/><i>script.py</i>` (file name only; a shared script gets `common/` in front). Rounded boxes are files, one line, no script. Dashed arrows are optional steps. The first line of a label is under 20 characters; about 15 nodes at most. Only `<br/>` and `<i>` HTML; no theme, no colors, no `classDef` or `style`. One caption line under the chart.
5. **Main pipeline**: a table with 3 columns: Script | What it does (at most 12 words) | When to run (at most 6 words). Only the scripts needed to run the pipeline; the script name is a link.
6. **How to run**: one bash block, a comment line and a command per step, in running order (about 20 lines).
7. **Optional tools**: one list; each item has a bold name, a one-line purpose and its command.
8. **Results**: a table with the row names written out, numbers copied from the cards, one-line caveats.
9. Other short sections only if needed (two lines each).
10. **Known issues**: collapsed, button text "Show N known issues". Each issue is checked to be still true.
11. **All files**: collapsed full tree that names every script with a one-line comment.

At most two collapsed blocks (Known issues, All files); about 100 visible lines.

Thresholds, flags, defaults, and the files a script reads and writes belong in that script's module docstring and `--help` text, not in the README.

## Root README

The root README is shorter: one chart with one node per layer (no script detail), a table of the layers with the Hugging Face model and dataset (visibility read from the Hub) and a link to each folder README, a short quick start, how a layer is built, and links to these docs.

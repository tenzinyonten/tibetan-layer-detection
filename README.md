# Tibetan layer detection

One token-classification model per layer of a Tibetan pecha (OpenPecha `.opf`). Every layer has its own folder with the same structure (see [`docs/layer_template.md`](docs/layer_template.md)). Shared training and fetching live in `common/`. Inference over a finished book, loading the published models, is the `detect` package (`tibetan-detect`).

## Layers

| Layer | What it marks | Folder | Dataset | Model |
|---|---|---|---|---|
| Tsawa | root text (རྩ་བ) | [`tsawa/`](tsawa/) | [BDRC/Tibetan-Tsawa-Detection](https://huggingface.co/datasets/BDRC/Tibetan-Tsawa-Detection) (private) | [BDRC/Bo-Tsawa-Detection](https://huggingface.co/BDRC/Bo-Tsawa-Detection) |
| Sabche | outline headings | [`sabche/`](sabche/) | [BDRC/Tibetan-sabche-detection](https://huggingface.co/datasets/BDRC/Tibetan-sabche-detection) (private) | [BDRC/Bo-Sabche-Detection](https://huggingface.co/BDRC/Bo-Sabche-Detection) (private) |
| Chapter | chapter breaks | [`chapter/`](chapter/) | [BDRC/Bo-Chapter-Dataset](https://huggingface.co/datasets/BDRC/Bo-Chapter-Dataset) (private) | [BDRC/Bo-Chapter-Detection](https://huggingface.co/BDRC/Bo-Chapter-Detection) (private) |
| Quotation | quotations in commentaries | [`quotation/`](quotation/) | [BDRC/Bo-Quotation-Dataset](https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset) | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) (private) |
| Yigchung | small-print interlinear notes (ཡིག་ཆུང་) | [`yigchung/`](yigchung/) | [BDRC/yigchung_detection_v2](https://huggingface.co/datasets/BDRC/yigchung_detection_v2) | [BDRC/Bo-Yigchung-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Detection) |
| Title and author | title and author spans | [`title_author/`](title_author/) | [BDRC/Bo-Title-Dataset](https://huggingface.co/datasets/BDRC/Bo-Title-Dataset), [BDRC/Bo-Author-Dataset](https://huggingface.co/datasets/BDRC/Bo-Author-Dataset) (private) | [BDRC/Bo-Title-Detection](https://huggingface.co/BDRC/Bo-Title-Detection), [BDRC/Bo-Author-Detection](https://huggingface.co/BDRC/Bo-Author-Detection) (private) |
| Spelling | spelling errors | separate repo: [tenzinyonten/TiSpell](https://github.com/tenzinyonten/TiSpell) | | |

If a page returns 404, the repo is private and needs access from its owner. The Tsawa and Sabche datasets still have their older `Tibetan-...-Detection` names. Yigchung also has a genre model (liturgy, prayer, sadhana, songs): `Yontenn/mmbert-yigchung-genre-v3` (private), trained on `Yontenn/yigchung-genre-bio-lotsawa-v2` (private).

## Layer folder

Each layer folder has `README.md`, `src/`, `data/processed/`, `docs/` (dataset card, model card, Hugging Face names), `results/` and, where it exists, `tests/`. The template is [`docs/layer_template.md`](docs/layer_template.md). Quotation's code lives in a separate repository (see [`quotation/README.md`](quotation/README.md)).

Book text paths and the frozen document split are shared, because more than one layer reads them: `data/shared/tsawa_audit.csv`, `data/shared/split_v3_frozen.csv`. Cloned `.opf` repos are in `data/raw_opf/` (gitignored). A loose OCR sample is in `data/ocr/`.

## Setup

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

`common/train_layer.py` is the training loop (one label name per run). On a fresh Vast.ai box, `bash setup_vast.sh`, then for example `python common/train_layer.py --smoke-test --output-dir /workspace/runs/smoke`. The exact command for each model is on its model card in the layer's `docs/`.

## More

- OpenPecha `.opf` layout, data batches and the original Tsawa pipeline: [`docs/pipeline_and_data.md`](docs/pipeline_and_data.md)
- Sabche zero-shot evaluation (Gemini and Claude against mmBERT): [`tenzinyonten/sabche-zeroshot-eval`](https://github.com/tenzinyonten/sabche-zeroshot-eval)

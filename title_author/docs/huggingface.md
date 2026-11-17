# Title and author on Hugging Face

| | HF link | Visibility | Card |
|---|---|---|---|
| Model, title | `BDRC/Bo-Title-Detection` | private | [`hf_cards/Bo-Title-Detection.md`](hf_cards/Bo-Title-Detection.md) |
| Model, author | `BDRC/Bo-Author-Detection` | private | [`hf_cards/Bo-Author-Detection.md`](hf_cards/Bo-Author-Detection.md) |
| Dataset for the title model | `BDRC/Bo-Title-Dataset` | private | [`hf_cards/Bo-Title-Dataset.md`](hf_cards/Bo-Title-Dataset.md) |
| Dataset for the author model | `BDRC/Bo-Author-Dataset` | private | [`hf_cards/Bo-Author-Dataset.md`](hf_cards/Bo-Author-Dataset.md) |
| Joint data, tokenized, 5 labels | [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) | public | [`dataset_card.md`](dataset_card.md) |
| Text and span tables | [Yontenn/formatting-title-author](https://huggingface.co/datasets/Yontenn/formatting-title-author) | public | yes (older version) |
| Joint model, run 2 (experiment) | `Yontenn/mmbert-title-author-run2` | private | [`model_cards/mmbert-title-author-run2.md`](model_cards/mmbert-title-author-run2.md) |

Pairs: `Bo-Title-Detection` was trained on `Bo-Title-Dataset`, and `Bo-Author-Detection` on `Bo-Author-Dataset`. Both datasets are relabelled copies of `title-author-training` (same rows and splits). The model weights are in the `best/` subfolder of each model repo.

If a page returns 404, the repo is private and needs access from its owner.

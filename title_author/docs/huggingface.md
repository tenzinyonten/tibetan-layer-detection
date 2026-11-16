# Title and author on Hugging Face

| | HF link | Visibility | Card |
|---|---|---|---|
| Dataset, tokenized, 5 labels | [Yontenn/title-author-training](https://huggingface.co/datasets/Yontenn/title-author-training) | public | [`dataset_card.md`](dataset_card.md) |
| Dataset, raw text and span tables | [Yontenn/formatting-title-author](https://huggingface.co/datasets/Yontenn/formatting-title-author) | public | yes |
| Dataset, title only | `Yontenn/title-only-training` | private | short |
| Dataset, author only | `Yontenn/author-only-training` | private | short |
| Model, title only | `Yontenn/mmbert-title-only` | private | [`model_cards/mmbert-title-only.md`](model_cards/mmbert-title-only.md) |
| Model, author only | `Yontenn/mmbert-author-only` | private | [`model_cards/mmbert-author-only.md`](model_cards/mmbert-author-only.md) |
| Model, joint (run 2) | `Yontenn/mmbert-title-author-run2` | private | [`model_cards/mmbert-title-author-run2.md`](model_cards/mmbert-title-author-run2.md) |

Pairs: `mmbert-title-only` was trained on `title-only-training`; `mmbert-author-only` on `author-only-training`; both views derive from the current `title-author-training`. `mmbert-title-author-run2` was trained on an earlier version of `title-author-training` that has since been overwritten (see its card).

The model weights are in the `best/` subfolder of each model repo. If a page returns 404, the repo is private and needs access from its owner.

The model cards are in this folder and have not been uploaded to the Hub yet. Names follow the current repos; the naming scheme is still being decided in `WORKPLAN.md`.

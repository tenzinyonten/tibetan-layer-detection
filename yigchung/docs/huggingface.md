# Yigchung on Hugging Face

Two systems, each a model with its dataset.

| | HF link | Visibility | Card |
|---|---|---|---|
| General model | [`BDRC/Bo-Yigchung-Detection`](https://huggingface.co/BDRC/Bo-Yigchung-Detection) | public | on the Hub |
| General dataset | [`BDRC/yigchung_detection_v2`](https://huggingface.co/datasets/BDRC/yigchung_detection_v2) | public | on the Hub |
| Classified model | `BDRC/Bo-Yigchung-Classified-Detection` | private | [`classified_model_card.md`](classified_model_card.md) |
| Classified dataset | `BDRC/Bo-Yigchung-Classified-Dataset` | private | [`classified_dataset_card.md`](classified_dataset_card.md) |

The classified pair is for four genres only (liturgy, prayer, sadhana, songs): books were classified into genres with Gemini first, and the genre "other" is excluded. `Bo-Yigchung-Classified-Detection` was trained on `Bo-Yigchung-Classified-Dataset`. Both were moved from `Yontenn/mmbert-yigchung-genre-v3` and `Yontenn/yigchung-genre-bio-lotsawa-v2` on 2026-11-17; the old names redirect. `detect/layer_config.py` still loads the general model, `BDRC/Bo-Yigchung-Detection`.

The two cards in this folder equal the Hub READMEs of the classified pair. The general pair's cards are not in this folder.

If a page returns 404, the repo is private and needs access from its owner.

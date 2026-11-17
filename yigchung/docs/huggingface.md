# Yigchung on Hugging Face

Two systems, each a model with its dataset.

| | HF link | Visibility | Card |
|---|---|---|---|
| General model | [`BDRC/Bo-Yigchung-Detection`](https://huggingface.co/BDRC/Bo-Yigchung-Detection) | public | [`model_card.md`](model_card.md) |
| General dataset | [`BDRC/Bo-Yigchung-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Dataset) | public | [`dataset_card.md`](dataset_card.md) |
| Classified model | `BDRC/Bo-Yigchung-Classified-Detection` | private | [`classified_model_card.md`](classified_model_card.md) |
| Classified dataset | `BDRC/Bo-Yigchung-Classified-Dataset` | private | [`classified_dataset_card.md`](classified_dataset_card.md) |

The classified pair is for four genres only (liturgy, prayer, sadhana, songs): books were classified into genres with Gemini first, and the genre "other" is excluded. `Bo-Yigchung-Classified-Detection` was trained on `Bo-Yigchung-Classified-Dataset`. Both were moved from `Yontenn/mmbert-yigchung-genre-v3` and `Yontenn/yigchung-genre-bio-lotsawa-v2` on 2026-11-17; the old names redirect. 

The general model is v2 (from `karma689/mmbert-yigchung-v2`). It replaced the earlier model in `BDRC/Bo-Yigchung-Detection` on 2026-11-17; the earlier model is kept under the git tag `v1`. The dataset was renamed from `BDRC/yigchung_detection_v2`; the old name redirects. `detect/layer_config.py` loads the general model with break penalty 8, stride 4914 and first-window stitching.

The four cards in this folder equal the Hub READMEs of both pairs.

If a page returns 404, the repo is private and needs access from its owner.

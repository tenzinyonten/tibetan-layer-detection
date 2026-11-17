---
language:
- bo
library_name: transformers
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Yigchung-Classified-Dataset
pipeline_tag: token-classification
tags:
- tibetan
- layer-detection
- yigchung
---

# Bo-Yigchung-Classified-Detection

Token classifier for Yigchung (the small-print interlinear note) in Tibetan texts of four classified genres: liturgy, prayer, sadhana and songs. It is not for other texts. It reads a flat text string and returns character spans. It is a fine-tune of [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (`ModernBertForTokenClassification`, 22 layers, 8,192 positions), trained only on books of those four genres.

## Labels

| id | label |
|---:|---|
| 0 | `O` |
| 1 | `B-YIGCHUNG` |
| 2 | `I-YIGCHUNG` |

## Training data

[BDRC/Bo-Yigchung-Classified-Dataset](https://huggingface.co/datasets/BDRC/Bo-Yigchung-Classified-Dataset): books were classified into genres with Gemini, and books of the genre "other" were left out. 3-class BIO windows of 8,192 tokens with stride 5,120.

| Split | Documents | Gold spans | Windows |
|---|---:|---:|---:|
| train | 49 | 9,306 | 1,388 |
| validation | 8 | 1,290 | 137 |
| test | 10 | 1,806 | 229 |

## Training

| | |
|---|---|
| Base model | `jhu-clsp/mmBERT-base` |
| Epochs | 20 planned, 20 run; best checkpoint at epoch 17 |
| Learning rate | 1e-5, linear decay |
| Batch size | 2, gradient accumulation 4 (effective 8) |
| Weight decay | 0.01 |
| Gradient clip | 0.3 |
| Warmup | 6% of steps |
| Precision | bf16 |
| Seed | 42 |
| Loss | cross-entropy with inverse-frequency class weights: `O` 1.0, `B-YIGCHUNG` 700.9, `I-YIGCHUNG` 11.2 |
| Early stopping | patience 3, one evaluation per epoch |
| Best checkpoint | highest validation span F1 (IoU >= 0.5, per window) |
| Input length | 8,192 tokens (8,190 content tokens) |

```bash
python common/train_layer.py --label-name YIGCHUNG --scheme bio \
    --dataset BDRC/Bo-Yigchung-Classified-Dataset --epochs 20 --lr 1e-5 \
    --break-penalty 5 --stitched-eval --skip-test \
    --run-name yigchung-genre-lotsawa-v3-swap --output-dir /workspace/runs/yigchung_genre_v3
```

The code is in the [layer repository](https://github.com/tenzinyonten/tibetan-layer-detection/tree/main/yigchung).

## Results

Metric: one-to-one span matching at IoU >= 0.5; precision, recall and F1 over spans. Windows overlap and are averaged per token, then the whole document is Viterbi-decoded with break penalty 5.

| Split | Gold spans | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| validation (8 documents) | 1,290 | 1,200 | 0.765 | 0.712 | 0.737 |
| test (10 documents) | 1,806 | 1,871 | 0.772 | 0.800 | 0.786 |

Test by genre:

| Genre | Documents | Gold spans | F1 |
|---|---:|---:|---:|
| liturgy | 4 | 1,594 | 0.808 |
| prayer | 4 | 69 | 0.862 |
| sadhana | 1 | 142 | 0.654 |
| songs | 1 | 1 | no score |

Caveats: the test set is small and uneven; two documents hold 86% of the test spans. Songs has no score because its test set has one gold span.

## Usage

Cut windows of 8,192 tokens with stride 5,120, average the logits of overlapping windows per token, then Viterbi-decode the whole text once with break penalty 5 (leaving a span costs 5; `O` to `I` is forbidden). Spans are inclusive: `text[start:end + 1]`. The repo includes the tokenizer files, copied unchanged from `jhu-clsp/mmBERT-base`.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("BDRC/Bo-Yigchung-Classified-Detection")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Yigchung-Classified-Detection")
```

## Limits

- It is for liturgy, prayer, sadhana and songs only. On other genres it scores very low: an earlier run of this model family scored 0.06 on books of the genre "other".
- Songs: the validation and test sets have only 4 and 1 gold spans, so no score is claimed for songs.
- The Lotsawa House documents are labelled prayer as an assumption (about 90% of those books are prayers by a rough title check).
- The test set has 10 documents, and its score depends mostly on two of them.
- The model sees the text only, not the page image, and boundary errors of a few syllables fail the IoU 0.5 match.
- Use the decoding above; the stride and break penalty differ from other Yigchung settings.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

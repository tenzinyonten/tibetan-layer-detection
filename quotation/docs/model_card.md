---
language:
- bo
library_name: transformers
pipeline_tag: token-classification
tags:
- tibetan
- quotation-detection
- token-classification
- bioe
base_model: jhu-clsp/mmBERT-base
datasets:
- BDRC/Bo-Quotation-Dataset
---

# Bo-Quotation-Detection

[`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) fine-tuned to mark quotation spans in classical Tibetan. One model per layer: this model detects quotations only. Base model license: Apache-2.0. Data: [`BDRC/Bo-Quotation-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset) (`windowed_w8192_s4579_bioe` @ `w2.0`).

## Labels

| ID | Label |
|---:|---|
| 0 | `O` |
| 1 | `B-QUOTE` |
| 2 | `I-QUOTE` |
| 3 | `E-QUOTE` |

## Training data

Trained on [`BDRC/Bo-Quotation-Dataset`](https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset), config `windowed_w8192_s4579_bioe`, version tag `w2.0`.

| Split | Books | Windows | Quotation spans |
|---|---:|---:|---:|
| Train | 210 | 6,195 | 18,753 |
| Validation | 25 | 645 | 2,005 |
| Test | 19 | 657 | 2,337 |

Windows are 8,192 tokens with a step of 4,579 (overlap 3,613). Each token is labelled in one window only, so a book can be rebuilt exactly. See the dataset card for the sources, cleaning and how the split was made.

## Training

| | |
|---|---|
| Input length / step | 8192 tokens / 4579 (overlap 3613) |
| LR / weight decay / grad clip | `1e-5` / `0.01` / `1.0` |
| Batch | 2 × 4 accumulation (effective 8) |
| Epochs / warmup / seed | 15 / 0.2 (10-epoch reference) / 42 |
| Precision | bf16 |
| Class weights | O 0.07, B 5.0, I 1.0, E 5.0 |
| Loss | weighted token CE |
| Selection | val IoU@0.5 F1, early stopping 4 epochs |
| Best checkpoint | step 6562 |

## Results

Held-out test (19 books). Decoder picked on validation, then frozen (Viterbi break penalty 0).

| Metric | Test |
|---|---:|
| IoU@0.5 F1 | **0.853** |
| Precision | 0.831 |
| Recall | 0.876 |
| Pred / gold spans | 2423 / 2299 |

**Viterbi + IoU@0.5.** The head emits a score per token per label. Argmax can jump to an illegal BIOE tag (e.g. `I` with no `B`). Viterbi picks the highest-scoring *legal* path (`B I* E`). A predicted span is a hit if it overlaps a gold span by IoU ≥ 0.5 — at least half the union of the two token ranges — then precision/recall/F1 are micro-averaged over those span matches.

## Usage

The tokenizer is the one of the base model, `jhu-clsp/mmBERT-base`. Load it from there: the tokenizer file stored in this repo was written by a newer version of `transformers` and does not load with version 4.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained("BDRC/Bo-Quotation-Detection")
```

Decode with Viterbi rather than argmax (only legal `B I* E` paths, break penalty 0). Long texts need 8,192-token windows with a step of 4,579. The full pipeline is `detect/` in [tibetan-layer-detection](https://github.com/tenzinyonten/tibetan-layer-detection).

## Limits

- Some small boundary errors remain in the training books, and some real quotations are unlabelled (mostly in books with 1-3% quotation density).
- The data comes from well-annotated books (7.4% quotation density), which is higher than typical Tibetan text.
- The same quotation can appear in different books: 16.7% of held-out spans also appear in train. Check the score on `seen_in_train == false` spans too.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

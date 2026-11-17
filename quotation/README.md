# Quotation

One token-classification model that marks quotation spans in classical Tibetan commentaries (one text quoting a scripture, a named teacher, or a stock phrase).

The training and evaluation code for this layer is in a separate repository, [Webuddhist-tech/Tibetan-quotation-detection](https://github.com/Webuddhist-tech/Tibetan-quotation-detection). This folder holds only the cards and the Hugging Face names, so that every layer can be found here.

## Hugging Face

| | Repo | Visibility |
|---|---|---|
| Dataset | [BDRC/Bo-Quotation-Dataset](https://huggingface.co/datasets/BDRC/Bo-Quotation-Dataset) | public |
| Model | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) | private |

Cards: [`docs/dataset_card.md`](docs/dataset_card.md), [`docs/model_card.md`](docs/model_card.md). They are copies of the READMEs on the Hub.

## Data

254 books with human-reviewed quotation spans, split by book into train, validation and test. The sources, the cleaning steps and the split are described in the dataset card.

## Train

The training settings are in the model card. The training code is in the repository above.

## Results

Test set, 19 books, span F1 at IoU 0.5 (from the model card):

| F1 | Precision | Recall |
|---|---|---|
| 0.853 | 0.831 | 0.876 |

## Inference

`detect/` in this repository loads `BDRC/Bo-Quotation-Detection` like the other layers.

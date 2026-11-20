# Encoder model — mmBERT quotation

Fine-tuned token classifier that marks quotation spans in classical Tibetan.
Backbone is [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base).
The trained weights and a model card live at
[`karma689/tibetan-quotation-detection`](https://huggingface.co/karma689/tibetan-quotation-detection).

This page is the user-facing write-up: what the model is, how to train it, how
to run inference, and the numbers that came out. Run commands are also in the
root [README](../README.md). Earlier experiments are summarised at the end of this page.

## What it does

The head is a 4-label BIOE classifier over 8,192-token sliding windows:

| ID | Label | Meaning |
|---:|---|---|
| 0 | `O` | not a quotation |
| 1 | `B-QUOTE` | first token of a quotation |
| 2 | `I-QUOTE` | inside a quotation |
| 3 | `E-QUOTE` | last token of a quotation |

Argmax can emit an illegal sequence (for example `I` with no `B`). At score
time a Viterbi decode keeps only legal paths (`B I* E`). A predicted span is a
hit if it overlaps a gold span by IoU ≥ 0.5. Precision, recall, and F1 are
micro-averaged over those span matches.

Windows overlap. Tokens that already appeared in the previous window's stride
prefix get label `-100` so they are context only and are not counted twice.

Spans in this repo use inclusive ends: `text[start:end+1]`,
`span_length = end - start + 1`.

## Data

| | |
|---|---|
| Dataset | [`karma689/tibetan-quotation-detection`](https://huggingface.co/datasets/karma689/tibetan-quotation-detection) |
| Config | `windowed_w8192_s4579_bioe` |
| Revision (pinned) | `w2.0` — unpinned loads are refused |
| Geometry | max length 8192, HF stride 3613, step 4579 |
| Split | 210 train / 25 val / 19 test books (6,195 / 645 / 657 windows) |
| Gold spans | 18,753 / 2,005 / 2,337 |

The book-level source (raw text + character offsets) is the same Hub repo.
Windowing and BIOE relabel are regenerable; see [Building the windowed set](#building-the-windowed-set) if you need to rebuild rather than load `w2.0`.

## Training recipe

Config: [`quotation/configs/train_quotation_model.yaml`](../configs/train_quotation_model.yaml).
Script: [`quotation/src/train_quotation_model.py`](../src/train_quotation_model.py).

| Knob | Value |
|---|---|
| Learning rate / weight decay / grad clip | `1e-5` / `0.01` / `1.0` |
| Batch | 2 × 4 accumulation (effective 8) |
| Epochs / warmup / seed | 15 / 0.2 of 10 reference epochs / 42 |
| Precision | bf16 |
| Attention | Flash Attention 2, falls back to SDPA |
| Class weights | O 0.07, B-QUOTE 5.0, I-QUOTE 1.0, E-QUOTE 5.0 |
| Loss | weighted token cross-entropy |
| Selection | val IoU@0.5 F1, early stopping 4 epochs (16 evals) |
| In-loop decode | Viterbi, break penalty 2.0 |
| Best checkpoint | step 6562 |

GPU: a 24 GB card (RTX 4090 / 3090) is enough. On a Vast.ai box run
`bash quotation/src/setup_quotation_train_env.sh` first so torch and Flash Attention land
in the right order.

### Commands

```bash
# optional: prove the loop without a full run
cd quotation
python train.py --prepare-only          # data + schedule + weight assertions, no GPU
python train.py --smoke                 # 200 steps + one eval
python train.py                         # the run
```

`--prepare-only` checks the pinned recipe: 4 labels, revision `w2.0`,
6,195 / 645 / 657 windows, weights `[0.07, 5.0, 1.0, 5.0]`, warmup 1,548,
eval every 193 steps, patience 16 evals.

Training writes `quotation/outputs/`, saves the best
checkpoint as `outputs/final/`, sweeps decode settings on validation, then
scores test at that frozen setting. `test_results.json` is the artifact to
quote. If `hub_model_id` is set in the config, the final folder is pushed
to the Hub at the end.

To skip the end-of-run test pass (the 19-book split is meant to be spent
once):

```bash
python train.py --no-test
```

### Building the windowed set

Skip this if you are loading `w2.0` from the Hub. Rebuild only when the
book-level source or the window geometry changes.

```bash
# 1. Fetch OpenPecha repos listed in Data/tsadra_repo_list.txt
python quotation/src/fetch_quotation_books.py

# 2. Token windows (BIO) — config name carries the step, not the HF stride
python quotation/src/build_quotation_windows.py --config quotation/configs/build_quotation_windows.yaml

# 3. Relabel BIO → BIOE and (optionally) push
python quotation/src/relabel_quotation_bioe.py --push
```

`relabel_quotation_bioe.py` defaults to Hub repo
`karma689/tibetan-quotation-detection`, config `windowed_w8192_s4579_bioe`,
tag `w2.0`.

## Inference

There is no separate predict binary. Scoring a checkpoint *is* the inference
path: load weights, run every window, Viterbi-decode, report spans.

### Held-out splits

Choose the decode on validation, then freeze it before looking at test.

```bash
# validation only (decode is chosen on this split)
python quotation/src/eval_quotation_model.py \
    --config quotation/configs/train_quotation_model.yaml \
    --checkpoint quotation/outputs/final \
    --split validation

# test — decode tuned on validation, then applied unchanged
python quotation/src/eval_quotation_model.py \
    --config quotation/configs/train_quotation_model.yaml \
    --checkpoint quotation/outputs/final \
    --split test --tune-on validation
```

`--checkpoint` can be a local folder (`outputs/final` or
`outputs/checkpoint-6562`) or any Hugging Face model id that has the 4-label
BIOE head.

What it prints: a break-penalty × logit-bias sweep, IoU@0.5 P/R/F1 (pooled and
per book), the pooled score with the largest book removed, MUC COR / PAR / INC
/ MIS / SPU, boundary MAE, a coverage histogram, argmax vs Viterbi, BIO-downgrade
and book-stitched IoU@0.5, and per-marker-class scores if
`Data/audit/quotation_gold_merged.json` is present.

JSON lands next to the checkpoint as `eval_<split>.json`, or at `--out`.

### DharmaBench QUDT (external)

```bash
python quotation/src/eval_quotation_dharmabench.py \
    --config quotation/configs/train_quotation_model.yaml \
    --checkpoint quotation/outputs/checkpoint-6562
```

Needs the vendored file
`third_party/DharmaBench/data/Tibetan/QUDT/test.jsonl` (406 documents, 335
gold quotations). No windowing: QUDT docs fit in one 8,192-token context.
Writes `metrics.json` and `predictions.jsonl` under
`quotation/results/dharmabench_qudt/` unless `--out-dir` is set.

`--close-edge-spans` is a diagnostic. QUDT cuts anthologies at ~1,000
characters, so many gold quotes run off the document edge. The faithful decode
drops a dangling BIOE span that never saw `E-QUOTE`; this flag closes it
instead. Do not quote the closed number as the headline.

### Loading the Hub checkpoint in Python

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "karma689/tibetan-quotation-detection"  # or a local outputs/final
tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForTokenClassification.from_pretrained(model_id)
```

For anything you will report, use `quotation/src/eval_quotation_model.py` so the decode
sweep, IoU matcher, and book stitching stay identical to the published run.

## Results

Numbers below are copied from run artifacts, not retyped from memory.

### In-house test (19 books) — the headline

Source: [`quotation/docs/model_card.md`](../docs/model_card.md).
Decoder picked on validation and frozen (Viterbi break penalty 0).

| Metric | Test |
|---|---:|
| IoU@0.5 F1 | **0.853** |
| Precision | 0.831 |
| Recall | 0.876 |
| Pred / gold spans | 2,423 / 2,299 |

This score is **per window, in tokens**, against 2,299 gold pieces. The
zero-shot Gemini/Claude numbers in [zero-shot.md](zero-shot.md) are **per book,
in characters**, against 2,337 gold spans. Do not put 0.853 next to 0.696 and
call it a comparison. Re-scoring the encoder with the Gemini scorer (character
offsets, same stitch) is the number that belongs on that table; it is not in
the artifacts yet.

### DharmaBench QUDT

The trained checkpoint `checkpoint-6562`, BIOE, Viterbi penalty 0. Source:
[`quotation/results/dharmabench_qudt/metrics.json`](../results/dharmabench_qudt/metrics.json)
(eval 2026-09-11). Chance baseline for MUC type is **0.6009** — 244 of 406
documents have no gold quotation, and empty-vs-empty scores 1.0.

| Decoder | MUC type F1 | vs 0.6009 | IoU@0.5 F1 | IoU P | IoU R | matches / gold / pred |
|---|---:|---:|---:|---:|---:|---:|
| argmax | 0.626 | +0.025 | 0.080 | 0.824 | 0.042 | 14 / 335 / 17 |
| **Viterbi p=0** | **0.630** | **+0.029** | **0.130** | 0.706 | 0.072 | 24 / 335 / 34 |

Closing dangling edge spans (`quotation/results/dharmabench_qudt_closed_edges/metrics.json`) moves
Viterbi MUC type to 0.637 and IoU F1 to 0.150. That is a diagnostic, not the
faithful number.

Earlier v1.3 (15-label BIO head, different test books) reached MUC type
**0.653** (Viterbi p=4) and IoU@0.5 **0.181** (Viterbi p=1). Details are in the source repo: [dharmabench-qudt-evaluation.md](https://github.com/WeBuddhist/Tibetan-quotation-detection/blob/master/docs/dharmabench-qudt-evaluation.md)
and [experiments/eval-03-dharmabench-qudt.md](https://github.com/WeBuddhist/Tibetan-quotation-detection/blob/master/docs/experiments/eval-03-dharmabench-qudt.md).
v1.3 looked stronger on the published board mainly because it stayed quiet on
the empty majority; restricted to the 162 documents that contain a quotation,
IoU recall was 0.099.

The current model is more conservative still (34 predicted spans vs 335 gold). Precision on
the spans it does emit is high; recall on this short-document, differently
annotated set is the gap.

### Earlier in-house checkpoints (for context)

From the source repo's [experiments log](https://github.com/WeBuddhist/Tibetan-quotation-detection/blob/master/docs/experiments/README.md). These used the older
seven-layer split and exact-span / later IoU scoring — not the current test number.

| Run | What changed | Headline |
|---|---|---|
| v1 | mmBERT-base baseline | exact span F1 0.013 |
| v1.2 | lower LR, grad clip, class-weight cap, O pinned 0.07 | exact span F1 0.078 |
| v1.3 | identical config, re-run | exact span F1 0.071 |
| eval-01 | IoU matching + Viterbi, no retrain | IoU@0.5 **0.427** |
| v2 | focal + Tversky + boundary + EER | in-loop IoU@0.5 0.311; rejected |

The current model is the first run trained on the quotation-only BIOE windows at `w2.0`.

## Caveats

- Test is 19 books. One large book can move the pooled F1; the eval script also
  prints the score with that book removed.
- 0.853 and the Gemini F1s are not on the same basis (see above).
- DharmaBench annotation conventions are not Tsadra's, and documents average
  ~1,000 characters against training windows of 8,192 tokens.
- Class weights, decode penalty, and checkpoint selection were all chosen on
  validation. Do not retune them after seeing test.

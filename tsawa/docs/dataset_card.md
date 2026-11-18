---
pretty_name: Bo-Tsawa-Dataset
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- tsawa
- bio
- mmbert
dataset_info:
  features:
  - name: pecha_id
    dtype: string
  - name: source_batch
    dtype: string
  - name: window_index
    dtype: int32
  - name: n_tokens_doc
    dtype: int32
  - name: coverage_pct
    dtype: float64
  - name: token_start
    dtype: int32
  - name: token_end
    dtype: int32
  - name: char_start
    dtype: int32
  - name: char_end
    dtype: int32
  - name: input_ids
    list: int32
  - name: attention_mask
    list: int8
  - name: labels
    list: int32
  splits:
  - name: train
    num_bytes: 357899251
    num_examples: 4850
  - name: validation
    num_bytes: 33648600
    num_examples: 456
  - name: classical_validation
    num_bytes: 47378958
    num_examples: 642
  - name: classical_test
    num_bytes: 46788566
    num_examples: 634
  - name: test
    num_bytes: 47152396
    num_examples: 639
  download_size: 33154758
  dataset_size: 532867771
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: classical_validation
    path: data/classical_validation-*
  - split: classical_test
    path: data/classical_test-*
  - split: test
    path: data/test-*
---

# Bo-Tsawa-Dataset

Windowed BIO token-classification data for **Tsawa** (རྩ་བ་, root text) in Tibetan commentaries. It trains [`BDRC/Bo-Tsawa-Detection`](https://huggingface.co/BDRC/Bo-Tsawa-Detection).

## Format

Tokenized with `jhu-clsp/mmBERT-base`. Windows of 8,192 tokens, stride 5,120. Labels: `O` = 0, `B-TSAWA` = 1, `I-TSAWA` = 2, `-100` = special tokens, padding and masked tokens. A token gets B if a span starts in it, I for the following tokens of the span.

Columns: `pecha_id`, `source_batch`, `window_index`, `n_tokens_doc`, `coverage_pct`, `token_start`, `token_end`, `char_start`, `char_end`, `input_ids`, `attention_mask`, `labels`.

Span ends follow the shad-inside convention: a span includes its closing shad. Repeated verses in the classical commentaries are masked (-100).

## Splits

| split | books | windows | spans |
|---|---:|---:|---:|
| train | 92 | 4,850 | 16,716 |
| validation (patched in v11) | 18 | 456 | 1,582 |
| test (benchmark, frozen) | 22 | 639 | 1,976 |
| classical_validation | 6 | 642 | 3,652 |
| classical_test | 6 | 634 | 4,397 |

Train: 53 existing books (6,649 spans), 26 re-annotated books (2,240 spans), 13 classical commentaries (7,827 spans). `test` is the benchmark and must stay untouched; select models on `validation`. The classical splits are whole-commentary holdouts (one test and one validation commentary per root text) and share root verses with train commentaries (about 90% of classical_validation and 99% of classical_test verses also occur in a train commentary), so they measure recall of seen root text, not generalisation.


## Data source and preparation

**Source.** Two sources, in three groups:

| group | books | source | text |
|---|---:|---|---|
| existing | 53 | BDRC volumes, OpenPecha `.opf` format | commentaries annotated in earlier rounds |
| re-annotated | 26 | BDRC volumes, OpenPecha `.opf` format | books re-annotated for span edges and boundaries |
| classical | 13 train + 6 validation + 6 test | WeBuddhist `*-rails` repositories (github.com/WeBuddhist), public | classical commentaries on six root texts (`WB...` ids, 25 documents) |

The six classical root texts are the Vinaya sutra, the Pramāṇavārttika (Valid Cognition), the Madhyamakāvatāra (Introduction to the Middle Way), the Abhidharmakośa (Treasury of Abhidharma), the Bodhicaryāvatāra and the Mahāyānasūtrālaṃkāra. Their repos carry `license: public` in the front matter; several files say the edition is "to be verified".

**Classical commentaries received from WeBuddhist** (25 documents; WeBuddhist repository, split, merged Tsawa spans):

| root text | WeBuddhist repo | doc id | commentary | split | spans |
|---|---|---|---|---|---:|
| Pramāṇavārttika (Valid Cognition) | `Commentary_on_Valid_Cognition-rails` | WB952E49A0 | ཚད་མ་རྣམ་འགྲེལ་གྱི་གཞུང་གསལ་པོར་བཤད་པ་ལེགས་བཤད་སྣང་བའི་གཏེར། | train | 1,323 |
| Pramāṇavārttika (Valid Cognition) | `Commentary_on_Valid_Cognition-rails` | WB538B6B7E | ཚད་མ་རྣམ་འགྲེལ་གྱི་དཀའ་འགྲེལ། | test | 1,157 |
| Pramāṇavārttika (Valid Cognition) | `Commentary_on_Valid_Cognition-rails` | WB221367FC | ཚད་མ་རྣམ་འགྲེལ་གྱི་ཚིག་ལེའུར་བྱས་པའི་རྣམ་བཤད་ཐར་ལམ་ཕྱིན་ཅི་མ་ལོག་པར་གསལ་བར་བྱེད་པ། | validation | 995 |
| Pramāṇavārttika (Valid Cognition) | `Commentary_on_Valid_Cognition-rails` | WBD11654CE | རྒྱས་པའི་བསྟན་བཅོས་ཚད་མ་རྣམ་འགྲེལ་གྱི་དཀའ་འགྲེལ་དགོངས་པ་རབ་གསལ། | train | 779 |
| Madhyamakāvatāra (Introduction to the Middle Way) | `Introduction_to_the_Middle_Way-rails` | WB293AAB0E | དབུ་མ་ལ་འཇུག་པའི་ཊཱི་ཀ་ཟབ་རྒྱས་ལམ་མཆོག་རབ་གསལ་སྣང་བའི་རྒྱན། | train | 332 |
| Madhyamakāvatāra (Introduction to the Middle Way) | `Introduction_to_the_Middle_Way-rails` | WB6605DC96 | དབུ་མ་ལ་འཇུག་པའི་བཤད་པ། | test | 331 |
| Madhyamakāvatāra (Introduction to the Middle Way) | `Introduction_to_the_Middle_Way-rails` | WBFD9A3D42 | དབུ་མ་ལ་འཇུག་པའི་རྒྱ་ཆེར་བཤད་པ་དགོངས་པ་རབ་གསལ། | validation | 325 |
| Madhyamakāvatāra (Introduction to the Middle Way) | `Introduction_to_the_Middle_Way-rails` | WB95443AC2 | དབུ་མ་ལ་འཇུག་པའི་འགྲེལ་བ་ཟླ་བའི་ཞལ་ལུང་དྲི་མེད་ཤེལ་ཕྲེང་། | train | 322 |
| Abhidharmakośa (Treasury of Abhidharma) | `Treasury_of_Abhidharma-rails` | WB08FDF935 | ཆོས་མངོན་པ་མཛོད་ཀྱི་ཚིག་ལེའུར་བྱས་པའི་འགྲེལ་པ་མངོན་པའི་རྒྱན། | train | 713 |
| Abhidharmakośa (Treasury of Abhidharma) | `Treasury_of_Abhidharma-rails` | WBB84F7E33 | ཆོས་མངོན་པའི་མཛོད་ཀྱི་རྣམ་པར་བཤད་པ་ཆོས་མངོན་རྒྱ་མཚོའི་སྙིང་པོ་མཁྱེན་བརྩེའི་ཞལ་ལུང་གཞོན་ནུ་རྣམ་རོལ་ལེགས་བཤད་ཆོས་མིག་རྣམ་འབྱེད་གྲུབ་བདེའི་ཤིང་རྟ། | test | 672 |
| Abhidharmakośa (Treasury of Abhidharma) | `Treasury_of_Abhidharma-rails` | WB3AF96C03 | ཆོས་མངོན་པ་མཛོད་ཀྱི་དཀའ་གནས་ལེགས་པར་བཤད་པ་ནོར་བུའི་བང་མཛོད། | validation | 390 |
| Abhidharmakośa (Treasury of Abhidharma) | `Treasury_of_Abhidharma-rails` | WB6EECC200 | ཆོས་མངོན་པ་མཛོད་ཀྱི་འགྲེལ་བཤད་མཚན་ཉིད་ཀྱི་རྗེས་སུ་འབྲང་བ། | train | 113 |
| Vinaya sutra | `Vinaya_sutra-rails` | WB6C2D1D2B | འདུལ་བའི་མདོའི་རྒྱ་ཆེར་འགྲེལ་པ། | train | 830 |
| Vinaya sutra | `Vinaya_sutra-rails` | WB8B136A96 | ལེགས་པར་གསུངས་པའི་དམ་པའི་ཆོས་འདུལ་བ་མཐའ་དག་གི་སྙིང་པོའི་དོན་ལེགས་པར་བཤད་པ་རིན་པོ་ཆེའི་འཕྲེང་བ་ཞེས་བྱ་བ་ལས་གཞི་སྟོད། | test | 737 |
| Vinaya sutra | `Vinaya_sutra-rails` | WBD9437B61 | ལེགས་པར་གསུངས་པའི་དམ་པའི་ཆོས་འདུལ་བ་མཐའ་དག་གི་སྙིང་པོའི་དོན་ལེགས་པར་བཤད་པ་རིན་པོ་ཆེའི་ཕྲེང་བ་ཞེས་བྱ་བའི་སྨད་ཆ། | validation | 638 |
| Vinaya sutra | `Vinaya_sutra-rails` | WBEEAE0D5D | བསྟན་བཅོས་འདུལ་བ་མདོ་རྩ་བའི་སྤྱི་དོན་ཊཱིཀ་ཆེན་རིན་ཆེན་ཕྲེང་བའི་དགོངས་རྒྱན། | train | 422 |
| Bodhicaryāvatāra | `bodhisattvacharyavatara-rails` | WBEC89D928 | ཀུན་དཔལ་སྤྱོད་འཇུག་ཚིག་འགྲེལ་འཇམ་དབྱངས་བླ་མའི་ཞལ་ལུང་། | train | 919 |
| Bodhicaryāvatāra | `bodhisattvacharyavatara-rails` | WB933FBB1E | བྱང་ཆུབ་སེམས་དཔའི་སྤྱོད་པ་ལ་འཇུག་པ་ཞེས་བྱ་བའི་འགྲེལ་པ། | test | 909 |
| Bodhicaryāvatāra | `bodhisattvacharyavatara-rails` | WB335D682D | བྱང་ཆུབ་ཀྱི་སྤྱོད་པ་ལ་འཇུག་པའི་དཀའ་འགྲེལ། | validation | 820 |
| Bodhicaryāvatāra | `bodhisattvacharyavatara-rails` | WB4FCCAAFE | བྱང་ཆུབ་སེམས་དཔའི་སྤྱོད་པ་ལ་འཇུག་པའི་འགྲེལ་བ | train | 565 |
| Mahāyānasūtrālaṃkāra (མདོ་སྡེ་རྒྱན།) | `mdo-sde-rgyan-rails` | WB1F92DD15 | ཐེག་པ་ཆེན་པོ་མདོ་སྡེའི་རྒྱན་གྱི་དགོངས་དོན་རྣམ་པར་བཤད་པ་ཐེག་མཆོག་བདུད་རྩིའི་དགའ་སྟོན། | train | 714 |
| Mahāyānasūtrālaṃkāra (མདོ་སྡེ་རྒྱན།) | `mdo-sde-rgyan-rails` | WB2415B174 | ཐེག་པ་ཆེན་པོ་མདོ་སྡེའི་རྒྱན་གྱི་འགྲེལ་པ་རིན་པོ་ཆེའི་ཕྲེང་བ། | test | 591 |
| Mahāyānasūtrālaṃkāra (མདོ་སྡེ་རྒྱན།) | `mdo-sde-rgyan-rails` | WB37FA7231 | མདོ་སྡེ་རྒྱན་གྱི་འགྲེལ་བཤད། སྟོད་ཆ། | validation | 484 |
| Mahāyānasūtrālaṃkāra (མདོ་སྡེ་རྒྱན།) | `mdo-sde-rgyan-rails` | WB72607BDE | མདོ་སྡེ་རྒྱན་གྱི་འགྲེལ་བཤད། སྨད་ཆ། | train | 407 |
| Mahāyānasūtrālaṃkāra (མདོ་སྡེ་རྒྱན།) | `mdo-sde-rgyan-rails` | WB1A8896DB | ཐེག་པ་ཆེན་པོ་མདོ་སྡེའི་རྒྱན་ཞེས་བྱ་བའི་མཆན་འགྲེལ། | train | 388 |

Each document is one commentary interleaved with the root verses it cites. Split: one validation and one test commentary per root text, the rest train (13 train, 6 validation, 6 test).

**Preparation**, in order (scripts in `tsawa/src/` and `common/` of the layer-detection repo):

1. Take the annotator spans for each book (character offsets); normalise span ends to the shad-inside convention (`tsawa_spans_merged_reannotated_normalized.csv`).
2. Snap span edges to syllable boundaries, merge neighbouring spans separated only by punctuation, drop books with fewer than 10 spans, and mask quotation-like spans out of the loss.
3. Set aside five interlinear or short-span books (IDAD44BA2, I1637B774, I9FF2B59B, I9CB71958, IB522F095): they are in no split.
4. Classical commentaries: build one interleaved document per aligned commentary from the WeBuddhist repos (the root verse from the root-text file, then the commentary that follows it, in reading order; the pairing comes from the `![[root#^id]]` transclusion links; outline headings, footnotes and markup removed; one span per run of adjacent root blocks; inline re-quotes of the verse found by fuzzy syllable matching and listed as ignore spans). Output: `interleaved_v3` (25 documents, 6 root texts; script `build_interleaved.py`, not yet saved in the repo, see the list above). Then convert and split: `convert_tsawa_classical.py`, `prepare_tsawa_classical_split.py`.
5. Collapse blank lines in the classical texts (43,660 removed) and remap all offsets: `clean_tsawa_classical_blank_lines.py`.
6. Split books: frozen book-level split in `split_v3_frozen_excl_set_aside.csv`.
7. Tokenize, window and label: `build_tsawa_dataset.py` (uses `common/bio_dataset.py`).
8. Patch validation labels for the six re-annotated books: `patch_tsawa_validation.py`, then `push_tsawa_dataset.py`.

`build_report_v10.json`, `patch_report_v11.json`, `classical_split.csv` and `collapse_report.json` in the repo describe the exact build.

## Annotation

**BDRC books (existing and re-annotated).** Tsawa spans were marked by human annotators on the commentary text; 26 books were re-annotated and the edges corrected. The code normalises ends and merges spans, but does not add new ones.

**Classical commentaries.** Not hand-annotated. The labels come from the WeBuddhist alignment: each root verse that a commentary links to by transclusion is inserted into the document from the root-text file and labelled Tsawa. The labels are therefore only as good as those links and the root-text files. Verses are also cited several times: the 25 classical documents together hold about 21,000 root-block citations of about 5,600 distinct verses (counting each root block as one citation), because each root text has 4 or 5 commentaries. Inline re-quotes of a verse inside the commentary are masked (-100), found by fuzzy matching; a few masks may include some commentary words, and short partial quotes are not masked.

## Known limits

- Quotations from other works look like Tsawa and cause most false positives.
- Positive tokens are about 8% of scored train tokens; use class weights.
- The test benchmark is only 22 books.
- The text comes from BDRC volumes and, for the classical commentaries, from public WeBuddhist repositories. Data-rights questions are still open for both, so check the terms before sharing any text derived from this dataset.

## Citation

```bibtex
@misc{bo_tsawa_dataset,
  title  = {Bo-Tsawa-Dataset: Tibetan Tsawa token classification data},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/BDRC/Bo-Tsawa-Dataset}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).

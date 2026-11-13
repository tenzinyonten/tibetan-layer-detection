#!/usr/bin/env python3
"""Push tsawa_dataset_v8 (local DatasetDict) to the Hugging Face Hub, then verify by reloading.

Private by default. Checks split sizes against build_report.json first, appends a short card
to the auto-generated README, and uploads build_report.json and classical_split.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "tsawa/data/processed/tsawa_dataset_v8"
SPLIT_CSV = ROOT / "tsawa/data/processed/classical_v3/classical_split.csv"

NOTES = """

## Notes (v8)

Binary BIO tsawa dataset, mmBERT tokenizer (`jhu-clsp/mmBERT-base`), windows of 8192 tokens with stride 5120.
Labels: `O`=0, `B-TSAWA`=1, `I-TSAWA`=2, `-100` = special tokens, padding and masked tokens.

| split | docs | windows | spans |
|---|---:|---:|---:|
| train | 92 | 4,850 | 16,716 |
| validation (.opf books, frozen) | 18 | 456 | 1,700 |
| test (.opf benchmark, frozen) | 22 | 639 | 1,976 |
| classical_validation | 6 | 642 | 3,652 |
| classical_test | 6 | 634 | 4,397 |

Train sources: 53 existing books, 26 re-annotated books, 13 classical commentaries (7,827 spans).
Span ends follow the shad-inside convention. The 5 interlinear/short-span books IDAD44BA2, I1637B774,
I9FF2B59B, I9CB71958 and IB522F095 are not in any split. Repeated verses in the classical docs are masked (-100).

Classical splits are whole-commentary holdouts (one test and one validation commentary per root text).
They share root verses with the train commentaries (about 90% of classical_validation / 99% of classical_test
verses also occur in a train commentary), so they measure recall of seen root text, not generalization.
`validation` and `test` are the real-book sets; `test` is the benchmark and must stay untouched.
`build_report.json` and `classical_split.csv` describe the exact build.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Yontenn/formatting-tsawa-v8")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    ds = load_from_disk(str(LOCAL))
    rep = json.loads((LOCAL / "build_report.json").read_text())
    for split, n in rep["windows"].items():
        name = split
        if len(ds[name]) != n:
            raise SystemExit(f"{name}: {len(ds[name])} rows, build report says {n}")
    print({k: len(v) for k, v in ds.items()})

    ds.push_to_hub(args.repo_id, private=not args.public)
    api = HfApi()
    readme = Path(hf_hub_download(args.repo_id, "README.md", repo_type="dataset"))
    card = readme.read_text(encoding="utf-8")
    if "## Notes (v8)" not in card:
        readme.write_text(card.rstrip("\n") + NOTES, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.repo_id, repo_type="dataset")
    for src, dst in ((LOCAL / "build_report.json", "build_report.json"), (SPLIT_CSV, "classical_split.csv")):
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=args.repo_id, repo_type="dataset")

    # verify: reload from the Hub and compare against the local copy
    info = api.dataset_info(args.repo_id)
    print("repo:", info.id, "private" if info.private else "public", "| files:", sorted(s.rfilename for s in info.siblings))
    for split in ("validation", "classical_test"):
        hub = load_dataset(args.repo_id, split=split)
        loc = ds[split]
        same = len(hub) == len(loc) and all(hub[i][c] == loc[i][c] for i in (0, len(loc) // 2, len(loc) - 1)
                                            for c in ("pecha_id", "input_ids", "labels", "window_index"))
        print(f"reload {split}: hub rows {len(hub)} / local {len(loc)} | sampled rows identical: {same}")
        if not same:
            raise SystemExit(f"{split} does not match after upload")
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

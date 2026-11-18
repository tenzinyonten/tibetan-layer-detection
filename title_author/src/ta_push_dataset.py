#!/usr/bin/env python3
"""Push the title/author DatasetDict + card to the Hugging Face Hub (private unless --public).
    python title_author/src/ta_push_dataset.py --repo-id REPO [--local-dir DIR] [--card FILE] [--public]

Reads the DatasetDict in --local-dir (default title_author/data/processed/gold_dataset_all) and the card --card (default title_author/docs/dataset_card.md;
for the single-entity datasets pass the card from title_author/docs/hf_cards/). Writes to the Hub repo: the data/ parquet files, README.md (the card),
label_map.json; data files of splits that no longer exist are deleted. --repo-id is required and has no default. Needs a Hugging Face login with write access.
"""
import argparse, json
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
LABEL2ID = {"O": 0, "B-TITLE": 1, "I-TITLE": 2, "B-AUTHOR": 3, "I-AUTHOR": 4}

p = argparse.ArgumentParser()
p.add_argument("--local-dir", type=Path, default=ROOT / "title_author/data/processed/gold_dataset_all", help="DatasetDict folder to push")
p.add_argument("--card", type=Path, default=ROOT / "title_author/docs/dataset_card.md", help="markdown file uploaded as README.md of the dataset repo")
p.add_argument("--repo-id", required=True,
               help="no default on purpose: pushing over Yontenn/title-author-training replaces the data that models were trained on")
p.add_argument("--public", action="store_true", help="make the repo public (default: private)")
a = p.parse_args()
dset = load_from_disk(str(a.local_dir))
print(f"Pushing {a.repo_id} private={not a.public}: " + ", ".join(f"{k}={len(v)}" for k, v in dset.items()), flush=True)
dset.push_to_hub(a.repo_id, private=not a.public)
api = HfApi()
api.upload_file(path_or_fileobj=str(a.card), path_in_repo="README.md", repo_id=a.repo_id, repo_type="dataset")
api.upload_file(path_or_fileobj=json.dumps(LABEL2ID, indent=2).encode(), path_in_repo="label_map.json",
                repo_id=a.repo_id, repo_type="dataset")
stale = [f.rfilename for f in api.dataset_info(a.repo_id).siblings
         if f.rfilename.startswith("data/") and f.rfilename.split("/")[1].split("-")[0] not in dset]
if stale:
    print("removing stale files:", stale)
    api.delete_files(a.repo_id, delete_patterns=stale, repo_type="dataset")
info = api.dataset_info(a.repo_id)
print("private:", info.private, "| https://huggingface.co/datasets/" + a.repo_id)

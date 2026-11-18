#!/usr/bin/env python3
"""Download the default config of BDRC/yigchung_detection_v2 (book text + spans) into paths.HUB
as train.parquet / validation.parquet / test.parquet.

Classified and general systems both start here. The dataset is renamed BDRC/Bo-Yigchung-Dataset (the old name
redirects); the code keeps the old name and pins revision 778c059b so the genre sets stay reproducible.
Usage: python yigchung/src/fetch_yigchung_hub.py     (no arguments)
Reads:  the Hub (network). Writes: <YIGCHUNG_DATA>/hub/{train,validation,test}.parquet (default yigchung/data/local/hub).
Needed by merge_yigchung_lotsawa.py, build_yigchung_genre_datasets.py, stratify_yigchung_genre.py, build_yigchung_benchmark_after.py, score_yigchung_benchmark.py."""
import shutil

from huggingface_hub import hf_hub_download

from paths import HUB

REPO, REVISION = "BDRC/yigchung_detection_v2", "778c059b1a6d93f549160ec06de6567ce020fd86"   # revision the genre sets were built from
HUB.mkdir(parents=True, exist_ok=True)
for split in ["train", "validation", "test"]:
    p = hf_hub_download(REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    shutil.copy(p, HUB / f"{split}.parquet")
    print("saved", HUB / f"{split}.parquet")

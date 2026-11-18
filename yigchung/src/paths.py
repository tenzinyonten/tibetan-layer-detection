"""Where the Yigchung scripts read and write.

Small tables (catalog, split, hand corrections) are tracked in yigchung/data/processed/.
Large inputs and built datasets live in yigchung/data/local/ (gitignored). Set YIGCHUNG_DATA
to use another folder.

Imported by every other script in this folder (classified and general systems); it is not run itself.
Environment: YIGCHUNG_DATA = folder for local inputs and outputs (default yigchung/data/local).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "yigchung" / "data" / "processed"
LOCAL = Path(os.environ.get("YIGCHUNG_DATA", REPO / "yigchung" / "data" / "local"))

# tracked tables
CATALOG = PROCESSED / "yigchung_catalog.csv"          # one row per book: genre label and genre split
CORRECTIONS = PROCESSED / "span_corrections.csv"      # hand drops / edge edits
ADDITIONS = PROCESSED / "span_additions.csv"          # hand-added spans
SPLIT_PROPOSAL = PROCESSED / "genre_split_proposal.csv"

# local inputs (not in git)
HUB = LOCAL / "hub"                                   # BDRC/yigchung_detection_v2 default config: <split>.parquet (fetch_yigchung_hub.py)
LOTSAWA = LOCAL / "lotsawa_combined.parquet"          # merge_yigchung_lotsawa.py
ANNOTATOR_SPANS = LOCAL / "annotator_spans"           # <book_id>.json: annotator-reviewed spans for books in the HF set
EXTRA_BOOKS = LOCAL / "extra_books"                   # <book_id>.json: annotator-reviewed books not in the HF set
ANNOTATOR_YAML = LOCAL / "annotator_review" / "yaml"  # annotator layer files for the 5 BDRC benchmark books
ANNOTATOR_ZIPS = LOCAL / "annotator_review" / "zips"  # the review OPF zips those offsets refer to

# local outputs (not in git)
GENRE = LOCAL / "genre_training"                      # with_lotsawa/, without_lotsawa/, bio_<name>/
BENCHMARK_AFTER = LOCAL / "benchmark_after"           # corrected 77-book benchmark

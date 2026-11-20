from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download

# Two names imported from src/layer_detection/quotation_windows.py
from quotation_lib.quotation_windows import (
    GateFailure,
    _ensure_windowed_config_yaml,
    load_windowed_dataset,
)

# Constants
LABEL_O = 0
LABEL_B_QUOTE = 1
LABEL_I_QUOTE = 2
LABEL_E = 3

SRC_CONFIG_NAME = "windowed_w8192_s4579"
SRC_TAG = "w1.0"

EXPECTED_SPANS = {
    "train": 18753,
    "validation": 2005,
    "test": 2337,
}

EXPECTED_BOUNDARY = {
    "i_at_boundary_total": 512,
    "i_continues": 507,
    "i_closes_at_boundary": 5,
    "b_continues": 6,
    "books_ending_mid_span": 0,
}

@dataclass
class BookShard:
    split: str
    book_index: int
    book_id: str
    filename: Path
    table: pa.Table

@dataclass
class Report:
    ok: bool
    failures: list[str]
    gate_lines: list[str]
    class_balance: dict[str, float]
    gate_counts: dict[str, int]

def _list_column_arrays(col: pa.ChunkedArray) -> list[np.ndarray]:
    """Split a ListArray into per-row 1-d numpy views without to_pylist."""
    chunk = col.combine_chunks()
    values = chunk.values.to_numpy(zero_copy_only=False)
    offsets = chunk.offsets.to_numpy()
    return [values[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]

def _offset_mapping_arrays(col: pa.ChunkedArray) -> list[np.ndarray]:
    """Split list<list<int>> offset_mapping into (n_tokens, 2) arrays."""
    outer = col.combine_chunks()
    inner = outer.values
    inner_vals = inner.values.to_numpy(zero_copy_only=False)
    inner_off = inner.offsets.to_numpy()
    outer_off = outer.offsets.to_numpy()
    out = []
    for i in range(len(outer_off) - 1):
        start_list = int(outer_off[i])
        end_list = int(outer_off[i + 1])
        start = int(inner_off[start_list])
        end = int(inner_off[end_list])
        pairs = inner_vals[start:end]
        if pairs.size % 2:
            raise GateFailure(
                f"offset_mapping row {i} has {pairs.size} scalars, expected pairs"
            )
        out.append(pairs.reshape(-1, 2))
    return out

def load_shards(src_dir: Path) -> Iterator[BookShard]:
    shards_dir = src_dir / "shards"
    for split in ["train", "validation", "test"]:
        split_dir = shards_dir / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.parquet")):
            name = path.stem
            if "_" in name:
                parts = name.split("_", 1)
                book_index = int(parts[0])
                book_id = parts[1]
            else:
                book_index = -1
                book_id = name
            
            table = pq.read_table(path)
            yield BookShard(
                split=split,
                book_index=book_index,
                book_id=book_id,
                filename=path,
                table=table,
            )

def relabel_book(shard: BookShard) -> BookShard:
    table = shard.table
    # Sort rows by window_index
    indices = pc.sort_indices(table, sort_keys=[("window_index", "ascending")])
    sorted_table = table.take(indices)
    
    labels_col = sorted_table.column("labels")
    offset_mapping_col = sorted_table.column("offset_mapping")
    
    labels_list = _list_column_arrays(labels_col)
    offset_mapping_list = _offset_mapping_arrays(offset_mapping_col)
    
    owned_masks = []
    owned_labels_list = []
    owned_offsets_list = []
    
    for row_labels, row_offsets in zip(labels_list, offset_mapping_list):
        mask = (row_labels != -100)
        owned_masks.append(mask)
        owned_labels_list.append(row_labels[mask])
        owned_offsets_list.append(row_offsets[mask])
        
    owned = np.concatenate(owned_labels_list)
    owned_offsets = np.concatenate(owned_offsets_list)
    
    nxt = np.empty_like(owned)
    if len(owned) > 0:
        nxt[:-1] = owned[1:]
        nxt[-1] = LABEL_O # end of book
        
    # Gate 4 check: No single-token spans
    bad_mask = (owned == LABEL_B_QUOTE) & (nxt != LABEL_I_QUOTE)
    if np.any(bad_mask):
        bad_indices = np.where(bad_mask)[0]
        idx = bad_indices[0]
        char_start, char_end = owned_offsets[idx]
        raise GateFailure(
            f"Gate 4 Failure: Book {shard.book_id} has a single-token span "
            f"at owned token index {idx} with character range {char_start}:{char_end}"
        )
        
    # Relabel I to E where next token is not I
    owned[(owned == LABEL_I_QUOTE) & (nxt != LABEL_I_QUOTE)] = LABEL_E
    
    # Scatter back
    offset = 0
    new_labels_list = []
    for row_labels, mask in zip(labels_list, owned_masks):
        n_owned = np.sum(mask)
        new_row_labels = row_labels.copy()
        new_row_labels[mask] = owned[offset : offset + n_owned]
        new_labels_list.append(new_row_labels)
        offset += n_owned
        
    # Rebuild ListArray for labels column
    labels_idx = sorted_table.schema.get_field_index("labels")
    source_field = sorted_table.schema.field(labels_idx)
    source_type = source_field.type
    value_type = source_type.value_type
    
    flat_new_labels = np.concatenate(new_labels_list)
    lengths = [len(arr) for arr in new_labels_list]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    
    flat_values_arr = pa.array(flat_new_labels, type=value_type)
    new_list_arr = pa.ListArray.from_arrays(offsets, flat_values_arr).cast(source_type)
    
    # Replace column in sorted_table
    relabeled_table = sorted_table.set_column(labels_idx, source_field, new_list_arr)
    
    # Preserve schema metadata if any
    if sorted_table.schema.metadata:
        relabeled_table = relabeled_table.replace_schema_metadata(sorted_table.schema.metadata)
        
    return BookShard(
        split=shard.split,
        book_index=shard.book_index,
        book_id=shard.book_id,
        filename=shard.filename,
        table=relabeled_table,
    )

def write_shards(shards: list[BookShard], out_dir: Path) -> None:
    for shard in shards:
        out_path = out_dir / "shards" / shard.split / shard.filename.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            shard.table,
            out_path,
            compression="SNAPPY",
            use_dictionary=True,
            version="2.6",
        )

def verify(src_dir: Path, out_dir: Path) -> Report:
    failures = []
    gate_lines = []
    
    # Load source and target shards
    src_shards = {s.book_id: s for s in load_shards(src_dir)}
    out_shards = {s.book_id: s for s in load_shards(out_dir)}
    
    if len(src_shards) != len(out_shards):
        failures.append(f"Shard count mismatch: src={len(src_shards)}, out={len(out_shards)}")
        raise GateFailure("\n".join(failures))
        
    # Check Gate 1 & 2
    for book_id, src_shard in src_shards.items():
        if book_id not in out_shards:
            failures.append(f"Book {book_id} missing from output shards")
            continue
        out_shard = out_shards[book_id]
        
        # Gate 2: Passthrough identity
        for col in ["book_id", "window_index", "input_ids", "attention_mask", "offset_mapping"]:
            src_col = src_shard.table.column(col)
            out_col = out_shard.table.column(col)
            if src_col.type != out_col.type:
                failures.append(f"Book {book_id} column {col} type mismatch: src={src_col.type}, out={out_col.type}")
            if not src_col.combine_chunks().equals(out_col.combine_chunks()):
                failures.append(f"Book {book_id} column {col} value mismatch")
                
        # Gate 1: Downgrade identity
        src_labels_list = _list_column_arrays(src_shard.table.column("labels"))
        out_labels_list = _list_column_arrays(out_shard.table.column("labels"))
        
        if len(src_labels_list) != len(out_labels_list):
            failures.append(f"Book {book_id} window count mismatch: src={len(src_labels_list)}, out={len(out_labels_list)}")
            continue
            
        for i, (src_arr, out_arr) in enumerate(zip(src_labels_list, out_labels_list)):
            downgraded = np.where(out_arr == LABEL_E, LABEL_I_QUOTE, out_arr)
            if not np.array_equal(downgraded, src_arr):
                failures.append(f"Book {book_id} window {i} labels downgrade mismatch")
                
    if failures:
        raise GateFailure("\n".join(failures))
    gate_lines.append("Gate 1 Downgrade identity: OK")
    gate_lines.append("Gate 2 Passthrough identity: OK")
    
    # Let's compute split statistics and check Gate 3, 4, 5, 6
    split_b_counts = {"train": 0, "validation": 0, "test": 0}
    split_e_counts = {"train": 0, "validation": 0, "test": 0}
    
    # Boundary accounting counters
    i_at_boundary_total = 0
    i_continues = 0
    i_closes_at_boundary = 0
    b_continues = 0
    books_ending_mid_span = 0
    
    # Class balance counters
    total_tokens_by_class = {LABEL_O: 0, LABEL_B_QUOTE: 0, LABEL_I_QUOTE: 0, LABEL_E: 0}
    
    for book_id, out_shard in out_shards.items():
        src_shard = src_shards[book_id]
        
        # Get sorted labels
        src_labels_list = _list_column_arrays(src_shard.table.column("labels"))
        out_labels_list = _list_column_arrays(out_shard.table.column("labels"))
        
        # Owned masks
        src_owned_list = [arr[arr != -100] for arr in src_labels_list]
        out_owned_list = [arr[arr != -100] for arr in out_labels_list]
        
        # Boundary accounting (on source labels)
        for k in range(len(src_owned_list) - 1):
            if len(src_owned_list[k]) == 0 or len(src_owned_list[k+1]) == 0:
                continue
            last_val = src_owned_list[k][-1]
            first_val = src_owned_list[k+1][0]
            
            if last_val == LABEL_I_QUOTE:
                i_at_boundary_total += 1
                if first_val == LABEL_I_QUOTE:
                    i_continues += 1
                elif first_val == LABEL_O:
                    i_closes_at_boundary += 1
            elif last_val == LABEL_B_QUOTE:
                if first_val == LABEL_I_QUOTE:
                    b_continues += 1
                    
        # Check if book ends mid-span (on source labels)
        book_src_owned = np.concatenate(src_owned_list) if src_owned_list else np.array([])
        if len(book_src_owned) > 0:
            last_owned_token_of_book = book_src_owned[-1]
            if last_owned_token_of_book in (LABEL_I_QUOTE, LABEL_B_QUOTE):
                books_ending_mid_span += 1
                
        # Stitch out labels for Gate 3, 5
        book_out_owned = np.concatenate(out_owned_list) if out_owned_list else np.array([])
        
        # Count B and E
        b_count = np.sum(book_out_owned == LABEL_B_QUOTE)
        e_count = np.sum(book_out_owned == LABEL_E)
        split_b_counts[out_shard.split] += b_count
        split_e_counts[out_shard.split] += e_count
        
        # Count classes for balance
        for label in [LABEL_O, LABEL_B_QUOTE, LABEL_I_QUOTE, LABEL_E]:
            total_tokens_by_class[label] += np.sum(book_out_owned == label)
            
        # Gate 5: Legality
        # Walk stitched stream with virtual O at each end
        stream = np.concatenate([[LABEL_O], book_out_owned, [LABEL_O]])
        for i in range(len(stream) - 1):
            curr = stream[i]
            nxt_val = stream[i+1]
            if curr == LABEL_O and nxt_val == LABEL_I_QUOTE:
                failures.append(f"Book {book_id} illegal transition O -> I at index {i}")
            if curr == LABEL_O and nxt_val == LABEL_E:
                failures.append(f"Book {book_id} illegal transition O -> E at index {i}")
            if curr == LABEL_E and nxt_val == LABEL_I_QUOTE:
                failures.append(f"Book {book_id} illegal transition E -> I at index {i}")
            if curr == LABEL_I_QUOTE and nxt_val == LABEL_O:
                failures.append(f"Book {book_id} illegal transition I -> O at index {i}")
            if curr == LABEL_I_QUOTE and nxt_val == LABEL_B_QUOTE:
                failures.append(f"Book {book_id} illegal transition I -> B at index {i}")
            if curr == LABEL_B_QUOTE and nxt_val == LABEL_O:
                failures.append(f"Book {book_id} illegal transition B -> O at index {i}")
            if curr == LABEL_B_QUOTE and nxt_val == LABEL_B_QUOTE:
                failures.append(f"Book {book_id} illegal transition B -> B at index {i}")
            if curr == LABEL_E and nxt_val == LABEL_E:
                failures.append(f"Book {book_id} illegal transition E -> E at index {i}")
                
    # Gate 3: One end per start
    for split in ["train", "validation", "test"]:
        b_cnt = split_b_counts[split]
        e_cnt = split_e_counts[split]
        expected = EXPECTED_SPANS[split]
        if b_cnt != e_cnt:
            failures.append(f"Split {split} B/E count mismatch: B={b_cnt}, E={e_cnt}")
        if b_cnt != expected:
            failures.append(f"Split {split} span count mismatch: got {b_cnt}, expected {expected}")
            
    if failures:
        raise GateFailure("\n".join(failures))
    gate_lines.append("Gate 3 One end per start: OK")
    gate_lines.append("Gate 4 No single-token spans: OK")
    gate_lines.append("Gate 5 Legality: OK")
    
    # Gate 6: Boundary accounting
    if i_at_boundary_total != EXPECTED_BOUNDARY["i_at_boundary_total"]:
        failures.append(f"Boundary i_at_boundary_total mismatch: got {i_at_boundary_total}, expected {EXPECTED_BOUNDARY['i_at_boundary_total']}")
    if i_continues != EXPECTED_BOUNDARY["i_continues"]:
        failures.append(f"Boundary i_continues mismatch: got {i_continues}, expected {EXPECTED_BOUNDARY['i_continues']}")
    if i_closes_at_boundary != EXPECTED_BOUNDARY["i_closes_at_boundary"]:
        failures.append(f"Boundary i_closes_at_boundary mismatch: got {i_closes_at_boundary}, expected {EXPECTED_BOUNDARY['i_closes_at_boundary']}")
    if b_continues != EXPECTED_BOUNDARY["b_continues"]:
        failures.append(f"Boundary b_continues mismatch: got {b_continues}, expected {EXPECTED_BOUNDARY['b_continues']}")
    if books_ending_mid_span != EXPECTED_BOUNDARY["books_ending_mid_span"]:
        failures.append(f"Boundary books_ending_mid_span mismatch: got {books_ending_mid_span}, expected {EXPECTED_BOUNDARY['books_ending_mid_span']}")
        
    if failures:
        raise GateFailure("\n".join(failures))
    gate_lines.append("Gate 6 Boundary accounting: OK")
    
    # Compute class balance percentages
    total_tokens = sum(total_tokens_by_class.values())
    class_balance = {}
    if total_tokens > 0:
        class_balance = {
            "O": total_tokens_by_class[LABEL_O] / total_tokens,
            "B": total_tokens_by_class[LABEL_B_QUOTE] / total_tokens,
            "I": total_tokens_by_class[LABEL_I_QUOTE] / total_tokens,
            "E": total_tokens_by_class[LABEL_E] / total_tokens,
        }
        
    gate_counts = {
        "i_at_boundary_total": i_at_boundary_total,
        "i_continues": i_continues,
        "i_closes_at_boundary": i_closes_at_boundary,
        "b_continues": b_continues,
        "books_ending_mid_span": books_ending_mid_span,
    }
    
    return Report(
        ok=True,
        failures=[],
        gate_lines=gate_lines,
        class_balance=class_balance,
        gate_counts=gate_counts,
    )

def write_stats(report: Report, out_dir: Path) -> None:
    stats_path = out_dir / "relabel_stats.json"
    stats = {
        "source_config": SRC_CONFIG_NAME,
        "source_tag": SRC_TAG,
        "gate_counts": report.gate_counts,
        "class_balance": {
            "O": f"{report.class_balance.get('O', 0.0) * 100:.4f}%",
            "I": f"{report.class_balance.get('I', 0.0) * 100:.4f}%",
            "B": f"{report.class_balance.get('B', 0.0) * 100:.4f}%",
            "E": f"{report.class_balance.get('E', 0.0) * 100:.4f}%",
        },
        "closer_statistics": {
            "provenance": "measured 2026-09-10 over Data/audit/quotation_gold_merged.json + quotation_split.json",
            "table": [
                {"class": "all", "spans": 23095, "ends_with_closer": "20,923 (90.6%)"},
                {"class": "TEXT", "spans": 13164, "ends_with_closer": "94.5%"},
                {"class": "FORMULA", "spans": 1007, "ends_with_closer": "94.3%"},
                {"class": "CHAINED", "spans": 934, "ends_with_closer": "94.4%"},
                {"class": "PERSON", "spans": 3636, "ends_with_closer": "85.6%"},
                {"class": "MISC", "spans": 4354, "ends_with_closer": "81.4%"}
            ]
        }
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def build_card_section(report: Report) -> str:
    section = f"""<!-- quotation-bioe-card -->
## Quotation Windows BIOE (Config `windowed_w8192_s4579_bioe`)

This config contains the same token windows as `windowed_w8192_s4579`, but with quotation labels relabeled from BIO to BIOE format.

### Label Set

| id | label | note |
|---:|---|---|
| 0 | `O` | unchanged |
| 1 | `B-QUOTE` | unchanged |
| 2 | `I-QUOTE` | unchanged |
| 3 | `E-QUOTE` | **new** |

### Relabeling Rule
Within each book, windows are sorted by `window_index` and their owned tokens (where `labels != -100`) are treated as a single continuous stream.
For each owned token `t`:
- If `label[t] == I` and the next owned token's label is not `I` (or `O` if `t` is the last owned token of the book), then `label[t]` becomes `E`.
- If `label[t] == B` and the next owned token's label is not `I`, a single-token span is detected (which is illegal and does not occur in this corpus).
- Otherwise, the label is unchanged.

This lookahead is evaluated across window seams (i.e. the next owned token after the last owned token of window $k$ is the first owned token of window $k+1$).
Mapping `E -> I` recovers the original BIO config exactly.

### Spans and Token Lengths
There are no single-token or two-token spans in this corpus (the minimum span length is 3 tokens), so no single-token (`S`) tag is used.

### Class Balance (Owned Tokens)
- `O`: {report.class_balance.get('O', 0.0) * 100:.4f}%
- `I`: {report.class_balance.get('I', 0.0) * 100:.4f}%
- `B`: {report.class_balance.get('B', 0.0) * 100:.4f}%
- `E`: {report.class_balance.get('E', 0.0) * 100:.4f}%

### Boundary Accounting
- `I` then `I` (span continues across boundary): {report.gate_counts.get('i_continues', 0)}
- `I` then `O` (span closes at boundary): {report.gate_counts.get('i_closes_at_boundary', 0)}
- `B` then `I` (span starts at boundary): {report.gate_counts.get('b_continues', 0)}
- Books ending mid-span: {report.gate_counts.get('books_ending_mid_span', 0)}

### Closer Statistics (Source Provenance)
Measured over `Data/audit/quotation_gold_merged.json` + `quotation_split.json`:

| | spans | ends with a closer |
|---|---:|---:|
| **all** | **23,095** | **20,923 (90.6%)** |
| TEXT | 13,164 | 94.5% |
| FORMULA | 1,007 | 94.3% |
| CHAINED | 934 | 94.4% |
| PERSON | 3,636 | 85.6% |
| MISC | 4,354 | 81.4% |
"""
    return section

def push_to_hub(
    out_dir: Path,
    repo: str,
    config_name: str,
    tag: str,
    *,
    card_section: str,
) -> None:
    import os
    if not config_name or config_name in {"default", "data", "windowed_w8192_s4579"}:
        raise GateFailure(
            f"refusing push_to_hub with config_name={config_name} (would overwrite or corrupt existing configs)"
        )
    if tag == "w1.0":
        raise GateFailure("refusing push_to_hub with tag='w1.0' (must not overwrite or move w1.0)")
        
    if os.environ.get("HF_TOKEN") is None:
        from huggingface_hub import get_token
        if get_token() is None:
            raise GateFailure("HF_TOKEN or huggingface-cli login required for --push")
            
    api = HfApi()
    
    # 1. Snapshot refs before upload
    refs_before = api.list_repo_refs(repo, repo_type="dataset")
    tags_before = {t.name: t.target_commit for t in refs_before.tags}
    main_before = next((b.target_commit for b in refs_before.branches if b.name == "main"), "")
    
    # 2. Snapshot windowed_w8192_s4579 files before upload
    files_before = api.list_repo_tree(repo, repo_type="dataset", recursive=True)
    w1_files_before = {f.path: f.blob_id for f in files_before if getattr(f, "type", None) == "file" and f.path.startswith("windowed_w8192_s4579/")}
    
    # 3. Upload the new config shards
    shard_root = out_dir / "shards"
    if not shard_root.is_dir():
        raise GateFailure(f"no shards at {shard_root}")
        
    print(f"pre-push: main={main_before}")
    api.upload_folder(
        folder_path=str(shard_root),
        path_in_repo=config_name,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns="*.parquet",
        commit_message=(
            f"Add named config {config_name} (quotation windows BIOE). "
            "Default book-level config and windowed_w8192_s4579 unchanged."
        ),
    )
    
    # 4. Rewrite README.md
    try:
        readme_path = hf_hub_download(repo, "README.md", repo_type="dataset")
        existing = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        existing = ""
        
    # Ensure windowed config yaml entry exists
    existing = _ensure_windowed_config_yaml(existing, config_name)
    
    bioe_marker = "<!-- quotation-bioe-card -->"
    windows_marker = "<!-- quotation-windows-card -->"
    
    if bioe_marker in existing:
        start_bioe = existing.index(bioe_marker)
        if windows_marker in existing:
            end_bioe = existing.index(windows_marker)
            existing = existing[:start_bioe] + card_section.strip() + "\n\n" + existing[end_bioe:]
        else:
            existing = existing[:start_bioe] + card_section.strip() + "\n"
    else:
        if windows_marker in existing:
            idx = existing.index(windows_marker)
            existing = existing[:idx] + card_section.strip() + "\n\n" + existing[idx:]
        else:
            existing = existing.rstrip() + "\n\n" + card_section.strip() + "\n"
            
    api.upload_file(
        path_or_fileobj=existing.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Document named config {config_name}",
    )
    
    # 5. Create tag
    refs_after = api.list_repo_refs(repo, repo_type="dataset")
    branches_after = {b.name: b.target_commit for b in refs_after.branches}
    tags_after = {t.name: t.target_commit for t in refs_after.tags}
    
    revision = branches_after.get("main")
    if not revision:
        raise GateFailure("main branch missing after README upload")
        
    if tags_after.get(tag) != revision:
        if tag in tags_after:
            api.delete_tag(repo, tag=tag, repo_type="dataset")
        api.create_tag(
            repo,
            tag=tag,
            revision=revision,
            repo_type="dataset",
            tag_message=f"{config_name} quotation windows BIOE",
        )
        
    # 6. Re-snapshot and assert
    refs_final = api.list_repo_refs(repo, repo_type="dataset")
    tags_final = {t.name: t.target_commit for t in refs_final.tags}
    
    # Assert w1.0 still points to 843e7f3c
    if "w1.0" in tags_final:
        if not tags_final["w1.0"].startswith("843e7f3c"):
            raise GateFailure(f"w1.0 tag changed after push! Points to {tags_final['w1.0']}, expected 843e7f3c")
    else:
        raise GateFailure("w1.0 tag missing after push!")
        
    # Assert windowed_w8192_s4579 files are unchanged
    files_final = api.list_repo_tree(repo, repo_type="dataset", recursive=True)
    w1_files_final = {f.path: f.blob_id for f in files_final if getattr(f, "type", None) == "file" and f.path.startswith("windowed_w8192_s4579/")}
    
    for path, blob_id in w1_files_before.items():
        if path not in w1_files_final:
            raise GateFailure(f"File {path} was deleted or moved during push!")
        if w1_files_final[path] != blob_id:
            raise GateFailure(f"File {path} was modified during push! (blob_id changed from {blob_id} to {w1_files_final[path]})")
            
    print(f"post-push: successfully pushed {config_name}@{tag}. w1.0 and windowed_w8192_s4579 are completely untouched.")

def verify_hub_round_trip(
    repo: str,
    *,
    config_name: str,
    tag: str,
    hub_verify_full: bool = False,
) -> None:
    """Gate 1 against Hub configs ``windowed_w8192_s4579@w1.0`` vs ``{config_name}@{tag}``."""
    print("Starting Hub round-trip verification...")
    splits = ["train", "validation", "test"] if hub_verify_full else ["test"]

    for split in splits:
        print(f"Verifying split '{split}' from Hub...")
        ds_src = load_windowed_dataset(repo, SRC_CONFIG_NAME, SRC_TAG)[split]
        ds_tgt = load_windowed_dataset(repo, config_name, tag)[split]

        src_labels: dict[tuple[str, int], list[int]] = {}
        for row in ds_src:
            src_labels[(row["book_id"], row["window_index"])] = row["labels"]

        for row in ds_tgt:
            key = (row["book_id"], row["window_index"])
            if key not in src_labels:
                raise GateFailure(
                    f"Hub round-trip error: window {key} in target but not in source"
                )
            src_lbl = np.array(src_labels[key])
            tgt_lbl = np.array(row["labels"])
            downgraded = np.where(tgt_lbl == LABEL_E, LABEL_I_QUOTE, tgt_lbl)
            if not np.array_equal(downgraded, src_lbl):
                raise GateFailure(
                    f"Hub round-trip error: window {key} labels downgrade mismatch"
                )

    print("Hub round-trip verification OK!")


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = argparse.ArgumentParser(description="Relabel quotation windows from BIO to BIOE.")
    parser.add_argument("--src", type=Path, default=Path("Data/quotation_windows_w8192_s4579"),
                        help="Source directory containing BIO shards")
    parser.add_argument("--out", type=Path, default=Path("Data/quotation_windows_w8192_s4579_bioe"),
                        help="Output directory for BIOE shards")
    parser.add_argument("--repo", type=str, default="karma689/tibetan-quotation-detection",
                        help="HuggingFace dataset repository")
    parser.add_argument("--config-name", type=str, default="windowed_w8192_s4579_bioe",
                        help="HuggingFace dataset config name")
    parser.add_argument("--tag", type=str, default="w2.0",
                        help="HuggingFace dataset tag")
    parser.add_argument("--push", action="store_true",
                        help="Push the built config to HuggingFace Hub")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification on the output directory")
    parser.add_argument("--hub-verify-full", action="store_true",
                        help="Verify all 254 books from the Hub round-trip (instead of only the test split)")
                        
    args = parser.parse_args(argv)
    
    if args.verify_only:
        print(f"Running verification on {args.out}...")
        report = verify(args.src, args.out)
        print("Verification OK!")
        for line in report.gate_lines:
            print(f"  {line}")
        return 0
        
    print(f"Loading shards from {args.src}...")
    shards = list(load_shards(args.src))
    print(f"Loaded {len(shards)} shards.")
    
    print("Relabeling shards to BIOE...")
    relabeled_shards = []
    for shard in shards:
        relabeled_shards.append(relabel_book(shard))
        
    print(f"Writing relabeled shards to {args.out}...")
    write_shards(relabeled_shards, args.out)
    
    print("Running verification...")
    report = verify(args.src, args.out)
    print("Verification OK!")
    for line in report.gate_lines:
        print(f"  {line}")
        
    print("Writing statistics...")
    write_stats(report, args.out)
    
    if args.push:
        print(f"Pushing to HuggingFace Hub ({args.repo})...")
        card_section = build_card_section(report)
        push_to_hub(
            out_dir=args.out,
            repo=args.repo,
            config_name=args.config_name,
            tag=args.tag,
            card_section=card_section,
        )
        
        verify_hub_round_trip(
            args.repo,
            config_name=args.config_name,
            tag=args.tag,
            hub_verify_full=args.hub_verify_full,
        )
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

# Skip tests if pyarrow is not installed
pa = pytest.importorskip("pyarrow")

from quotation_lib.quotation_windows import GateFailure
from quotation_lib.quotation_bioe import (
    BookShard,
    relabel_book,
    LABEL_O,
    LABEL_B_QUOTE,
    LABEL_I_QUOTE,
    LABEL_E,
)

def make_synthetic_shard(
    book_id: str,
    windows: list[dict[str, Any]],
    split: str = "train",
    book_index: int = 0,
) -> BookShard:
    book_ids = [book_id] * len(windows)
    window_indices = [w["window_index"] for w in windows]
    input_ids = [w["input_ids"] for w in windows]
    attention_mask = [w["attention_mask"] for w in windows]
    labels = [w["labels"] for w in windows]
    offset_mapping = [w["offset_mapping"] for w in windows]
    
    schema = pa.schema([
        ("book_id", pa.string()),
        ("window_index", pa.int64()),
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
        ("labels", pa.list_(pa.int64())),
        ("offset_mapping", pa.list_(pa.list_(pa.int64()))),
    ])
    
    table = pa.Table.from_pydict({
        "book_id": book_ids,
        "window_index": window_indices,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
    }, schema=schema)
    
    return BookShard(
        split=split,
        book_index=book_index,
        book_id=book_id,
        filename=Path(f"{book_index:04d}_{book_id}.parquet"),
        table=table,
    )

def test_span_continues_across_boundary():
    """Case 1: A span ending exactly on a window's last owned token with the next window continuing it.

    The last owned token of window 0 is I, and the first owned token of window 1 is I.
    It must stay I.
    """
    windows = [
        {
            "window_index": 0,
            "input_ids": [100, 101, 102, 103, 104],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, LABEL_B_QUOTE, LABEL_I_QUOTE, LABEL_I_QUOTE, -100],
            "offset_mapping": [[0, 0], [1, 5], [6, 10], [11, 15], [0, 0]],
        },
        {
            "window_index": 1,
            "input_ids": [103, 104, 105, 106, 107],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [LABEL_I_QUOTE, LABEL_I_QUOTE, LABEL_O, LABEL_O, -100],
            "offset_mapping": [[11, 15], [16, 20], [21, 25], [26, 30], [0, 0]],
        }
    ]
    shard = make_synthetic_shard("B01", windows)
    relabeled = relabel_book(shard)
    
    # Read the labels back
    chunk = relabeled.table.column("labels").combine_chunks()
    values = chunk.values.to_numpy()
    offsets = chunk.offsets.to_numpy()
    
    row0_labels = values[offsets[0]:offsets[1]]
    row1_labels = values[offsets[1]:offsets[2]]
    
    # Window 0: [-100, B, I, I, -100] -> last owned is index 3 (LABEL_I_QUOTE)
    # Since window 1 starts with LABEL_I_QUOTE, window 0's last owned token must stay LABEL_I_QUOTE.
    assert row0_labels[3] == LABEL_I_QUOTE
    # Window 1: [I, I, O, O, -100] -> the last I (index 1) is followed by O (index 2), so it must become E
    assert row1_labels[0] == LABEL_I_QUOTE
    assert row1_labels[1] == LABEL_E
    assert row1_labels[2] == LABEL_O

def test_span_closes_at_boundary():
    """Case 2: A span ending exactly on a window's last owned token with the next window starting O.

    The last owned token of window 0 is I, and the first owned token of window 1 is O.
    It must become E.
    """
    windows = [
        {
            "window_index": 0,
            "input_ids": [100, 101, 102, 103, 104],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, LABEL_B_QUOTE, LABEL_I_QUOTE, LABEL_I_QUOTE, -100],
            "offset_mapping": [[0, 0], [1, 5], [6, 10], [11, 15], [0, 0]],
        },
        {
            "window_index": 1,
            "input_ids": [103, 104, 105, 106, 107],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [LABEL_O, LABEL_O, LABEL_O, LABEL_O, -100],
            "offset_mapping": [[11, 15], [16, 20], [21, 25], [26, 30], [0, 0]],
        }
    ]
    shard = make_synthetic_shard("B02", windows)
    relabeled = relabel_book(shard)
    
    chunk = relabeled.table.column("labels").combine_chunks()
    values = chunk.values.to_numpy()
    offsets = chunk.offsets.to_numpy()
    
    row0_labels = values[offsets[0]:offsets[1]]
    
    # Window 0: [-100, B, I, I, -100] -> last owned is index 3 (LABEL_I_QUOTE)
    # Since window 1 starts with LABEL_O, window 0's last owned token must become LABEL_E.
    assert row0_labels[3] == LABEL_E

def test_b_on_last_owned_token():
    """Case 3: A B on the last owned token of a window.

    The last owned token of window 0 is B, and the first owned token of window 1 is I.
    It must stay B.
    """
    windows = [
        {
            "window_index": 0,
            "input_ids": [100, 101, 102],
            "attention_mask": [1, 1, 1],
            "labels": [LABEL_O, LABEL_B_QUOTE, -100],
            "offset_mapping": [[0, 5], [6, 10], [0, 0]],
        },
        {
            "window_index": 1,
            "input_ids": [101, 102, 103],
            "attention_mask": [1, 1, 1],
            "labels": [LABEL_I_QUOTE, LABEL_I_QUOTE, LABEL_O],
            "offset_mapping": [[6, 10], [11, 15], [16, 20]],
        }
    ]
    shard = make_synthetic_shard("B03", windows)
    relabeled = relabel_book(shard)
    
    chunk = relabeled.table.column("labels").combine_chunks()
    values = chunk.values.to_numpy()
    offsets = chunk.offsets.to_numpy()
    
    row0_labels = values[offsets[0]:offsets[1]]
    row1_labels = values[offsets[1]:offsets[2]]
    
    assert row0_labels[1] == LABEL_B_QUOTE
    assert row1_labels[0] == LABEL_I_QUOTE
    assert row1_labels[1] == LABEL_E

def test_one_window_book():
    """Case 4: A book of one window."""
    windows = [
        {
            "window_index": 0,
            "input_ids": [100, 101, 102, 103, 104],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, LABEL_B_QUOTE, LABEL_I_QUOTE, LABEL_I_QUOTE, LABEL_O],
            "offset_mapping": [[0, 0], [1, 5], [6, 10], [11, 15], [16, 20]],
        }
    ]
    shard = make_synthetic_shard("B04", windows)
    relabeled = relabel_book(shard)
    
    chunk = relabeled.table.column("labels").combine_chunks()
    values = chunk.values.to_numpy()
    offsets = chunk.offsets.to_numpy()
    
    row0_labels = values[offsets[0]:offsets[1]]
    
    # [-100, B, I, I, O] -> [-100, B, I, E, O]
    assert row0_labels[1] == LABEL_B_QUOTE
    assert row0_labels[2] == LABEL_I_QUOTE
    assert row0_labels[3] == LABEL_E
    assert row0_labels[4] == LABEL_O

def test_single_token_span_fails():
    """Case 5: A synthetic single-token span that must fail gate 4."""
    windows = [
        {
            "window_index": 0,
            "input_ids": [100, 101, 102],
            "attention_mask": [1, 1, 1],
            "labels": [LABEL_O, LABEL_B_QUOTE, LABEL_O],
            "offset_mapping": [[0, 5], [6, 10], [11, 15]],
        }
    ]
    shard = make_synthetic_shard("B05", windows)
    
    with pytest.raises(GateFailure) as exc_info:
        relabel_book(shard)
        
    assert "Gate 4 Failure" in str(exc_info.value)
    assert "B05" in str(exc_info.value)
    assert "6:10" in str(exc_info.value)

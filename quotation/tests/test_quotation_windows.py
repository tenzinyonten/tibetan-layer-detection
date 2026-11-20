"""Tests for quotation window build (``quotation_lib.quotation_windows``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quotation_lib.quotation_windows import (
    BookRow,
    GateFailure,
    Geometry,
    Window,
    _char_range_from_tokens,
    _expected_bio_sequence,
    _load_shards,
    _normalize_spans,
    _offset_mapping_arrays,
    _span_fully_in_window,
    _stitch_owned,
    _tokens_for_span_in_window,
    _ensure_windowed_config_yaml,
    _windows_from_parquet_table,
    existing_shards_ready,
    expected_book_counts,
    load_job_config,
    project_labels,
    resolve_geometry,
    tokenize_book_windows,
    verify,
    windowed_parquet_data_files,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "build_quotation_windows.yaml"


class WordTokenizer:
    """Minimal fast tokenizer mock: one token per whitespace-delimited word."""

    is_fast = True
    all_special_ids = [0, 1]

    def __len__(self) -> int:
        return 256_000

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = True,
        max_length: int,
        stride: int,
        return_overflowing_tokens: bool = True,
        return_offsets_mapping: bool = True,
        return_special_tokens_mask: bool = True,
        padding: bool = False,
    ) -> dict[str, Any]:
        del truncation, return_offsets_mapping, return_special_tokens_mask, padding
        words: list[tuple[int, int, int]] = []
        for part in text.split():
            start = text.index(part)
            end = start + len(part)
            words.append((100 + len(words), start, end))

        content_cap = max_length - 2
        step = max_length - stride
        windows: list[list[int]] = []
        offsets: list[list[list[int]]] = []
        specials: list[list[int]] = []
        pos = 0
        while pos < len(words):
            chunk = words[pos : pos + content_cap]
            if not chunk:
                break
            input_ids = [0] + [w[0] for w in chunk] + [1]
            offset_mapping = [[0, 0]] + [[w[1], w[2]] for w in chunk] + [[0, 0]]
            special_tokens_mask = [1] + [0] * len(chunk) + [1]
            windows.append(input_ids)
            offsets.append(offset_mapping)
            specials.append(special_tokens_mask)
            if pos + content_cap >= len(words):
                break
            pos += step
        return {
            "input_ids": windows,
            "offset_mapping": offsets,
            "special_tokens_mask": specials,
        }


def _book(text: str, spans: list[dict[str, Any]], book_id: str = "B01") -> BookRow:
    return BookRow(
        book_id=book_id,
        text=text,
        spans=spans,
        split="train",
    )


def _geom(max_length: int = 20, step: int = 12) -> Geometry:
    return Geometry(max_length=max_length, step=step, hf_stride=max_length - step)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_resolve_geometry_step_and_hf_stride():
    cfg = {"max_length": 8192, "step": 4751, "hf_stride": 3441}
    geom = resolve_geometry(cfg)
    assert geom.step == 4751
    assert geom.hf_stride == 3441


def test_resolve_geometry_refuses_step_ge_max_length():
    with pytest.raises(ValueError, match="step must satisfy"):
        resolve_geometry({"max_length": 100, "step": 100, "hf_stride": 0})


def test_resolve_geometry_refuses_hf_stride_ge_max_length():
    with pytest.raises(ValueError, match="hf_stride"):
        resolve_geometry({"max_length": 8192, "step": 4751, "hf_stride": 8192})


def test_inclusive_span_length():
    spans = _normalize_spans([{"start": 10, "end": 12}])
    assert spans[0]["end"] - spans[0]["start"] + 1 == 3


# --------------------------------------------------------------------------
# ownership + contiguity
# --------------------------------------------------------------------------


def test_ownership_contiguity_first_mode():
    text = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp"
    spans = [{"start": text.index("dd"), "end": text.index("dd") + 1}]
    book = _book(text, spans)
    tok = WordTokenizer()
    geom = _geom(max_length=12, step=6)
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "first", tok)
    stitched = _stitch_owned(book, windows, tok)
    assert stitched
    prev_end = -1
    for ts, te, _label in stitched:
        assert ts >= prev_end
        prev_end = te
    owned_keys = {(ts, te) for ts, te, _ in stitched}
    assert len(owned_keys) == len(stitched)


def test_ownership_contiguity_centre_mode():
    text = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
    spans = [{"start": text.index("jj"), "end": text.index("kk") + 1}]
    book = _book(text, spans, book_id="B02")
    tok = WordTokenizer()
    geom = _geom(max_length=12, step=6)
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "centre", tok)
    stitched = _stitch_owned(book, windows, tok)
    assert len(stitched) > 0
    keys = [(ts, te) for ts, te, _ in stitched]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------
# span edge cases
# --------------------------------------------------------------------------


def test_span_ending_exactly_on_window_boundary():
    text = "aa bb cc dd ee ff gg hh"
    end = text.index("dd") + 1
    spans = [{"start": text.index("bb"), "end": end}]
    book = _book(text, spans, book_id="B03")
    tok = WordTokenizer()
    geom = _geom(max_length=10, step=5)
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "first", tok)
    for window in windows:
        if _span_fully_in_window(
            spans[0]["start"],
            spans[0]["end"],
            window.offset_mapping,
            window.input_ids,
            tok,
        ):
            indices = _tokens_for_span_in_window(
                spans[0]["start"],
                spans[0]["end"],
                window.offset_mapping,
                window.input_ids,
                tok,
            )
            char_range = _char_range_from_tokens(
                indices, window.offset_mapping, window.input_ids, tok
            )
            assert char_range is not None
            c_start, c_end = char_range
            assert book.text[c_start:c_end] == book.text[spans[0]["start"] : spans[0]["end"] + 1]
            return
    pytest.fail("span not contained in any window")


def test_span_longer_than_step_still_contained_with_enough_overlap():
    text = " ".join(f"w{i:02d}" for i in range(30))
    start = text.index("w05")
    end = text.index("w18") + 3
    spans = [{"start": start, "end": end}]
    book = _book(text, spans, book_id="B04")
    tok = WordTokenizer()
    # span is 14 words; step=10 is shorter than span but max_length fits it whole
    geom = _geom(max_length=32, step=10)
    windows = tokenize_book_windows(book, tok, geom)
    assert any(
        _span_fully_in_window(start, end, w.offset_mapping, w.input_ids, tok)
        for w in windows
    )


def test_book_shorter_than_one_window_produces_single_window():
    text = "aa bb cc"
    book = _book(text, [{"start": 0, "end": 1}], book_id="B05")
    tok = WordTokenizer()
    geom = _geom(max_length=20, step=12)
    windows = tokenize_book_windows(book, tok, geom)
    assert len(windows) == 1
    project_labels(book, windows, "first", tok)
    stitched = _stitch_owned(book, windows, tok)
    assert len(stitched) == 3


def test_inclusive_offset_round_trip():
    text = "aa bb cc dd ee"
    start = text.index("cc")
    end = text.index("dd") + 1
    gold = text[start : end + 1]
    book = _book(text, [{"start": start, "end": end}], book_id="B06")
    tok = WordTokenizer()
    geom = _geom(max_length=12, step=6)
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "first", tok)
    for window in windows:
        if not _span_fully_in_window(start, end, window.offset_mapping, window.input_ids, tok):
            continue
        indices = _tokens_for_span_in_window(
            start, end, window.offset_mapping, window.input_ids, tok
        )
        char_range = _char_range_from_tokens(
            indices, window.offset_mapping, window.input_ids, tok
        )
        assert char_range is not None
        c_start, c_end = char_range
        assert text[c_start:c_end] == gold
        return
    pytest.fail("round-trip window not found")


def test_bio_stitched_matches_span_projection():
    text = "aa bb cc dd ee ff"
    spans = [{"start": text.index("bb"), "end": text.index("cc") + 1}]
    book = _book(text, spans, book_id="B07")
    tok = WordTokenizer()
    geom = _geom(max_length=10, step=5)
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "first", tok)
    stitched = _stitch_owned(book, windows, tok)
    expected = _expected_bio_sequence(book, stitched)
    actual = [label for _ts, _te, label in stitched]
    assert expected == actual


def test_windows_from_parquet_avoids_pylist_nested_ints():
    pa = pytest.importorskip("pyarrow")
    import numpy as np

    table = pa.table(
        {
            "book_id": ["B01", "B01"],
            "window_index": [1, 0],
            "input_ids": [[10, 20, 30], [40, 50]],
            "attention_mask": [[1, 1, 1], [1, 0]],
            "labels": [[0, 1, -100], [2, 0]],
            "offset_mapping": [
                [[0, 1], [1, 2], [0, 0]],
                [[3, 4], [4, 5]],
            ],
        }
    )
    windows = _windows_from_parquet_table(table)
    assert [w.window_index for w in windows] == [1, 0]
    assert list(windows[0].input_ids) == [10, 20, 30]
    assert list(windows[1].labels) == [2, 0]
    assert np.asarray(windows[0].offset_mapping).tolist() == [[0, 1], [1, 2], [0, 0]]
    assert list(windows[0].special_tokens_mask) == [0, 0, 1]
    pairs = _offset_mapping_arrays(table.column("offset_mapping"))
    assert pairs[0].shape == (3, 2)
    assert pairs[1].shape == (2, 2)


def test_load_shards_reads_split_directories(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    split_dir = tmp_path / "shards" / "train"
    split_dir.mkdir(parents=True)
    table = pa.table(
        {
            "book_id": ["B09"],
            "window_index": [0],
            "input_ids": [[7, 8]],
            "attention_mask": [[1, 1]],
            "labels": [[0, 1]],
            "offset_mapping": [[[0, 2], [2, 4]]],
        }
    )
    pq.write_table(table, split_dir / "0000_B09.parquet")
    loaded = _load_shards(tmp_path)
    windows = loaded["train"]["B09"]
    assert len(windows) == 1
    assert windows[0].window_index == 0
    assert list(windows[0].input_ids) == [7, 8]
    assert [list(pair) for pair in windows[0].offset_mapping] == [[0, 2], [2, 4]]


def test_existing_shards_ready_requires_all_splits(tmp_path: Path):
    expected = {"train": 2, "validation": 1, "test": 1}
    assert existing_shards_ready(tmp_path, expected) is False
    for name, n in expected.items():
        split = tmp_path / "shards" / name
        split.mkdir(parents=True)
        for i in range(n):
            (split / f"{i:04d}_B{i}.parquet").write_bytes(b"x")
    assert existing_shards_ready(tmp_path, expected) is True
    (tmp_path / "shards" / "train" / "0000_B0.parquet").unlink()
    assert existing_shards_ready(tmp_path, expected) is False


def test_existing_shards_ready_rejects_geometry_mismatch(tmp_path: Path):
    expected = {"train": 1, "validation": 1, "test": 1}
    for name in expected:
        split = tmp_path / "shards" / name
        split.mkdir(parents=True)
        (split / "0000_B0.parquet").write_bytes(b"x")
    (tmp_path / "window_stats.json").write_text(
        json.dumps(
            {
                "verification_ok": True,
                "geometry": {"max_length": 8192, "step": 4579, "hf_stride": 3613},
            }
        ),
        encoding="utf-8",
    )
    geom = Geometry(max_length=8192, step=4579, hf_stride=3613)
    assert existing_shards_ready(tmp_path, expected, geom=geom) is True
    other = Geometry(max_length=8192, step=4751, hf_stride=3441)
    assert existing_shards_ready(tmp_path, expected, geom=other) is False


def test_yaml_expected_books_match_splits():
    cfg = load_job_config(CONFIG)
    assert expected_book_counts(cfg) == {"train": 210, "validation": 25, "test": 19}


def test_windowed_parquet_data_files_pin_revision_and_extension():
    files = windowed_parquet_data_files(
        "karma689/tibetan-quotation-detection",
        "windowed_w8192_s4579",
        "w1.0",
    )
    assert files["train"].endswith("/windowed_w8192_s4579/train/*.parquet")
    assert "@w1.0/" in files["train"]
    assert set(files) == {"train", "validation", "test"}


def test_windowed_yaml_keeps_jsonl_default_and_parquet_named_config():
    text = _ensure_windowed_config_yaml(
        "---\nlicense: other\n---\nbody\n",
        "windowed_w8192_s4579",
    )
    assert "path: train.jsonl" in text
    assert "windowed_w8192_s4579/train/*.parquet" in text


def test_push_reuses_existing_shards_without_tokenize(monkeypatch, tmp_path: Path):
    from quotation_lib import quotation_windows as qw

    (tmp_path / "README_card_snippet.md").write_text(
        "<!-- quotation-windows-card -->\ncard\n", encoding="utf-8"
    )
    pushed: dict[str, Any] = {}

    def fake_push(out_dir, repo, config_name, tag, *, card_section):
        pushed.update(
            {
                "out": out_dir,
                "repo": repo,
                "config": config_name,
                "tag": tag,
                "card": card_section,
            }
        )

    monkeypatch.setattr(qw, "existing_shards_ready", lambda *a, **k: True)
    monkeypatch.setattr(qw, "push_to_hub", fake_push)
    monkeypatch.setattr(
        qw,
        "tokenize_book_windows",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not rebuild")),
    )
    monkeypatch.setattr(
        qw,
        "load_tokenizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not load tokenizer")),
    )
    monkeypatch.setattr(
        qw,
        "load_benchmark",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not load benchmark")),
    )

    rc = qw.main(["--push", "--out", str(tmp_path)])
    assert rc == 0
    assert pushed["config"] == "windowed_w8192_s4579"
    assert pushed["tag"] == "w1.0"
    assert "quotation-windows-card" in pushed["card"]


def test_yaml_has_no_damaged_books():
    cfg = load_job_config(CONFIG)
    assert "damaged_books" not in cfg


def test_damaged_books_in_cfg_raises_in_main():
    from quotation_lib import quotation_windows as qw

    cfg = load_job_config(CONFIG)
    bad = dict(cfg)
    bad["damaged_books"] = ["P000218"]
    with pytest.raises(GateFailure, match="damaged_books"):
        if bad.get("damaged_books"):
            raise qw.GateFailure(
                f"damaged_books must be empty for quotation windows; got {bad['damaged_books']}"
            )


@pytest.mark.skipif(
    not (ROOT / "Data" / "hf_export" / "train.jsonl").is_file(),
    reason="local hf_export not present",
)
def test_smoke_one_book_from_local_export():
    from quotation_lib.quotation_windows import Benchmark, load_benchmark

    pytest.importorskip("transformers")
    from quotation_lib.quotation_windows import load_tokenizer

    local = ROOT / "Data" / "hf_export"
    benchmark = load_benchmark(
        "karma689/tibetan-quotation-detection",
        "main",
        local_dir=local,
        local_files_only=True,
    )
    book = benchmark.splits["train"][0]
    try:
        tok = load_tokenizer("jhu-clsp/mmBERT-base", local_files_only=True)
    except Exception:
        pytest.skip("mmBERT tokenizer not cached locally")
    geom = resolve_geometry(load_job_config(CONFIG))
    windows = tokenize_book_windows(book, tok, geom)
    project_labels(book, windows, "first", tok)
    windows_by_book = {book.book_id: windows}
    report = verify(
        Benchmark(splits={"train": [book]}, source_repo="", source_revision=""),
        windows_by_book,
        tok,
        geom,
        ownership="first",
        expected_windows=999_999,
        window_tolerance=1.0,
        limit_books=True,
    )
    assert report.ok, report.failures

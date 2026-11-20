import pytest
import json
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

from types import SimpleNamespace

from quotation_lib.gemini_benchmark import (
    make_units,
    fill_anchor_prompt,
    wait_for_pacing,
    update_ledger,
    get_sha1,
    cmd_locate,
    evaluate_predictions,
    expand_split_units,
    classify_gemini_run_error,
    next_server_wait_seconds,
    unit_failure_action,
    write_retry_list,
    count_gemini_tokens,
    COUNT_TOKEN_RETRY_WAIT_SECONDS,
    is_cached_complete,
    normalize_finish_reason,
    claude_batch_snapshot,
    claude_batch_is_ready,
    split_unit_in_half,
)

# --------------------------------------------------------------------------- #
# 1. Unit Cutter Tests
# --------------------------------------------------------------------------- #

def test_make_units_short():
    # Short text <= 250k chars -> 1 piece
    text = "Short Tibetan text །" * 100
    units = make_units(text, "mock_book_1")
    assert len(units) == 1
    assert units[0]["start"] == 0
    assert units[0]["end"] == len(text)
    assert units[0]["piece"] == 0

def test_make_units_long():
    # Long text (751k characters) -> 4 near-equal pieces
    # We construct a synthetic long text with shads every 100 characters
    block = "Tibetan text shad ། " + ("a" * 80) + " ། "
    text = block * 7515  # total length around 751k
    units = make_units(text, "mock_book_2")
    assert len(units) == 4
    
    # Check overlap is >= 2000
    for i in range(len(units) - 1):
        overlap = units[i]["end"] - units[i+1]["start"]
        assert overlap >= 2000
        
    # Check pieces are near-equal
    lengths = [u["end"] - u["start"] for u in units]
    mean_len = sum(lengths) / 4
    for length in lengths:
        assert abs(length - mean_len) < 500  # very well balanced

# --------------------------------------------------------------------------- #
# 2. Fill Prompt Test
# --------------------------------------------------------------------------- #

def test_fill_prompt():
    template = "You are given a chunk of running text. Find every QUOTE.\nText:"
    unit_text = "བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་པ།"
    filled = fill_anchor_prompt(template, unit_text)
    # Check there is no second "Text:" and ends with the text
    assert filled.count("Text:") == 1
    assert filled.endswith(unit_text)
    assert filled == f"{template}\n{unit_text}"

# --------------------------------------------------------------------------- #
# 3. Pacing and Ledger Tests
# --------------------------------------------------------------------------- #

def test_wait_for_pacing_tpm():
    pacing_log = []
    # Test TPM limit
    this_tokens = 50000
    rpm_limit = 5
    tpm_limit = 100000
    
    # Pre-fill pacing log with 80k tokens in-flight (so this request would exceed)
    now = time.time()
    pacing_log.append((now - 10, 40000))
    pacing_log.append((now - 5, 40000))
    
    with patch("time.sleep") as mock_sleep:
        # It should hit sleep because total (80k) + this (50k) + 30k (reserve) = 160k > 100k
        # Wait, inside wait_for_pacing, sleep is called to let entries age out
        def side_effect(sec):
            pacing_log.pop(0)  # simulate time passing by removing the oldest entry
        mock_sleep.side_effect = side_effect
        
        wait_for_pacing(pacing_log, this_tokens, rpm_limit, tpm_limit)
        assert len(mock_sleep.mock_calls) >= 1

def test_wait_for_pacing_rpm():
    pacing_log = []
    # Test RPM limit
    this_tokens = 1000
    rpm_limit = 2
    tpm_limit = 100000
    
    now = time.time()
    pacing_log.append((now - 10, 1000))
    pacing_log.append((now - 5, 1000))
    
    with patch("time.sleep") as mock_sleep:
        def side_effect(sec):
            pacing_log.pop(0)
        mock_sleep.side_effect = side_effect
        
        wait_for_pacing(pacing_log, this_tokens, rpm_limit, tpm_limit)
        assert len(mock_sleep.mock_calls) >= 1

def test_update_ledger_success(tmp_path):
    ledger_file = tmp_path / "ledger.json"
    p_date = "2026-09-11"
    # Success increment
    update_ledger(ledger_file, p_date, max_rpd=10)
    with ledger_file.open() as f:
        data = json.load(f)
    assert data[p_date] == 1

def test_update_ledger_limit(tmp_path):
    ledger_file = tmp_path / "ledger.json"
    p_date = "2026-09-11"
    # Pre-fill with limit (90% of 10 is 9)
    with ledger_file.open("w") as f:
        json.dump({p_date: 9}, f)
        
    with pytest.raises(SystemExit):
        update_ledger(ledger_file, p_date, max_rpd=10)

# --------------------------------------------------------------------------- #
# 4. PerDay 429 Dispatch Test
# --------------------------------------------------------------------------- #

class _FakeAPIError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


def test_classify_gemini_run_error_kinds():
    overloaded = _FakeAPIError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}",
        code=503,
    )
    assert classify_gemini_run_error(overloaded) == "server"
    assert classify_gemini_run_error(_FakeAPIError("429 RESOURCE_EXHAUSTED PerDay quota", code=429)) == "daily"
    assert classify_gemini_run_error(_FakeAPIError("429 RESOURCE_EXHAUSTED PerMinute quota", code=429)) == "rate"
    assert classify_gemini_run_error(_FakeAPIError("400 INVALID_ARGUMENT", code=400)) == "bad_request"


def test_unit_failure_action_defers_503_instead_of_raising():
    assert unit_failure_action("server", 4, second_pass=False) == "wait_server"
    assert unit_failure_action("server", 5, second_pass=False) == "defer"
    assert unit_failure_action("server", 5, second_pass=True) == "give_up"
    assert unit_failure_action("daily", 1, second_pass=False) == "stop_daily"
    assert unit_failure_action("bad_request", 1, second_pass=False) == "stop_bad_request"
    assert unit_failure_action("rate", 8, second_pass=False) == "wait_rate"


def test_next_server_wait_seconds_backoff():
    assert next_server_wait_seconds(1, jitter=0.0) == 30.0
    assert next_server_wait_seconds(2, jitter=0.0) == 60.0
    assert next_server_wait_seconds(3, jitter=0.0) == 120.0
    assert next_server_wait_seconds(4, jitter=0.0) == 240.0
    assert next_server_wait_seconds(5, jitter=0.0) == 240.0


def test_write_retry_list_strips_pass_marker(tmp_path):
    path = tmp_path / "retry_validation.jsonl"
    write_retry_list(path, [{"book_id": "I6380CEC4", "piece": 0, "start": 0, "end": 10, "_retry_pass": True}])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["book_id"] == "I6380CEC4"
    assert "_retry_pass" not in saved


def test_count_gemini_tokens_retries_timeout_after_a_minute():
    timeout = Exception("[Errno 60] Operation timed out")
    ok = MagicMock(total_tokens=16143)
    client = MagicMock()
    client.models.count_tokens.side_effect = [timeout, timeout, ok]

    with patch("quotation_lib.gemini_benchmark.time.sleep") as mock_sleep:
        tokens = count_gemini_tokens(client, "gemini-3.1-flash-lite", "prompt")

    assert tokens == 16143
    assert client.models.count_tokens.call_count == 3
    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        assert call.args[0] == COUNT_TOKEN_RETRY_WAIT_SECONDS


def test_count_gemini_tokens_raises_bad_request_without_retry():
    client = MagicMock()
    client.models.count_tokens.side_effect = _FakeAPIError("400 INVALID_ARGUMENT", code=400)
    with patch("quotation_lib.gemini_benchmark.time.sleep") as mock_sleep:
        with pytest.raises(_FakeAPIError):
            count_gemini_tokens(client, "gemini-3.1-flash-lite", "prompt")
    mock_sleep.assert_not_called()
    assert client.models.count_tokens.call_count == 1


def test_count_gemini_tokens_stops_on_daily_quota():
    client = MagicMock()
    client.models.count_tokens.side_effect = _FakeAPIError(
        "429 RESOURCE_EXHAUSTED PerDay quota", code=429
    )
    with patch("quotation_lib.gemini_benchmark.time.sleep") as mock_sleep:
        with pytest.raises(SystemExit):
            count_gemini_tokens(client, "gemini-3.1-flash-lite", "prompt")
    mock_sleep.assert_not_called()
    assert client.models.count_tokens.call_count == 1

# --------------------------------------------------------------------------- #
# 5. Overlap Join and Locate Tests
# --------------------------------------------------------------------------- #

def test_overlap_join(tmp_path):
    # Setup mock data directory structure
    exp_dir = tmp_path / "results" / "zeroshot"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock book list and split config
    split_data = {"book1": "test"}
    split_file = tmp_path / "Data" / "audit" / "quotation_split.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w") as f:
        json.dump(split_data, f)
        
    # Mock gold export test.jsonl
    book_text = "བཅོམ་ལྡན་འདས་ཀྱིས་ཆོས་ཐམས་ཅད་རྫོགས་པར་གསུངས་སོ། །" * 10
    gold_data = {
        "book_id": "book1",
        "text": book_text,
        "spans": [{"start": 10, "end": 40, "cls": "TEXT"}]
    }
    export_file = tmp_path / "Data" / "hf_export" / "test.jsonl"
    export_file.parent.mkdir(parents=True, exist_ok=True)
    with export_file.open("w") as f:
        f.write(json.dumps(gold_data) + "\n")
        
    # Mock repos base text
    repo_file = tmp_path / "Data" / "repos" / "book1" / "book1.opf" / "base" / "v001.txt"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text(book_text, encoding="utf-8")
    
    # Mock units_test.jsonl
    units_file = exp_dir / "units_test.jsonl"
    with units_file.open("w") as f:
        json.dump({
            "book_id": "book1",
            "piece": 0,
            "start": 0,
            "end": len(book_text),
            "sha1": get_sha1(book_text)
        }, f)
        f.write("\n")
        
    # Mock cached raw response
    # It has valid JSON spans
    raw_response = {
        "finish_reason": "STOP",
        "raw_text": '{"spans": [{"label": "QUOTE", "head": "བཅོམ་ལྡན་འདས་ཀྱིས་ཆོས་", "tail": "རྫོགས་པར་གསུངས་སོ། །"}]}'
    }
    raw_cache_file = exp_dir / "gemini-3.8-flash" / "raw" / "book1" / "0.json"
    raw_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_cache_file.open("w") as f:
        json.dump(raw_response, f)
        
    # Patch all globals to point to tmp_path
    with patch("quotation_lib.gemini_benchmark.EXP_DIR", exp_dir), \
         patch("quotation_lib.gemini_benchmark.ROOT", tmp_path):
        
        # Test cmd_locate
        args = MagicMock()
        args.model = "gemini-3.8-flash"
        args.split = "test"
        args.source = "units"
        
        cmd_locate(args)
        
        # Verify spans.jsonl is created
        spans_file = exp_dir / "gemini-3.8-flash" / "spans.jsonl"
        assert spans_file.is_file()
        
        with spans_file.open() as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["book_id"] == "book1"


def _write_locate_fixture(tmp_path, book_text, raw_by_piece, *, source="units"):
    exp_dir = tmp_path / "results" / "zeroshot"
    exp_dir.mkdir(parents=True, exist_ok=True)

    split_file = tmp_path / "Data" / "audit" / "quotation_split.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps({"book1": "test"}), encoding="utf-8")

    export_file = tmp_path / "Data" / "hf_export" / "test.jsonl"
    export_file.parent.mkdir(parents=True, exist_ok=True)
    export_file.write_text(
        json.dumps({
            "book_id": "book1",
            "text": book_text,
            "spans": [{"start": 10, "end": 40, "cls": "TEXT"}],
        })
        + "\n",
        encoding="utf-8",
    )

    repo_file = tmp_path / "Data" / "repos" / "book1" / "book1.opf" / "base" / "v001.txt"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text(book_text, encoding="utf-8")

    units_file = exp_dir / f"{source}_test.jsonl"
    with units_file.open("w") as f:
        json.dump({
            "book_id": "book1",
            "piece": 0,
            "start": 0,
            "end": len(book_text),
            "sha1": get_sha1(book_text),
        }, f)
        f.write("\n")

    cache_sub = "raw" if source == "units" else "raw_windows"
    for piece, raw in raw_by_piece.items():
        raw_cache_file = exp_dir / "gemini-3.5-flash-lite" / cache_sub / "book1" / f"{piece}.json"
        raw_cache_file.parent.mkdir(parents=True, exist_ok=True)
        raw_cache_file.write_text(
            json.dumps({"finish_reason": raw.get("finish_reason", "STOP"), "raw_text": raw["raw_text"]}),
            encoding="utf-8",
        )
    return exp_dir


def test_cmd_locate_salvages_truncated_json(tmp_path):
    book_text = "AAAA STARTQUOTE body body ENDQUOTE BBBB " * 20
    truncated = (
        '{\n  "spans": [\n'
        '    {\n      "label": "QUOTE",\n'
        '      "head": "STARTQUOTE",\n'
        '      "tail": "ENDQUOTE"\n'
        '    },\n'
        '    {\n      "label": "QUOTE",\n'
        '      "head": "UNFINISHED'
    )
    exp_dir = _write_locate_fixture(
        tmp_path,
        book_text,
        {"0": {"finish_reason": "MAX_TOKENS", "raw_text": truncated}},
    )

    with patch("quotation_lib.gemini_benchmark.EXP_DIR", exp_dir), \
         patch("quotation_lib.gemini_benchmark.ROOT", tmp_path):
        args = MagicMock()
        args.model = "gemini-3.5-flash-lite"
        args.split = "test"
        args.source = "units"
        cmd_locate(args)

    spans_file = exp_dir / "gemini-3.5-flash-lite" / "spans.jsonl"
    assert spans_file.is_file()
    lines = [json.loads(line) for line in spans_file.read_text().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["book_id"] == "book1"


def test_expand_split_units_replaces_parent_when_children_exist(tmp_path):
    block = "Tibetan text shad ། " + ("a" * 80) + " ། "
    text = block * 80
    unit = {
        "book_id": "book1",
        "piece": 6,
        "start": 0,
        "end": len(text),
        "sha1": get_sha1(text),
    }
    cache_dir = tmp_path / "book1"
    cache_dir.mkdir()
    (cache_dir / "6_0.json").write_text("{}", encoding="utf-8")
    (cache_dir / "6_1.json").write_text("{}", encoding="utf-8")

    assert expand_split_units(unit, tmp_path / "missing", text) == [unit]
    expanded = expand_split_units(unit, cache_dir, text)
    assert [item["piece"] for item in expanded] == ["6_0", "6_1"]
    half1, half2 = split_unit_in_half(unit, text)
    assert expanded[0]["start"] == half1["start"]
    assert expanded[1]["end"] == half2["end"]


def test_cmd_locate_uses_split_children_instead_of_truncated_parent(tmp_path):
    book_text = ("STARTQUOTE middle ENDQUOTE ། " + ("z" * 80)) * 80
    broken_parent = '{"spans":[{"label":"QUOTE","head":"UNFINISHED'
    child_json = json.dumps({
        "spans": [{"label": "QUOTE", "head": "STARTQUOTE", "tail": "ENDQUOTE"}]
    })
    exp_dir = _write_locate_fixture(
        tmp_path,
        book_text,
        {
            "0": {"finish_reason": "MAX_TOKENS", "raw_text": broken_parent},
            "0_0": {"finish_reason": "STOP", "raw_text": child_json},
            "0_1": {"finish_reason": "STOP", "raw_text": child_json},
        },
    )

    with patch("quotation_lib.gemini_benchmark.EXP_DIR", exp_dir), \
         patch("quotation_lib.gemini_benchmark.ROOT", tmp_path):
        args = MagicMock()
        args.model = "gemini-3.5-flash-lite"
        args.split = "test"
        args.source = "units"
        cmd_locate(args)

    spans_file = exp_dir / "gemini-3.5-flash-lite" / "spans.jsonl"
    lines = [json.loads(line) for line in spans_file.read_text().splitlines() if line]
    assert len(lines) >= 1
    assert lines[0]["book_id"] == "book1"


# --------------------------------------------------------------------------- #
# 6. Scorer and 20-span round-trip Tests
# --------------------------------------------------------------------------- #

def test_normalize_finish_reason_strips_sdk_enum_prefix():
    assert normalize_finish_reason("FinishReason.STOP") == "STOP"
    assert normalize_finish_reason("STOP") == "STOP"
    assert normalize_finish_reason("blocked") == "BLOCKED"


def test_is_cached_complete_accepts_sdk_enum_strings():
    assert is_cached_complete("FinishReason.STOP")
    assert is_cached_complete("STOP")
    assert is_cached_complete("FinishReason.MAX_TOKENS")
    assert is_cached_complete("blocked")
    assert not is_cached_complete("errored")
    assert not is_cached_complete("")
    assert not is_cached_complete(None)


def test_evaluate_predictions_round_trip():
    # Verify that evaluating predictions with whitespace trim and stitch ON/OFF is working
    gold_book = {
        "book_id": "book1",
        "text": "  བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་སོ། །  ",
        "spans": [{"start": 2, "end": 26, "seen_in_train": False}]
    }
    
    preds = [
        {"book_id": "book1", "label": "QUOTATION", "start": 0, "end": 28} # untrimmed prediction
    ]
    
    base_texts = {"book1": "  བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་སོ། །  "}
    
    score_stitch_on = evaluate_predictions(preds, [gold_book], base_texts, stitch=True)
    assert score_stitch_on["f1"] == 1.0  # it trims first, so they match perfectly!


def test_claude_batch_snapshot_uses_processing_status():
    batch = SimpleNamespace(
        id="msgbatch_1",
        processing_status="in_progress",
        created_at="2026-09-21",
        request_counts=SimpleNamespace(
            processing=29, succeeded=0, errored=0, canceled=0, expired=0
        ),
    )
    snap = claude_batch_snapshot(batch)
    assert snap["id"] == "msgbatch_1"
    assert snap["status"] == "in_progress"
    assert snap["request_counts"]["processing"] == 29


def test_claude_batch_is_ready_when_ended():
    assert claude_batch_is_ready(SimpleNamespace(processing_status="ended"))
    assert not claude_batch_is_ready(SimpleNamespace(processing_status="in_progress"))
    assert not claude_batch_is_ready(SimpleNamespace(processing_status="completed"))

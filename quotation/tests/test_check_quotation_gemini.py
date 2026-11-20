"""Empty Gemini dumps and adjacent-span stitching for scoring (no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import check_quotation_gemini  # noqa: E402
from quotation_lib.span_metrics import match_iou  # noqa: E402


class StitchTests(unittest.TestCase):
    def test_stitch_merges_two_adjacent_intervals(self):
        text = "AAAA ། BBBB"
        pred = [("QUOTATION", 0, 3), ("QUOTATION", 7, 10)]
        stitched = check_quotation_gemini.stitch_adjacent_spans(text, pred, max_gap_len=15)
        self.assertEqual(stitched, [("QUOTATION", 0, 10)])

        gold = [("QUOTATION", 0, 10)]
        self.assertEqual(len(match_iou(gold, pred, threshold=0.5)), 0)
        self.assertEqual(len(match_iou(gold, stitched, threshold=0.5)), 1)
        self.assertEqual(len(pred), 2)
        self.assertEqual(len(stitched), 1)

    def test_stitch_does_not_merge_across_real_text(self):
        text = "AAAA xx BBBB"
        pred = [("QUOTATION", 0, 3), ("QUOTATION", 7, 10)]
        stitched = check_quotation_gemini.stitch_adjacent_spans(text, pred)
        self.assertEqual(len(stitched), 2)


class EmptyDumpScoreTests(unittest.TestCase):
    def test_empty_dump_is_zero_pred_not_an_error(self):
        gold_spans = [{"label": "QUOTATION", "start": 0, "end": 4}]
        with tempfile.TemporaryDirectory() as tmp:
            quotes_path = Path(tmp) / "empty.json"
            quotes_path.write_text(json.dumps({"spans": []}), encoding="utf-8")
            with patch.object(
                check_quotation_gemini,
                "load_book_row",
                return_value=("hello world", gold_spans),
            ):
                result = check_quotation_gemini.score_quotes("P000000", quotes_path)
        self.assertNotIn("error", result)
        self.assertEqual(result["n_parsed"], 0)
        self.assertEqual(result["n_located"], 0)
        self.assertEqual(result["n_pred"], 0)
        self.assertEqual(result["n_gold"], 1)
        self.assertEqual(result["n_match"], 0)
        self.assertEqual(result["iou50_precision"], 0.0)
        self.assertEqual(result["iou50_recall"], 0.0)

    def test_empty_gold_and_pred_is_perfect(self):
        with tempfile.TemporaryDirectory() as tmp:
            quotes_path = Path(tmp) / "empty.json"
            quotes_path.write_text(json.dumps({"spans": []}), encoding="utf-8")
            with patch.object(
                check_quotation_gemini,
                "load_book_row",
                return_value=("hello", []),
            ):
                result = check_quotation_gemini.score_quotes("P000000", quotes_path)
        self.assertEqual(result["iou50_f1"], 1.0)

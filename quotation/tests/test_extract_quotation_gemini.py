"""Anchor dump identity, salvage, and skip-existing helpers (no API)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import extract_quotation_gemini  # noqa: E402


class SpanKeyAndSalvageTests(unittest.TestCase):
    def test_dedup_keeps_distinct_head_tail_with_empty_text(self):
        a = {"label": "QUOTE", "text": "", "head": "འདི་ནི་", "tail": "ཡིན་ནོ།", "frame": "x"}
        b = {"label": "QUOTE", "text": "", "head": "གཞན་ནི་", "tail": "མ་ཡིན།", "frame": "y"}
        keys = {extract_quotation_gemini._span_key(a), extract_quotation_gemini._span_key(b)}
        self.assertEqual(len(keys), 2)

    def test_same_head_tail_collide_even_if_frame_differs(self):
        a = {"text": "", "head": "HEAD", "tail": "TAIL", "frame": "one"}
        b = {"text": "", "head": "HEAD", "tail": "TAIL", "frame": "two"}
        self.assertEqual(extract_quotation_gemini._span_key(a), extract_quotation_gemini._span_key(b))

    def test_full_text_identity_still_used_for_old_prompt(self):
        span = {"label": "QUOTE", "text": "the whole quote"}
        self.assertEqual(extract_quotation_gemini._span_key(span), "text:the whole quote")

    def test_salvage_keeps_head_spans_without_text(self):
        raw = (
            '{"spans":['
            '{"label":"QUOTE","frame":"f","head":"aaa","tail":"bbb"},'
            '{"label":"QUOTE","text":"","head":"ccc","tail":"ddd"},'
            '{"label":"QUOTE","head":"eee","tail":"fff"'  # truncated
        )
        spans = extract_quotation_gemini.salvage_spans(raw)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["head"], "aaa")
        self.assertEqual(spans[1]["tail"], "ddd")


class DumpValidTests(unittest.TestCase):
    def test_dump_is_valid_requires_spans_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            path.write_text(json.dumps({"spans": []}), encoding="utf-8")
            self.assertTrue(extract_quotation_gemini.dump_is_valid(path))
            path.write_text(json.dumps({"quotes": []}), encoding="utf-8")
            self.assertFalse(extract_quotation_gemini.dump_is_valid(path))

    def test_skip_existing_returns_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "out.json"
            dump.write_text(json.dumps({"spans": [{"head": "a", "tail": "b"}]}), encoding="utf-8")
            prompt = ROOT / "prompts" / "quotation_anchor_prompt.md"
            rc = extract_quotation_gemini.main(
                ["--book", "P000069", "--prompt", str(prompt), "--out", str(dump), "--skip-existing"]
            )
            self.assertEqual(rc, 0)

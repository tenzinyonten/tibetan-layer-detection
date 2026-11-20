"""Locate Gemini-emitted quote strings in book text."""

from __future__ import annotations

import json
import unittest

from quotation_lib.gemini_locate import (
    ParsedQuote,
    locate_from_anchors,
    locate_parsed_quote,
    locate_span,
    parse_quote_dump,
    quotes_from_items,
    recover_offsets,
)
from quotation_lib.span_metrics import inclusive_iou, match_iou


def _anchor_quote(head: str, tail: str) -> ParsedQuote:
    return ParsedQuote(
        quote_id="1", label="QUOTATION", text="", start_text=head, end_text=tail
    )


class LocateSpanTests(unittest.TestCase):
    def test_exact_find_returns_inclusive_offsets(self):
        hay = "aaaབཀའ་སྩལ་པ།bbb"
        needle = "བཀའ་སྩལ་པ།"
        hit = locate_span(hay, needle)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.method, "exact")
        self.assertEqual(hay[hit.start : hit.end + 1], needle)
        self.assertEqual(hit.score, 1.0)

    def test_normalized_ignores_newlines_and_maps_shad_variant(self):
        hay = "ཆོས་རྣམས་\nརང་བཞིན་རྣམ་དག་པས༑"
        needle = "ཆོས་རྣམས་རང་བཞིན་རྣམ་དག་པས།"
        hit = locate_span(hay, needle)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.method, "normalized")
        self.assertEqual(hit.start, 0)
        self.assertEqual(hit.end, len(hay) - 1)

    def test_ellipsis_uses_start_and_end_fragments(self):
        hay = "xxxxསྡིག་པ་ཅི་ཡང་མི་བྱ་སྟེ། །MIDDLE JUNKཅུང་ཟད་བསགས་པ་ཡི།yyyy"
        needle = "སྡིག་པ་ཅི་ཡང་མི་བྱ་སྟེ། །...ཅུང་ཟད་བསགས་པ་ཡི།"
        hit = locate_span(hay, needle)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.method.startswith("anchors"))
        recovered = hay[hit.start : hit.end + 1]
        self.assertTrue(recovered.startswith("སྡིག་པ་"))
        self.assertTrue(recovered.endswith("ཅུང་ཟད་བསགས་པ་ཡི།"))

    def test_fuzzy_recovers_small_drift(self):
        hay = "prefix ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་གོད་པ་དང༌། suffix"
        needle = "ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌།"
        hit = locate_span(hay, needle, min_score=0.7)
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit.score, 0.7)
        self.assertIn("ལེ་ལོ", hay[hit.start : hit.end + 1])

    def test_missing_needle_returns_none(self):
        self.assertIsNone(locate_span("abcdef", "zzzzzz"))

    def test_recover_offsets_adds_chunk_start(self):
        chunk = "....QUOTEHERE...."
        offsets = recover_offsets(chunk, "QUOTEHERE", chunk_start=1000)
        self.assertEqual(offsets, (1004, 1012))


class AnchorAndParseTests(unittest.TestCase):
    def test_anchors_take_end_after_start(self):
        hay = "END first START body body END second"
        hit = locate_from_anchors(hay, "START", "END")
        self.assertEqual(hay[hit.start : hit.end + 1], "START body body END")

    def test_max_span_rejects_a_far_away_tail(self):
        hay = "START body" + ("x" * 500) + "END"
        self.assertIsNotNone(locate_from_anchors(hay, "START", "END"))
        self.assertIsNone(locate_from_anchors(hay, "START", "END", max_span=100))

    def test_normalized_drops_tsheg_before_shad(self):
        hay = "xxཚོགས་བཞིན་འཁོར། །yy"
        hit = locate_span(hay, "ཚོགས་བཞིན་འཁོར་། །")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.method, "normalized")
        self.assertEqual(hay[hit.start : hit.end + 1], "ཚོགས་བཞིན་འཁོར། །")

    def test_parse_head_tail_anchors(self):
        raw = json.dumps(
            {
                "spans": [
                    {"label": "QUOTE", "head": "ལེ་ལོ་དང་ནི་", "tail": "ལྔར་འདོད་དོ།"},
                    {"label": "QUOTE", "head": "ཆོས་རྣམས།", "tail": ""},
                ]
            }
        )
        quotes = parse_quote_dump(raw)
        self.assertEqual(quotes[0].start_text, "ལེ་ལོ་དང་ནི་")
        self.assertEqual(quotes[0].end_text, "ལྔར་འདོད་དོ།")
        # an anchor head is not the span, so it must not be used as full text
        self.assertEqual(quotes[0].text, "")
        # an empty tail means the head holds the whole span
        self.assertEqual(quotes[1].text, "ཆོས་རྣམས།")

    def test_from_index_skips_an_earlier_lookalike(self):
        hay = "HEAD one TAIL ... HEAD two TAIL"
        first = locate_parsed_quote(hay, _anchor_quote("HEAD", "TAIL"))
        self.assertEqual(first.start, 0)
        second = locate_parsed_quote(
            hay, _anchor_quote("HEAD", "TAIL"), from_index=first.end + 1
        )
        self.assertEqual(hay[second.start : second.end + 1], "HEAD two TAIL")

    def test_parse_human_dump_and_json(self):
        dump = (
            "[Quote 24] Frame: མདོ་ལས། Starts at:\n"
            "AAA\nEnds at:\nZZZ\nCloser: ཞེས་གསུངས།\n\n"
            "[Quote 29] Frame: ལས། Text: ཆོས་རྣམས། Closer: ཅེས་གསུངས།\n"
        )
        quotes = parse_quote_dump(dump)
        self.assertEqual(len(quotes), 2)
        self.assertEqual(quotes[0].quote_id, "24")
        self.assertEqual(quotes[0].start_text.strip(), "AAA")
        self.assertEqual(quotes[0].end_text.strip(), "ZZZ")
        self.assertEqual(quotes[1].text, "ཆོས་རྣམས།")

        as_json = json.dumps({"spans": [{"label": "QUOTE", "text": "ཆོས་རྣམས།"}]})
        json_quotes = parse_quote_dump(as_json)
        self.assertEqual(json_quotes[0].label, "QUOTATION")
        self.assertEqual(json_quotes[0].text, "ཆོས་རྣམས།")

    def test_quotes_from_items_keeps_head_tail_anchors(self):
        quotes = quotes_from_items(
            [{"label": "QUOTE", "head": "AAA", "tail": "ZZZ"}]
        )
        self.assertEqual(quotes[0].start_text, "AAA")
        self.assertEqual(quotes[0].end_text, "ZZZ")
        self.assertEqual(quotes[0].text, "")

    def test_locate_parsed_quote_prefers_full_text(self):
        hay = "frame XXX quoted words YYY closer"
        from quotation_lib.gemini_locate import ParsedQuote

        quote = ParsedQuote(
            quote_id="1",
            label="QUOTATION",
            text="quoted words",
            start_text="XXX",
            end_text="YYY",
        )
        hit = locate_parsed_quote(hay, quote)
        self.assertEqual(hay[hit.start : hit.end + 1], "quoted words")


class ScoreAgainstGoldTests(unittest.TestCase):
    def test_iou_match_at_half(self):
        gold = [("QUOTATION", 10, 20)]
        pred = [("QUOTATION", 12, 22)]
        matches = match_iou(gold, pred, threshold=0.5)
        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(matches[0][2], 0.5)
        self.assertLess(inclusive_iou(gold[0], ("QUOTATION", 100, 110)), 0.5)


if __name__ == "__main__":
    unittest.main()

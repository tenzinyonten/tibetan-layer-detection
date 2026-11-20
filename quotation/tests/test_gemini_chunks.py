"""Tests for Tibetan text chunking and gold span validation."""

from __future__ import annotations

import unittest
from quotation_lib.gemini_chunks import (
    snap,
    make_chunks,
    compute_span_lengths,
    compute_percentiles,
    verify_chunks_identity,
    verify_chunks_coverage,
    verify_span_coverage_guarantee,
    verify_round_trip,
    SHAD,
)


class GeminiChunksTests(unittest.TestCase):
    def test_snap_finds_nearest_shad_and_cuts_after_it(self):
        # Index:  0123456789012345
        text = "abc།def།gh།ijk"
        # Nearest shad to 5 is at 3. Cut after it is 4.
        self.assertEqual(snap(text, 5, radius=3), 4)
        # Nearest shad to 9 is at 7 (cut after is 8). Nearest shad to 9 is also at 10 (cut after is 11).
        # Distance of cut 8 to 9 is 1 (abs(8 - 9)). Distance of cut 11 to 9 is 2 (abs(11 - 9)).
        # So it should pick 8.
        self.assertEqual(snap(text, 9, radius=3), 8)
        # Out of radius
        self.assertEqual(snap(text, 14, radius=2), 14)

    def test_make_chunks_never_stalls_and_splits_properly(self):
        text = "a།b།c།d།e"
        # W = 4, S = 2.
        # pos = 0. target = 4. min(4, 9) = 4.
        # radius=100. lo=0, hi=9.
        # text: "a(1)།(2)b(3)།(4)c(5)།(6)d(7)།(8)e(9)"
        # indices:
        # 0: a
        # 1: །
        # 2: b
        # 3: །
        # 4: c
        # 5: །
        # 6: d
        # 7: །
        # 8: e
        # Let's verify that make_chunks completes and does not stall.
        chunks = make_chunks(text, "book1", W=4, S=2)
        self.assertGreater(len(chunks), 0)
        # Ensure never stall works even if text is long and has no shads.
        text_no_shads = "x" * 100
        chunks_no_shads = make_chunks(text_no_shads, "book2", W=10, S=5)
        self.assertEqual(len(chunks_no_shads), 19)  # 0 to 100 with step 5
        self.assertEqual(chunks_no_shads[0]["start"], 0)
        self.assertEqual(chunks_no_shads[0]["end"], 10)

    def test_compute_span_lengths_and_percentiles(self):
        spans = [
            {"start": 10, "end": 19, "label": "QUOTATION"},  # len = 10
            {"start": 100, "end": 119, "label": "QUOTATION"},  # len = 20
            {"start": 50, "end": 54, "label": "TSAWA"},  # len = 5
        ]
        lengths = compute_span_lengths(spans, ["QUOTATION", "TSAWA"])
        self.assertEqual(lengths["QUOTATION"], [10, 20])
        self.assertEqual(lengths["TSAWA"], [5])

        p_quote = compute_percentiles(lengths["QUOTATION"])
        self.assertAlmostEqual(p_quote["p50"], 15.0)
        self.assertAlmostEqual(p_quote["p95"], 19.5)
        self.assertEqual(p_quote["max"], 20)

    def test_verifications_identity_and_coverage(self):
        text = "abcdefghij"
        chunks = [
            {"start": 0, "end": 6, "text": "abcdef"},
            {"start": 4, "end": 10, "text": "efghij"},
        ]
        self.assertTrue(verify_chunks_identity(text, chunks))
        self.assertTrue(verify_chunks_coverage(len(text), chunks))

        # Break identity
        chunks_bad_id = [
            {"start": 0, "end": 6, "text": "abcdeX"},
            {"start": 4, "end": 10, "text": "efghij"},
        ]
        self.assertFalse(verify_chunks_identity(text, chunks_bad_id))

        # Break coverage
        chunks_bad_cov = [
            {"start": 0, "end": 3, "text": "abc"},
            {"start": 5, "end": 10, "text": "fghij"},
        ]
        self.assertFalse(verify_chunks_coverage(len(text), chunks_bad_cov))

        # Non-zero start
        chunks_bad_start = [
            {"start": 1, "end": 10, "text": "bcdefghij"},
        ]
        self.assertFalse(verify_chunks_coverage(len(text), chunks_bad_start))

    def test_verify_span_coverage_guarantee(self):
        spans = [
            {"start": 2, "end": 5, "label": "QUOTATION"},  # len = 4, lies within [0, 6) and [4, 10) -> covered by [0, 6)
            {"start": 5, "end": 8, "label": "QUOTATION"},  # len = 4, lies within [4, 10) -> covered
            {"start": 1, "end": 7, "label": "QUOTATION"},  # len = 7, cannot lie in any chunk of size 6.
        ]
        chunks = [
            {"start": 0, "end": 6, "text": "abcdef"},
            {"start": 4, "end": 10, "text": "efghij"},
        ]
        # W = 6, S = 2. W - S = 4.
        # Target layers are QUOTATION.
        # Spans with len <= 4: first two.
        # First two are covered by chunks. The third has len 7 > W-S, so it is ignored.
        passed, uncovered = verify_span_coverage_guarantee(spans, chunks, W=6, S=2, layers=["QUOTATION"])
        self.assertTrue(passed)
        self.assertEqual(len(uncovered), 0)

        # Let's add a short span that is split
        spans_bad = spans + [
            {"start": 3, "end": 6, "label": "QUOTATION"}  # len = 4. span text covers indices 3, 4, 5, 6.
            # In [0, 6), range is [0, 5], so index 6 is missing.
            # In [4, 10), range is [4, 9], so index 3 is missing.
            # This short span is split across chunks!
        ]
        passed_bad, uncovered_bad = verify_span_coverage_guarantee(spans_bad, chunks, W=6, S=2, layers=["QUOTATION"])
        self.assertFalse(passed_bad)
        self.assertEqual(len(uncovered_bad), 1)
        self.assertEqual(uncovered_bad[0]["start"], 3)

    def test_verify_round_trip(self):
        text = "abcdefghij"
        spans = [
            {"start": 2, "end": 5, "label": "QUOTATION"},  # cdef
            {"start": 5, "end": 8, "label": "QUOTATION"},  # fghi
        ]
        chunks = [
            {"start": 0, "end": 6, "text": "abcdef"},
            {"start": 4, "end": 10, "text": "efghij"},
        ]
        self.assertTrue(verify_round_trip(spans, chunks, text, layers=["QUOTATION"]))

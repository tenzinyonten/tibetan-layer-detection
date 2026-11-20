from __future__ import annotations

import unittest
from pathlib import Path
from quotation_lib.quotation_benchmark import (
    Span, Book, Gold, merge_continued_spans, check_structural_leakage,
    normalize_string, get_book_density_bucket_idx, compute_split_cost,
    run_single_split, evaluate_assignment, audit_leakage
)


class TestQuotationBenchmark(unittest.TestCase):
    def test_merge_continued_spans_with_and_without_guard(self):
        text = (
            "ཆེན་པོ་ཡིན་པར་ཤེས་པར་བྱ།། ཞེས་སོ།། "  # Span 1 (ends on closer)
            "ཡང་བཤེས་གཉེན་དུལ་བ་ཞི་བ་... ཡང་། "      # Span 2 (gap to s3 contains letters)
            "བྱང་ཆུབ་སེམས་དང་སྡོམ་པ་དང་།།\n"        # Span 3 (no closer at end)
            "བགྱིད་དང་བགྱིད་དུ་སྩལ་བ་དང་།།"         # Span 4 (continuation)
        )
        s1 = Span("ann1", 0, 33, "MISC", True)
        s2 = Span("ann2", 35, 61, "PERSON", False)
        s3 = Span("ann3", 68, 95, "TEXT", False)
        s4 = Span("ann4", 97, 125, "MISC", True)
        
        book = Book(
            layer="Citation",
            batch="new",
            verdict="accept",
            title="Mock Book",
            doc_chars=len(text),
            closed_pct=50.0,
            misaligned_pct=0.0,
            both_mid_pct=0.0,
            line_split_pct=50.0,
            n_original=4,
            spans=[s1, s2, s3, s4],
            text=text
        )
        
        gold = Gold(
            generated="2026-09-10",
            policy="test",
            kept_books=1,
            kept_spans=4,
            dropped_spans=0,
            snapped_spans=0,
            class_counts={"MISC": 2, "PERSON": 1, "TEXT": 1},
            books={"B01": book}
        )
        
        gold_merged = merge_continued_spans(gold)
        
        self.assertEqual(len(gold_merged.books["B01"].spans), 3)
        
        merged_spans = gold_merged.books["B01"].spans
        self.assertEqual(merged_spans[0].ann, "ann1")
        self.assertEqual(merged_spans[0].end, 33)
        self.assertEqual(merged_spans[1].ann, "ann2")
        self.assertEqual(merged_spans[1].start, 35)
        
        self.assertEqual(merged_spans[2].start, 68)
        self.assertEqual(merged_spans[2].end, 125)
        self.assertEqual(merged_spans[2].cls, "TEXT")
        self.assertTrue(merged_spans[2].closed)
        self.assertEqual(merged_spans[2].merged_from, 2)

    def test_normalize_string(self):
        raw = "བྱང་ཆུབ་ སེམས་དང་། ཞེས་སོ།།"
        self.assertEqual(normalize_string(raw), "བྱངཆུབསེམསདངཞེསསོ")

    def test_structural_leakage_checks(self):
        split = {"B01": "train", "B02": "val", "B03": "test"}
        gold = Gold(
            generated="", policy="", kept_books=3, kept_spans=0, dropped_spans=0, snapped_spans=0,
            class_counts={}, books={
                "B01": Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 0, []),
                "B02": Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 0, []),
                "B03": Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 0, []),
            }
        )
        
        check_structural_leakage(gold, split)
        
        # Test missing a book in assignment should raise AssertionError
        bad_split = {"B01": "train", "B03": "test"}
        with self.assertRaises(AssertionError):
            check_structural_leakage(gold, bad_split)

    def test_seen_in_train_and_leakage_audit(self):
        text_train = "བྱང་ཆུབ་སེམས་དཔའ་ཆེན་པོ་ཉིད་"
        text_val = "བྱང་ཆུབ་སེམས་དཔའ་ཆེན་པོ་ཉིད་"
        text_test = "གཞན་ཞིག་གོ།"
        text_test_short = "བྱང་ཆུབ་"
        
        b_train = Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 1, [Span("a1", 0, len(text_train)-1, "TEXT", False)], text=text_train)
        b_val = Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 1, [Span("a2", 0, len(text_val)-1, "TEXT", False)], text=text_val)
        b_test = Book("Citation", "new", "accept", "", 100, 0, 0, 0, 0, 2, [
            Span("a3", 0, len(text_test)-1, "PERSON", False),
            Span("a4", 0, len(text_test_short)-1, "TEXT", False)
        ], text=text_test + text_test_short)
        
        gold = Gold(
            generated="", policy="", kept_books=3, kept_spans=4, dropped_spans=0, snapped_spans=0,
            class_counts={}, books={"B01": b_train, "B02": b_val, "B03": b_test}
        )
        split = {"B01": "train", "B02": "val", "B03": "test"}
        
        leakage = audit_leakage(gold, split)
        
        self.assertTrue(b_val.spans[0].seen_in_train)
        self.assertFalse(b_test.spans[0].seen_in_train)
        self.assertFalse(b_test.spans[1].seen_in_train)
        self.assertEqual(leakage["seen_in_train_count"], 1)
        self.assertEqual(leakage["total_held_out_spans"], 3)

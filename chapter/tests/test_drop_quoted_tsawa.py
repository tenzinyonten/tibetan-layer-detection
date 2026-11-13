"""Unit tests for Stage 3 drop_quoted_tsawa (predicted quotation, not gold)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tibetan_layer_detection.infer import layers_for_run  # noqa: E402
from tibetan_layer_detection.postprocess import drop_quoted_tsawa  # noqa: E402


def span(start, end, confidence=1.0):
    return {"start": start, "end": end, "confidence": confidence}


class TestDropQuotedTsawa(unittest.TestCase):
    def test_drops_when_half_covered(self):
        layers = {
            "tsawa": [span(0, 10), span(20, 30)],
            "quotation": [span(0, 8)],
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 1)
        self.assertEqual([(s["start"], s["end"]) for s in out["tsawa"]], [(20, 30)])
        self.assertEqual(len(out["quotation"]), 1)

    def test_keeps_partial_overlap_under_threshold(self):
        layers = {
            "tsawa": [span(0, 10)],
            "quotation": [span(0, 4)],  # 40%
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(len(out["tsawa"]), 1)

    def test_noop_without_quotation_layer(self):
        layers = {"tsawa": [span(0, 10)]}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertIs(out, layers)

    def test_noop_when_quotation_model_found_nothing(self):
        layers = {"tsawa": [span(0, 10)], "quotation": []}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(out["tsawa"], layers["tsawa"])

    def test_ignores_a_gold_key_if_someone_passed_one(self):
        layers = {
            "tsawa": [span(0, 10)],
            "quotation": [],
            "gold_quotation": [span(0, 10)],
        }
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertEqual(len(out["tsawa"]), 1)

    def test_noop_without_tsawa(self):
        layers = {"quotation": [span(0, 10)]}
        out, n = drop_quoted_tsawa(layers)
        self.assertEqual(n, 0)
        self.assertIs(out, layers)


class TestLayersForRun(unittest.TestCase):
    def test_tsawa_only_also_runs_quotation_model(self):
        self.assertEqual(layers_for_run(["tsawa"]), ["tsawa", "quotation"])

    def test_skip_quotation(self):
        self.assertEqual(layers_for_run(["tsawa"], skip_quotation=True), ["tsawa"])

    def test_sabche_only_unchanged(self):
        self.assertEqual(layers_for_run(["sabche"]), ["sabche"])


if __name__ == "__main__":
    unittest.main()

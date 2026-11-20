"""In-loop IoU@0.5, token recall, ASRF, and MUC helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from quotation_lib.training_utils import LABEL2ID, LABELS
from quotation_lib.span_metrics import (
    asrf_decode,
    compute_eval_metrics,
    flatten_window_metrics,
    iou50_from_sequences,
    muc_taxonomy,
    token_recall_inside_gold,
    viterbi_decode_torch,
)
from quotation_lib.loss_objective import N_BIO, N_LAYERS


def _confident_logits(label_ids: np.ndarray, magnitude: float = 20.0) -> np.ndarray:
    seq = label_ids.shape[0]
    logits = np.zeros((1, seq, N_BIO), dtype=np.float32)
    for index, label in enumerate(label_ids):
        logits[0, index, int(label)] = magnitude
    return logits


class IouAndRecallTests(unittest.TestCase):
    def test_perfect_viterbi_scores_one(self):
        seq = 16
        gold = np.zeros((1, seq), dtype=np.int64)
        gold[0, 4] = LABEL2ID["B-QUOTE"]
        gold[0, 5:10] = LABEL2ID["I-QUOTE"]
        logits = _confident_logits(gold[0])
        metrics = compute_eval_metrics(logits, gold, break_penalty=2.0)
        self.assertGreater(metrics["iou50_f1"], 0.99)
        self.assertGreater(metrics["token_recall_inside_gold"], 0.99)

    def test_token_recall_ignores_o(self):
        gold = [["O", "B-QUOTE", "I-QUOTE", "O"]]
        pred = [["O", "B-QUOTE", "O", "B-SABCHE"]]
        metrics = token_recall_inside_gold(pred, gold)
        self.assertAlmostEqual(metrics["token_recall_inside_gold"], 0.5)

    def test_iou_zero_when_layers_differ(self):
        gold = [["B-QUOTE", "I-QUOTE"]]
        pred = [["B-SABCHE", "I-SABCHE"]]
        metrics = iou50_from_sequences(pred, gold)
        self.assertEqual(metrics["iou50_f1"], 0.0)


class ViterbiConstraintTests(unittest.TestCase):
    def test_illegal_i_is_not_emitted(self):
        seq = 8
        logits = torch.zeros(1, seq, N_BIO)
        # Strong I-QUOTE everywhere, including token 0 where I is illegal.
        logits[0, :, LABEL2ID["I-QUOTE"]] = 10.0
        logits[0, 0, LABEL2ID["O"]] = 0.0
        path = viterbi_decode_torch(logits, break_penalty=2.0)[0].tolist()
        labels = [LABELS[index] for index in path]
        for previous, current in zip(labels, labels[1:]):
            if current.startswith("I-"):
                tag = current[2:]
                self.assertIn(previous, {f"B-{tag}", f"I-{tag}"})


class AsrfAndMucTests(unittest.TestCase):
    def test_asrf_cuts_at_start_end_peaks(self):
        seq = 20
        bio = np.zeros((seq, N_BIO), dtype=np.float32)
        bio[:, LABEL2ID["I-QUOTE"]] = 5.0
        bio[:3, LABEL2ID["O"]] = 8.0
        bio[12:, LABEL2ID["O"]] = 8.0
        start = np.full((seq, N_LAYERS), -5.0, dtype=np.float32)
        end = np.full((seq, N_LAYERS), -5.0, dtype=np.float32)
        start[3, 0] = 5.0
        end[11, 0] = 5.0
        path = asrf_decode(bio, start, end)
        self.assertEqual(int(path[3]), LABEL2ID["B-QUOTE"])
        self.assertTrue(all(int(path[i]) == LABEL2ID["I-QUOTE"] for i in range(4, 11)))
        self.assertEqual(int(path[0]), LABEL2ID["O"])

    def test_muc_counts_exact_and_partial(self):
        gold = [("QUOTE", 0, 9), ("SABCHE", 20, 24)]
        pred = [("QUOTE", 0, 9), ("QUOTE", 18, 22)]
        counts = muc_taxonomy(gold, pred)
        self.assertEqual(counts["COR"], 1)
        self.assertEqual(counts["INC"], 1)
        self.assertEqual(counts["MIS"], 0)
        self.assertEqual(counts["SPU"], 0)

    def test_flatten_reports_quote_overproposal(self):
        gold = np.zeros((1, 8), dtype=np.int64)
        gold[0, 1] = LABEL2ID["B-QUOTE"]
        gold[0, 2] = LABEL2ID["I-QUOTE"]
        pred = gold.copy()
        pred[0, 5] = LABEL2ID["B-QUOTE"]
        pred[0, 6] = LABEL2ID["I-QUOTE"]
        metrics = flatten_window_metrics(pred, gold)
        self.assertAlmostEqual(metrics["quote_overproposal"], 2.0)


if __name__ == "__main__":
    unittest.main()

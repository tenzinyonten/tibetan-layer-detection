"""Behaviour tests for the Training objective."""

from __future__ import annotations

import unittest

import torch

from quotation_lib.training_utils import LABEL2ID
from quotation_lib.loss_objective import (
    BOOKTITLE_INDEX,
    N_BIO,
    N_LAYERS,
    EerPrior,
    V2LossWeights,
    boundary_loss,
    compute_loss_objective,
    eer_loss,
    lambda_at_step,
    layer_probs_bct,
    predicted_density,
    span_tversky_loss,
)


def _empty_spans(batch: int = 1):
    return [[] for _ in range(batch)]


def _logits(batch: int, seq: int, fill: float = 0.0) -> torch.Tensor:
    return torch.full((batch, seq, N_BIO), fill)


def _one_hot_bio(seq: int, spans: list[tuple[int, int, int]]) -> torch.Tensor:
    """Hard BIO labels: spans are ``(layer, start, end)`` half-open."""
    labels = torch.zeros(seq, dtype=torch.long)
    for layer, start, end in spans:
        if end <= start:
            continue
        b_id = 1 + 2 * layer
        i_id = 2 + 2 * layer
        labels[start] = b_id
        if end - start > 1:
            labels[start + 1 : end] = i_id
    return labels


def _confident_logits(labels: torch.Tensor, magnitude: float = 50.0) -> torch.Tensor:
    seq = labels.size(0)
    logits = torch.zeros(1, seq, N_BIO)
    logits[0, torch.arange(seq), labels] = magnitude
    return logits


class LambdaRampTests(unittest.TestCase):
    def test_ramps_from_zero_over_2000_steps(self):
        self.assertEqual(lambda_at_step(1.0, 0, 2000), 0.0)
        self.assertAlmostEqual(lambda_at_step(1.0, 1000, 2000), 0.5)
        self.assertEqual(lambda_at_step(1.0, 2000, 2000), 1.0)
        self.assertEqual(lambda_at_step(1.0, 5000, 2000), 1.0)

    def test_no_ramp_returns_target(self):
        self.assertEqual(lambda_at_step(0.1, 1, 0), 0.1)


class PerfectPredictionTests(unittest.TestCase):
    def test_perfect_span_and_boundary_are_near_zero(self):
        seq = 40
        gold = [(0, 10, 20)]
        labels = _one_hot_bio(seq, gold).unsqueeze(0)
        logits = _confident_logits(labels[0])
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": 10,
                    "end": 20,
                    "truncated": False,
                    "span_id": "t:QUOTATION:0:0:0",
                }
            ]
        ]
        l_span = span_tversky_loss(logits, labels, spans)
        self.assertLess(l_span.item(), 1e-4)

        start_logits = torch.full((1, seq, N_LAYERS), -50.0)
        end_logits = torch.full((1, seq, N_LAYERS), -50.0)
        start_logits[0, 10, 0] = 50.0
        end_logits[0, 19, 0] = 50.0
        start_targets = torch.zeros(1, N_LAYERS, seq)
        end_targets = torch.zeros(1, N_LAYERS, seq)
        start_targets[0, 0, 10] = 1.0
        end_targets[0, 0, 19] = 1.0
        mask = torch.ones(1, seq)
        pos_weight = torch.ones(N_LAYERS)
        l_bnd = boundary_loss(
            start_logits, end_logits, start_targets, end_targets, mask, pos_weight
        )
        self.assertLess(l_bnd.item(), 1e-4)


class TverskyAlphaBetaTests(unittest.TestCase):
    def test_overshoot_penalised_less_than_undershoot(self):
        """α=0.3 < β=0.7: false positives cost less than false negatives."""
        seq = 80
        gold_start, gold_end = 10, 30
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": gold_start,
                    "end": gold_end,
                    "truncated": False,
                    "span_id": "t:QUOTATION:0:0:0",
                }
            ]
        ]
        gold_labels = _one_hot_bio(seq, [(0, gold_start, gold_end)]).unsqueeze(0)

        overshoot = _one_hot_bio(seq, [(0, gold_start - 5, gold_end + 5)])
        undershoot = _one_hot_bio(seq, [(0, gold_start + 5, gold_end - 5)])
        l_over = span_tversky_loss(
            _confident_logits(overshoot), gold_labels, spans
        )
        l_under = span_tversky_loss(
            _confident_logits(undershoot), gold_labels, spans
        )
        self.assertGreater(l_over.item(), 0.0)
        self.assertGreater(l_under.item(), l_over.item())

    def test_truncated_spans_are_skipped(self):
        seq = 32
        labels = _one_hot_bio(seq, [(0, 8, 16)]).unsqueeze(0)
        logits = torch.zeros(1, seq, N_BIO)
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": 8,
                    "end": 16,
                    "truncated": True,
                    "span_id": "t:QUOTATION:0:0:0",
                }
            ]
        ]
        loss = span_tversky_loss(logits, labels, spans)
        self.assertEqual(loss.item(), 0.0)

    def test_fp_keep_ignores_neighbouring_same_layer_span(self):
        seq = 64
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": 10,
                    "end": 20,
                    "truncated": False,
                    "span_id": "a",
                },
                {
                    "layer": "QUOTATION",
                    "start": 24,
                    "end": 34,
                    "truncated": False,
                    "span_id": "b",
                },
            ]
        ]
        labels = _one_hot_bio(seq, [(0, 10, 20), (0, 24, 34)]).unsqueeze(0)
        perfect = span_tversky_loss(_confident_logits(labels[0]), labels, spans)
        self.assertLess(perfect.item(), 1e-4)

    def test_fn_keep_ignores_tokens_collapsed_to_another_layer(self):
        seq = 40
        # Gold quotation [10, 20), but BIO collapse gave those tokens to Sabche.
        labels = _one_hot_bio(seq, [(1, 10, 20)]).unsqueeze(0)
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": 10,
                    "end": 20,
                    "truncated": False,
                    "span_id": "t:QUOTATION:0:0:0",
                }
            ]
        ]
        # Predict nothing for quotation (all O). Without fn_keep this is a large FN.
        logits = _confident_logits(torch.zeros(seq, dtype=torch.long))
        loss = span_tversky_loss(logits, labels, spans)
        self.assertLess(loss.item(), 1e-4)


class EerDensityTests(unittest.TestCase):
    def test_rho_hat_matches_hand_calculation(self):
        seq = 100
        n_quote = 10
        labels = torch.zeros(seq, dtype=torch.long)
        labels[:n_quote] = LABEL2ID["I-QUOTE"]
        logits = _confident_logits(labels)
        attention = torch.ones(1, seq)
        # 20 padding tokens must be excluded; 15 loss_mask=0 tokens must be included.
        attention[0, 80:] = 0
        rho = predicted_density(logits, attention)
        # 10 positives / 80 non-padding
        self.assertAlmostEqual(rho[0].item(), 10 / 80, places=4)
        self.assertAlmostEqual(rho[1].item(), 0.0, places=4)

        priors = [EerPrior(rho=0.0, gamma=0.0) for _ in range(N_LAYERS)]
        priors[0] = EerPrior(rho=0.125, gamma=0.0)
        self.assertLess(eer_loss(logits, attention, priors).item(), 1e-4)

        priors[0] = EerPrior(rho=0.0, gamma=0.0)
        hinge = eer_loss(logits, attention, priors)
        self.assertAlmostEqual(hinge.item(), 10 / 80, places=4)

    def test_booktitle_channel_does_not_enter_eer(self):
        seq = 20
        labels = torch.zeros(seq, dtype=torch.long)
        labels[:] = LABEL2ID["I-BOOKTITLE"]
        logits = _confident_logits(labels)
        attention = torch.ones(1, seq)
        priors = [EerPrior(rho=0.0, gamma=0.0) for _ in range(N_LAYERS)]
        # If BOOKTITLE were included the hinge would be ~1.0.
        self.assertEqual(BOOKTITLE_INDEX, 6)
        self.assertLess(eer_loss(logits, attention, priors).item(), 1e-4)


class CombinedObjectiveTests(unittest.TestCase):
    def test_terms_are_finite_and_logged_separately(self):
        seq = 32
        labels = _one_hot_bio(seq, [(0, 8, 16)]).unsqueeze(0)
        logits = _confident_logits(labels[0])
        spans = [
            [
                {
                    "layer": "QUOTATION",
                    "start": 8,
                    "end": 16,
                    "truncated": False,
                    "span_id": "x",
                }
            ]
        ]
        start_logits = torch.zeros(1, seq, N_LAYERS)
        end_logits = torch.zeros(1, seq, N_LAYERS)
        start_t = torch.zeros(1, N_LAYERS, seq)
        end_t = torch.zeros(1, N_LAYERS, seq)
        start_t[0, 0, 8] = 1.0
        end_t[0, 0, 15] = 1.0
        mask = torch.ones(1, seq)
        out = compute_loss_objective(
            logits=logits,
            labels=labels,
            spans_batch=spans,
            start_logits=start_logits,
            end_logits=end_logits,
            start_targets=start_t,
            end_targets=end_t,
            loss_mask=mask,
            attention_mask=mask,
            pos_weight=torch.ones(N_LAYERS),
            priors=[EerPrior(0.025, 0.01)] * N_LAYERS,
            weights=V2LossWeights(ramp_span_eer_steps=0),
            step=1,
        )
        for key in ("loss", "loss_token", "loss_span", "loss_boundary", "loss_eer"):
            value = out[key]
            self.assertTrue(torch.isfinite(value), key)
            self.assertFalse(torch.isnan(value), key)


if __name__ == "__main__":
    unittest.main()

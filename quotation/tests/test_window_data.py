"""Pinned load, collator, pos_weight, and step-count helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from quotation_lib.window_data import (
    JOINT_MODEL_WINDOWS,
    DatasetRevisionError,
    V2DataCollator,
    assert_columns,
    boundary_pos_weight,
    early_stopping_eval_patience,
    report_window_step_counts,
    require_pinned_revision,
    steps_per_epoch,
    token_density_by_layer,
)
from quotation_lib.loss_objective import BOOKTITLE_INDEX, N_LAYERS
from quotation_lib.training_utils import LABEL2ID


class RevisionAndColumnsTests(unittest.TestCase):
    def test_missing_revision_raises(self):
        for value in (None, "", "null", "None"):
            with self.assertRaises(DatasetRevisionError):
                require_pinned_revision(value)

    def test_pinned_revision_passes(self):
        self.assertEqual(require_pinned_revision("v2.0"), "v2.0")

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError) as ctx:
            assert_columns(["input_ids", "labels"])
        self.assertIn("spans", str(ctx.exception))

    def test_required_columns_pass(self):
        assert_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
                "spans",
                "start_targets",
                "end_targets",
                "loss_mask",
                "book_id",
            ]
        )


class PosWeightAndDensityTests(unittest.TestCase):
    def test_pos_weight_is_n_neg_over_n_pos_and_clamped(self):
        seq = 10
        start = np.zeros((N_LAYERS, seq), dtype=np.float32)
        end = np.zeros((N_LAYERS, seq), dtype=np.float32)
        start[0, 2] = 1.0
        end[0, 5] = 1.0
        mask = np.ones(seq, dtype=np.float32)
        rows = [
            {
                "start_targets": start,
                "end_targets": end,
                "loss_mask": mask,
            }
        ]
        weights = boundary_pos_weight(rows, cap=100.0)
        # 2 heads × 10 tokens = 20; 2 positives → 18/2 = 9
        self.assertAlmostEqual(weights[0].item(), 9.0)
        self.assertEqual(weights[BOOKTITLE_INDEX].item(), 1.0)

    def test_pos_weight_respects_loss_mask(self):
        seq = 4
        start = np.zeros((N_LAYERS, seq), dtype=np.float32)
        end = np.zeros((N_LAYERS, seq), dtype=np.float32)
        start[0, 0] = 1.0
        mask = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rows = [{"start_targets": start, "end_targets": end, "loss_mask": mask}]
        weights = boundary_pos_weight(rows, cap=100.0)
        # masked-out positive must not count; 2 heads × 3 kept = 6 negatives, 0 pos → cap path uses max(n_pos,1)
        self.assertGreaterEqual(weights[0].item(), 1.0)

    def test_density_includes_loss_mask_zero_tokens(self):
        seq = 10
        labels = np.zeros(seq, dtype=np.int64)
        labels[:2] = LABEL2ID["I-QUOTE"]
        rows = [
            {
                "labels": labels,
                "attention_mask": np.ones(seq),
                "loss_mask": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
            }
        ]
        density = token_density_by_layer(rows)
        self.assertAlmostEqual(density["Quotation"], 2 / 10)
        self.assertEqual(density["__content_tokens__"], 10.0)


class _PadTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def pad(self, encoded_inputs, return_tensors=None, padding=True, pad_to_multiple_of=None, **kwargs):
        rows = encoded_inputs if isinstance(encoded_inputs, list) else [encoded_inputs]
        max_len = max(len(row["input_ids"]) for row in rows)
        if pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        batch = {"input_ids": [], "attention_mask": []}
        if any("labels" in row for row in rows):
            batch["labels"] = []
        for row in rows:
            pad = max_len - len(row["input_ids"])
            batch["input_ids"].append(list(row["input_ids"]) + [0] * pad)
            batch["attention_mask"].append(list(row.get("attention_mask", [1] * len(row["input_ids"]))) + [0] * pad)
            if "labels" in batch:
                batch["labels"].append(list(row.get("labels", [])) + [-100] * pad)
        if return_tensors == "pt":
            return {key: torch.tensor(value) for key, value in batch.items()}
        return batch


class CollatorTests(unittest.TestCase):
    def test_collator_pads_targets_and_keeps_spans(self):
        collator = V2DataCollator(_PadTokenizer(), pad_to_multiple_of=None)
        start_a = np.zeros((N_LAYERS, 3), dtype=np.float32)
        start_b = np.zeros((N_LAYERS, 5), dtype=np.float32)
        start_a[0, 1] = 1.0
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [0, 1, 2],
                "spans": [{"layer": "QUOTATION", "start": 1, "end": 2, "truncated": False}],
                "start_targets": start_a,
                "end_targets": np.zeros((N_LAYERS, 3), dtype=np.float32),
                "loss_mask": [1.0, 1.0, 0.0],
                "book_id": "P1",
            },
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [0, 0, 0, 0, 0],
                "spans": [],
                "start_targets": start_b,
                "end_targets": np.zeros((N_LAYERS, 5), dtype=np.float32),
                "loss_mask": [1.0, 1.0, 1.0, 1.0, 1.0],
                "book_id": "P2",
            },
        ]
        batch = collator(features)
        self.assertEqual(tuple(batch["start_targets"].shape), (2, N_LAYERS, 5))
        self.assertEqual(tuple(batch["loss_mask"].shape), (2, 5))
        self.assertEqual(batch["loss_mask"][0, 2].item(), 0.0)
        self.assertEqual(batch["start_targets"][0, 0, 1].item(), 1.0)
        self.assertEqual(len(batch["spans"][0]), 1)
        self.assertEqual(batch["book_id"], ["P1", "P2"])


class StepCountTests(unittest.TestCase):
    def test_joint_steps_match_plan(self):
        # ceil(14338/2) // 4 = 7169 // 4 = 1792
        self.assertEqual(steps_per_epoch(JOINT_MODEL_WINDOWS, 2, 4), 1792)

    def test_report_prints_eval_four_times_per_epoch(self):
        schedule = report_window_step_counts(
            9060, 1400, batch_size=2, grad_accum=4, num_epochs=10, warmup_ratio=0.2
        )
        self.assertEqual(schedule["eval_steps"], schedule["steps_per_epoch"] // 4)
        self.assertGreaterEqual(schedule["steps_per_epoch"] / schedule["eval_steps"], 4)
        self.assertEqual(schedule["evals_per_epoch"], 4)

    def test_early_stopping_patience_is_epochs_not_evals(self):
        # 4 evals/epoch × 2 epochs = 8. Passing 2 to HF would stop after 0.5 epoch.
        self.assertEqual(
            early_stopping_eval_patience(2, evals_per_epoch=4),
            8,
        )
        self.assertEqual(
            early_stopping_eval_patience(2, evals_per_epoch=1),
            2,
        )


if __name__ == "__main__":
    unittest.main()

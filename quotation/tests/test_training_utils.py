from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quotation_lib.audit_utils import BookRecord, MODEL_LAYERS, in_scope_layers
from quotation_lib.dataset_utils import apply_confirmed_scope
from quotation_lib.training_utils import (
    LABEL2ID,
    LABELS,
    align_window_labels,
    bio_spans,
    build_windowed_dataset,
    class_weights_from_dataset,
    normalize_spans,
    scope_mask,
    soft_span_metrics,
    tokenize_book_windows,
    window_cache_key,
)
from quotation_lib.window_supervision import (
    attach_window_supervision,
    build_window_fields,
    overlapping_token_range,
    paint_edge,
    resolve_windowing,
    round_trip_span,
    span_id_for,
    window_count_status,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        return {
            "input_ids": [[101, 10, 11, 102], [101, 11, 12, 102]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "token_type_ids": [[0, 0, 0, 0], [0, 0, 0, 0]],
            "offset_mapping": [
                [(0, 0), (0, 2), (2, 4), (0, 0)],
                [(0, 0), (2, 4), (4, 6), (0, 0)],
            ],
            "special_tokens_mask": [[1, 0, 0, 1], [1, 0, 0, 1]],
            "overflow_to_sample_mapping": [0, 0],
        }


class TrainingUtilsTests(unittest.TestCase):
    def test_bio_spans_treats_orphaned_i_as_new_span(self):
        self.assertEqual(
            bio_spans(["I-QUOTE", "I-QUOTE", "O", "B-AUTHOR"]),
            [("QUOTE", 0, 1), ("AUTHOR", 3, 3)],
        )

    def test_soft_span_metrics_accepts_one_token_boundary_error(self):
        gold = [["O", "B-QUOTE", "I-QUOTE", "O"]]
        predicted = [["B-QUOTE", "I-QUOTE", "O", "O"]]

        self.assertEqual(soft_span_metrics(gold, predicted, tolerance=0)["f1"], 0.0)
        self.assertEqual(soft_span_metrics(gold, predicted, tolerance=1)["f1"], 1.0)

    def test_soft_span_metrics_requires_matching_layer(self):
        metrics = soft_span_metrics(
            [["B-QUOTE"]],
            [["B-SABCHE"]],
            tolerance=1,
        )
        self.assertEqual(metrics, {"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_booktitle_labels_can_be_masked_from_loss(self):
        windows = tokenize_book_windows(
            {
                "book_id": "PTEST",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "BOOKTITLE"}],
                "in_scope_layers": ["BookTitle"],
            },
            FakeTokenizer(),
            max_length=4,
            stride=2,
            masked_layers=["BookTitle"],
        )
        self.assertTrue(
            all(
                label in (-100, LABEL2ID["O"])
                for window in windows["labels"]
                for label in window
            )
        )
        self.assertNotIn(LABEL2ID["B-BOOKTITLE"], windows["labels"][0])
        self.assertNotIn(LABEL2ID["I-BOOKTITLE"], windows["labels"][0])

    def test_o_class_weight_is_pinned_before_cap(self):
        class FakeDataset:
            def __getitem__(self, key):
                assert key == "labels"
                # Heavy O, one rare span label — without pinning, O would get a
                # tiny inverse-frequency weight after mean-normalization.
                return [
                    [LABEL2ID["O"]] * 100 + [LABEL2ID["B-AUTHOR"]],
                ]

        weights = class_weights_from_dataset(FakeDataset(), cap=5.0, o_weight=0.07)
        self.assertEqual(weights[LABEL2ID["O"]], 0.07)
        self.assertLessEqual(max(weights), 5.0)
        self.assertEqual(len(weights), len(LABELS))

    def test_class_weights_share_one_value_across_b_and_i(self):
        class FakeDataset:
            def __getitem__(self, key):
                assert key == "labels"
                # One long QUOTE span: a single B against many I. Weighting the
                # two independently would make B far heavier than I.
                return [
                    [LABEL2ID["O"]] * 500
                    + [LABEL2ID["B-QUOTE"]]
                    + [LABEL2ID["I-QUOTE"]] * 199
                    + [LABEL2ID["B-AUTHOR"]]
                    + [LABEL2ID["I-AUTHOR"]] * 4
                ]

        weights = class_weights_from_dataset(FakeDataset(), cap=5.0, o_weight=0.07)
        for tag in ("QUOTE", "AUTHOR"):
            self.assertEqual(
                weights[LABEL2ID[f"B-{tag}"]], weights[LABEL2ID[f"I-{tag}"]]
            )
        # The rarer layer still outweighs the common one.
        self.assertGreater(weights[LABEL2ID["I-AUTHOR"]], weights[LABEL2ID["I-QUOTE"]])

    def test_span_interior_outweighs_o(self):
        """O must not outweigh the interior of a common layer.

        When it does, continuing a span costs more than dropping out of it and
        long spans fragment into token-sized pieces.
        """

        class FakeDataset:
            def __getitem__(self, key):
                assert key == "labels"
                return [
                    [LABEL2ID["O"]] * 34_000
                    + [LABEL2ID["B-QUOTE"]] * 22
                    + [LABEL2ID["I-QUOTE"]] * 2_100
                    + [LABEL2ID["B-AUTHOR"]]
                    + [LABEL2ID["I-AUTHOR"]] * 7
                ]

        weights = class_weights_from_dataset(FakeDataset(), cap=5.0, o_weight=0.07)
        o = weights[LABEL2ID["O"]]
        for tag in ("QUOTE", "AUTHOR"):
            self.assertGreater(weights[LABEL2ID[f"I-{tag}"]], o)

    def test_confirmed_scope_treats_undeclared_layers_as_negatives(self):
        record = BookRecord(
            book="PTEST",
            doc_len=10,
            meta_path=None,
            base_path=None,
            declared_layers={"Quotation"},
            layer_files={},
        )
        scope = in_scope_layers(record, "confirmed")
        self.assertEqual(scope, set(MODEL_LAYERS) - {"Quotation"})

    def test_legacy_hub_scope_is_repaired(self):
        rows = [
            {
                "book_id": "P000010",
                "in_scope_layers": ["BookTitle"],
            }
        ]
        self.assertEqual(apply_confirmed_scope(rows), 1)
        self.assertEqual(
            set(rows[0]["in_scope_layers"]),
            set(MODEL_LAYERS) - {"Quotation"},
        )

    def test_priority_only_replaces_overlapping_part_of_lower_span(self):
        spans = normalize_spans(
            [
                {"start": 0, "end": 7, "label": "SABCHE"},
                {"start": 2, "end": 3, "label": "QUOTATION"},
            ]
        )

        labels, _ = align_window_labels(
            [(0, 2), (2, 4), (4, 6), (6, 8)],
            [0, 0, 0, 0],
            spans,
        )

        self.assertEqual(
            labels,
            [
                LABEL2ID["B-SABCHE"],
                LABEL2ID["B-QUOTE"],
                LABEL2ID["B-SABCHE"],
                LABEL2ID["I-SABCHE"],
            ],
        )

    def test_longer_non_quotation_span_wins(self):
        spans = normalize_spans(
            [
                {"start": 0, "end": 3, "label": "TSAWA"},
                {"start": 1, "end": 6, "label": "CHAPTER"},
            ]
        )
        labels, _ = align_window_labels([(2, 3)], [0], spans)
        self.assertEqual(labels, [LABEL2ID["B-CHAPTER"]])

    def test_stride_prefix_is_context_only(self):
        spans = normalize_spans([{"start": 2, "end": 7, "label": "QUOTATION"}])
        labels, covered = align_window_labels(
            [(2, 4), (4, 6), (6, 8)],
            [0, 0, 0],
            spans,
            already_covered_through=6,
        )
        self.assertEqual(labels, [-100, -100, LABEL2ID["I-QUOTE"]])
        self.assertEqual(covered, 8)

    def test_align_window_labels_matches_naive_on_dense_book(self):
        """Sweep alignment must match per-token scan; stay near-linear in spans."""
        import time

        from quotation_lib.training_utils import LAYER_TO_TAG, pick_winning_span

        # ~5k spans × ~20k tokens: naive O(tokens×spans) is many seconds;
        # the sweep must finish well under a second and agree token-for-token.
        spans = normalize_spans(
            [
                {
                    "start": i * 20,
                    "end": i * 20 + 9,
                    "label": "QUOTATION" if i % 5 else "SABCHE",
                }
                for i in range(5000)
            ]
        )
        offsets = [(i, i + 5) for i in range(0, 80000, 4)]
        specials = [0] * len(offsets)

        def naive() -> list[int]:
            labels: list[int] = []
            active: int | None = None
            for token_start, token_end in offsets:
                winner = pick_winning_span(spans, token_start, token_end)
                if winner is None:
                    labels.append(LABEL2ID["O"])
                    active = None
                else:
                    tag = "I" if active == winner["_identity"] else "B"
                    labels.append(
                        LABEL2ID[f"{tag}-{LAYER_TO_TAG[winner['layer']]}"]
                    )
                    active = winner["_identity"]
            return labels

        t0 = time.perf_counter()
        fast, _ = align_window_labels(offsets, specials, spans)
        elapsed = time.perf_counter() - t0
        self.assertEqual(fast, naive())
        self.assertLess(elapsed, 1.0, f"alignment too slow: {elapsed:.2f}s")

    def test_scope_mask_allows_only_o_and_assessed_layers(self):
        allowed = scope_mask(["Quotation", "BOOKTITLE"])
        self.assertEqual(allowed[LABEL2ID["O"]], 1)
        self.assertEqual(allowed[LABEL2ID["B-QUOTE"]], 1)
        self.assertEqual(allowed[LABEL2ID["I-BOOKTITLE"]], 1)
        self.assertEqual(allowed[LABEL2ID["B-SABCHE"]], 0)

    def test_tokenize_windows_preserves_model_inputs_and_masks_overlap(self):
        tokenizer = FakeTokenizer()
        windows = tokenize_book_windows(
            {
                "book_id": "PTEST",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
                "in_scope_layers": ["Quotation"],
            },
            tokenizer,
            max_length=4,
            stride=2,
        )
        self.assertIn("token_type_ids", windows)
        self.assertNotIn("offset_mapping", windows)
        self.assertEqual(windows["labels"][1][1], -100)
        self.assertEqual(windows["labels"][1][2], LABEL2ID["I-QUOTE"])
        self.assertFalse(tokenizer.kwargs["padding"])

    def test_include_metadata_keeps_offsets_and_book_id(self):
        windows = tokenize_book_windows(
            {
                "book_id": "PTEST",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
                "in_scope_layers": ["Quotation"],
            },
            FakeTokenizer(),
            max_length=4,
            stride=2,
            include_metadata=True,
        )
        self.assertIn("offset_mapping", windows)
        self.assertIn("special_tokens_mask", windows)
        self.assertEqual(windows["book_id"], ["PTEST", "PTEST"])
        self.assertEqual(windows["window_index"], [0, 1])
        self.assertEqual(windows["labels"][1][1], -100)

    def test_positive_span_must_be_in_scope(self):
        with self.assertRaisesRegex(ValueError, "positive spans outside"):
            tokenize_book_windows(
                {
                    "book_id": "PTEST",
                    "text": "ab",
                    "spans": [{"start": 0, "end": 1, "label": "AUTHOR"}],
                    "in_scope_layers": ["Quotation"],
                },
                FakeTokenizer(),
                max_length=4,
                stride=2,
            )

    def test_window_cache_key_changes_with_inputs(self):
        row = {
            "book_id": "PTEST",
            "text": "abcdef",
            "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
            "in_scope_layers": ["Quotation"],
        }
        base = window_cache_key(
            row,
            tokenizer_name="tok-a",
            max_length=4,
            stride=2,
            masked_layers=[],
        )
        self.assertNotEqual(
            base,
            window_cache_key(
                row,
                tokenizer_name="tok-b",
                max_length=4,
                stride=2,
                masked_layers=[],
            ),
        )
        changed = {**row, "text": "abcdefg"}
        self.assertNotEqual(
            base,
            window_cache_key(
                changed,
                tokenizer_name="tok-a",
                max_length=4,
                stride=2,
                masked_layers=[],
            ),
        )

    def test_build_windowed_dataset_caches_and_resumes_partial(self):
        rows = [
            {
                "book_id": "P001",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
                "in_scope_layers": ["Quotation"],
            },
            {
                "book_id": "P002",
                "text": "ghijkl",
                "spans": [{"start": 0, "end": 3, "label": "QUOTATION"}],
                "in_scope_layers": ["Quotation"],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            tokenizer = FakeTokenizer()
            with mock.patch(
                "quotation_lib.training_utils.tokenize_book_windows",
                wraps=tokenize_book_windows,
            ) as mocked:
                first = build_windowed_dataset(
                    rows[:1],
                    tokenizer,
                    max_length=4,
                    stride=2,
                    cache_dir=cache_dir,
                    tokenizer_name="fake-tok",
                )
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(len(first), 2)

                second = build_windowed_dataset(
                    rows,
                    tokenizer,
                    max_length=4,
                    stride=2,
                    cache_dir=cache_dir,
                    tokenizer_name="fake-tok",
                )
                # P001 served from cache; only P002 is tokenized.
                self.assertEqual(mocked.call_count, 2)
                self.assertEqual(len(second), 4)

    def test_build_windowed_dataset_parallel_matches_serial(self):
        rows = [
            {
                "book_id": f"P{i:03d}",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
                "in_scope_layers": ["Quotation"],
            }
            for i in range(4)
        ]
        serial = build_windowed_dataset(
            rows, FakeTokenizer(), max_length=4, stride=2, num_workers=1
        )
        parallel = build_windowed_dataset(
            rows, FakeTokenizer(), max_length=4, stride=2, num_workers=2
        )
        self.assertEqual(serial.column_names, parallel.column_names)
        for column in serial.column_names:
            self.assertEqual(list(serial[column]), list(parallel[column]))


class WindowSupervisionFieldTests(unittest.TestCase):
    def test_cli_stride_is_step_not_hf_overlap(self):
        window_len, step, hf_stride = resolve_windowing(8192, 4916)
        self.assertEqual((window_len, step, hf_stride), (8192, 4916, 3276))
        with self.assertRaises(ValueError):
            resolve_windowing(8192, 8192)
        with self.assertRaises(ValueError):
            resolve_windowing(8192, 0)

    def test_outward_rounding_includes_partial_edge_tokens(self):
        # Tokens cover chars [0,4) and [4,8). Gold [2,5] starts and ends mid-token.
        offsets = [(0, 0), (0, 4), (4, 8), (0, 0)]
        specials = [1, 0, 0, 1]
        self.assertEqual(overlapping_token_range(offsets, specials, 2, 5), (1, 3))
        self.assertEqual(overlapping_token_range(offsets, specials, 0, 3), (1, 2))
        self.assertEqual(overlapping_token_range(offsets, specials, 4, 7), (2, 3))

    def test_truncated_vs_whole_and_in_window_edges_only(self):
        offsets = [(0, 0), (10, 14), (14, 18), (18, 22), (0, 0)]
        specials = [1, 0, 0, 0, 1]
        labels = [-100, LABEL2ID["O"], LABEL2ID["O"], LABEL2ID["O"], -100]
        whole = {
            "source_index": 0,
            "char_start": 12,
            "char_end": 16,
            "hub_label": "QUOTATION",
            "layer": "Quotation",
        }
        truncated = {
            "source_index": 1,
            "char_start": 20,
            "char_end": 30,
            "hub_label": "SABCHE",
            "layer": "Sabche",
        }
        fields = build_window_fields(
            book_id="PTEST",
            offsets=offsets,
            specials=specials,
            labels=labels,
            gold_spans=[whole, truncated],
            damaged=False,
        )
        by_id = {span["span_id"]: span for span in fields["spans"]}
        self.assertFalse(by_id["PTEST:QUOTATION:12:16:0"]["truncated"])
        self.assertTrue(by_id["PTEST:SABCHE:20:30:1"]["truncated"])
        # Whole span: both edges in-window → 1.0 peaks.
        self.assertEqual(fields["start_targets"][0][1], 1.0)  # Quotation start
        self.assertEqual(fields["end_targets"][0][2], 1.0)
        # Truncated: start is in-window (char 20 ∈ [18,22)), end is outside.
        self.assertEqual(fields["start_targets"][1][3], 1.0)  # Sabche start
        self.assertTrue(all(value == 0.0 for value in fields["end_targets"][1]))

    def test_smoothing_uses_max_not_sum(self):
        row = [0.0] * 10
        paint_edge(row, 5, (1.0, 0.3, 0.1))
        paint_edge(row, 6, (1.0, 0.3, 0.1))
        self.assertEqual(row[5], 1.0)
        self.assertEqual(row[6], 1.0)
        self.assertEqual(row[4], 0.3)
        self.assertEqual(row[7], 0.3)
        self.assertEqual(row[3], 0.1)
        self.assertEqual(row[8], 0.1)
        self.assertLessEqual(max(row), 1.0)

    def test_damaged_book_masks_o_but_keeps_gold_positives(self):
        offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)]
        specials = [1, 0, 0, 0, 1]
        labels = [
            -100,
            LABEL2ID["B-QUOTE"],
            LABEL2ID["I-QUOTE"],
            LABEL2ID["O"],
            -100,
        ]
        gold = [
            {
                "source_index": 0,
                "char_start": 0,
                "char_end": 3,
                "hub_label": "QUOTATION",
                "layer": "Quotation",
            }
        ]
        fields = build_window_fields(
            book_id="P000218",
            offsets=offsets,
            specials=specials,
            labels=labels,
            gold_spans=gold,
            damaged=True,
        )
        self.assertEqual(fields["loss_mask"][0], 0.0)
        self.assertEqual(fields["loss_mask"][1], 1.0)
        self.assertEqual(fields["loss_mask"][2], 1.0)
        self.assertEqual(fields["loss_mask"][3], 0.0)
        self.assertEqual(fields["loss_mask_zeros"]["damaged_o"], 1)
        self.assertEqual(fields["loss_mask_zeros"]["special_pad"], 2)

    def test_booktitle_tokens_are_zero_on_loss_mask_only(self):
        windows = tokenize_book_windows(
            {
                "book_id": "PTEST",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "BOOKTITLE"}],
                "in_scope_layers": ["BookTitle"],
            },
            FakeTokenizer(),
            max_length=4,
            stride=2,
            include_metadata=True,
        )
        attached, _stats = attach_window_supervision(
            windows,
            {
                "book_id": "PTEST",
                "text": "abcdef",
                "spans": [{"start": 2, "end": 5, "label": "BOOKTITLE"}],
            },
            damaged_books=(),
        )
        # BIO labels stay BookTitle (not -100); loss_mask zeros those tokens.
        booktitle_ids = {LABEL2ID["B-BOOKTITLE"], LABEL2ID["I-BOOKTITLE"]}
        self.assertTrue(
            any(label in booktitle_ids for window in attached["labels"] for label in window)
        )
        n_zeroed = 0
        for labels, mask in zip(attached["labels"], attached["loss_mask"]):
            for label, weight in zip(labels, mask):
                if label in booktitle_ids:
                    self.assertEqual(weight, 0.0)
                    n_zeroed += 1
        self.assertGreater(n_zeroed, 0)

    def test_same_layer_overlaps_are_kept(self):
        offsets = [(0, 0), (0, 4), (4, 8), (8, 12), (0, 0)]
        specials = [1, 0, 0, 0, 1]
        labels = [-100, LABEL2ID["B-QUOTE"], LABEL2ID["I-QUOTE"], LABEL2ID["I-QUOTE"], -100]
        gold = [
            {
                "source_index": 0,
                "char_start": 0,
                "char_end": 7,
                "hub_label": "QUOTATION",
                "layer": "Quotation",
            },
            {
                "source_index": 1,
                "char_start": 4,
                "char_end": 11,
                "hub_label": "QUOTATION",
                "layer": "Quotation",
            },
        ]
        fields = build_window_fields(
            book_id="PTEST",
            offsets=offsets,
            specials=specials,
            labels=labels,
            gold_spans=gold,
            damaged=False,
        )
        self.assertEqual(len(fields["spans"]), 2)
        self.assertEqual(fields["n_same_layer_overlaps"], 1)
        self.assertEqual({span["span_id"] for span in fields["spans"]}, {
            "PTEST:QUOTATION:0:7:0",
            "PTEST:QUOTATION:4:11:1",
        })

    def test_span_id_is_stable_across_two_windows(self):
        gold_row = {
            "book_id": "PTEST",
            "text": "abcdefghij",
            "spans": [{"start": 2, "end": 5, "label": "QUOTATION"}],
            "in_scope_layers": ["Quotation"],
        }
        windows = tokenize_book_windows(
            gold_row, FakeTokenizer(), max_length=4, stride=2, include_metadata=True
        )
        attached, _ = attach_window_supervision(
            windows, gold_row, damaged_books=(), min_span_chars=1
        )
        expected = span_id_for("PTEST", "QUOTATION", 2, 5, 0)
        ids_by_window = [
            {span["span_id"] for span in window_spans}
            for window_spans in attached["spans"]
        ]
        self.assertGreaterEqual(len(ids_by_window), 2)
        self.assertIn(expected, ids_by_window[0])
        self.assertIn(expected, ids_by_window[1])
        self.assertEqual(ids_by_window[0] & ids_by_window[1], {expected})

    def test_round_trip_allows_outward_token_edges(self):
        text = " abcdef།"
        # Token covers the leading space and trailing shad; gold is the interior.
        ok, gold, reconstructed = round_trip_span(
            text,
            [(0, 0), (0, 8), (0, 0)],
            [1, 0, 1],
            {"start": 1, "end": 2, "char_start": 1, "char_end": 6},
        )
        self.assertTrue(ok)
        self.assertEqual(gold, "abcdef")
        self.assertEqual(reconstructed, " abcdef།")

    def test_window_count_rejects_9103_and_8906(self):
        status, err = window_count_status(9060, predicted=9060)
        self.assertEqual(status, "PASS")
        self.assertIsNone(err)
        _status, err = window_count_status(9103, predicted=None)
        self.assertIn("9103", err or "")
        _status, err = window_count_status(8906, predicted=None)
        self.assertIn("8906", err or "")
        # Ceil drift that happens to sit nearer 9103 is allowed if L_i explains it.
        status, err = window_count_status(9082, predicted=9082)
        self.assertEqual(status, "PASS")
        self.assertIsNone(err)

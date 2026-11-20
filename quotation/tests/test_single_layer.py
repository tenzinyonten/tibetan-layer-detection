"""Tests for the single-annotation training path (quotation/src/train_quotation_model.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quotation_lib.single_layer import (
    build_scheme,
    compute_metrics_factory,
    in_scope,
    iou50_metrics,
    project_labels,
    sequences_from_ids,
    single_layer_class_weights,
    token_recall_inside_gold,
    transition_matrix,
    joint_model_equivalent_weight,
    viterbi_decode,
)
from quotation_lib.training_utils import LABEL2ID, LABELS

SCHEME = build_scheme("Yigchung")
B15, I15 = LABEL2ID["B-YIGCHUNG"], LABEL2ID["I-YIGCHUNG"]
BQ15, IQ15 = LABEL2ID["B-QUOTE"], LABEL2ID["I-QUOTE"]
O15 = LABEL2ID["O"]


# --------------------------------------------------------------------------
# scheme
# --------------------------------------------------------------------------


def test_scheme_is_three_labels_in_bio_order():
    assert SCHEME.labels == ("O", "B-YIGCHUNG", "I-YIGCHUNG")
    assert (SCHEME.o_id, SCHEME.b_id, SCHEME.i_id) == (0, 1, 2)
    assert (SCHEME.source_b_id, SCHEME.source_i_id) == (B15, I15)


def test_scheme_accepts_lowercase_layer_names():
    assert build_scheme("yigchung").tag == "YIGCHUNG"


# --------------------------------------------------------------------------
# label projection
# --------------------------------------------------------------------------


def test_other_layers_become_O_and_the_target_keeps_its_bio_prefix():
    labels15 = [O15, BQ15, IQ15, B15, I15, I15, O15]
    out, recovered = project_labels(labels15, [], SCHEME)
    assert out == [0, 0, 0, 1, 2, 2, 0]
    assert recovered == 0


def test_minus_100_is_never_overwritten():
    labels15 = [-100, -100, B15, I15, -100]
    out, _ = project_labels(labels15, [], SCHEME)
    assert out == [-100, -100, 1, 2, -100]


def test_a_span_continuing_past_the_duplicated_prefix_keeps_its_I_prefix():
    # align_window_labels computes BIO before masking, so the first supervised
    # token of a span that started in the previous window is I-, not B-.
    labels15 = [-100, -100, I15, I15, O15]
    out, _ = project_labels(labels15, [], SCHEME)
    assert out == [-100, -100, 2, 2, 0]


def test_nested_gold_hidden_by_winner_takes_all_is_recovered():
    # Quotation wins the 15-class collapse; the Yigchung span inside it is
    # invisible in `labels` and would otherwise train as a negative.
    labels15 = [O15, BQ15, IQ15, IQ15, IQ15, O15]
    spans = [{"layer": "YIGCHUNG", "start": 2, "end": 4}]  # half-open
    out, recovered = project_labels(labels15, spans, SCHEME)
    assert out == [0, 0, 1, 2, 0, 0]
    assert recovered == 2


def test_recovery_continues_an_adjacent_visible_span_rather_than_restarting_it():
    labels15 = [B15, I15, IQ15, O15]
    spans = [{"layer": "YIGCHUNG", "start": 0, "end": 3}]
    out, recovered = project_labels(labels15, spans, SCHEME)
    assert out == [1, 2, 2, 0]
    assert recovered == 1


def test_recovery_skips_masked_positions_and_other_layers_spans():
    labels15 = [-100, BQ15, IQ15]
    spans = [
        {"layer": "YIGCHUNG", "start": 0, "end": 2},
        {"layer": "TSAWA", "start": 0, "end": 3},
    ]
    out, recovered = project_labels(labels15, spans, SCHEME)
    assert out == [-100, 1, 0]
    assert recovered == 1


def test_recovery_clamps_a_span_that_runs_past_the_window():
    labels15 = [O15, O15]
    spans = [{"layer": "YIGCHUNG", "start": 1, "end": 99}]
    out, recovered = project_labels(labels15, spans, SCHEME)
    assert out == [0, 1]
    assert recovered == 1


def test_projection_is_lossless_for_the_target_layer():
    rng = np.random.default_rng(0)
    labels15 = rng.integers(0, len(LABELS), size=4096).tolist()
    out, _ = project_labels(labels15, [], SCHEME)
    for old, new in zip(labels15, out, strict=True):
        if old == B15:
            assert new == SCHEME.b_id
        elif old == I15:
            assert new == SCHEME.i_id
        else:
            assert new == SCHEME.o_id


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------


def test_in_scope_reads_the_target_layers_entries_only():
    full = [1] * len(LABELS)
    assert in_scope(full, SCHEME)
    without = list(full)
    without[B15] = 0
    without[I15] = 0
    assert not in_scope(without, SCHEME)
    other = list(full)
    other[BQ15] = 0
    assert in_scope(other, SCHEME)


def test_in_scope_rejects_a_mask_of_the_wrong_width():
    with pytest.raises(ValueError):
        in_scope([1, 1, 1], SCHEME)


# --------------------------------------------------------------------------
# Viterbi
# --------------------------------------------------------------------------


def test_transition_matrix_forbids_I_without_a_matching_B_or_I():
    matrix = transition_matrix(SCHEME.labels, break_penalty=0.0)
    assert matrix[SCHEME.o_id, SCHEME.i_id] < -1e8
    assert matrix[SCHEME.b_id, SCHEME.i_id] == 0.0
    assert matrix[SCHEME.i_id, SCHEME.i_id] == 0.0
    assert matrix[SCHEME.o_id, SCHEME.b_id] == 0.0


def test_break_penalty_is_charged_on_span_exit_only():
    matrix = transition_matrix(SCHEME.labels, break_penalty=3.0)
    assert matrix[SCHEME.i_id, SCHEME.o_id] == -3.0
    assert matrix[SCHEME.i_id, SCHEME.b_id] == -3.0
    assert matrix[SCHEME.i_id, SCHEME.i_id] == 0.0
    assert matrix[SCHEME.o_id, SCHEME.o_id] == 0.0


def _one_hot(path: list[int], strength: float = 8.0) -> np.ndarray:
    logits = np.zeros((1, len(path), SCHEME.n_labels), dtype=np.float32)
    for index, label in enumerate(path):
        logits[0, index, label] = strength
    return logits


def test_viterbi_recovers_a_confident_legal_path():
    path = [0, 1, 2, 2, 0, 1, 2]
    decoded = viterbi_decode(_one_hot(path), SCHEME, break_penalty=0.0)
    assert decoded[0].tolist() == path


def test_viterbi_repairs_an_orphaned_I():
    # An I- with no B- before it is illegal; the decode must not emit one.
    # Position 0 has no predecessor and is unconstrained, exactly as in
    # span_metrics; bio_spans starts a new span on an orphaned I there.
    decoded = viterbi_decode(_one_hot([0, 2, 2, 0]), SCHEME, break_penalty=0.0)
    labels = [SCHEME.id2label[i] for i in decoded[0].tolist()]
    assert labels[1] == "B-YIGCHUNG"
    for previous, current in zip(labels[:-1], labels[1:], strict=True):
        if current == "I-YIGCHUNG":
            assert previous in ("B-YIGCHUNG", "I-YIGCHUNG")


def test_a_large_break_penalty_merges_two_near_spans_into_one():
    logits = _one_hot([1, 2, 0, 1, 2], strength=1.0)
    loose = viterbi_decode(logits, SCHEME, break_penalty=0.0)[0].tolist()
    tight = viterbi_decode(logits, SCHEME, break_penalty=8.0)[0].tolist()
    assert loose.count(SCHEME.b_id) == 2
    assert tight.count(SCHEME.b_id) == 1


def test_viterbi_forces_O_on_padding():
    logits = _one_hot([1, 2, 2, 2])
    mask = np.array([[1, 1, 0, 0]])
    decoded = viterbi_decode(logits, SCHEME, break_penalty=0.0, attention_mask=mask)
    assert decoded[0].tolist()[2:] == [SCHEME.o_id, SCHEME.o_id]


def test_viterbi_rejects_the_wrong_number_of_classes():
    with pytest.raises(ValueError):
        viterbi_decode(np.zeros((1, 4, 15), dtype=np.float32), SCHEME)


def test_viterbi_chunking_does_not_change_the_result():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(9, 24, 3)).astype(np.float32)
    whole = viterbi_decode(logits, SCHEME, break_penalty=2.0, chunk=64)
    chunked = viterbi_decode(logits, SCHEME, break_penalty=2.0, chunk=2)
    assert np.array_equal(whole, chunked)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_iou50_is_perfect_when_prediction_equals_gold():
    gold = [["O", "B-YIGCHUNG", "I-YIGCHUNG", "O", "B-YIGCHUNG"]]
    metrics = iou50_metrics(gold, gold)
    assert metrics["iou50_f1"] == 1.0
    assert metrics["iou50_n_pred"] == metrics["iou50_n_gold"] == 2.0
    assert metrics["iou50_overproposal"] == 1.0


def test_iou50_counts_a_span_off_by_one_token_as_a_match_but_not_a_half_span():
    gold = [["B-YIGCHUNG"] + ["I-YIGCHUNG"] * 9]  # 10 tokens
    near = [["B-YIGCHUNG"] + ["I-YIGCHUNG"] * 8 + ["O"]]  # 9/10 → IoU 0.9
    half = [["B-YIGCHUNG"] + ["I-YIGCHUNG"] * 3 + ["O"] * 6]  # 4/10 → IoU 0.4
    assert iou50_metrics(near, gold)["iou50_f1"] == 1.0
    assert iou50_metrics(half, gold)["iou50_f1"] == 0.0


def test_overproposal_is_predicted_over_gold_span_count():
    gold = [["B-YIGCHUNG", "I-YIGCHUNG", "O", "O"]]
    pred = [["B-YIGCHUNG", "O", "B-YIGCHUNG", "B-YIGCHUNG"]]
    assert iou50_metrics(pred, gold)["iou50_overproposal"] == 3.0


def test_token_recall_counts_only_tokens_inside_gold_spans():
    gold = [["B-YIGCHUNG", "I-YIGCHUNG", "I-YIGCHUNG", "O"]]
    pred = [["B-YIGCHUNG", "I-YIGCHUNG", "O", "B-YIGCHUNG"]]
    out = token_recall_inside_gold(pred, gold)
    assert out["token_recall_inside_gold"] == pytest.approx(2 / 3)
    assert out["token_recall_n"] == 3.0


def test_sequences_from_ids_drops_masked_positions_from_both_sides():
    pred_ids = [[1, 2, 0, 0]]
    gold_ids = [[1, 2, -100, 0]]
    pred, gold = sequences_from_ids(pred_ids, gold_ids, SCHEME)
    assert pred == [["B-YIGCHUNG", "I-YIGCHUNG", "O"]]
    assert gold == [["B-YIGCHUNG", "I-YIGCHUNG", "O"]]


def test_compute_metrics_scores_a_perfect_model_at_one():
    gold_ids = np.array([[1, 2, 2, 0, 0, 1, 2, -100]])
    logits = np.zeros((1, 8, 3), dtype=np.float32)
    for index, label in enumerate(gold_ids[0]):
        logits[0, index, max(int(label), 0)] = 10.0
    metrics = compute_metrics_factory(SCHEME, break_penalty=2.0)((logits, gold_ids))
    assert metrics["iou50_f1"] == 1.0
    assert metrics["token_recall_inside_gold"] == 1.0
    assert metrics["argmax_iou50_f1"] == 1.0
    assert metrics["viterbi_break_penalty"] == 2.0


def test_compute_metrics_reports_zero_rather_than_dividing_by_zero():
    gold_ids = np.array([[0, 0, 0]])
    logits = np.zeros((1, 3, 3), dtype=np.float32)
    logits[0, :, 0] = 10.0
    metrics = compute_metrics_factory(SCHEME)((logits, gold_ids))
    assert metrics["iou50_f1"] == 0.0
    assert metrics["iou50_n_gold"] == 0.0
    assert metrics["token_recall_inside_gold"] == 0.0


def test_compute_metrics_accepts_a_tuple_of_outputs():
    gold_ids = np.array([[1, 2, 0]])
    logits = np.zeros((1, 3, 3), dtype=np.float32)
    logits[0, 0, 1] = logits[0, 1, 2] = logits[0, 2, 0] = 10.0
    metrics = compute_metrics_factory(SCHEME)(((logits, None), gold_ids))
    assert metrics["iou50_f1"] == 1.0


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------


class _FakeDataset:
    def __init__(self, labels):
        self._labels = labels

    def __getitem__(self, key):
        assert key == "labels"
        return self._labels


def test_class_weights_pin_the_layer_at_one_and_O_at_the_configured_value():
    dataset = _FakeDataset([[0] * 100 + [1, 2, 2], [-100, 0, 0]])
    weights = single_layer_class_weights(dataset, SCHEME, o_weight=0.07)
    assert weights == [0.07, 1.0, 1.0]


def test_class_weight_override_reproduces_a_chosen_positive_weight():
    dataset = _FakeDataset([[0] * 10 + [1, 2]])
    weights = single_layer_class_weights(
        dataset, SCHEME, o_weight=0.07, positive_weight=5.0
    )
    assert weights == [0.07, 5.0, 5.0]


def test_class_weights_respect_the_cap():
    dataset = _FakeDataset([[0] * 10 + [1, 2]])
    weights = single_layer_class_weights(
        dataset, SCHEME, cap=2.0, o_weight=0.07, positive_weight=9.0
    )
    assert weights == [0.07, 2.0, 2.0]


def test_class_weights_refuse_a_split_with_no_positives():
    with pytest.raises(ValueError):
        single_layer_class_weights(_FakeDataset([[0, 0, -100]]), SCHEME)


def test_joint_model_equivalent_weight_matches_the_15_class_formula():
    from quotation_lib.training_utils import class_weights_from_dataset

    counts = {
        LABEL2ID["O"]: 1_000_000,
        LABEL2ID["B-QUOTE"]: 3_000,
        LABEL2ID["I-QUOTE"]: 60_000,
        LABEL2ID["B-YIGCHUNG"]: 700,
        LABEL2ID["I-YIGCHUNG"]: 4_000,
        LABEL2ID["B-TSAWA"]: 500,
        LABEL2ID["I-TSAWA"]: 20_000,
    }
    flat = []
    for label_id, count in counts.items():
        flat.extend([label_id] * count)
    expected = class_weights_from_dataset(
        _FakeDataset([flat]), cap=5.0, o_weight=0.07
    )[LABEL2ID["B-YIGCHUNG"]]
    assert joint_model_equivalent_weight(counts, SCHEME, cap=5.0) == pytest.approx(expected)


# --------------------------------------------------------------------------
# SplitBundle — dataset and book ids must not drift apart
# --------------------------------------------------------------------------


class _FakeArrow:
    """Minimal stand-in for a datasets.Dataset for slicing tests."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return _FakeArrow([self._rows[i] for i in indices])


def _bundle(n):
    from quotation_lib.single_layer import SplitBundle

    return SplitBundle(
        dataset=_FakeArrow(range(n)),
        book_ids=[f"P{i:03d}" for i in range(n)],
        stats={},
    )


def test_select_slices_dataset_and_book_ids_together():
    small = _bundle(10).select(4)
    assert len(small) == 4
    assert small.book_ids == ["P000", "P001", "P002", "P003"]
    small.check_alignment()


def test_select_clamps_to_the_available_rows():
    assert len(_bundle(3).select(99)) == 3
    assert len(_bundle(3).select(-1)) == 0


def test_check_alignment_catches_drift():
    from quotation_lib.single_layer import SplitBundle

    bad = SplitBundle(dataset=_FakeArrow(range(5)), book_ids=["a", "b"], stats={})
    with pytest.raises(ValueError):
        bad.check_alignment()


def test_empty_book_ids_are_allowed():
    from quotation_lib.single_layer import SplitBundle

    SplitBundle(dataset=_FakeArrow(range(5)), book_ids=[], stats={}).check_alignment()


# --------------------------------------------------------------------------
# per-book metrics
# --------------------------------------------------------------------------


def _seqs(*specs):
    return [list(spec) for spec in specs]


G = "B-YIGCHUNG"
I = "I-YIGCHUNG"
O = "O"


def test_per_book_pools_counts_across_a_books_windows():
    from quotation_lib.single_layer import per_book_metrics

    gold = [[G, O], [G, O]]
    pred = [[G, O], [O, O]]
    metrics = per_book_metrics(pred, gold, ["A", "A"])
    assert metrics["book_A_n_gold"] == 2.0
    assert metrics["book_A_iou50_recall"] == 0.5


def test_per_book_ignores_books_with_no_gold_spans():
    from quotation_lib.single_layer import per_book_metrics

    metrics = per_book_metrics([[G, O], [O, O]], [[G, O], [O, O]], ["A", "B"])
    assert "book_A_iou50_f1" in metrics
    assert "book_B_iou50_f1" not in metrics
    assert metrics["n_books_with_gold"] == 1.0


def test_excl_top_book_removes_the_largest_contributor():
    from quotation_lib.single_layer import per_book_metrics

    # Book A: 3 gold, all found. Book B: 1 gold, missed.
    gold = [[G, O], [G, O], [G, O], [G, O]]
    pred = [[G, O], [G, O], [G, O], [O, O]]
    metrics = per_book_metrics(pred, gold, ["A", "A", "A", "B"])
    assert metrics["book_A_iou50_f1"] == 1.0
    assert metrics["iou50_f1_excl_top_book"] == 0.0
    assert metrics["excl_top_book_n_gold"] == 1.0


def test_excl_top_book_is_zero_when_only_one_book_has_gold():
    from quotation_lib.single_layer import per_book_metrics

    metrics = per_book_metrics([[G, O]], [[G, O]], ["A"])
    assert metrics["iou50_f1_excl_top_book"] == 0.0
    assert metrics["excl_top_book_n_gold"] == 0.0


def test_per_book_returns_nothing_without_book_ids():
    from quotation_lib.single_layer import per_book_metrics

    assert per_book_metrics([[G]], [[G]], []) == {}


# --------------------------------------------------------------------------
# decode sweep and selection
# --------------------------------------------------------------------------


def test_bias_vector_lifts_both_positive_classes_only():
    from quotation_lib.single_layer import bias_vector

    bias = bias_vector(SCHEME, 1.5)
    assert bias.tolist() == [0.0, 1.5, 1.5]


def test_a_positive_bias_can_only_help_recall():
    from quotation_lib.single_layer import bias_vector, evaluate_logits

    gold = np.array([[1, 2, 2, 0, 0, 0]])
    logits = np.zeros((1, 6, 3), dtype=np.float32)
    logits[0, :, 0] = 0.6  # weakly prefers O everywhere — the quiet model
    logits[0, 0, 1] = 0.5  # B at the span start, just not quite enough
    logits[0, 1:3, 2] = 0.5  # I for the continuation
    plain = evaluate_logits(logits, gold, SCHEME, break_penalty=0.0)
    lifted = evaluate_logits(
        logits, gold, SCHEME, break_penalty=0.0, label_bias=bias_vector(SCHEME, 3.0)
    )
    assert plain["iou50_recall"] == 0.0
    assert lifted["iou50_recall"] > plain["iou50_recall"]


def test_bias_does_not_move_the_argmax_reference():
    from quotation_lib.single_layer import bias_vector, evaluate_logits

    gold = np.array([[1, 2, 0, 0]])
    logits = np.zeros((1, 4, 3), dtype=np.float32)
    logits[0, :, 0] = 1.0
    plain = evaluate_logits(logits, gold, SCHEME, break_penalty=0.0)
    lifted = evaluate_logits(
        logits, gold, SCHEME, break_penalty=0.0, label_bias=bias_vector(SCHEME, 5.0)
    )
    assert plain["argmax_iou50_f1"] == lifted["argmax_iou50_f1"]


def test_sweep_covers_every_pair_and_records_both_settings():
    from quotation_lib.single_layer import sweep_decode

    gold = np.array([[1, 2, 0, 0]])
    logits = np.zeros((1, 4, 3), dtype=np.float32)
    logits[0, 0, 1] = logits[0, 1, 2] = 5.0
    rows = sweep_decode(
        logits, gold, SCHEME, penalties=(0.0, 2.0), biases=(0.0, 1.0)
    )
    assert len(rows) == 4
    assert {(r["break_penalty"], r["bias"]) for r in rows} == {
        (0.0, 0.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)
    }


def test_best_decode_picks_the_highest_f1():
    from quotation_lib.single_layer import best_decode

    rows = [
        {"iou50_f1": 0.30, "break_penalty": 0.0, "bias": 0.0},
        {"iou50_f1": 0.42, "break_penalty": 4.0, "bias": 1.0},
        {"iou50_f1": 0.41, "break_penalty": 2.0, "bias": 0.0},
    ]
    assert best_decode(rows)["break_penalty"] == 4.0


def test_best_decode_breaks_ties_toward_the_gentler_setting():
    from quotation_lib.single_layer import best_decode

    rows = [
        {"iou50_f1": 0.40, "break_penalty": 8.0, "bias": 2.0},
        {"iou50_f1": 0.40, "break_penalty": 1.0, "bias": 0.0},
    ]
    chosen = best_decode(rows)
    assert (chosen["break_penalty"], chosen["bias"]) == (1.0, 0.0)


def test_best_decode_refuses_an_empty_sweep():
    from quotation_lib.single_layer import best_decode

    with pytest.raises(ValueError):
        best_decode([])


# --------------------------------------------------------------------------
# compute_metrics with book ids
# --------------------------------------------------------------------------


def test_compute_metrics_adds_per_book_keys_without_touching_the_selection_key():
    gold_ids = np.array([[1, 2, 0], [1, 0, 0]])
    logits = np.zeros((2, 3, 3), dtype=np.float32)
    for b in range(2):
        for t in range(3):
            logits[b, t, max(int(gold_ids[b, t]), 0)] = 10.0
    metrics = compute_metrics_factory(SCHEME, book_ids=["A", "B"])((logits, gold_ids))
    assert metrics["iou50_f1"] == 1.0          # selection key, unchanged
    assert metrics["book_A_iou50_f1"] == 1.0
    assert metrics["book_B_iou50_f1"] == 1.0
    assert metrics["n_books_with_gold"] == 2.0


def test_compute_metrics_refuses_misaligned_book_ids():
    gold_ids = np.array([[1, 2, 0], [1, 0, 0]])
    logits = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="per-book metrics would be wrong"):
        compute_metrics_factory(SCHEME, book_ids=["A"])((logits, gold_ids))


def test_compute_metrics_without_book_ids_emits_no_per_book_keys():
    gold_ids = np.array([[1, 2, 0]])
    logits = np.zeros((1, 3, 3), dtype=np.float32)
    metrics = compute_metrics_factory(SCHEME)((logits, gold_ids))
    assert not any(key.startswith("book_") for key in metrics)

"""Tests for the BIOE quotation detection layer (src/layer_detection/bioe_layer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quotation_lib.bioe_layer import (
    BioeScheme,
    build_scheme,
    transition_matrix,
    viterbi_decode,
    bioe_spans,
    downgrade_to_bio,
    stitch_book_logits,
    e_recall,
    pinned_class_weights,
    evaluate_logits,
    compute_metrics_factory,
    bias_vector,
)
from quotation_lib.training_utils import bio_spans as reference_bio_spans
import quotation_lib.single_layer as single_layer

SCHEME = build_scheme("Quotation")


# --------------------------------------------------------------------------
# scheme
# --------------------------------------------------------------------------


def test_scheme_is_four_labels_in_bioe_order():
    assert SCHEME.labels == ("O", "B-QUOTE", "I-QUOTE", "E-QUOTE")
    assert (SCHEME.o_id, SCHEME.b_id, SCHEME.i_id, SCHEME.e_id) == (0, 1, 2, 3)
    assert SCHEME.n_labels == 4


def test_single_layer_still_builds_three_class_scheme():
    # Assert single_layer's 3-class scheme still builds (as required by test plan)
    yigchung_scheme = single_layer.build_scheme("Yigchung")
    assert yigchung_scheme.n_labels == 3
    assert yigchung_scheme.labels == ("O", "B-YIGCHUNG", "I-YIGCHUNG")


# --------------------------------------------------------------------------
# transition legality
# --------------------------------------------------------------------------


def test_transition_matrix_legality_and_break_penalty():
    # Every forbidden pair is NEG, every legal pair is not NEG
    matrix = transition_matrix(SCHEME.labels, break_penalty=3.0)
    
    # Legal transitions:
    # ("O", "O"), ("O", "B"), ("B", "I"), ("B", "E"), ("I", "I"), ("I", "E"), ("E", "O"), ("E", "B")
    assert matrix[SCHEME.o_id, SCHEME.o_id] == 0.0
    assert matrix[SCHEME.o_id, SCHEME.b_id] == 0.0
    assert matrix[SCHEME.b_id, SCHEME.i_id] == 0.0
    assert matrix[SCHEME.b_id, SCHEME.e_id] == 0.0
    assert matrix[SCHEME.i_id, SCHEME.i_id] == 0.0
    assert matrix[SCHEME.i_id, SCHEME.e_id] == 0.0
    assert matrix[SCHEME.e_id, SCHEME.o_id] == 0.0
    assert matrix[SCHEME.e_id, SCHEME.b_id] == -3.0  # break penalty subtracted on E -> B only
    
    # Forbidden transitions:
    # Everything else is NEG
    assert matrix[SCHEME.o_id, SCHEME.i_id] < -1e8
    assert matrix[SCHEME.o_id, SCHEME.e_id] < -1e8
    assert matrix[SCHEME.b_id, SCHEME.o_id] < -1e8
    assert matrix[SCHEME.b_id, SCHEME.b_id] < -1e8
    assert matrix[SCHEME.i_id, SCHEME.o_id] < -1e8
    assert matrix[SCHEME.i_id, SCHEME.b_id] < -1e8
    assert matrix[SCHEME.e_id, SCHEME.i_id] < -1e8
    assert matrix[SCHEME.e_id, SCHEME.e_id] < -1e8


# --------------------------------------------------------------------------
# Viterbi Decode
# --------------------------------------------------------------------------


def _one_hot(path: list[int], strength: float = 10.0) -> np.ndarray:
    logits = np.zeros((1, len(path), SCHEME.n_labels), dtype=np.float32)
    for index, label in enumerate(path):
        logits[0, index, label] = strength
    return logits


def test_viterbi_never_emits_illegal_sequences():
    # Even with random/noisy logits, Viterbi must only emit legal transitions
    rng = np.random.default_rng(42)
    for _ in range(20):
        logits = rng.normal(size=(1, 50, SCHEME.n_labels)).astype(np.float32)
        decoded = viterbi_decode(logits, SCHEME, break_penalty=2.0)[0].tolist()
        
        # Check start of sequence: must be O or B
        assert decoded[0] in (SCHEME.o_id, SCHEME.b_id)
        
        # Check end of sequence: must be O or E
        assert decoded[-1] in (SCHEME.o_id, SCHEME.e_id)
        
        # Check transitions
        for i in range(len(decoded) - 1):
            p, n = decoded[i], decoded[i+1]
            p_lbl = SCHEME.id2label[p].split("-")[0]
            n_lbl = SCHEME.id2label[n].split("-")[0]
            
            # Legal pairs:
            # ("O", "O"), ("O", "B"), ("B", "I"), ("B", "E"), ("I", "I"), ("I", "E"), ("E", "O"), ("E", "B")
            assert (p_lbl, n_lbl) in {
                ("O", "O"), ("O", "B"),
                ("B", "I"), ("B", "E"),
                ("I", "I"), ("I", "E"),
                ("E", "O"), ("E", "B")
            }


# --------------------------------------------------------------------------
# BIOE Spans
# --------------------------------------------------------------------------


def test_bioe_spans_edge_cases():
    # B I* E is a valid span
    assert bioe_spans(["B-QUOTE", "I-QUOTE", "E-QUOTE"]) == [("QUOTE", 0, 2)]
    
    # B E is a valid span
    assert bioe_spans(["B-QUOTE", "E-QUOTE"]) == [("QUOTE", 0, 1)]
    
    # Drops orphan I
    assert bioe_spans(["I-QUOTE", "E-QUOTE"]) == []
    assert bioe_spans(["O", "I-QUOTE", "O"]) == []
    
    # Drops unterminated B I*
    assert bioe_spans(["B-QUOTE", "I-QUOTE", "O"]) == []
    assert bioe_spans(["B-QUOTE", "I-QUOTE"]) == []
    assert bioe_spans(["B-QUOTE"]) == []
    
    # Multiple spans
    labels = ["O", "B-QUOTE", "I-QUOTE", "E-QUOTE", "O", "B-QUOTE", "E-QUOTE"]
    assert bioe_spans(labels) == [("QUOTE", 1, 3), ("QUOTE", 5, 6)]


def test_downgrade_to_bio_equivalence():
    # Downgrading a BIOE sequence and running bio_spans equals bioe_spans on the original
    # Only valid BIOE sequences where every span is terminated are equivalent.
    sequences = [
        ["B-QUOTE", "I-QUOTE", "E-QUOTE"],
        ["B-QUOTE", "E-QUOTE"],
        ["O", "B-QUOTE", "I-QUOTE", "E-QUOTE", "O", "B-QUOTE", "E-QUOTE"],
    ]
    for seq in sequences:
        bioe_res = bioe_spans(seq)
        bio_seq = downgrade_to_bio(seq)
        bio_res = reference_bio_spans(bio_seq)
        assert bioe_res == bio_res


# --------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------


def test_stitch_book_logits_and_assertions():
    # Construct synthetic 3-window book
    book_ids = ["book_1", "book_1", "book_1"]
    window_indices = [0, 1, 2]
    
    # 4 classes
    logits = np.zeros((3, 5, 4), dtype=np.float32)
    # Window 0: O, B, I, I, E
    logits[0, :, :] = _one_hot([0, 1, 2, 2, 3])[0]
    # Window 1: -100, -100, O, B, E (first two are -100)
    logits[1, :, :] = _one_hot([0, 0, 0, 1, 3])[0]
    # Window 2: -100, O, O, O, O (first is -100)
    logits[2, :, :] = _one_hot([0, 0, 0, 0, 0])[0]
    
    gold_ids = np.array([
        [0, 1, 2, 2, 3],
        [-100, -100, 0, 1, 3],
        [-100, 0, 0, 0, 0]
    ])
    
    pred_seqs, gold_seqs = stitch_book_logits(
        logits, gold_ids, book_ids, window_indices, SCHEME, break_penalty=0.0
    )
    
    assert len(pred_seqs) == 1
    assert len(gold_seqs) == 1
    
    # Expected concatenated owned tokens:
    # Window 0: 5 tokens
    # Window 1: 3 tokens (indices 2, 3, 4)
    # Window 2: 4 tokens (indices 1, 2, 3, 4)
    # Total owned tokens = 5 + 3 + 4 = 12
    assert len(pred_seqs[0]) == 12
    assert len(gold_seqs[0]) == 12
    
    # Test gap failure (e.g. missing window 1)
    with pytest.raises(ValueError, match="Gap detected"):
        stitch_book_logits(
            logits[:2], gold_ids[:2], ["book_1", "book_1"], [0, 2], SCHEME
        )
        
    # Test length assertion failure
    with pytest.raises(AssertionError, match="does not match owned-token count"):
        stitch_book_logits(
            logits, gold_ids, book_ids, window_indices, SCHEME,
            book_owned_counts={"book_1": 99}  # Expect 12, but we say 99
        )


# --------------------------------------------------------------------------
# e_recall
# --------------------------------------------------------------------------


def test_e_recall_on_known_confusion():
    gold = [["O", "B-QUOTE", "I-QUOTE", "E-QUOTE"]]
    pred = [["O", "B-QUOTE", "I-QUOTE", "I-QUOTE"]]  # Missed E-QUOTE
    assert e_recall(pred, gold) == 0.0
    
    pred_perfect = [["O", "B-QUOTE", "I-QUOTE", "E-QUOTE"]]
    assert e_recall(pred_perfect, gold) == 1.0


# --------------------------------------------------------------------------
# Class Weights
# --------------------------------------------------------------------------


def test_pinned_class_weights_ordering_and_refusal():
    cfg = {
        "O": 0.07,
        "B-QUOTE": 5.0,
        "I-QUOTE": 1.0,
        "E-QUOTE": 5.0,
    }
    weights = pinned_class_weights(cfg, SCHEME)
    assert weights == [0.07, 5.0, 1.0, 5.0]
    
    # Missing key must raise ValueError
    bad_cfg = {
        "O": 0.07,
        "B-QUOTE": 5.0,
        "I-QUOTE": 1.0,
    }
    with pytest.raises(ValueError, match="Missing class weight for label"):
        pinned_class_weights(bad_cfg, SCHEME)

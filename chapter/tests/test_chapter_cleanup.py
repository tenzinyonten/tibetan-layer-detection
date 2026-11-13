"""Unit + frozen-dump tests for Chapter post-Viterbi cleanup.

Unit tests cover the ༼…༽ stub-extend rule and Chapter LayerConfig flags.
If scratch/chapter/chapter_test_spans.jsonl is present, also reconstructs
the same char spans the measurement script used and checks:
  - Chapter rule_flags + apply_linguistic_rules reproduce F1 ~0.8448
  - P000110 and P000275 ༼ stubs extend to an exact gold match
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from detect.decode import (  # noqa: E402
    apply_linguistic_rules,
    extend_to_closing_bracket,
)
from detect.infer import rule_flags  # noqa: E402
from detect.layer_config import LAYERS  # noqa: E402

DUMP = ROOT / "scratch/chapter/chapter_test_spans.jsonl"
CLEAN = ROOT / "chapter/data/processed/chapter_spans_clean.csv"
AUDIT = ROOT / "data/shared/tsawa_audit.csv"
PRED_KEY = "viterbi_bp4.0"
EXPECTED_F1 = 0.8448
F1_TOL = 0.0005
EXACT_BOOKS = ("P000110", "P000275")


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter else 0.0


def merge_overlaps(spans):
    out = []
    for s, e in sorted(spans):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def match(gold, pred, thr=0.5):
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gold) for j, p in enumerate(pred)
                    if iou(g, p) >= thr), reverse=True)
    mg, mp = set(), set()
    for _, i, j in pairs:
        if i not in mg and j not in mp:
            mg.add(i)
            mp.add(j)
    return mg, mp


def score(gold_by_book, pred_by_book):
    tp = fn = fp = 0
    for p, gold in gold_by_book.items():
        pred = pred_by_book.get(p, [])
        mg, mp = match(gold, pred)
        tp += len(mg)
        fn += len(gold) - len(mg)
        fp += len(pred) - len(mp)
    n_gold, n_pred = tp + fn, tp + fp
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fn": fn, "fp": fp}


class TestExtendToClosingBracket(unittest.TestCase):
    def test_short_open_extends_to_just_before_close(self):
        text = "abc༼le'u dummy heading here༽def"
        s = text.index("༼")
        close = text.index("༽")
        out = extend_to_closing_bracket([(s, s + 3, 1.0, 1)], text)
        self.assertEqual(out, [(s, close, 1.0, 1)])
        self.assertEqual(text[out[0][0]:out[0][1]], "༼le'u dummy heading here")

    def test_already_long_span_is_left_alone(self):
        text = "༼abcdefghij more text here༽"
        out = extend_to_closing_bracket([(0, 10, 1.0, 1)], text)
        self.assertEqual(out, [(0, 10, 1.0, 1)])

    def test_no_closer_in_lookahead_is_left_alone(self):
        text = "༼ab" + ("x" * 200)
        out = extend_to_closing_bracket([(0, 3, 1.0, 1)], text)
        self.assertEqual(out, [(0, 3, 1.0, 1)])

    def test_does_not_touch_spans_that_do_not_start_with_open(self):
        text = "x༼short༽"
        out = extend_to_closing_bracket([(0, 3, 1.0, 1)], text)
        self.assertEqual(out, [(0, 3, 1.0, 1)])

    def test_apply_linguistic_rules_grows_stub_before_short_drop(self):
        text = "༼ab longer heading text༽"
        spans, n = apply_linguistic_rules(
            [(0, 3, 1.0, 1)], text,
            extend_to_tsheg=False, merge_gaps=False, extend_brackets=True)
        close = text.index("༽")
        self.assertEqual(n, 0)
        self.assertEqual(spans, [(0, close, 1.0, 1)])

    def test_apply_linguistic_rules_off_by_default(self):
        text = "༼ab longer heading text༽"
        spans, n = apply_linguistic_rules(
            [(0, 3, 1.0, 1)], text,
            extend_to_tsheg=False, merge_gaps=False)
        self.assertEqual(n, 1)
        self.assertEqual(spans, [])


class TestChapterConfig(unittest.TestCase):
    def test_chapter_matches_measured_switches(self):
        cfg = LAYERS["chapter"]
        self.assertFalse(cfg.extend_to_tsheg)
        self.assertFalse(cfg.merge_small_gaps)
        self.assertTrue(cfg.extend_to_closing_bracket)
        self.assertFalse(cfg.repair_fragments)
        self.assertFalse(hasattr(cfg, "drop_sanskrit") and cfg.drop_sanskrit)

    def test_sabche_unchanged(self):
        cfg = LAYERS["sabche"]
        self.assertFalse(cfg.extend_to_tsheg)
        self.assertFalse(cfg.merge_small_gaps)
        self.assertTrue(cfg.repair_fragments)
        self.assertFalse(cfg.extend_to_closing_bracket)

    def test_other_layers_keep_original_defaults(self):
        for name in ("tsawa", "quotation", "yigchung"):
            cfg = LAYERS[name]
            self.assertTrue(cfg.extend_to_tsheg, name)
            self.assertTrue(cfg.merge_small_gaps, name)
            self.assertFalse(cfg.extend_to_closing_bracket, name)

    def test_rule_flags_maps_chapter_bracket(self):
        flags = rule_flags(LAYERS["chapter"])
        self.assertEqual(flags, {
            "extend_to_tsheg": False,
            "merge_gaps": False,
            "repair": False,
            "extend_fragments": False,
            "extend_brackets": True,
        })


@unittest.skipUnless(DUMP.is_file() and CLEAN.is_file() and AUDIT.is_file(),
                     "frozen chapter test dump / gold / audit not on disk")
class TestChapterCleanupOnFrozenDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
        tok.model_max_length = int(1e12)
        audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}

        pred_tok = defaultdict(set)
        books = set()
        for line in DUMP.open(encoding="utf-8"):
            r = json.loads(line)
            books.add(r["pecha_id"])
            for a, b in r[f"{PRED_KEY}_tok"]:
                pred_tok[r["pecha_id"]].add((a, b))

        gold_all = defaultdict(list)
        for r in csv.DictReader(CLEAN.open(encoding="utf-8")):
            if r["dropped"] != "True":
                gold_all[r["pecha_id"]].append((int(r["start"]), int(r["end"])))

        texts, raw, gold = {}, {}, {}
        for p in sorted(books):
            text = Path(audit[p]["base_path"]).read_text(encoding="utf-8")
            texts[p] = text
            offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            raw[p] = merge_overlaps(
                [(offs[a][0], offs[b][1]) for a, b in pred_tok[p] if 0 <= a <= b < len(offs)])
            gold[p] = sorted(gold_all[p])

        flags = rule_flags(LAYERS["chapter"])
        cleaned = {}
        for p in books:
            spans4 = [(s, e, 1.0, 1) for s, e in raw[p]]
            out, _ = apply_linguistic_rules(spans4, texts[p], **flags)
            cleaned[p] = [(s, e) for s, e, _, _ in out]

        cls.texts = texts
        cls.raw = raw
        cls.gold = gold
        cls.cleaned = cleaned
        cls.score = score(gold, cleaned)

    def test_chapter_f1_matches_measured(self):
        self.assertAlmostEqual(self.score["f1"], EXPECTED_F1, delta=F1_TOL,
                               msg=self.score)

    def test_bracket_stubs_exact_on_p000110_and_p000275(self):
        # Measurement: the recovered headings on these two books are exact gold
        # matches after extend (one extra ༼ stub on P000275 is not gold and is
        # left as a false positive — net still +0.12 F1).
        recovered = []
        for p in EXACT_BOOKS:
            text = self.texts[p]
            gold_set = set(self.gold[p])
            for s, e in self.raw[p]:
                if e - s >= 10 or text[s:s + 1] != "༼":
                    continue
                j = text.find("༽", s, min(len(text), s + 150))
                if j != -1 and (s, j) in gold_set:
                    recovered.append(p)
        self.assertEqual(sorted(recovered), ["P000110", "P000275"])


if __name__ == "__main__":
    unittest.main()

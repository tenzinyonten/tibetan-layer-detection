#!/usr/bin/env python3
"""Fit candidate weights on TRAIN (never validation or test).

    python title_author/src/ta_rule_fit.py author [N]
    python title_author/src/ta_rule_fit.py title  [N]
    python title_author/src/ta_rule_fit.py both   [N]

Author weights → ta_rule_weights.json
Title weights  → ta_rule_title_weights.json
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import ta_rule_extract as T

which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("author", "title", "both") else "author"
N = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 6000


def fit_author(rows):
    X, y, names = [], [], None
    for r in rows:
        head, tail = T.region_text(r)
        g_reg, g = T.gold_span(r, "author")
        for region, text in [("head", head)] + ([("tail", tail)] if tail else []):
            for f, i, name in T.author_candidates_v2(text, region, r.whole_work):
                names = names or sorted(f)
                X.append([f[k] for k in names])
                y.append(int(region == g_reg and T.iou((i, i + len(name)), g) >= 0.5))
    return np.array(X), np.array(y), names


def fit_title(rows):
    X, y, names = [], [], None
    hits = 0
    for r in rows:
        head, tail = T.region_text(r)
        g_reg, g = T.gold_span(r, "title")
        ok = False
        for f, i, span, region in T.title_candidates(head, tail):
            names = names or sorted(f)
            hit = int(region == g_reg and T.iou((i, i + len(span)), g) >= 0.5)
            X.append([f[k] for k in names])
            y.append(hit)
            ok = ok or bool(hit)
        hits += int(ok)
    print("title oracle coverage (a candidate IoU>=0.5)", round(hits / len(rows), 3),
          "on", len(rows), "works")
    return np.array(X), np.array(y), names


def dump(kind, X, y, names, path):
    print(kind, "candidates", len(y), "positive", int(y.sum()),
          "pos rate", round(float(y.mean()), 3) if len(y) else 0)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight=None).fit(X, y)
    W = {k: round(float(c), 3) for k, c in zip(names, clf.coef_[0])}
    W["bias"] = round(float(clf.intercept_[0]), 3)
    path.write_text(json.dumps(W, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(W, indent=1))
    print("wrote", path)


rows = list(T.load("train", N, 1).itertuples())
print("works", len(rows))
if which in ("author", "both"):
    dump("author", *fit_author(rows), T.WEIGHTS_FILE)
if which in ("title", "both"):
    dump("title", *fit_title(rows), T.TITLE_WEIGHTS_FILE)

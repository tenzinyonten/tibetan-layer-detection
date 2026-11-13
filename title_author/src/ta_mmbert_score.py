#!/usr/bin/env python3
"""Score the trained mmBERT (saved logits from ta_predict.py, flattened by the converter) with the SAME scorer as the Gemini answers.
Viterbi over 5 labels (O, B-TITLE, I-TITLE, B-AUTHOR, I-AUTHOR), break penalty 12 (picked on validation earlier); one span per field
(title = first span, author = last span, as in the benchmark; "conf" = the most confident span); the token span is decoded to text and located in the
head or tail of the work, so the answer has the same shape as a Gemini answer {title, title_region, author, author_region}.
A gold-label decode is run first as a sanity check of the token-to-text step (it should score ~1.0).

    python3 title_author/src/ta_mmbert_score.py SPLIT [N_PREFIX_WORKS]       SPLIT = validation | test (test: ALLOW_TEST=1)"""
import os, sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, score = _g["load"], _g["score"]
from transformers import AutoTokenizer
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
if SPLIT == "test" and os.environ.get("ALLOW_TEST") != "1": sys.exit("test needs ALLOW_TEST=1 (final comparison only, no tuning)")
PEN = 12.0; NEG = -1e9
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"); SPECIAL = set(tok.all_special_ids)

trans = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        if j == 2 and i not in (1, 2): trans[i, j] = NEG; continue          # I-TITLE only after B-TITLE / I-TITLE
        if j == 4 and i not in (3, 4): trans[i, j] = NEG; continue          # I-AUTHOR only after B-AUTHOR / I-AUTHOR
        cont = 2 if i in (1, 2) else 4 if i in (3, 4) else None
        if i != 0 and j != cont: trans[i, j] -= PEN                          # leaving a span costs the break penalty

def viterbi(lg):
    T = len(lg); dp = np.full((T, 5), NEG); bp = np.zeros((T, 5), dtype=np.int64); dp[0] = lg[0]; dp[0, (2, 4)] = NEG
    for t in range(1, T):
        s = dp[t - 1][:, None] + trans; bp[t] = s.argmax(0); dp[t] = s.max(0) + lg[t]
    p = np.zeros(T, dtype=np.int64); p[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1): p[t - 1] = bp[t, p[t]]
    return p

def spans(path, b, i):
    out, st = [], None
    for k, v in enumerate(path):
        if v == b:
            if st is not None: out.append((st, k - 1))
            st = k
        elif v == i:
            if st is None: st = k
        else:
            if st is not None: out.append((st, k - 1)); st = None
    if st is not None: out.append((st, len(path) - 1))
    return out

F = np.load(ROOT / f"title_author/data/mmbert_preds/{SPLIT}_flat.npz", allow_pickle=False)
off, LG, IDS, LAB, WID = F["offsets"], F["logits"], F["ids"], F["labels"], F["work_id"]
rows = {r.work_id: r for r in load(SPLIT, 0, 0).itertuples()}

def locate(text_ids_span, head, tail, last, boundary, a):
    """decode token span -> text; find it in head/tail. returns (text, region) or None."""
    s = tok.decode(text_ids_span).strip()
    if not s: return None
    cands = []
    for rg, text in (("head", head), ("tail", tail)):
        if text and s in text: cands.append(rg)
    if not cands:
        sq = "".join(s.split());
        for rg, text in (("head", head), ("tail", tail)):
            if text and sq in "".join(text.split()): cands.append(rg)
        if not cands: return None
        import ta_rule_extract as T
        r = T.repair_span(s, head, tail, ("head" if a < boundary else "tail") if len(cands) > 1 else cands[0], last=last)
        return r
    rg = cands[0] if len(cands) == 1 else ("head" if a < boundary else "tail")
    return (s, rg)

N = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
res = {"gold": {"title": 0, "author": 0}, "first_last": {"title": 0, "author": 0}, "conf": {"title": 0, "author": 0}}; n = 0; miss = 0; out_rows = {}
for k, w in enumerate(WID):
    if w not in rows or k >= N: continue
    r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
    lg = LG[off[k]:off[k + 1]].astype(np.float32); ids = IDS[off[k]:off[k + 1]]; lab = LAB[off[k]:off[k + 1]]
    real = np.array([i not in SPECIAL for i in ids]); ri = np.where(real)[0]
    if len(ri) == 0: continue
    lgr, idr, labr = lg[ri], ids[ri], lab[ri]
    nh = len(tok(head, add_special_tokens=False)["input_ids"]) if tail else 10**9
    path = viterbi(lgr); n += 1
    for name, p in (("first_last", path), ("gold", np.where(labr < 0, 0, labr))):
        ans = {}
        for f, b, i, last in (("title", 1, 2, False), ("author", 3, 4, True)):
            sp = spans(p, b, i)
            if not sp: continue
            if name == "first_last": a, z = sp[-1] if last else sp[0]
            else: a, z = sp[-1] if last else sp[0]
            loc = locate(idr[a:z + 1], head, tail, last, nh, a)
            if loc: ans[f], ans[f + "_region"] = loc
            else: miss += 1
        for f in ("title", "author"): res[name][f] += score(r, ans, f)[0] == "ok"
        if name == "first_last": out_rows[w] = ans
    pr = np.exp(lgr - lgr.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); ans = {}
    for f, b, i in (("title", 1, 2), ("author", 3, 4)):
        sp = spans(path, b, i)
        if not sp: continue
        a, z = max(sp, key=lambda s: pr[s[0]:s[1] + 1, [b, i]].sum(1).mean())
        loc = locate(idr[a:z + 1], head, tail, f == "author", nh, a)
        if loc: ans[f], ans[f + "_region"] = loc
    for f in ("title", "author"): res["conf"][f] += score(r, ans, f)[0] == "ok"
print(f"{SPLIT}: {n} works, spans I could not locate in the text: {miss}")
for name in ("gold", "first_last", "conf"):
    print(f"  {name:11} title {res[name]['title']/n:.3f}  author {res[name]['author']/n:.3f}")
json.dump(out_rows, open(ROOT / f"title_author/data/mmbert_preds/{SPLIT}_answers.json", "w"), ensure_ascii=False)

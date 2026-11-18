#!/usr/bin/env python3
"""Score a SINGLE-ENTITY mmBERT model (3 labels: O, B, I) with the same scorer and the same token->text step as ta_mmbert_score.py (the joint model).
Reads data/mmbert_preds/{FIELD}_only/{SPLIT}_flat.npz (flattened logits from ta_predict.py + the converter), decodes with Viterbi at several break penalties,
takes one span per work (title: the FIRST span, author: the LAST span; or the most confident span) and scores it (IoU >= 0.5, same block).

    python3 ta_mmbert_single_score.py FIELD SPLIT PENALTIES [write PENALTY SELECT]
    python3 ta_mmbert_single_score.py author validation 3,6,12,20                  # tune the penalty on validation
    ALLOW_TEST=1 python3 ta_mmbert_single_score.py author test 12 write 12 first_last    # final: ONE penalty chosen on validation, writes test_answers.json
FIELD = title | author. test needs ALLOW_TEST=1 (final comparison only).
PENALTIES = comma-separated break penalties to try (the models on the Hub use 6). SELECT = first_last | conf (which span to keep; the Hub models use first_last).
The work texts and gold spans come from the parquet folder TA_PQ (default title_author/data/processed/parquet_export_strat_both_notrans_nous).
Prints one line per penalty (share of works with IoU >= 0.5). With `write` it saves title_author/data/mmbert_preds/{FIELD}_only/{SPLIT}_answers.json for that penalty."""
import os, sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, score = _g["load"], _g["score"]
from transformers import AutoTokenizer
FIELD, SPLIT = sys.argv[1], sys.argv[2]; PENS = [float(x) for x in sys.argv[3].split(",")]
WRITE = sys.argv[4:] and sys.argv[4] == "write"
if SPLIT == "test" and os.environ.get("ALLOW_TEST") != "1": sys.exit("test needs ALLOW_TEST=1 (final comparison only, no tuning)")
LAST = FIELD == "author"
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"); SPECIAL = set(tok.all_special_ids)
NEG = -1e9

def trans_for(pen):
    t = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if j == 2 and i not in (1, 2): t[i, j] = NEG; continue          # I only after B or I
            if i != 0 and j != (2 if i in (1, 2) else None): t[i, j] -= pen  # leaving a span costs the break penalty
    return t

def viterbi(lg, trans):
    T = len(lg); dp = np.full((T, 3), NEG); bp = np.zeros((T, 3), dtype=np.int64); dp[0] = lg[0]; dp[0, 2] = NEG
    for t in range(1, T):
        s = dp[t - 1][:, None] + trans; bp[t] = s.argmax(0); dp[t] = s.max(0) + lg[t]
    p = np.zeros(T, dtype=np.int64); p[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1): p[t - 1] = bp[t, p[t]]
    return p

def spans(path):
    out, st = [], None
    for k, v in enumerate(path):
        if v == 1:
            if st is not None: out.append((st, k - 1))
            st = k
        elif v == 2:
            if st is None: st = k
        elif st is not None: out.append((st, k - 1)); st = None
    if st is not None: out.append((st, len(path) - 1))
    return out

def locate(ids_span, head, tail, last, boundary, a):
    """token span -> text, found in the head or tail (same logic as the joint model's script)"""
    s = tok.decode(ids_span).strip()
    if not s: return None
    cands = [rg for rg, text in (("head", head), ("tail", tail)) if text and s in text]
    if not cands:
        sq = "".join(s.split())
        for rg, text in (("head", head), ("tail", tail)):
            if text and sq in "".join(text.split()): cands.append(rg)
        if not cands: return None
        import ta_rule_extract as T
        return T.repair_span(s, head, tail, ("head" if a < boundary else "tail") if len(cands) > 1 else cands[0], last=last)
    return (s, cands[0] if len(cands) == 1 else ("head" if a < boundary else "tail"))

F = np.load(ROOT / f"title_author/data/mmbert_preds/{FIELD}_only/{SPLIT}_flat.npz", allow_pickle=False)
off, LG, IDS, WID = F["offsets"], F["logits"], F["ids"], F["work_id"]
rows = {r.work_id: r for r in load(SPLIT, 0, 0).itertuples()}
print(f"{FIELD}-only model, {SPLIT}: {len(WID)} works, penalties {PENS}", flush=True)
for pen in PENS:
    trans = trans_for(pen); ok = {"first_last": 0, "conf": 0}; n = miss = 0; answers = {"first_last": {}, "conf": {}}
    for k, w in enumerate(WID):
        if w not in rows: continue
        r = rows[w]; head = r.text_head; tail = "" if r.whole_work else (r.text_tail or "")
        lg = LG[off[k]:off[k + 1]].astype(np.float32); ids = IDS[off[k]:off[k + 1]]
        ri = np.where(np.array([i not in SPECIAL for i in ids]))[0]
        if len(ri) == 0: continue
        lgr, idr = lg[ri], ids[ri]; nh = len(tok(head, add_special_tokens=False)["input_ids"]) if tail else 10**9
        path = viterbi(lgr, trans); sp = spans(path); n += 1
        pr = np.exp(lgr - lgr.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        for sel in ("first_last", "conf"):
            ans = {}
            if sp:
                a, z = (sp[-1] if LAST else sp[0]) if sel == "first_last" else max(sp, key=lambda s: pr[s[0]:s[1] + 1, [1, 2]].sum(1).mean())
                loc = locate(idr[a:z + 1], head, tail, LAST, nh, a)
                if loc: ans[FIELD], ans[FIELD + "_region"] = loc
                else: miss += 1
            ok[sel] += score(r, ans, FIELD)[0] == "ok"; answers[sel][w] = ans
    print(f"  penalty {pen:5.1f}: {FIELD} first_last {ok['first_last']/n:.3f}   conf {ok['conf']/n:.3f}   (works {n}, spans not located in the text {miss})", flush=True)
    if WRITE and pen == float(sys.argv[5]):
        sel = sys.argv[6]; outp = ROOT / f"title_author/data/mmbert_preds/{FIELD}_only/{SPLIT}_answers.json"
        outp.write_text(json.dumps(answers[sel], ensure_ascii=False), encoding="utf-8"); print("  wrote", outp.name, f"(penalty {pen}, {sel})")

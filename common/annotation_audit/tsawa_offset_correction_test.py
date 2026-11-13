"""Exploratory: correct tsawa offsets in the Sabche-realigned test books and re-score the saved v6 test predictions.
Read-only on data/ and scratch/. Writes tsawa_offset_correction_test.md and tsawa_offset_corrected_spans.csv here.

Correction rule (per span, no labels from the model): a clean tsawa span in this corpus starts at a line start and ends right
before a shad or newline (checked on 10 clean test books: start ok 97-100%, end ok 86-99%). For each span in a suspect book
choose the shift k in [-K, K] that maximises  2*start_ok + 2*end_ok + share of 7/9/11/13-syllable clauses  - |k|/10000
(ties go to the smaller shift). The span length is unchanged.
Scoring: the saved window-level predictions (scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl, Viterbi bp 4.0, the file
behind the reported test F1); gold per window is rebuilt as the book's gold spans clipped to the window's character range, and
the same rebuild is used for original and corrected gold so the two are comparable."""
import glob, json, re, collections
import numpy as np, pandas as pd

OUT = "common/annotation_audit"; K = 200
SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}
SUSPECT = ["P000027", "P000083", "P000144", "P000242", "P000269"]
def base(p): return open([x for x in glob.glob(f"data/raw_opf/{p}.opf/*") if x.endswith(".opf")][0] + "/base/v001.txt", encoding="utf-8").read()
def syl(s): return len([x for x in s.split("་") if x.strip()])
def metre(t):
    c, s = [], 0
    for m in re.finditer(f"[{SH}]", t):
        n = syl(t[s:m.start()]); s = m.end()
        if n: c.append(n)
    n = syl(t[s:])
    if n: c.append(n)
    return sum(x in ISO for x in c) / len(c) if c else 0.0
start_ok = lambda b, s: s == 0 or b[s - 1] == "\n"
end_ok = lambda b, e: e < len(b) and b[e] in ("།", "\n")
def best_shift(b, s, e):
    best = (-9, 0)
    for k in range(-K, K + 1):
        s2, e2 = s + k, e + k
        if s2 < 0 or e2 > len(b): continue
        v = 2 * start_ok(b, s2) + 2 * end_ok(b, e2) + metre(b[s2:e2]) - abs(k) / 10000
        if v > best[0]: best = (v, k)
    return best[1]

T = pd.read_csv("tsawa/data/processed/tsawa_spans_merged.csv"); T = T[~T.dropped.astype(bool)]
split = pd.read_csv("data/shared/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
test_books = sorted(p for p, s in split.items() if s == "test")
CONTROL = [p for p in test_books if p not in SUSPECT and p != "IF3ACC3E1"]  # IF3ACC3E1: 747 short word-fragment spans, not verse
# --- diagnostics: how often does the search move spans in clean books (false-move rate) vs suspect books
rows, corrected = [], {}
for p in SUSPECT + CONTROL:
    b = base(p); g = T[T.pecha_id == p].sort_values("start")
    ks = [best_shift(b, int(s), int(e)) for s, e in zip(g.start, g.end)]
    new = [(int(s) + k, int(e) + k) for (s, e), k in zip(zip(g.start, g.end), ks)]
    corrected[p] = (list(zip(map(int, g.start), map(int, g.end))), new)
    so0 = np.mean([start_ok(b, int(s)) for s in g.start]); eo0 = np.mean([end_ok(b, int(e)) for e in g.end])
    so1 = np.mean([start_ok(b, s) for s, _ in new]); eo1 = np.mean([end_ok(b, e) for _, e in new])
    kk = collections.Counter(ks)
    rows.append(dict(pecha_id=p, group="suspect" if p in SUSPECT else "control", spans=len(g), moved=sum(k != 0 for k in ks), moved_pct=round(100 * np.mean([k != 0 for k in ks]), 1),
                     start_ok_before=round(so0, 2), start_ok_after=round(so1, 2), end_ok_before=round(eo0, 2), end_ok_after=round(eo1, 2),
                     median_shift_of_moved=int(np.median([k for k in ks if k])) if any(ks) else 0, common_shifts=dict(kk.most_common(4))))
diag = pd.DataFrame(rows)
pd.DataFrame([dict(pecha_id=p, old_start=o[0], old_end=o[1], new_start=n[0], new_end=n[1], shift=n[0] - o[0]) for p, (olds, news) in corrected.items() if p in SUSPECT for o, n in zip(olds, news)]).to_csv(f"{OUT}/tsawa_offset_corrected_spans.csv", index=False)

# --- scoring (token space, like the reported evaluation)
# The saved prediction file stores spans in "scored index" space (token positions inside the window, ignore positions dropped).
# For these test books nothing is ignored (v6_ignore_spans.csv has no test books), so scored index = book token - token_start.
import sys
sys.path.insert(0, "common")
from build_tsawa_dataset import label_tokens           # the dataset builder's own labelling rule
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
ign = pd.read_csv("tsawa/data/processed/v6_ignore_spans.csv")
assert not (set(ign.pecha_id) & set(test_books)), "ignore spans touch test books"

def spans_from_labels(lab):  # inclusive token spans, a span may start at an I label (window cut)
    out, st = [], None
    for i, v in enumerate(lab):
        if v == 0:
            if st is not None: out.append((st, i - 1)); st = None
        elif v == 1:
            if st is not None: out.append((st, i - 1))
            st = i
        else:
            if st is None: st = i
    if st is not None: out.append((st, len(lab) - 1))
    return out
OFFS = {}
def gold_tokens(pid, char_spans):
    if pid not in OFFS:
        OFFS[pid] = [tuple(x) for x in tok(base(pid), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    sp = sorted((s, e, "x") for s, e in char_spans)
    return spans_from_labels(label_tokens(OFFS[pid], sp, {"O": 0, "B-TSAWA": 1, "I-TSAWA": 2}) if False else
                             label_tokens(OFFS[pid], [(s, e, "x", "TSAWA") for s, e, _ in sp], {"O": 0, "B-TSAWA": 1, "I-TSAWA": 2}))

def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo: return 0.0
    i = hi - lo + 1; return i / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - i)
def match(gold, pred, thr=0.5):
    c = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred) if iou(g, p) >= thr), reverse=True)
    ug, up, n = set(), set(), 0
    for _, gi, pi in c:
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi); n += 1
    return n
W = [json.loads(l) for l in open("scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl", encoding="utf-8")]
def f1(tp, npred, ng):
    p = tp / npred if npred else 0; r = tp / ng if ng else 0
    return (2 * p * r / (p + r) if p + r else 0), p, r
allT = {p: list(zip(map(int, g.start), map(int, g.end))) for p, g in T.groupby("pecha_id")}
def window_gold(w, doc_gold):
    t0, n = int(w["token_start"]), int(w["n_scored"]); out = []
    for a, b in doc_gold:
        lo, hi = max(a, t0), min(b, t0 + n - 1)
        if hi >= lo: out.append((lo - t0, hi - t0))
    return out
def score(char_spans_by_book, books=None):
    DG = {p: gold_tokens(p, sp) for p, sp in char_spans_by_book.items()}
    tp = npd = ng = 0; per = collections.defaultdict(lambda: [0, 0, 0]); mism = 0
    for w in W:
        p = w["pecha_id"]
        if books and p not in books: continue
        G = window_gold(w, DG[p]); P = [tuple(x) for x in w["viterbi_bp4.0"]]
        if char_spans_by_book is ORIG_ALL and sorted(G) != sorted(tuple(x) for x in w["gold"]): mism += 1
        m = match(G, P); tp += m; npd += len(P); ng += len(G)
        q = per[p]; q[0] += m; q[1] += len(P); q[2] += len(G)
    return f1(tp, npd, ng), (tp, npd, ng), per, mism
ORIG_ALL = {p: allT[p] for p in test_books}
def with_(variant):   # variant: dict book -> corrected char spans
    d = dict(ORIG_ALL); d.update(variant); return d
# gold exactly as saved
tp = npd = ng = 0
for w in W:
    G = [tuple(x) for x in w["gold"]]; P = [tuple(x) for x in w["viterbi_bp4.0"]]
    tp += match(G, P); npd += len(P); ng += len(G)
f_file = f1(tp, npd, ng); c_file = (tp, npd, ng)
f_orig, c_orig, per_orig, mism = score(ORIG_ALL)
# variant A: shift only. variant B: shift, then move the start to the nearest line start within 40 characters
def snap_start(b, s, e):
    for d_ in range(0, 41):
        for c_ in (s - d_, s + d_):
            if 0 <= c_ < e and start_ok(b, c_): return c_
    return s
VA = {p: corrected[p][1] for p in SUSPECT}
VB = {p: [(snap_start(base(p), s, e), e) for s, e in corrected[p][1]] for p in SUSPECT}
res = {}
for name, v in (("A: shift only", VA), ("B: shift + snap start to line start", VB)):
    f_all, c_all, per_c, _ = score(with_(v)); f_s, c_s, _, _ = score(with_(v), set(SUSPECT))
    res[name] = (f_all, c_all, f_s, c_s, per_c, v)
f_orig_s, c_orig_s, _, _ = score(ORIG_ALL, set(SUSPECT))
pb = []
for p in SUSPECT:
    r = {"pecha_id": p, "pred": per_orig[p][1], "gold_original": per_orig[p][2], "tp_original": per_orig[p][0], "F1_original": round(f1(*per_orig[p][0:3][::1])[0] if False else f1(per_orig[p][0], per_orig[p][1], per_orig[p][2])[0], 3)}
    for name, (fa, ca, fs, cs, per_c, v) in res.items():
        k = name[0]; r[f"gold_{k}"] = per_c[p][2]; r[f"tp_{k}"] = per_c[p][0]; r[f"F1_{k}"] = round(f1(per_c[p][0], per_c[p][1], per_c[p][2])[0], 3)
    pb.append(r)
pb = pd.DataFrame(pb)
for name, (fa, ca, fs, cs, per_c, v) in res.items():
    so = np.mean([start_ok(base(p), s) for p in SUSPECT for s, e in v[p]]); eo = np.mean([end_ok(base(p), e) for p in SUSPECT for s, e in v[p]])
    res[name] = res[name] + (so, eo)

def table(df):
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
L = ["# Tsawa test labels: offset correction test\n",
     "Exploratory. Read-only. Script: `tsawa_offset_correction_test.py`; corrected spans (variant A) for the suspect books: `tsawa_offset_corrected_spans.csv`.\n",
     "## Method\n",
     f"- Suspect books: {', '.join(SUSPECT)}. The first four are the test books whose Sabche offsets were realigned; `P000269` is added because its tsawa spans fit the text as badly (start ok 31%, below). Control: the other {len(CONTROL)} verse tsawa test books (`IF3ACC3E1` left out: 747 short word-fragment spans, not verse).",
     "- In clean books a tsawa span starts at a line start and ends right before a shad or newline. The correction picks, per span, the shift in [-200, 200] with the best line-start and line-end fit plus verse metre (details in the script docstring). Span length is unchanged (variant A). Variant B also moves each start to the nearest line start within 40 characters.",
     "- Scoring: the saved window-level v6 predictions (Viterbi, break penalty 4.0, token space), IoU >= 0.5, one-to-one matching, as in the reported test F1. Gold per window is rebuilt from character spans with the dataset builder's own `label_tokens` and the mmBERT tokenizer, and the same rebuild is used for original and corrected spans.\n",
     "## Do the labels fit the text? Line-start / line-end fit before and after the shift\n",
     table(diag.drop(columns=["common_shifts"])),
     "\nMost common shifts per suspect book (shift: spans): " + "; ".join(f"`{r.pecha_id}` {r.common_shifts}" for r in diag[diag.group == 'suspect'].itertuples()) + ".\n",
     "## Test F1\n",
     f"- Windows where the rebuilt gold (original spans) differs from the saved gold: {mism} of {len(W)}. Saved gold gives F1 {f_file[0]:.4f}.",
     table(pd.DataFrame([{"gold": "saved window gold (as reported)", "scope": "all 22 test books", "F1": round(f_file[0], 4), "precision": round(f_file[1], 4), "recall": round(f_file[2], 4), "tp": c_file[0], "pred": c_file[1], "gold spans": c_file[2]},
                         {"gold": "rebuilt, original spans", "scope": "all 22 test books", "F1": round(f_orig[0], 4), "precision": round(f_orig[1], 4), "recall": round(f_orig[2], 4), "tp": c_orig[0], "pred": c_orig[1], "gold spans": c_orig[2]}] +
                        [{"gold": f"rebuilt, suspect books corrected ({n})", "scope": "all 22 test books", "F1": round(r[0][0], 4), "precision": round(r[0][1], 4), "recall": round(r[0][2], 4), "tp": r[1][0], "pred": r[1][1], "gold spans": r[1][2]} for n, r in res.items()] +
                        [{"gold": "rebuilt, original spans", "scope": f"the {len(SUSPECT)} suspect books", "F1": round(f_orig_s[0], 4), "precision": round(f_orig_s[1], 4), "recall": round(f_orig_s[2], 4), "tp": c_orig_s[0], "pred": c_orig_s[1], "gold spans": c_orig_s[2]}] +
                        [{"gold": f"rebuilt, corrected ({n})", "scope": f"the {len(SUSPECT)} suspect books", "F1": round(r[2][0], 4), "precision": round(r[2][1], 4), "recall": round(r[2][2], 4), "tp": r[3][0], "pred": r[3][1], "gold spans": r[3][2]} for n, r in res.items()])),
     "\nLine-start / line-end fit of the corrected spans in the suspect books: " + "; ".join(f"{n}: start ok {r[6]:.2f}, end ok {r[7]:.2f}" for n, r in res.items()) + ".\n",
     "Per suspect book:\n", table(pb)]
open(f"{OUT}/tsawa_offset_correction_test.md", "w", encoding="utf-8").write("\n".join(L))
print(diag.drop(columns=["common_shifts"]).to_string(index=False))
print("rebuild mismatches:", mism, "/", len(W))
print(f"{'saved gold':34} F1 {f_file[0]:.4f} P {f_file[1]:.4f} R {f_file[2]:.4f} tp {c_file[0]} pred {c_file[1]} gold {c_file[2]}")
print(f"{'rebuilt original':34} F1 {f_orig[0]:.4f} P {f_orig[1]:.4f} R {f_orig[2]:.4f} tp {c_orig[0]} pred {c_orig[1]} gold {c_orig[2]}")
for n, r in res.items(): print(f"{'corrected '+n[:1]:34} F1 {r[0][0]:.4f} P {r[0][1]:.4f} R {r[0][2]:.4f} tp {r[1][0]} pred {r[1][1]} gold {r[1][2]} | suspect-only F1 {r[2][0]:.4f} (orig {f_orig_s[0]:.4f}) start_ok {r[6]:.2f} end_ok {r[7]:.2f}")
print(pb.to_string(index=False))

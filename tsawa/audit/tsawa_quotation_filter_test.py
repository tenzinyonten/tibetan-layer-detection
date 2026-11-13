"""Would dropping predicted tsawa spans that overlap a Quotation/Citation span help? Read-only post-processing study.
Predictions: scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl (v6, Viterbi bp 4.0, window level, token space).
Gold: the patched test set scratch/tsawa/analysis/formatting-tsawa-v6_test_patched_full22_final (reproduces P 0.629 / R 0.921 / F1 0.748).
Quotation + Citation spans come from the books' own layer files (gold annotations = an upper bound for what a quotation detector could give).
Writes tsawa_quotation_filter_test.md and tsawa_quotation_filter_by_book.csv here. Run from the repo root."""
import glob, json, collections
import numpy as np, pandas as pd, yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

OUT = "tsawa/audit"; R = "data/raw_opf"
W = {(w["pecha_id"], int(w["window_index"])): w for w in map(json.loads, open("scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl", encoding="utf-8"))}
ds = load_from_disk("scratch/tsawa/analysis/formatting-tsawa-v6_test_patched_full22_final")
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)

def opf(p): return [x for x in glob.glob(f"{R}/{p}.opf/*") if x.endswith(".opf")][0]
def layer(p, n):
    f = f"{opf(p)}/layers/v001/{n}.yml"
    try: y = (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("annotations") or {}
    except FileNotFoundError: return []
    return sorted((a["span"]["start"], a["span"]["end"]) for a in (y if isinstance(y, list) else y.values()))
OFF, QS = {}, {}
def book(p):
    if p not in OFF:
        OFF[p] = tok(open(opf(p) + "/base/v001.txt", encoding="utf-8").read(), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        QS[p] = layer(p, "Quotation") + layer(p, "Citation")
    return OFF[p], QS[p]

def spans_of(seq):
    out, st = [], None
    for i, v in enumerate(seq):
        if v == 1:
            if st is not None: out.append((st, i - 1))
            st = i
        elif v == 2:
            if st is None: st = i
        elif st is not None: out.append((st, i - 1)); st = None
    if st is not None: out.append((st, len(seq) - 1))
    return out
def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo: return 0.0
    i = hi - lo + 1; return i / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - i)
def match(g, p):
    c = sorted(((iou(a, b), gi, pi) for gi, a in enumerate(g) for pi, b in enumerate(p) if iou(a, b) >= .5), reverse=True)
    ug, up = set(), set()
    for _, gi, pi in c:
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi)
    return up

# ---- one record per predicted span
recs = []; GOLD = {}
pid = ds["pecha_id"]; wi = ds["window_index"]; labs = ds["labels"]
for i in range(len(ds)):
    w = W[(pid[i], int(wi[i]))]; lab = np.array(labs[i]); G = spans_of(lab[lab != -100].tolist())
    P = [tuple(x) for x in w["viterbi_bp4.0"]]; PT = [tuple(x) for x in w["viterbi_bp4.0_tok"]]
    GOLD[(pid[i], int(wi[i]))] = (G, P)
    up = match(G, P); off, Q = book(pid[i])
    for k, (a, b) in enumerate(PT):
        s, e = off[a][0], off[b][1]
        cov = sum(max(0, min(e, qe) - max(s, qs)) for qs, qe in Q) / max(1, e - s)
        miou = max([max(0, min(e, qe) - max(s, qs)) / max(1, (e - s) + (qe - qs) - max(0, min(e, qe) - max(s, qs))) for qs, qe in Q], default=0.0)
        recs.append(dict(pecha_id=pid[i], window=int(wi[i]), k=k, tp=k in up, cov=min(cov, 1.0), max_iou=miou, chars=e - s, has_quotes=bool(Q)))
df = pd.DataFrame(recs)

def score(keep):   # re-run the greedy matching on the kept predictions
    tp = npd = ng = 0
    kd = df[keep].groupby(["pecha_id", "window"]).k.apply(set).to_dict()
    for key, (G, P) in GOLD.items():
        ks = kd.get(key, set()); Pk = [p for j, p in enumerate(P) if j in ks]
        tp += len(match(G, Pk)); npd += len(Pk); ng += len(G)
    pr = tp / npd if npd else 0; rc = tp / ng
    return dict(tp=tp, pred=npd, fp=npd - tp, fn=ng - tp, P=pr, R=rc, F1=2 * pr * rc / (pr + rc) if pr + rc else 0)

base = score(pd.Series(True, index=df.index))
fp, tp = df[~df.tp], df[df.tp]
def share(d, mask): return f"{int(mask.sum())} ({100 * mask.mean():.1f}%)" if len(d) else "0"
rows = []
for name, m in (("any overlap (coverage > 0)", lambda d: d["cov"] > 0), ("at least 25% of the span is quotation", lambda d: d["cov"] >= .25),
                ("at least 50% of the span is quotation", lambda d: d["cov"] >= .5), ("at least 90% of the span is quotation", lambda d: d["cov"] >= .9),
                ("IoU >= 0.5 with one Quotation/Citation span", lambda d: d["max_iou"] >= .5)):
    rows.append({"overlap with a Quotation/Citation span": name, f"false positives ({len(fp)})": share(fp, m(fp)), f"true positives ({len(tp)})": share(tp, m(tp))})
ov = pd.DataFrame(rows)
sim = []
for kind, col, ths in (("coverage", "cov", (0.01, .25, .5, .75, .9)), ("IoU with one span", "max_iou", (.3, .5, .7))):
    for t in ths:
        keep = ~(df[col] >= t); r = score(keep); drop = df[~keep]
        sim.append({"rule: drop prediction if": f"{kind} >= {t}", "dropped": len(drop), "FPs removed": int((~drop.tp).sum()), "TPs lost": int(drop.tp.sum()),
                    "P": round(r["P"], 3), "R": round(r["R"], 3), "F1": round(r["F1"], 3), "dF1": round(r["F1"] - base["F1"], 3)})
sim = pd.DataFrame(sim)
byb = df.groupby("pecha_id").apply(lambda d: pd.Series(dict(preds=len(d), fp=int((~d.tp).sum()), fp_cov50=int(((~d.tp) & (d["cov"] >= .5)).sum()), tp=int(d.tp.sum()), tp_cov50=int((d.tp & (d["cov"] >= .5)).sum()), has_quotation_layer=bool(d.has_quotes.iloc[0])))).reset_index()
byb.sort_values("fp_cov50", ascending=False).to_csv(f"{OUT}/tsawa_quotation_filter_by_book.csv", index=False)

def table(d):
    return "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in d.itertuples(index=False))
nq = int(df.groupby("pecha_id").has_quotes.first().sum())
L = ["# Quotation filter for tsawa predictions: read-only check\n",
     "Predictions are window-level (neighbouring windows overlap, so a span can be counted in two windows, as in the reported score). A prediction is a true positive when it matches a gold span at IoU >= 0.5 (one-to-one, greedy). Quotation/Citation spans are the books' own gold layer spans.\n",
     f"Baseline: {base['tp']} TP, {base['fp']} FP, {base['fn']} FN; P {base['P']:.3f}, R {base['R']:.3f}, F1 {base['F1']:.3f}. {nq} of the 22 test books have a Quotation or Citation layer.\n",
     "## 1 and 2: how many false positives and true positives overlap quotations\n", table(ov),
     "\n## 3: simulated filter (predictions dropped, matching re-run)\n", table(sim),
     "\n## By book (quotation overlap >= 50% of the span)\n", table(byb.sort_values("fp_cov50", ascending=False).head(12))]
open(f"{OUT}/tsawa_quotation_filter_test.md", "w", encoding="utf-8").write("\n".join(L))
print(f"baseline tp {base['tp']} fp {base['fp']} fn {base['fn']} P {base['P']:.3f} R {base['R']:.3f} F1 {base['F1']:.3f} | books with a quotation/citation layer {nq}/22")
print(ov.to_string(index=False)); print(sim.to_string(index=False)); print(byb.sort_values("fp_cov50", ascending=False).head(10).to_string(index=False))

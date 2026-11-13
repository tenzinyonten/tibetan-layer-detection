"""For the real tsawa predictions the predicted-quotation filter drops (coverage >= 0.5): is there a GOLD quotation at that spot?
Reuses scratch/postprocess_round2/score_pred_quotes.py (same windows, same patched gold, same predicted quotes), read-only.
Writes tsawa_dropped_tp_gold_quote_check.md and .csv here. Run from the repo root."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, yaml
sys.path.insert(0, "scratch/postprocess_round2"); sys.path.insert(0, ".")
import score_pred_quotes as S
from transformers import AutoTokenizer

THR = 0.5; OUT = "common/annotation_audit"
books = sorted({json.loads(l)["pecha_id"] for l in S.DUMP.open(encoding="utf-8")})
quotes = S.load_quotes(books)                                        # predicted quotation spans (chars)
rows, recs = S.window_level(quotes)
base, filt = rows[0][1], rows[2][1]
print(f"baseline right {base['right']}, filter >= {THR} right {filt['right']}  (net TP lost {base['right'] - filt['right']}); predictions dropped {rows[2][2]}")

W = {(w["pecha_id"], int(w["window_index"])): w for w in map(json.loads, S.DUMP.open(encoding="utf-8"))}
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True); tok.model_max_length = int(1e12)
OFF, TXT, GQ = {}, {}, {}
def book(p):
    if p not in OFF:
        TXT[p] = Path(S.AUDIT[p]["base_path"]).read_text(encoding="utf-8")
        OFF[p] = tok(TXT[p], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        g = {}
        for n in ("Quotation", "Citation"):
            f = Path(S.AUDIT[p]["base_path"]).parents[1] / "layers" / "v001" / f"{n}.yml"
            try: y = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("annotations") or {}
            except FileNotFoundError: y = {}
            g[n] = S.merge_overlaps([(a["span"]["start"], a["span"]["end"]) for a in (y if isinstance(y, list) else y.values())])
        GQ[p] = g
    return OFF[p], TXT[p], GQ[p]
def inter_len(a, B): return sum(max(0, min(a[1], e) - max(a[0], s)) for s, e in B)
def isect(a, B): return [(max(a[0], s), min(a[1], e)) for s, e in B if min(a[1], e) > max(a[0], s)]

out = []
for r in recs:
    if not (r["tp"] and r["cov"] >= THR): continue
    w = W[(r["pecha_id"], r["window"])]; a, b = w["viterbi_bp4.0_tok"][r["k"]]
    off, text, gq = book(r["pecha_id"]); span = (off[a][0], off[b][1]); L = max(1, span[1] - span[0])
    pq = S.merge_overlaps(quotes[r["pecha_id"]])
    pq_in = isect(span, pq)                                          # where the predicted quotation covers this tsawa span
    pq_len = sum(e - s for s, e in pq_in)
    row = dict(pecha_id=r["pecha_id"], window=r["window"], start=span[0], end=span[1], chars=L, pred_quote_cov=round(r["cov"], 2), text=text[span[0]:span[1]].replace("\n", "⏎")[:60])
    for name, G in (("quote+citation", S.merge_overlaps(gq["Quotation"] + gq["Citation"])), ("quotation only", gq["Quotation"])):
        in_gold = sum(inter_len(x, G) for x in pq_in)                # part of the predicted-quote overlap that lies in gold quotation
        row[f"gold_cov_of_pred_overlap[{name}]"] = round(in_gold / max(1, pq_len), 2)
        row[f"gold_cov_of_span[{name}]"] = round(inter_len(span, G) / L, 2)
    out.append(row)
df = pd.DataFrame(out); df.to_csv(f"{OUT}/tsawa_dropped_tp_gold_quote_check.csv", index=False)
n = len(df)
def cls(v): return "real gold quotation (>= 50% of the overlap lies in a gold quotation)" if v >= .5 else ("partly (0 < share < 50%)" if v > 0 else "no gold quotation at all (false quotation prediction)")
tab = {}
for name in ("quote+citation", "quotation only"):
    c = Counter(cls(v) for v in df[f"gold_cov_of_pred_overlap[{name}]"]); tab[name] = c
print("dropped true-positive predictions:", n, "| distinct (book, start, end):", len(df.drop_duplicates(["pecha_id", "start", "end"])))
for name, c in tab.items(): print(name, dict(c))
anyov = {name: int((df[f"gold_cov_of_span[{name}]"] > 0).sum()) for name in tab}; print("any overlap of the tsawa span with gold:", anyov)
print(df.groupby("pecha_id").size().sort_values(ascending=False).head(8).to_dict())
fals = df[df["gold_cov_of_pred_overlap[quote+citation]"] == 0]
print("examples with NO gold quotation:"); print(fals.head(8)[["pecha_id", "start", "end", "chars", "pred_quote_cov", "text"]].to_string(index=False))
real = df[df["gold_cov_of_pred_overlap[quote+citation]"] >= .5]
print("examples with a real gold quotation:"); print(real.head(6)[["pecha_id", "start", "end", "chars", "pred_quote_cov", "text"]].to_string(index=False))
L = ["# Dropped true-positive tsawa spans vs the gold Quotation layer\n",
     f"Filter: drop a predicted tsawa span when predicted quotations cover at least {THR:.0%} of it (window level, patched test gold). The filter takes the true-positive count from {base['right']} to {filt['right']} (net {base['right'] - filt['right']} lost). The same windows hold {n} baseline true-positive predictions that the rule drops ({len(df.drop_duplicates(['pecha_id', 'start', 'end']))} distinct spans); the net figure is lower because a dropped duplicate can leave its gold span matched by another prediction.\n",
     "For each dropped span, the part covered by the predicted quotation is checked against the gold Quotation + Citation layer (and Quotation alone).\n",
     "| gold quotation at that spot | quotation + citation | quotation only |\n|---|---:|---:|\n" + "\n".join(f"| {k} | {tab['quote+citation'].get(k, 0)} | {tab['quotation only'].get(k, 0)} |" for k in [cls(1), cls(.25), cls(0)]),
     f"\nSpans whose own text overlaps any gold quotation at all: {anyov['quote+citation']} of {n} (quotation + citation), {anyov['quotation only']} (quotation only).\n"]
open(f"{OUT}/tsawa_dropped_tp_gold_quote_check.md", "w", encoding="utf-8").write("\n".join(L))

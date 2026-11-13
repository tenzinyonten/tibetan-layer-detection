"""Offset-drift scan of the tsawa train and val books (same detection as the test analysis). Read-only.
Start ok = the span starts at a line start (previous character is a newline); end ok = the character after the span is a shad or newline.
Clean books: start ok 97-100%. Flag < 60%.  Writes tsawa_drift_train_val.csv next to this file."""
import glob, re
import numpy as np, pandas as pd

OUT = "common/annotation_audit"; SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}
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
T = pd.read_csv("tsawa/data/processed/tsawa_spans_merged.csv"); T = T[~T.dropped.astype(bool)]
split = pd.read_csv("data/shared/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
SV = pd.read_csv("sabche/data/processed/sabche_book_verdicts.csv").set_index("pecha_id")
rows = []
for pid, g in T.groupby("pecha_id"):
    sp = split.get(pid)
    if sp is None: continue
    b = base(pid); S = list(zip(g.start.astype(int), g.end.astype(int)))
    so = [s == 0 or b[s - 1] == "\n" for s, e in S]; eo = [e < len(b) and b[e] in ("།", "\n") for s, e in S]
    def m(k):
        c = [metre(b[s + k:e + k]) for s, e in S if s + k >= 0 and e + k <= len(b)]
        return 100 * float(np.mean(c)) if c else 0.0
    best = max(range(-60, 61), key=m) if np.mean(so) < 0.6 else 0
    rows.append(dict(pecha_id=pid, split=sp, batch=g.batch.iloc[0], spans=len(S), start_ok_pct=round(100 * np.mean(so), 1), end_ok_pct=round(100 * np.mean(eo), 1),
                     spans_not_at_line_start=int(len(S) - sum(so)), sabche_realigned=bool(SV.realigned.get(pid, False)) if pid in SV.index else None,
                     metre_shift0=round(m(0), 1), best_shift_pm60=best, metre_best=round(m(best), 1)))
df = pd.DataFrame(rows).sort_values(["split", "start_ok_pct"])
df["flag"] = df.start_ok_pct < 60
df.to_csv(f"{OUT}/tsawa_drift_train_val.csv", index=False)
tv = df[df.split.isin(["train", "val"])]
print("train/val books:", len(tv), "| flagged (<60%):", int(tv.flag.sum()))
pd.set_option("display.width", 250)
print(tv[tv.flag][["pecha_id", "split", "batch", "spans", "start_ok_pct", "end_ok_pct", "spans_not_at_line_start", "sabche_realigned", "metre_shift0", "best_shift_pm60", "metre_best"]].to_string(index=False))
for sp in ("train", "val"):
    x = tv[tv.split == sp]; f = x[x.flag]
    print(f"{sp}: books {len(x)}, flagged {len(f)}, realigned among flagged {int(f.sabche_realigned.sum())}, spans total {x.spans.sum()}, spans in flagged books {f.spans.sum()} ({100*f.spans.sum()/x.spans.sum():.1f}%), of which not at line start {f.spans_not_at_line_start.sum()} ({100*f.spans_not_at_line_start.sum()/x.spans.sum():.1f}% of all {sp} spans)")
print("books 60-97% start ok (not flagged):", len(tv[(~tv.flag) & (tv.start_ok_pct < 97)]))
print(tv[(~tv.flag) & (tv.start_ok_pct < 97)][["pecha_id", "split", "spans", "start_ok_pct", "end_ok_pct", "sabche_realigned"]].to_string(index=False))
print("realigned train/val books NOT flagged:", tv[(~tv.flag) & (tv.sabche_realigned == True)][["pecha_id", "split", "spans", "start_ok_pct"]].to_string(index=False))
print("start_ok distribution (train+val):", tv.start_ok_pct.describe().round(1).to_dict())

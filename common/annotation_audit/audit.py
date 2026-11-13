"""Read-only annotation audit over data/raw_opf (tsawa, sabche, chapter). Writes only to common/annotation_audit/."""
import glob, os, bisect, csv
import yaml
import pandas as pd
import numpy as np

R = "data/raw_opf"; OUT = "common/annotation_audit"
LAYERS = {"tsawa": "Tsawa", "sabche": "Sabche", "chapter": "Chapter"}
SHORT = 5

def opf_dir(pid):
    g = glob.glob(f"{R}/{pid}.opf/*/layers/v001")
    return g[0] if g else None

def load_spans(d, name):
    p = f"{d}/{name}.yml"
    if not os.path.exists(p): return None
    y = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return [(a["span"]["start"], a["span"]["end"], k) for k, a in (y.get("annotations") or {}).items()]

books = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{R}/*.opf"))
data, no_v001 = {}, []
for pid in books:
    d = opf_dir(pid)
    if d is None: no_v001.append(pid); continue
    data[pid] = {l: load_spans(d, n) for l, n in LAYERS.items()}
    data[pid]["commentary"] = load_spans(d, "Commentary")
    data[pid]["base_path"] = glob.glob(f"{d}/../../base/v001.txt")[0]

def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]

# ---- TSAWA: commentary consistency ----
merged = pd.read_csv("tsawa/data/processed/tsawa_spans_merged.csv")
merged = merged[~merged.dropped.astype(bool)]
mcount = merged.groupby("pecha_id").size().to_dict()
split = pd.read_csv("data/shared/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
tsawa_books = [p for p in data if data[p]["tsawa"] is not None]
rows = []
for p in tsawa_books:
    ts, com = data[p]["tsawa"], data[p]["commentary"]
    n = len(ts)
    if com is None:
        rows.append(dict(pecha_id=p, n_tsawa=n, n_merged=mcount.get(p, 0), has_commentary=False, n_commentary=0, n_tsawa_touching_commentary="", pct_touching="", split=split.get(p, "dropped")))
    else:
        com_s = sorted((s, e) for s, e, _ in com)
        touch = sum(any(overlaps((s, e), c) for c in com_s) for s, e, _ in ts)
        rows.append(dict(pecha_id=p, n_tsawa=n, n_merged=mcount.get(p, 0), has_commentary=True, n_commentary=len(com), n_tsawa_touching_commentary=touch, pct_touching=round(100 * touch / n, 1) if n else 0.0, split=split.get(p, "dropped")))
tc = pd.DataFrame(rows)
nocom = tc[~tc.has_commentary]
q75 = nocom.n_tsawa.quantile(0.75)
tc["flag_zero_overlap"] = tc.has_commentary & (tc.n_tsawa > 0) & (tc.n_tsawa_touching_commentary == 0)
tc["flag_nocom_many"] = (~tc.has_commentary) & (tc.n_tsawa >= q75)
tc["flag_nocom_other"] = (~tc.has_commentary) & ~tc.flag_nocom_many
tc.sort_values(["flag_zero_overlap", "flag_nocom_many", "n_tsawa"], ascending=[False, False, False]).to_csv(f"{OUT}/tsawa_commentary_check.csv", index=False)

# ---- TSAWA: dropped books ----
dropped = tc[~tc.pecha_id.isin(split.keys())].copy()
dropped["note"] = np.where(dropped.n_merged == dropped.n_tsawa, "", "merged count differs from raw")
dropped.sort_values("n_tsawa").to_csv(f"{OUT}/tsawa_dropped_books.csv", index=False)

# ---- ALL LAYERS: short spans ----
short_rows = []
for p, d in data.items():
    base = open(d["base_path"], encoding="utf-8").read()
    for l in LAYERS:
        for s, e, k in d[l] or []:
            if e - s < SHORT:
                short_rows.append(dict(layer=l, pecha_id=p, ann_id=k, start=s, end=e, length=e - s, text=base[s:e].replace("\n", "\\n")))
sh = pd.DataFrame(short_rows)
sh.to_csv(f"{OUT}/short_spans.csv", index=False)

# ---- ALL LAYERS: cross-layer overlaps ----
ov_rows = []
pairs = [("tsawa", "sabche"), ("tsawa", "chapter"), ("sabche", "chapter")]
for p, d in data.items():
    for la, lb in pairs:
        A, B = d[la] or [], d[lb] or []
        if not A or not B: continue
        Bs = sorted(B); starts = [b[0] for b in Bs]
        maxlen = max(e - s for s, e, _ in Bs)
        for s, e, k in A:
            i = bisect.bisect_left(starts, s - maxlen)
            while i < len(Bs) and Bs[i][0] < e:
                bs, be, bk = Bs[i]
                if bs < e and s < be:
                    ov = min(e, be) - max(s, bs)
                    kind = "same" if (s, e) == (bs, be) else "a_in_b" if bs <= s and e <= be else "b_in_a" if s <= bs and be <= e else "partial"
                    ov_rows.append(dict(layer_a=la, layer_b=lb, pecha_id=p, a_ann=k, a_start=s, a_end=e, b_ann=bk, b_start=bs, b_end=be, overlap_chars=ov, kind=kind))
                i += 1
ovd = pd.DataFrame(ov_rows)
ovd.to_csv(f"{OUT}/cross_layer_overlaps.csv", index=False)

print("books", len(books), "no layers/v001:", no_v001)
print("tsawa books", len(tsawa_books), "| q75 of no-commentary span counts:", q75)
print(tc[["flag_zero_overlap", "flag_nocom_many", "flag_nocom_other"]].sum().to_dict(), "| dropped:", len(dropped))
print("short spans by layer:", sh.layer.value_counts().to_dict() if len(sh) else {})
print("overlap pairs:", ovd.groupby(["layer_a", "layer_b", "kind"]).size().to_dict() if len(ovd) else {})

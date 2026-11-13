"""Cross-layer analysis (read-only on data/). Writes cross_layer_analysis.md plus
commentary_no_tsawa_verse.csv and cross_layer_conflicts.csv next to this file.
Run from the repo root:  python common/annotation_audit/cross_layer_analysis.py"""
import bisect, glob, os, re
import numpy as np, pandas as pd, yaml

R = "data/raw_opf"; OUT = "common/annotation_audit"
SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}; MINRUN = 4
SHAD_RE = re.compile(f"[{SH}]")
def syl(s): return len([x for x in s.split("་") if x.strip()])

def opf(pid):
    g = [x for x in glob.glob(f"{R}/{pid}.opf/*") if x.endswith(".opf")]
    return g[0] if g else None
def layer(d, name, verse=False):
    p = f"{d}/layers/v001/{name}.yml"
    if not os.path.exists(p): return None
    y = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("annotations") or {}
    y = {i: a for i, a in enumerate(y)} if isinstance(y, list) else y   # some layers store a list, not an id map
    f = (lambda a: (a["span"]["start"], a["span"]["end"], a.get("isverse"))) if verse else (lambda a: (a["span"]["start"], a["span"]["end"]))
    return sorted(f(a) for a in y.values())
def overlaps(s, e, S): return any(a < e and s < b for a, b, *_ in S)
def runs(b, mask):
    out, cur, p = [], [], 0
    def flush():
        if len(cur) >= MINRUN: out.append((cur[0][0], cur[-1][1]))
    for m in SHAD_RE.finditer(b):
        n = syl(b[p:m.start()]); cs, ce = p, m.start(); p = m.end()
        if not n: continue
        ok = n in ISO and not any(mask[cs:ce])
        if ok and cur and n != cur[-1][2]: flush(); cur.clear()
        if ok: cur.append((cs, ce, n))
        else: flush(); cur.clear()
    flush(); return out

books = sorted(os.path.basename(x)[:-4] for x in glob.glob(f"{R}/*.opf"))
info = {}
for pid in books:
    d = opf(pid)
    if d: info[pid] = d
LAY = {n: {} for n in ("Tsawa", "Sabche", "Chapter", "BookTitle", "Author", "Commentary")}
for pid, d in info.items():
    for n in LAY:
        v = layer(d, n, verse=(n == "Commentary"))
        if v is not None: LAY[n][pid] = v

# ================= ANALYSIS 1 =================
cn = sorted(set(LAY["Commentary"]) - set(LAY["Tsawa"]))
neither = sorted(set(info) - set(LAY["Commentary"]) - set(LAY["Tsawa"]))
def verse_scan(pid):
    b = open(info[pid] + "/base/v001.txt", encoding="utf-8").read()
    Qt = (layer(info[pid], "Quotation") or []) + (layer(info[pid], "Citation") or [])
    C = LAY["Commentary"].get(pid, [])
    r = runs(b, bytearray(len(b)))
    q = [x for x in r if overlaps(x[0], x[1], Qt)]
    cv = [x for x in r if any(a < x[1] and x[0] < c and v for a, c, v in C)]
    un = [x for x in r if x not in q]
    return dict(pecha_id=pid, text_chars=len(b), n_commentary=len(C), commentary_verse_spans=sum(1 for *_, v in C if v),
                strict_runs=len(r), runs_in_quotation=len(q), runs_unexplained=len(un),
                runs_in_commentary_verse=len([x for x in un if x in cv]),
                run_chars_unexplained=sum(e - s for s, e in un),
                unexplained_per_100k=round(len(un) / len(b) * 1e5, 1) if len(b) else 0)
A1 = pd.DataFrame([verse_scan(p) for p in cn]).sort_values("runs_unexplained", ascending=False)
A1.to_csv(f"{OUT}/commentary_no_tsawa_verse.csv", index=False)
REF = pd.DataFrame([verse_scan(p) for p in neither])      # reference: books with neither layer
mt = pd.read_csv(f"{OUT}/missed_tsawa_by_book.csv")          # reference: tsawa books (runs outside tsawa)
mt["unexplained_per_100k"] = mt.strict_runs_unexplained / mt.text_chars * 1e5

# ================= ANALYSIS 2 =================
def split_of(path):
    d = pd.read_csv(path, comment="#"); d["split"] = d.split.replace({"validation": "val"}); return d.set_index("pecha_id").split.to_dict()
SP = {"tsawa": split_of("data/shared/split_v3_frozen.csv"),
      "sabche": split_of("sabche/data/processed/sabche_split_frozen.csv"),
      "chapter": split_of("chapter/data/processed/chapter_split_frozen.csv")}
test = {k: {p for p, s in v.items() if s == "test"} for k, v in SP.items()}
tsp = pd.read_csv("tsawa/data/processed/tsawa_spans_merged.csv"); tsp = tsp[~tsp.dropped.astype(bool)]
ssp = pd.read_csv("sabche/data/processed/sabche_spans_clean.csv"); ssp = ssp[~ssp.dropped.astype(bool)]
csp = pd.read_csv("chapter/data/processed/chapter_spans_clean.csv"); csp = csp[~csp.dropped.astype(bool)]
SPANS = {"tsawa": tsp, "sabche": ssp, "chapter": csp}
for k, v in SPANS.items(): v["len"] = v.end - v.start
def stats(k, books_):
    s = SPANS[k][SPANS[k].pecha_id.isin(books_)]
    per = s.groupby("pecha_id").size().reindex(sorted(books_), fill_value=0)
    bt = s.drop_duplicates("pecha_id").batch.value_counts().to_dict()
    L = s.len.to_numpy()
    q = np.percentile(L, [5, 25, 50, 75, 95]) if len(L) else [0]*5
    return dict(books=len(books_), spans=len(s), median_spans_per_book=float(per.median()),
                len_p5=int(q[0]), len_p25=int(q[1]), len_median=int(q[2]), len_p75=int(q[3]), len_p95=int(q[4]), len_max=int(L.max()) if len(L) else 0,
                short_under5=int((L < 5).sum()), old_books=bt.get("old", 0), new_books=bt.get("new", 0),
                old_spans=int((s.batch == "old").sum()), new_spans=int((s.batch == "new").sum()))
SV = pd.read_csv("sabche/data/processed/sabche_book_verdicts.csv").set_index("pecha_id")
ST = {k: {"test": stats(k, test[k]), "all": stats(k, set(SP[k]))} for k in SP}
pairs3 = [("tsawa", "sabche"), ("tsawa", "chapter"), ("sabche", "chapter")]
T_OV = {f"{a}&{b}": sorted(test[a] & test[b]) for a, b in pairs3}
T_OV["all three"] = sorted(test["tsawa"] & test["sabche"] & test["chapter"])
# where the books that are test in one layer sit in the other layers
cross_split = {}
for a in SP:
    for p in test[a]:
        cross_split.setdefault(a, []).append({b: SP[b].get(p, "not in dataset") for b in SP if b != a})
def final_overlaps(a, b, books_):
    rows = []
    for p in books_:
        A = SPANS[a][SPANS[a].pecha_id == p][["start", "end"]].to_numpy().tolist()
        B = SPANS[b][SPANS[b].pecha_id == p][["start", "end"]].to_numpy().tolist()
        for s, e in A:
            for bs, be in B:
                if bs < e and s < be:
                    kind = "same" if (s, e) == (bs, be) else "a_in_b" if bs <= s and e <= be else "b_in_a" if s <= bs and be <= e else "partial"
                    rows.append((p, a, s, e, b, bs, be, min(e, be) - max(s, bs), kind))
    return rows

# ================= ANALYSIS 3 =================
def pair_conflicts(na, nb):
    rows = []
    for p in sorted(set(LAY[na]) & set(LAY[nb])):
        A = [x[:2] for x in LAY[na][p]]; B = sorted(x[:2] for x in LAY[nb][p])
        if not A or not B: continue
        starts = [x[0] for x in B]; mx = max(e - s for s, e in B)
        for s, e in A:
            i = bisect.bisect_left(starts, s - mx)
            while i < len(B) and B[i][0] < e:
                bs, be = B[i]
                if bs < e and s < be:
                    kind = "same" if (s, e) == (bs, be) else "a_in_b" if bs <= s and e <= be else "b_in_a" if s <= bs and be <= e else "partial"
                    rows.append(dict(pecha_id=p, layer_a=na, a_start=s, a_end=e, layer_b=nb, b_start=bs, b_end=be, overlap_chars=min(e, be) - max(s, bs), kind=kind))
                i += 1
    return rows
CP = [("Tsawa", "Sabche"), ("Tsawa", "Chapter"), ("Sabche", "Chapter"), ("Chapter", "BookTitle"), ("Sabche", "BookTitle"),
      ("Tsawa", "BookTitle"), ("Chapter", "Author"), ("Sabche", "Author"), ("Tsawa", "Author")]
conf = pd.DataFrame([r for a, b in CP for r in pair_conflicts(a, b)])
splits_all = lambda p: {k: SP[k].get(p, "-") for k in SP}
conf["split_tsawa"] = conf.pecha_id.map(lambda p: SP["tsawa"].get(p, "-"))
conf["split_sabche"] = conf.pecha_id.map(lambda p: SP["sabche"].get(p, "-"))
conf["split_chapter"] = conf.pecha_id.map(lambda p: SP["chapter"].get(p, "-"))
conf.to_csv(f"{OUT}/cross_layer_conflicts.csv", index=False)

# ================= REPORT =================
def table(df, cols=None):
    df = df[cols] if cols else df
    return "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
L = ["# Cross-layer analysis: tsawa, sabche, chapter\n",
     "Read-only analysis of `data/raw_opf` and the frozen split / cleaned span files. Numbers only. Script: `cross_layer_analysis.py`; detail CSVs: `commentary_no_tsawa_verse.csv`, `cross_layer_conflicts.csv`, `tsawa_offset_shift_test.csv`.\n"]
# ---- A1
thr = {t: int((A1.runs_unexplained >= t).sum()) for t in (1, 20, 50, 100)}
L += ["## Analysis 1: books with a Commentary layer and no Tsawa layer\n",
      f"- Books with `Commentary.yml`: {len(LAY['Commentary'])}. With `Tsawa.yml`: {len(LAY['Tsawa'])}. Commentary and no Tsawa: **{len(cn)}**. Both: {len(set(LAY['Commentary']) & set(LAY['Tsawa']))}. Tsawa and no Commentary: {len(set(LAY['Tsawa']) - set(LAY['Commentary']))}.",
      "- Metre test: same as `missed_verse_scan.py`, a run is 4+ consecutive shad-delimited clauses with the same 7, 9, 11 or 13 syllable count. No tsawa exists in these books, so no text is masked. Runs overlapping a Quotation or Citation span are counted separately (`in Quotation`). `in commentary verse` counts runs that overlap a Commentary span flagged `isverse` (the annotator already marked that text as verse).",
      f"- Books by unlabelled runs (not in Quotation/Citation): at least 1: {thr[1]}; at least 20: {thr[20]}; at least 50: {thr[50]}; at least 100: {thr[100]}; none: {len(A1) - thr[1]}. Total runs {int(A1.runs_unexplained.sum()):,}; runs inside Quotation/Citation {int(A1.runs_in_quotation.sum()):,}.",
      f"- Reference rates, unlabelled runs per 100,000 characters (median over books): these {len(A1)} commentary-no-tsawa books **{A1.unexplained_per_100k.median():.1f}**; {len(REF)} books with neither a Commentary nor a Tsawa layer {REF.unexplained_per_100k.median():.1f}; tsawa books (runs outside their tsawa spans) {mt.unexplained_per_100k.median():.1f}.\n",
      "Top 30 by unlabelled runs:\n",
      table(A1.head(30).rename(columns={"runs_unexplained": "unlabelled runs", "runs_in_quotation": "in Quotation", "runs_in_commentary_verse": "in commentary verse", "run_chars_unexplained": "run chars", "unexplained_per_100k": "runs per 100k chars", "commentary_verse_spans": "commentary isverse spans"}),
            ["pecha_id", "text_chars", "n_commentary", "commentary isverse spans", "unlabelled runs", "in Quotation", "in commentary verse", "run chars", "runs per 100k chars"]),
      "\nAll books are in `commentary_no_tsawa_verse.csv`.\n"]
# ---- A2
L += ["## Analysis 2: test sets across the three layers\n",
      "### Test-book overlap (frozen splits)\n",
      table(pd.DataFrame([{"layer": k, "books in dataset": len(SP[k]), "train": sum(1 for s in SP[k].values() if s == 'train'), "val": sum(1 for s in SP[k].values() if s == 'val'), "test": len(test[k])} for k in SP])),
      "\n" + table(pd.DataFrame([{"test sets": k, "shared test books": len(v)} for k, v in T_OV.items()])),
      "\nSplit of each layer's test books in the other layers:\n"]
rows = []
for a in SP:
    for b in SP:
        if a == b: continue
        c = pd.Series([SP[b].get(p, "not in dataset") for p in test[a]]).value_counts().to_dict()
        rows.append({"test books of": a, "their split in": b, **{k: c.get(k, 0) for k in ("test", "val", "train", "not in dataset")}})
L += [table(pd.DataFrame(rows)), "\n### Per-layer benchmark statistics (cleaned spans, the files the datasets are built from)\n"]
def srow(name, d): return {"set": name, **{k: d[k] for k in d}}
srows = []
for k in SP:
    for part in ("test", "all"):
        srows.append({"layer": k, "set": part, **ST[k][part]})
sdf = pd.DataFrame(srows)
L += [table(sdf, ["layer", "set", "books", "spans", "median_spans_per_book", "old_books", "new_books", "old_spans", "new_spans"]),
      "\nSpan length in characters:\n", table(sdf, ["layer", "set", "len_p5", "len_p25", "len_median", "len_p75", "len_p95", "len_max", "short_under5"])]
ratio = lambda d: f"{d['old_books']}:{d['new_books']}"
L += ["\nOld:new batch ratio by books: " + "; ".join(f"{k} test {ratio(ST[k]['test'])}, all {ratio(ST[k]['all'])}" for k in SP) + ".\n"]
# shared test books: span-level disagreements in the cleaned views
L += ["### Disagreements in books that are test books of two or more layers (cleaned spans)\n"]
shared_any = sorted({p for a, b in pairs3 for p in (test[a] & test[b])})
fo = []
for a, b in pairs3:
    fo += final_overlaps(a, b, sorted(test[a] & test[b]))
fo = pd.DataFrame(fo, columns=["pecha_id", "layer_a", "a_start", "a_end", "layer_b", "b_start", "b_end", "overlap_chars", "kind"])
L += [f"Books that are test in at least two layers: {len(shared_any)} ({', '.join(shared_any) or 'none'}).",
      f"Overlapping span pairs between the layers' cleaned spans in those books: {len(fo)}." + (" By pair and kind: " + "; ".join(f"{a}/{b} {k}: {n}" for (a, b, k), n in fo.groupby(['layer_a', 'layer_b', 'kind']).size().items()) + "." if len(fo) else ""), ""]
if len(fo):
    L += ["Per book (all rows are in the shared-books comparison above; counts by kind):\n",
          table(fo.groupby("pecha_id").agg(pairs=("kind", "size"), tsawa_inside_sabche=("kind", lambda x: int((x == "a_in_b").sum())), sabche_inside_tsawa=("kind", lambda x: int((x == "b_in_a").sum())), partial=("kind", lambda x: int((x == "partial").sum()))).reset_index()
                .assign(sabche_verdict=lambda d: d.pecha_id.map(SV.verdict), sabche_realigned=lambda d: d.pecha_id.map(SV.realigned))), ""]
# also: cleaned-span overlaps over ALL books, per pair, for context
allfo = {f"{a}/{b}": final_overlaps(a, b, sorted(set(SPANS[a].pecha_id) & set(SPANS[b].pecha_id))) for a, b in pairs3}
L += ["For context, the same cleaned-span comparison over every book present in both layers' datasets: " + "; ".join(f"{k} {len(v)} overlapping pairs in {len({r[0] for r in v})} books" for k, v in allfo.items()) + ".\n"]
# ---- A2b: where the cleaned views disagree
allrows = pd.DataFrame(allfo["tsawa/sabche"], columns=["pecha_id", "layer_a", "a_start", "a_end", "layer_b", "b_start", "b_end", "overlap_chars", "kind"])
both = sorted(set(SPANS["tsawa"].pecha_id) & set(SPANS["sabche"].pecha_id))
rl = {p: bool(SV.realigned.get(p, False)) for p in both}
per = allrows.groupby("pecha_id").size().reindex(both, fill_value=0)
grp = pd.DataFrame({"pairs": per, "realigned": pd.Series(rl)})
L += ["### Where the cleaned tsawa and sabche spans disagree\n",
      "Books present in both the tsawa and sabche datasets, split by whether the Sabche cleaning step realigned the book (`sabche_book_verdicts.csv`, `realigned=True`: the raw Sabche offsets were shifted, 2-4 offset segments per book):\n",
      table(pd.DataFrame([{"Sabche realigned": k, "books": int((grp.realigned == k).sum()), "books with a tsawa/sabche overlap": int(((grp.realigned == k) & (grp.pairs > 0)).sum()), "overlapping pairs": int(grp[grp.realigned == k].pairs.sum())} for k in (True, False)])),
      "\nThe tsawa spans in the realigned books were not shifted. Test of whether they carry a similar offset: shift every tsawa span by k characters (k from -60 to 60) and measure the share of clauses inside the spans with a 7/9/11/13 syllable metre; `best shift` is the k with the highest share. Books where the best shift gains at least 5 points over shift 0:\n"]
ISO_ = ISO
def clauses_(t):
    o, s_ = [], 0
    for m in SHAD_RE.finditer(t):
        n = syl(t[s_:m.start()]); s_ = m.end()
        if n: o.append(n)
    n = syl(t[s_:])
    if n: o.append(n)
    return o
drows = []
for pid, g in SPANS["tsawa"].groupby("pecha_id"):
    if pid not in SP["tsawa"]: continue
    b_ = open(info[pid] + "/base/v001.txt", encoding="utf-8").read(); S_ = list(zip(g.start, g.end))
    def sc(k):
        c = [n for a_, e_ in S_ if a_ + k >= 0 and e_ + k <= len(b_) for n in clauses_(b_[a_ + k:e_ + k])]
        return 100 * sum(n in ISO_ for n in c) / len(c) if c else 0.0
    scs = {k: sc(k) for k in range(-60, 61)}; best = max(scs, key=scs.get)
    drows.append(dict(pecha_id=pid, split=SP["tsawa"][pid], tsawa_spans=len(S_), metre_pct_shift0=round(scs[0], 1), best_shift=best, metre_pct_best=round(scs[best], 1),
                      sabche_realigned=SV.realigned.get(pid, "-"), sabche_pairs_overlapping=int(per.get(pid, 0)) if pid in per.index else "-"))
dd = pd.DataFrame(drows)
dd["gain"] = (dd.metre_pct_best - dd.metre_pct_shift0).round(1)
dsel = dd[dd.gain >= 5].sort_values(["sabche_realigned", "gain"], ascending=[False, False])
L += [table(dsel.drop(columns="gain")),
      f"\nOf the {len(dsel)} books above, {int((dsel.sabche_realigned == True).sum())} are Sabche-realigned books (the tsawa dataset has {int((dd.sabche_realigned == True).sum())} realigned books in total). The other {int((dsel.sabche_realigned != True).sum())} are not realigned and are listed for completeness; `IF3ACC3E1` (747 short word-fragment spans, 0.6% metrical at shift 0) is not meaningful for this test.",
      "Test books that are Sabche-realigned books: " + "; ".join(f"`{r.pecha_id}` ({r.tsawa_spans} spans, shift 0: {r.metre_pct_shift0}%, best shift {r.best_shift:+d}: {r.metre_pct_best}%)" for r in dd[(dd.split == 'test') & (dd.sabche_realigned == True)].itertuples()) +
      f". Together {int(dd[(dd.split=='test') & (dd.sabche_realigned == True)].tsawa_spans.sum())} of the {int(dd[dd.split=='test'].tsawa_spans.sum())} tsawa test spans ({100*dd[(dd.split=='test') & (dd.sabche_realigned == True)].tsawa_spans.sum()/dd[dd.split=='test'].tsawa_spans.sum():.0f}%) are in realigned books. For `P000027` and `P000144` the gain is under 5 points (3.9 and 4.7), so the metre test is weak evidence for them; a book with several offset segments has no single shift and the search is capped at 60, so the numbers show that an offset is likely present, not what it is. The stronger evidence is the overlap table above: in the 23 realigned books the cleaned tsawa and sabche spans overlap in 19 books (910 pairs), against 13 of 175 books (199 pairs) elsewhere.",
      "Example, `P000027` (test): the cleaned Sabche heading at `[18207:18445]` starts at a line start (it was shifted +30 from the raw `[18177:18415]`); the tsawa span `[18414:18570]` starts with the last words of that same heading line (`…འཁོར་བའི་ཚུལ༽ནི།`) and runs into the verse that begins on the next line.\n"]
dd.to_csv(f"{OUT}/tsawa_offset_shift_test.csv", index=False)
# ---- A3
L += ["## Analysis 3: conflicts between layers on the same text\n",
      "Raw layer files (`layers/v001/`), all 539 books. Overlap = any shared character. `same`: identical span; `a in b` / `b in a`: one contains the other; `partial`: they cross.\n"]
g = conf.groupby(["layer_a", "layer_b", "kind"]).agg(pairs=("pecha_id", "size"), books=("pecha_id", "nunique")).reset_index()
tot = conf.groupby(["layer_a", "layer_b"]).agg(span_pairs=("pecha_id", "size"), books=("pecha_id", "nunique")).reset_index()
L += [table(tot), "\nBy kind:\n", table(g)]
for a, b in (("Sabche", "Chapter"), ("Chapter", "BookTitle")):
    c = conf[(conf.layer_a == a) & (conf.layer_b == b)]
    L += [f"\n### {a} vs {b}: {len(c)} overlapping pairs in {c.pecha_id.nunique()} books\n"]
    if len(c): L += [table(c.head(40), ["pecha_id", "a_start", "a_end", "b_start", "b_end", "overlap_chars", "kind", "split_tsawa", "split_sabche", "split_chapter"])]
c = conf[(conf.layer_a == "Tsawa") & (conf.layer_b == "Sabche")]
L += [f"\n### Tsawa vs Sabche: {len(c)} overlapping pairs in {c.pecha_id.nunique()} books\n", "Per book:\n",
      table(c.groupby("pecha_id").agg(pairs=("kind", "size"), same=("kind", lambda x: int((x == "same").sum())), tsawa_inside_sabche=("kind", lambda x: int((x == "a_in_b").sum())), sabche_inside_tsawa=("kind", lambda x: int((x == "b_in_a").sum())), partial=("kind", lambda x: int((x == "partial").sum()))).reset_index().assign(split_tsawa=lambda d: d.pecha_id.map(lambda p: SP['tsawa'].get(p, '-')), split_sabche=lambda d: d.pecha_id.map(lambda p: SP['sabche'].get(p, '-'))))]
bt = conf[conf.layer_b.isin(["BookTitle", "Author"]) | conf.layer_a.isin(["BookTitle", "Author"])]
L += ["\n### Layers overlapping BookTitle or Author: books and pairs\n",
      table(bt.groupby(["layer_a", "layer_b"]).agg(pairs=("pecha_id", "size"), books=("pecha_id", "nunique")).reset_index()),
      "\nFull list of every conflict row: `cross_layer_conflicts.csv`.\n"]
open(f"{OUT}/cross_layer_analysis.md", "w", encoding="utf-8").write("\n".join(L))
print("A1:", len(cn), "books; thresholds", thr, "| ref neither", len(REF), "| A3 rows", len(conf))

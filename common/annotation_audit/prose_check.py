"""For the books flagged 'commentary exists, 0% tsawa overlap': does the tsawa text look like verse or prose?
Read-only. Verse features follow tsawa/src/checks/audit_before_v6.py (shad clauses, ISO = {7,9,11,13} syllables)."""
import glob, re, statistics as st
import pandas as pd, yaml

R = "data/raw_opf"; OUT = "common/annotation_audit"
SHADS = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}
tc = pd.read_csv(f"{OUT}/tsawa_commentary_check.csv")
ids = tc[tc.flag_zero_overlap].pecha_id.tolist()

def syl(seg): return len([x for x in seg.split("་") if x.strip()])
def clauses(t):
    out, s = [], 0
    for m in re.finditer(f"[{SHADS}]", t):
        n = syl(t[s:m.start()])
        if n: out.append(n)
        s = m.end()
    tail = syl(t[s:])
    if tail: out.append(tail)
    return out
def spans(d, n):
    y = yaml.safe_load(open(f"{d}/layers/v001/{n}.yml", encoding="utf-8"))["annotations"]
    return sorted((a["span"]["start"], a["span"]["end"]) for a in y.values())
def stats(base, S):
    cl = [n for s, e in S for n in clauses(base[s:e])]
    iso = 100 * sum(n in ISO for n in cl) / len(cl) if cl else float("nan")
    ends = 100 * sum(base[s:e].rstrip().rstrip("་ ").endswith(tuple(SHADS)) for s, e in S) / len(S)
    return iso, ends, st.median(n for n in cl) if cl else float("nan"), st.median(e - s for s, e in S), cl

rows = []
for p in ids:
    d = glob.glob(f"{R}/{p}.opf/*")[0]
    d = [x for x in glob.glob(f"{R}/{p}.opf/*") if x.endswith(".opf")][0]
    base = open(f"{d}/base/v001.txt", encoding="utf-8").read()
    T, C = spans(d, "Tsawa"), spans(d, "Commentary")
    ti, te, tm, tl, tcl = stats(base, T)
    ci, ce, cm, cl_, ccl = stats(base, C)
    ex = [base[s:e].strip().replace("\n", " ")[:70] for s, e in (T[0], T[len(T)//2], T[-1])]
    rows.append(dict(pecha_id=p, split=tc.set_index("pecha_id").split[p], n_tsawa=len(T), n_comm=len(C),
                     tsawa_iso_pct=round(ti, 1), tsawa_ends_shad_pct=round(te, 1), tsawa_median_clause_syl=tm, tsawa_median_span_chars=tl,
                     comm_iso_pct=round(ci, 1), comm_median_clause_syl=cm, comm_median_span_chars=cl_, examples=" || ".join(ex)))
df = pd.DataFrame(rows)
def verdict(r):
    if r.tsawa_iso_pct >= 60: v = "VERSE"
    elif r.tsawa_iso_pct < 40: v = "PROSE"
    else: v = "MIXED"
    if r.n_tsawa < 5: v += " (few spans)"
    return v
df["verdict"] = df.apply(verdict, axis=1)
df.sort_values(["verdict", "n_tsawa"], ascending=[True, False]).to_csv(f"{OUT}/zero_overlap_prose_check.csv", index=False)
print(df.verdict.value_counts().to_dict())
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(df.sort_values(["verdict", "tsawa_iso_pct"])[["pecha_id", "split", "n_tsawa", "tsawa_iso_pct", "tsawa_ends_shad_pct", "tsawa_median_clause_syl", "tsawa_median_span_chars", "comm_iso_pct", "verdict"]].to_string(index=False))

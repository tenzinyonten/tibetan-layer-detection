"""Check 1: tsawa/commentary 4-way breakdown over all books. Check 2: tiny Sabche gold spans per split.
Read-only on data/. Writes layer_venn_and_sabche_tiny_gold.md and commentary_only_books.csv, sabche_tiny_gold.csv here.
Run from the repo root."""
import glob, os
import pandas as pd, yaml

R = "data/raw_opf"; OUT = "common/annotation_audit"
def opf(p): return [x for x in glob.glob(f"{R}/{p}.opf/*") if x.endswith(".opf")][0]
books = sorted(os.path.basename(x)[:-4] for x in glob.glob(f"{R}/*.opf"))
def nspans(d, n):
    f = f"{d}/layers/v001/{n}.yml"
    if not os.path.exists(f): return None
    y = (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("annotations") or {}
    return len(y)
rows = []
for p in books:
    d = opf(p); ts, cm = nspans(d, "Tsawa"), nspans(d, "Commentary")
    rows.append(dict(pecha_id=p, has_tsawa=ts is not None, n_tsawa=ts or 0, has_commentary=cm is not None, n_commentary=cm or 0,
                     text_chars=len(open(d + "/base/v001.txt", encoding="utf-8").read())))
df = pd.DataFrame(rows)
cat = lambda r: ("tsawa + commentary" if r.has_commentary else "tsawa only") if r.has_tsawa else ("commentary only" if r.has_commentary else "neither")
df["group"] = df.apply(cat, axis=1)
counts = df.group.value_counts().reindex(["tsawa + commentary", "tsawa only", "commentary only", "neither"])
# empty layer files, if any
empty_t = int(((df.has_tsawa) & (df.n_tsawa == 0)).sum()); empty_c = int(((df.has_commentary) & (df.n_commentary == 0)).sum())
co = df[df.group == "commentary only"].copy()
vs = pd.read_csv(f"{OUT}/commentary_no_tsawa_verse.csv").set_index("pecha_id")
co["unlabelled_verse_runs"] = co.pecha_id.map(vs.runs_unexplained)
co["runs_per_100k_chars"] = co.pecha_id.map(vs.unexplained_per_100k)
co = co.sort_values("unlabelled_verse_runs", ascending=False)
co.to_csv(f"{OUT}/commentary_only_books.csv", index=False)

# ---- Check 2
SP = pd.read_csv("sabche/data/processed/sabche_split_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
sc = pd.read_csv("sabche/data/processed/sabche_spans_clean.csv")
def text_of(p, s, e, w=45):
    b = open(opf(p) + "/base/v001.txt", encoding="utf-8").read()
    return b[max(0, s - w):s].replace("\n", "⏎"), b[s:e].replace("\n", "⏎"), b[e:e + w].replace("\n", "⏎")
sc["len"] = sc.end - sc.start
gold = sc[~sc.dropped.astype(bool)].copy()
gold["split"] = gold.pecha_id.map(SP)
tiny = gold[gold.len < 3].sort_values(["split", "pecha_id", "start"])
sizes = gold.groupby("split").size()
raw = []
for p in sorted(set(sc.pecha_id)):
    f = f"{opf(p)}/layers/v001/Sabche.yml"
    for k, a in ((yaml.safe_load(open(f, encoding="utf-8")) or {}).get("annotations") or {}).items():
        raw.append((p, k, a["span"]["start"], a["span"]["end"]))
raw = pd.DataFrame(raw, columns=["pecha_id", "ann_id", "start", "end"]); raw["len"] = raw.end - raw.start; raw["split"] = raw.pecha_id.map(SP)
out = []
for r in tiny.itertuples():
    pre, t, post = text_of(r.pecha_id, r.start, r.end)
    out.append(dict(split=r.split, pecha_id=r.pecha_id, start=r.start, end=r.end, length=r.len, raw_start=r.raw_start, raw_end=r.raw_end, action=r.action, text=t, before=pre, after=post))
tdf = pd.DataFrame(out); tdf.to_csv(f"{OUT}/sabche_tiny_gold.csv", index=False)

def table(d):
    return "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in d.itertuples(index=False))
L = ["# Layer breakdown and tiny Sabche gold spans\n", "Read-only. Script: `analyze_layer_venn.py`.\n",
     f"## Check 1: Tsawa vs Commentary across {len(df)} books\n",
     table(pd.DataFrame({"group": counts.index, "books": counts.values})),
     f"\nTotals: tsawa layer {int(df.has_tsawa.sum())}, commentary layer {int(df.has_commentary.sum())}. Layer files with zero annotations: tsawa {empty_t}, commentary {empty_c}.\n",
     f"### Commentary only ({len(co)} books)\n",
     f"Unlabelled verse runs = 4+ consecutive same-metre clauses (7/9/11/13 syllables) outside Quotation/Citation, from `commentary_no_tsawa_verse.csv`. Median runs per 100k characters: {co.runs_per_100k_chars.median():.1f}. Books with at least 20 runs: {(co.unlabelled_verse_runs >= 20).sum()}; at least 50: {(co.unlabelled_verse_runs >= 50).sum()}.\n",
     table(co[["pecha_id", "text_chars", "n_commentary", "unlabelled_verse_runs", "runs_per_100k_chars"]]),
     "\nAll book IDs with their numbers are also in `commentary_only_books.csv`.\n",
     "## Check 2: Sabche gold spans under 3 characters\n",
     "Gold = the cleaned Sabche spans the dataset is built from (`sabche_spans_clean.csv`, dropped=False), length = end - start, split from `sabche_split_frozen.csv`.\n",
     table(pd.DataFrame([{"split": s, "gold spans": int(sizes.get(s, 0)), "under 3 chars": int((tiny.split == s).sum()), "of which 1 char": int(((tiny.split == s) & (tiny.len == 1)).sum()), "of which 2 chars": int(((tiny.split == s) & (tiny.len == 2)).sum()), "of which 0 chars": int(((tiny.split == s) & (tiny.len == 0)).sum())} for s in ("train", "val", "test")])),
     f"\nFor comparison, spans under 5 characters in the cleaned gold: {int((gold.len < 5).sum())}; in the raw `Sabche.yml` files (all books in the dataset, before cleaning): under 3 chars {int((raw.len < 3).sum())}, under 5 chars {int((raw.len < 5).sum())}, zero or negative length {int((raw.len <= 0).sum())}.\n",
     "Each tiny gold span with its surrounding text (`⏎` newline; 45 characters before and after):\n",
     table(tdf[["split", "pecha_id", "start", "end", "length", "action", "before", "text", "after"]]) if len(tdf) else "(none)\n"]
nxt = []
for r in tdf.itertuples():
    g = gold[gold.pecha_id == r.pecha_id].sort_values("start")
    n_ = g[g.start >= r.end]
    nxt.append(int(n_.start.iloc[0] - r.end) if len(n_) else -1)
tdf["chars_to_next_gold_span"] = nxt
tdf.to_csv(f"{OUT}/sabche_tiny_gold.csv", index=False)
bk = tdf.groupby("pecha_id").size()
L += ["\nPer book: " + "; ".join(f"`{p_}` {n_} tiny spans, {int((gold.pecha_id == p_).sum())} gold spans in the book, median gold length {int(gold[gold.pecha_id == p_].len.median())}" for p_, n_ in bk.items()) + f". Median gold span length overall: {int(gold.len.median())} characters.",
      "Cleaning action of each tiny span: " + ", ".join(f"{k} {v}" for k, v in tdf.action.value_counts().items()) + " (`end+1` = the raw span was 1 character and the cleaning step extended its end to include the following ཾ; `keep` = unchanged).",
      "Characters from each tiny span's end to the next gold span: " + ", ".join(f"{r.pecha_id} {r.start}: {r.chars_to_next_gold_span}" for r in tdf.itertuples()) + ".\n"]
open(f"{OUT}/layer_venn_and_sabche_tiny_gold.md", "w", encoding="utf-8").write("\n".join(L))
print(counts.to_dict(), "empty", empty_t, empty_c)
print("co books", len(co)); print(sizes.to_dict()); print("tiny:", tiny.groupby("split").size().to_dict(), "len counts", tiny.len.value_counts().to_dict())
print("raw under 3:", int((raw.len < 3).sum()), "under 5:", int((raw.len < 5).sum()), "<=0:", int((raw.len <= 0).sum()))
print(tdf.to_string(index=False) if len(tdf) else "none")

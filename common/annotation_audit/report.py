"""Combine the audit CSVs into audit_report.md and list under-annotated tsawa books.

Run last, after audit.py, missed_verse_scan.py, cross_layer_analysis.py and prose_check.py, from the repo root.
Reads the CSVs in common/annotation_audit/. Writes under_annotated_tsawa_books.csv (5 or fewer tsawa spans and
50 or more strict verse runs no layer covers) and audit_report.md in the same folder.
"""
import pandas as pd
OUT = "common/annotation_audit"
tc = pd.read_csv(f"{OUT}/tsawa_commentary_check.csv")
dr = pd.read_csv(f"{OUT}/tsawa_dropped_books.csv")
sh = pd.read_csv(f"{OUT}/short_spans.csv")
ov = pd.read_csv(f"{OUT}/cross_layer_overlaps.csv")

mv = pd.read_csv(f"{OUT}/missed_tsawa_by_book.csv").fillna(0)
ua = mv[(mv.n_tsawa <= 5) & (mv.strict_runs_no_layer >= 50)].copy()
ua["ratio"] = (ua.strict_runs_no_layer / ua.n_tsawa).round(1)
ua = ua.sort_values(["ratio", "strict_runs_no_layer"], ascending=False).reset_index(drop=True)
ua.insert(0, "rank", ua.index + 1)
ua.to_csv(f"{OUT}/under_annotated_tsawa_books.csv", index=False)

def md(df, cols=None):
    df = df[cols] if cols else df
    h = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return h + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))

short_n = sh.groupby(["layer", "pecha_id"]).size().rename("n_short").reset_index()
ov_n = ov.groupby("pecha_id").size().rename("n_overlap").reset_index()
dropped_ids = set(dr.pecha_id)
pc = pd.read_csv(f"{OUT}/zero_overlap_prose_check.csv")
real_ids = set(pc[~pc.verdict.str.startswith("VERSE")].pecha_id)
tc["flag_zero_overlap_all"] = tc.flag_zero_overlap
tc["flag_zero_overlap"] = tc.flag_zero_overlap_all & tc.pecha_id.isin(real_ids)

# ---- tsawa master priority table ----
t = tc.copy()
t = t.merge(short_n[short_n.layer == "tsawa"][["pecha_id", "n_short"]], on="pecha_id", how="left").merge(ov_n, on="pecha_id", how="left").fillna({"n_short": 0, "n_overlap": 0})
t["dropped"] = t.pecha_id.isin(dropped_ids)
def reasons(r):
    out = []
    if r.flag_zero_overlap: out.append("commentary exists, 0% tsawa overlap, tsawa text is prose/mixed")
    if r.flag_nocom_many: out.append("no commentary, many tsawa spans")
    if r.dropped: out.append(f"dropped (<10 spans)")
    if r.n_short: out.append(f"{int(r.n_short)} short spans")
    if r.n_overlap: out.append(f"{int(r.n_overlap)} cross-layer overlaps")
    return "; ".join(out)
t["reasons"] = t.apply(reasons, axis=1)
t["tier"] = 4
t.loc[t.dropped, "tier"] = 2
t.loc[(t.n_short > 0) | (t.n_overlap > 0), "tier"] = t.loc[(t.n_short > 0) | (t.n_overlap > 0), "tier"].clip(upper=3)
t.loc[t.flag_zero_overlap | t.flag_nocom_many, "tier"] = 1
TOP = ["I23323023", "I4B1806B4", "I454D2699"]
SWAP = "I85485484"
t.loc[t.pecha_id == SWAP, "tier"] = 1
t.loc[t.pecha_id == SWAP, "reasons"] = "possible commentary/tsawa label swap (see its section); " + t.loc[t.pecha_id == SWAP, "reasons"]
t.loc[t.pecha_id.isin(TOP), "tier"] = 0
t.loc[t.pecha_id == "I23323023", "reasons"] = "TOP: train book, mislabeled (mantra, scribe note and colophon labeled as tsawa; ~175 question spans to confirm); " + t.loc[t.pecha_id == "I23323023", "reasons"]
t.loc[t.pecha_id.isin(["I4B1806B4", "I454D2699"]), "reasons"] = "TOP: val book, verify convention: mix of verse and prose, needs annotator judgment; " + t.loc[t.pecha_id.isin(["I4B1806B4", "I454D2699"]), "reasons"]
flagged = t[t.tier < 4].sort_values(["tier", "n_tsawa"], ascending=[True, False])
tier_name = {0: "0 TOP", 1: "1 HIGH", 2: "2 dropped", 3: "3 short/overlap"}
flagged["priority"] = flagged.tier.map(tier_name)

z = tc[tc.flag_zero_overlap].sort_values("n_tsawa", ascending=False).merge(pc[["pecha_id", "verdict", "tsawa_iso_pct", "examples"]], on="pecha_id")
fp = pc[pc.verdict.str.startswith("VERSE")].sort_values("n_tsawa", ascending=False)
m = tc[tc.flag_nocom_many].sort_values("n_tsawa", ascending=False)
allnc = tc[~tc.has_commentary].sort_values("n_tsawa", ascending=False)
q75 = int(tc[~tc.has_commentary].n_tsawa.quantile(0.75))

inv = sh[(sh.layer == "chapter") & (sh.length < 0)]
same = ov[ov.kind == "same"]
nm_ids = ", ".join(f"`{x}`" for x in m.pecha_id.head(8))
other_real = [x for x in z.pecha_id if x not in TOP]
L = []
L.append("# Annotation audit: tsawa, sabche, chapter\n")
L.append(f"""## One-page summary for annotators (about 2 minutes)

We checked every tsawa, sabche and chapter layer in the 539 books for things that look wrong. This page says what to review and why; the rest of the file has the full tables. Nothing has been changed in the data.

**Review in this order**

1. **Three books that directly affect the model (do these first).**
   - `I23323023` (train, 207 tsawa spans): **mislabeled.** Three tsawa spans are clearly not root text: the bare mantra `ༀ་སྭ་སྟི` (span 0), the scribe's note `འབྲི་བན་དཀོན་མཆོག་བརྟན་འཕེལ…` and the dated colophon `སྤྱི་ལོ་༢༠༢༡ ཟླ་བ་༡༡ ཚེས་༠༨། ཡོ་རོབ་ཝིའ་ན་མཐོ་སློབ` (spans 205 and 206). Remove or relabel those. About 175 more spans are numbered questions ending `ཞུས་པས།` followed by the answer; that looks like a consistent question-and-answer convention, so ask whether it counts as tsawa.
   - `I4B1806B4` (val, 69) and `I454D2699` (val, 46): **verify convention: mix of verse and prose, needs annotator judgment.** Their tsawa spans are quoted passages (prose in `I4B1806B4`, practice liturgy and mantras in `I454D2699`), each followed by a quote marker (`ཞེས་གསུངས་ཏེ`, `ཞེས་པ་སྟེ`) and an explanation. The labeling is consistent, so this may be intended. Decide whether quoted prose and liturgy count as tsawa.
   - Only 30-42% of the tsawa lines in these three books have a verse metre, against about 88% in a normal tsawa book. We tested for a position shift and found none; the spans are in the right place.
2. **Possible commentary/tsawa label swap: `I85485484`.** A 5,985-character prayer book with exactly 1 tsawa span (a closing verse) while 15 of its 22 commentary spans are verse quatrains marked `isverse`. The verses may have been put in the Commentary layer instead of Tsawa. Please check which layer the verse belongs to.
3. **Other tsawa spans on non-verse text:** {", ".join(f"`{x}`" for x in other_real)}. Mostly scribe or colophon lines and a mantra, all in small or dropped books, so lower impact.
4. **No Commentary layer but many tsawa spans** ({len(m)} books, at least {q75} tsawa spans each, for example {nm_ids}). Check whether the commentary exists in the text and was never annotated. Many of these may simply be root texts, so look before changing anything.
5. **{len(inv)} chapter spans whose end is before their start** in {inv.pecha_id.nunique()} books (list in the chapter section). These are always errors; fix or delete them.
6. **{len(same)} spans that sit exactly on a span of another layer** (5 tsawa = sabche, 1 sabche = chapter), in {same.pecha_id.nunique()} books (list in the overlaps section). One text region should not carry two layers; decide which one is right.
7. **Dropped books** ({len(dr)} books with a tsawa layer but fewer than 10 spans; {int((dr.n_tsawa<=3).sum())} have 3 or fewer). They were left out of training. If any should be fully annotated, they are the cheapest additions.
   - Highest priority among them: the {len(ua)} dropped books with 5 or fewer tsawa spans and 50 or more uncovered verse runs (`{", ".join(ua.pecha_id.head(4))}` first; see "Severely under-annotated tsawa books"). A few tsawa spans were marked in a book full of verse, and the rest of the verse is unlabeled.
8. **Short spans (under 5 characters)**, lowest priority: tsawa 1,499 (1,430 in six books that appear to mark word fragments, so ask first whether that is intended), sabche 28, chapter {int(((sh.layer=="chapter")&(sh.length>=0)).sum())} more (mostly bare 2-character markers such as `༼ཀ`).

**Do not spend time on:** the 52 books that have a Commentary layer with no tsawa overlap but verse-looking tsawa. That is the normal alternating pattern of verse and commentary (for example `I6D11E414`), not an error.
""")
L.append("Read-only audit of `data/raw_opf` (539 books with `layers/v001/`). Raw layer files are the source for every check; the dropped list uses `tsawa/data/processed/tsawa_spans_merged.csv` and `split_v3_frozen.csv`. Generated by `common/annotation_audit/audit.py` and `report.py`; detail tables are CSVs next to this file.\n")
L.append("## Summary\n")
L.append(f"""| check | books | notes |
|---|---:|---|
| HIGH: commentary exists, 0% tsawa overlap, tsawa text is prose or mixed | {len(z)} | of 59 zero-overlap books; the other {len(fp)} have verse-looking tsawa and were removed (see Tier 1a) |
| HIGH: no commentary, tsawa spans >= {q75} (top quartile of the {len(allnc)} no-commentary books) | {len(m)} | |
| Dropped tsawa books (<10 spans, not in split_v3) | {len(dr)} | {int((dr.n_tsawa==0).sum())} have 0 spans, {int((dr.n_tsawa<=3).sum())} have 3 or fewer |
| Short spans (<5 chars), tsawa | {sh[sh.layer=='tsawa'].pecha_id.nunique()} | {int((sh.layer=='tsawa').sum())} spans |
| Short spans (<5 chars), sabche | {sh[sh.layer=='sabche'].pecha_id.nunique()} | {int((sh.layer=='sabche').sum())} spans |
| Short spans (<5 chars), chapter | {sh[sh.layer=='chapter'].pecha_id.nunique()} | {int((sh.layer=='chapter').sum())} spans, {int(((sh.layer=='chapter')&(sh.length<0)).sum())} inverted (end < start) |
| Cross-layer overlaps | {ov.pecha_id.nunique()} | {len(ov)} span pairs; no partial (crossing) overlaps anywhere |
| All no-commentary tsawa books (secondary list) | {len(allnc)} | see the last section |
""")
L.append("Zero-overlap check: 59 tsawa books have a Commentary layer that no tsawa span touches. Each was tested for verse versus prose (see Tier 1a). 52 have verse-looking tsawa (the alternating verse/commentary pattern, as in `I6D11E414`) and are NOT priorities. 7 have prose-looking or mixed tsawa and stay HIGH. Shifting the tsawa spans by up to 400 characters in the four larger of those 7 never produced verse-like text, so none is an offset shift. The cases differ: some are clear mislabels (mantra, scribe note, colophon), others are quoted prose or liturgy that may be a convention (see Tier 1a).\n")

L.append("## Priority list: tsawa\n")
L.append("Sorted by tier (0 TOP: the three train/val books that affect model quality, 1 HIGH, 2 dropped, 3 short spans or overlap), then tsawa span count. Reasons are all flags a book carries.\n")
L.append(md(flagged.assign(has_commentary=flagged.has_commentary.map({True: "yes", False: "no"})).rename(columns={"n_tsawa": "tsawa spans", "n_commentary": "commentary spans", "pct_touching": "% tsawa touching commentary"}),
            ["priority", "pecha_id", "tsawa spans", "has_commentary", "commentary spans", "% tsawa touching commentary", "split", "reasons"]))

L.append("\n### Tier 1a: commentary exists, tsawa never touches it, tsawa text is prose or mixed\n")
L.append("Verse test: share of shad-delimited clauses inside tsawa spans with 7, 9, 11 or 13 syllables (corpus average about 88%). PROSE < 40%, MIXED 40-60%, VERSE >= 60%. Examples are the first, middle and last span (first 70 chars).\n")
L.append(md(z.rename(columns={"tsawa_iso_pct": "verse-like clauses %"}), ["pecha_id", "n_tsawa", "n_commentary", "split", "verse-like clauses %", "verdict", "examples"]))
L.append("\nWhat the examples show. **Mislabeled:** `I23323023` has tsawa on a bare mantra and on a scribe's note and dated colophon (`སྤྱི་ལོ་༢༠༢༡ ཟླ་བ་༡༡ ཚེས་༠༨། ཡོ་རོབ་ཝིའ་ན་མཐོ་སློབ`); `I069801F1` and `IC3A7006F` have scribe or colophon lines (`འབྲི་བན་དཀོན་མཆོག་བརྟན་འཕེལ`) labelled as tsawa; `IA3DD1ADA` is a single `མངྒ་ལཾ། བྷ་ཝནྟུ` span. **Verify convention (mix of verse and prose, needs annotator judgment):** `I4B1806B4` and `I454D2699` have tsawa spans that are quoted prose, liturgy and mantras followed by an explanation, mixed with real verse (homage, dedication). Also in `I23323023`, about 175 numbered question spans (`... ཞུས་པས།`) follow a consistent question-and-answer pattern.\n")
L.append("\n#### Removed as false positives (verse-looking tsawa, 52 books)\n")
L.append("Not errors. Two worth a glance anyway: `I88CF073C` (3 spans, median 1,816 chars, verse-like but very long spans) and `I85485484` (1 tsawa span; its commentary spans are 82% verse-like, so the commentary label may be what is off).\n")
L.append(md(fp, ["pecha_id", "split", "n_tsawa", "tsawa_iso_pct", "comm_iso_pct", "verdict"]))
L.append(f"\n### Tier 1b: no commentary, >= {q75} tsawa spans\n")
L.append(md(m, ["pecha_id", "n_tsawa", "split"]))

L.append("\n### Tier 2: dropped books that have tsawa annotations\n")
L.append("In the audit CSV these have a tsawa layer but fewer than 10 spans, so they were left out of training and evaluation. Possibly under-annotated; sorted by span count.\n")
dd = dr.assign(has_commentary=dr.has_commentary.map({True: "yes", False: "no"})).sort_values("n_tsawa", ascending=False)
L.append(md(dd, ["pecha_id", "n_tsawa", "has_commentary", "n_commentary"]))

L.append("\n### Tier 3: short spans and overlaps in tsawa\n")
s = sh[sh.layer == "tsawa"]
top = s.groupby("pecha_id").size().sort_values(ascending=False)
L.append(f"{len(s)} tsawa spans under 5 characters in {top.shape[0]} books. {int(top.head(6).sum())} of them sit in 6 books ({', '.join(top.head(6).index)}) where the spans look like mid-sentence word fragments such as `དང༌`, `མེད`, `དེ` (checked by eye in the top 3 books only). That points to a different annotation convention, systematic rather than noise.\n")
L.append(md(top.rename("short spans").reset_index().head(25).rename(columns={"index": "pecha_id"})))
L.append("\nLength histogram (chars): " + ", ".join(f"{k}: {v}" for k, v in s.length.value_counts().sort_index().items()) + ". Zero-length spans: " + str(int((s.length == 0).sum())) + ".\n")


L.append("\n## Severely under-annotated tsawa books\n")
L.append(f"Books with a Tsawa layer of **5 or fewer spans** and **50 or more verse runs that no layer covers** (4+ consecutive same-metre clauses of 7, 9, 11 or 13 syllables sharing no character with any layer, from `missed_verse_scan.py`). The ratio is uncovered runs per tsawa span; the list is ranked by it, most verse and fewest tsawa first. {len(ua)} of the {int((mv.n_tsawa <= 5).sum())} books with 5 or fewer tsawa spans qualify; together they have {int(ua.n_tsawa.sum())} tsawa spans and {int(ua.strict_runs_no_layer.sum()):,} uncovered runs.\n")
L.append(md(ua.rename(columns={"strict_runs_no_layer": "uncovered verse runs", "strict_runs_in_quotation": "runs in Quotation/Citation (not counted)", "text_chars": "book chars"}),
            ["rank", "pecha_id", "split", "n_tsawa", "tsawa_chars", "uncovered verse runs", "ratio", "runs in Quotation/Citation (not counted)", "has_commentary", "book chars"]))
L.append(f"\nAll {len(ua)} books have the split `dropped`: a book with fewer than 10 tsawa spans is left out of the train, validation and test sets, so none of these annotations is used in training or evaluation today. Books with 20 to 49 uncovered runs and 5 or fewer tsawa spans: {int(((mv.n_tsawa <= 5) & (mv.strict_runs_no_layer >= 20) & (mv.strict_runs_no_layer < 50)).sum())}; books with 6 to 9 tsawa spans and 50 or more uncovered runs: {int(((mv.n_tsawa >= 6) & (mv.n_tsawa <= 9) & (mv.strict_runs_no_layer >= 50)).sum())}. Full table: `under_annotated_tsawa_books.csv`.\n")
L.append("## Priority list: sabche\n")
s = sh[sh.layer == "sabche"]
L.append(f"{len(s)} short spans in {s.pecha_id.nunique()} books (lengths: {dict(s.length.value_counts().sort_index())}).\n")
L.append(md(s[["pecha_id", "ann_id", "start", "end", "length", "text"]].sort_values(["pecha_id", "start"])))
L.append("\n## Priority list: chapter\n")
s = sh[sh.layer == "chapter"]
L.append(f"{len(s)} short spans in {s.pecha_id.nunique()} books ({int((s.length<0).sum())} inverted with end < start, {int((s.length==2).sum())} two-character spans such as a bare `༼ཀ` marker, {int((s.length==0).sum())} zero-length). Books by count (all spans in `short_spans.csv`):\n")
L.append("All inverted chapter spans (end < start):\n")
L.append(md(s[s.length < 0].sort_values(["pecha_id", "start"]), ["pecha_id", "ann_id", "start", "end", "text"]))
L.append("\nShort spans by book:\n")
cs = s.groupby("pecha_id").agg(n=("length", "size"), inverted=("length", lambda x: int((x < 0).sum())), example=("text", "first")).reset_index().sort_values("n", ascending=False)
L.append(md(cs))

L.append("\n## Separate flag: possible commentary/tsawa label swap (`I85485484`)\n")
L.append("The book is a short prayer text (5,985 characters, dropped from training because it has 1 tsawa span). Its only tsawa span is `[5181:5341]`, a closing verse quatrain flagged `isverse`. Of its 22 Commentary spans, 15 are flagged `isverse` and are verse quatrains too (for example `[3024:3182]` and `[3251:3410]`), and 82% of their clauses have a verse metre. Verse in the Commentary layer and one verse in the Tsawa layer suggests the two labels were mixed up for this book. Needs a human read.\n")
L.append("\n## Cross-layer overlaps (all three layers)\n")
g = ov.groupby(["layer_a", "layer_b", "kind"]).agg(pairs=("pecha_id", "size"), books=("pecha_id", "nunique")).reset_index()
L.append(md(g))
L.append("\nNo tsawa/chapter overlaps and no partial overlaps exist. Tsawa spans sit entirely inside a sabche span (`a_in_b`) or coincide with one (`same`); one sabche span coincides exactly with a chapter span. Same-span coincidence between two layers is the most likely real conflict. Full list (these are the 6 exact conflicts, plus the tsawa-inside-sabche rows):\n")
L.append(md(ov.sort_values(["kind", "pecha_id"]), ["pecha_id", "layer_a", "a_start", "a_end", "layer_b", "b_start", "b_end", "kind"]))

L.append("\n## Secondary: all tsawa books with no Commentary layer\n")
L.append(f"{len(allnc)} books. Many are probably root texts, so absence of Commentary alone is a weak signal; the {len(m)} marked HIGH above are the ones to check first.\n")
L.append(md(allnc.assign(priority=allnc.flag_nocom_many.map({True: "HIGH", False: ""})), ["pecha_id", "n_tsawa", "split", "priority"]))
L.append("\n## Method\n")
L.append("- Commentary lives in `layers/v001/Commentary.yml` (annotation map with `span.start/end`, `isverse`). \"Touch\" means any character overlap with any Commentary span.\n- Short means `end - start < 5` on raw layer files, so inverted spans are included.\n- Overlap means any shared character between spans of two different layers in the same book.\n- Dropped = tsawa books absent from `split_v3_frozen.csv` (88), which matches the <10 merged-span rule.\n- Nothing under `data/` or `tsawa/` was modified.\n")
open(f"{OUT}/audit_report.md", "w", encoding="utf-8").write("\n".join(L))
print(flagged.tier.value_counts().sort_index().to_dict(), len("\n".join(L).splitlines()), "lines")

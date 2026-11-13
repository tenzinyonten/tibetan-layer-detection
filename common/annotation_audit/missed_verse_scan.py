"""Scan every tsawa book for verse-metre runs that are NOT labelled tsawa (possible missed tsawa).
Read-only on data/. Writes missed_tsawa_by_book.csv and missed_tsawa_summary.md next to this file.

A run = 4+ consecutive shad-delimited clauses, all with a 7/9/11/13 syllable metre, none touching a tsawa span.
  loose  : any mix of those four metres
  strict : every clause in the run has the SAME syllable count (isometric stanza), much rarer in prose
Runs overlapping a Quotation or Citation span are reported separately (verse quoted from another work is
legitimately not tsawa)."""
import glob, os, re
import pandas as pd, yaml

R = "data/raw_opf"; OUT = "common/annotation_audit"
SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}; MINRUN = 4
SHAD_RE = re.compile(f"[{SH}]")
def syl(s): return len([x for x in s.split("་") if x.strip()])

def spans(d, name, with_verse=False):
    p = f"{d}/layers/v001/{name}.yml"
    if not os.path.exists(p): return []
    y = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("annotations") or {}
    return sorted((a["span"]["start"], a["span"]["end"], a.get("isverse")) if with_verse else (a["span"]["start"], a["span"]["end"]) for a in y.values())

def overlaps(s, e, S): return any(a < e and s < b for a, b, *_ in S)

def all_layers(d, n):
    """{layer name: [(start, end), ...]} for every layer file in the book (some store a list, not an id map)."""
    out = {}
    for f in sorted(glob.glob(f"{d}/layers/v001/*.yml")):
        y = (yaml.safe_load(open(f, encoding="utf-8")) or {}).get("annotations") or {}
        items = y if isinstance(y, list) else y.values()
        out[os.path.basename(f)[:-4]] = [(a["span"]["start"], a["span"]["end"]) for a in items if a.get("span")]
    return out

def runs(b, mask, strict):
    out, cur, p = [], [], 0
    def flush():
        if len(cur) >= MINRUN: out.append((cur[0][0], cur[-1][1], cur[0][2]))
    for m in SHAD_RE.finditer(b):
        n = syl(b[p:m.start()]); cs, ce = p, m.start(); p = m.end()
        if not n: continue
        ok = n in ISO and not any(mask[cs:ce])
        if ok and cur and strict and n != cur[-1][2]:
            flush(); cur.clear()
        if ok: cur.append((cs, ce, n))
        else: flush(); cur.clear()
    flush()
    return out

audit = pd.read_csv("data/shared/tsawa_audit.csv")
split = pd.read_csv("data/shared/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
rows = []
for pid in audit[audit.has_tsawa_layer == True].pecha_id:
    d = [x for x in glob.glob(f"{R}/{pid}.opf/*") if x.endswith(".opf")][0]
    b = open(d + "/base/v001.txt", encoding="utf-8").read()
    T = spans(d, "Tsawa"); Qt = spans(d, "Quotation") + spans(d, "Citation"); C = spans(d, "Commentary", True)
    mask = bytearray(len(b))
    for s, e in T: mask[s:e] = b"\x01" * (e - s)
    tl = sum(e - s for s, e in T)
    rec = dict(pecha_id=pid, split=split.get(pid, "dropped"), text_chars=len(b), n_tsawa=len(T), tsawa_chars=tl,
               has_commentary=bool(C))
    LM = {}                                   # per-layer coverage masks, for the "no layer at all" count
    for ln, sp_ in all_layers(d, pid).items():
        m_ = bytearray(len(b))
        for a_, c_ in sp_: m_[max(0, a_):min(len(b), c_)] = b"\x01" * max(0, min(len(b), c_) - max(0, a_))
        LM[ln] = m_
    for name, strict in (("strict", True), ("loose", False)):
        R_ = runs(b, mask, strict)
        quoted = [r for r in R_ if overlaps(r[0], r[1], Qt)]
        cverse = [r for r in R_ if any(a < r[1] and r[0] < c and v for a, c, v in C)]
        chars = sum(e - s for s, e, _ in R_)
        if strict:
            old_un = [r for r in R_ if r not in quoted]
            none = [r for r in old_un if not any(any(m_[r[0]:r[1]]) for m_ in LM.values())]
            rec.update(strict_runs_no_layer=len(none), strict_chars_no_layer=sum(e - s for s, e, _ in none))
            for ln, m_ in LM.items():
                rec[f"overlap_{ln}"] = sum(1 for r in old_un if any(m_[r[0]:r[1]]))
        rec.update({f"{name}_runs": len(R_), f"{name}_chars": chars,
                    f"{name}_runs_in_quotation": len(quoted), f"{name}_runs_in_commentary_verse": len(cverse),
                    f"{name}_runs_unexplained": len([r for r in R_ if r not in quoted])})
    rows.append(rec)
df = pd.DataFrame(rows)
df["strict_chars_per_tsawa_char"] = (df.strict_chars / df.tsawa_chars.where(df.tsawa_chars > 0)).round(2)
df["strict_runs_per_100k"] = (df.strict_runs / df.text_chars * 1e5).round(1)
df.sort_values("strict_chars", ascending=False).to_csv(f"{OUT}/missed_tsawa_by_book.csv", index=False)
print(len(df), "books;", "strict runs total", int(df.strict_runs.sum()), "loose", int(df.loose_runs.sum()))

# ---------------- summary markdown ----------------
def md(t, cols, names):
    h = "| " + " | ".join(names) + " |\n|" + "|".join("---" for _ in names) + "|\n"
    return h + "".join("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], (int,)) and not isinstance(r[c], bool) else str(r[c]) for c in cols) + " |\n" for _, r in t.iterrows())
df["ratio"] = (df.strict_chars / df.tsawa_chars.where(df.tsawa_chars > 0)).round(1)
cols = ["pecha_id", "split", "n_tsawa", "tsawa_chars", "strict_runs_unexplained", "strict_runs_in_quotation", "strict_chars", "ratio", "text_chars"]
names = ["book", "split", "tsawa spans", "tsawa chars", "unlabelled verse runs", "runs inside Quotation/Citation (not counted)", "chars in all runs", "run chars per tsawa char", "book chars"]
dropped = df[df.split == "dropped"]
A = dropped[dropped.strict_runs_unexplained >= 20].sort_values("strict_runs_unexplained", ascending=False)
used = df[df.split != "dropped"]
B = used[(used.strict_runs_unexplained >= 50) & (used.ratio >= 5)].sort_values("strict_runs_unexplained", ascending=False)
Bev = B[B.split.isin(["val", "test"])]
ok = used[(used.strict_runs_unexplained >= 50) & (used.ratio < 2)]
L = ["# Where tsawa may be missing: unlabelled verse by book\n",
     f"Scan of all {len(df)} books with a Tsawa layer (`missed_verse_scan.py`; per-book numbers in `missed_tsawa_by_book.csv`). A **run** is 4 or more consecutive shad-delimited clauses that all have the same 7, 9, 11 or 13 syllable count (an isometric stanza) and do not touch any tsawa span. Runs that overlap a Quotation or Citation span are counted separately and left out of the main numbers, because verse quoted from another work is not tsawa.\n",
     "## What this does and does not show\n",
     f"- Totals: {int(df.strict_runs.sum()):,} runs in {len(df)} books (median {int(df.strict_runs.median())} per book; {int((df.strict_runs==0).sum())} books have none). By split: " + ", ".join(f"{k} {int(v):,}" for k, v in df.groupby('split').strict_runs.sum().items()) + ".",
     "- I read 9 random runs from 3 of the top books (`P000218`, `P000275`, `I2FCD4B1D`): all 9 are real verse, not noise. Several are introduced by a source marker (`བཤེས་སྤྲིངས་སུ།` 'in the Letter to a Friend', `སྡོམ་འབྱུང་ལས།` 'from the Sdom 'byung'), so they are quoted verse.",
     "- So a run is verse that nobody labelled as tsawa. It may be (a) missed tsawa, (b) a quotation from another text that was not marked Quotation, or (c) the author's own verse. The annotator decides which. Use the counts to choose where to look first, not as a count of errors.",
     "- To read a book's runs in context: `python common/annotation_audit/make_book_check.py <book id>` writes an HTML page with the tsawa spans highlighted and every run underlined in blue (the page uses the looser definition, any mix of the four metres, so it shows slightly more runs).\n",
     f"## 1. Dropped books: add tsawa from scratch ({len(A)} books with at least 20 unlabelled runs)\n",
     "These books have fewer than 10 tsawa spans, so they were left out of training. Most have hundreds of verse stanzas and almost no tsawa, so they are the clearest case for adding annotations.\n",
     md(A.head(30), cols, names),
     f"\n## 2. Training, validation and test books where tsawa is a small share of the verse ({len(B)} books)\n",
     "Criteria: at least 50 unlabelled runs and run characters at least 5 times the tsawa characters. Verse left unlabelled here teaches the model that verse is not tsawa (the `audit_before_v6` under-labelling concern). **The validation and test books matter most, since they also lower the measured score:** " + (", ".join(f"`{r.pecha_id}` ({r.split})" for r in Bev.itertuples()) or "none") + ".\n",
     md(B.head(30), cols, names),
     f"\n## 3. Probably fine\n",
     f"{len(ok)} training/validation/test books have at least 50 unlabelled runs but their tsawa already covers more than half as much text as the unlabelled verse (ratio under 2), for example `P000078` (ratio 0.9). Their unlabelled verse is likely quotation or the author's own stanzas next to a well-annotated root text. Listed in the CSV (`ratio` column).\n"]

# ---------------- every-layer version ----------------
d0 = df.fillna(0)
ovc = [c for c in d0.columns if c.startswith("overlap_")]
C = d0.sort_values("strict_runs_no_layer", ascending=False)
top = C.head(25)
t2 = pd.DataFrame({"book": top.pecha_id, "split": top.split, "tsawa spans": top.n_tsawa.astype(int), "has Commentary": top.has_commentary,
                   "runs outside Quotation/Citation": top.strict_runs_unexplained.astype(int), "runs outside every layer": top.strict_runs_no_layer.astype(int),
                   "chars": top.strict_chars_no_layer.astype(int), "book chars": top.text_chars.astype(int)})
sp_tab = d0.groupby("split")[["strict_runs", "strict_runs_unexplained", "strict_runs_no_layer"]].sum().astype(int).reset_index()
sp_tab.columns = ["split", "all runs outside tsawa", "excluding Quotation/Citation (version above)", "excluding every layer"]
thr = pd.DataFrame([{"books with at least N runs": f">= {t}", "excluding Quotation/Citation": int((d0.strict_runs_unexplained >= t).sum()), "excluding every layer": int((d0.strict_runs_no_layer >= t).sum())} for t in (1, 20, 50, 100)])
L2 = ["\n## 4. Verse that no layer covers\n",
      "Same runs as above, but a run is dropped if it shares any character with **any** layer in the book (every `layers/v001/*.yml`: Commentary, Sabche, Chapter, Yigchung, Footnote, Quotation, Citation, Author, BookTitle and so on). Tsawa spans were already excluded. What is left is verse that no annotation touches.\n",
      f"- Runs: **{int(d0.strict_runs_unexplained.sum()):,} excluding Quotation/Citation only (previous version) -> {int(d0.strict_runs_no_layer.sum()):,} excluding every layer** ({int(d0.strict_runs.sum()):,} before any exclusion). Characters in those runs: {int(d0.strict_chars_no_layer.sum()):,}.",
      f"- Books with no such run: {int((d0.strict_runs_no_layer == 0).sum())} of {len(d0)} (was {int((d0.strict_runs_unexplained == 0).sum())}). In {int(((d0.strict_runs_no_layer == 0) & (d0.strict_runs_unexplained > 0)).sum())} books every previously counted run lies inside some layer.",
      f"- Almost all of the drop is Commentary. Of the {int(d0.strict_runs_unexplained.sum()):,} previous runs, the number that overlap each layer (a run can overlap several): " + ", ".join(f"{c[8:]} {int(d0[c].sum())}" for c in ovc if d0[c].sum() > 0) + ". In books with a Commentary layer the count goes " + f"{int(d0[d0.has_commentary].strict_runs_unexplained.sum()):,} -> {int(d0[d0.has_commentary].strict_runs_no_layer.sum())}; in books without one {int(d0[~d0.has_commentary].strict_runs_unexplained.sum()):,} -> {int(d0[~d0.has_commentary].strict_runs_no_layer.sum()):,}.\n",
      md(sp_tab, list(sp_tab.columns), list(sp_tab.columns)), "\n", md(thr, list(thr.columns), list(thr.columns)),
      "\nTop 25 books by verse runs that no layer covers:\n", md(t2, list(t2.columns), list(t2.columns))]
L += L2
open(f"{OUT}/missed_tsawa_summary.md", "w", encoding="utf-8").write("\n".join(L))
print("summary:", len(A), "dropped,", len(B), "used (", len(Bev), "val/test ),", len(ok), "fine")

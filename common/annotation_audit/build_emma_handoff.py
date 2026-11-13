"""Build the tsawa re-annotation handoff: emma/train.txt, emma/eval.txt, emma/test.txt (one problem book per line, with what to do).
Read-only on the data. Default is a dry run (prints counts and the interlinear picks); pass --write to write the three files.
Dropped books (tsawa layer but fewer than 10 spans, not in split_v3) are listed in train.txt. Run from the repo root."""
import glob, sys
import numpy as np, pandas as pd

MIN_PLAIN_DROPPED_RUNS = 50     # a dropped book with fewer verse runs than this (and no other flag) is not listed; set 0 to list all 88
AUD = "common/annotation_audit"; OUTDIR = f"{AUD}/emma"
T = pd.read_csv("tsawa/data/processed/tsawa_spans_merged.csv"); T = T[~T.dropped.astype(bool)]; T["len"] = T.end - T.start
SPLIT = pd.read_csv("data/shared/split_v3_frozen.csv", comment="#").set_index("pecha_id").split.to_dict()
ALL = sorted(T.pecha_id.unique())
split_of = lambda p: {"val": "eval"}.get(SPLIT.get(p, "dropped"), SPLIT.get(p, "dropped"))   # train / eval / test / dropped
nspans = T.groupby("pecha_id").size().to_dict()
drift = pd.read_csv("tsawa/audit/tsawa_drift_train_val.csv").set_index("pecha_id")
flag_tv = pd.read_csv("tsawa/audit/tsawa_drift_train_val_flagged.csv").set_index("pecha_id")
mv = pd.read_csv(f"{AUD}/missed_tsawa_by_book.csv").fillna(0).set_index("pecha_id")
under = pd.read_csv(f"{AUD}/under_annotated_tsawa_books.csv").set_index("pecha_id")
reasons = {p: [] for p in ALL}        # book -> [(category, text)]
def add(p, cat, text): reasons[p].append((cat, text))

# 1. offset drift: Sabche-realigned old-batch books (train/val from the drift scan, categories A and B) + the five drifted test books
test_drift = ["P000027", "P000083", "P000144", "P000242", "P000269"]
for p, r in flag_tv.iterrows():
    if r.category[0] in "AB":
        add(p, "drift", f"re-align positions: only {r.start_ok_pct:.0f}% of its {int(r.spans)} tsawa spans start at a line start (clean books ~100%)"
                        + (f"; metre improves when spans move about {int(r.best_shift_pm60):+d} chars" if r.metre_best - r.metre_shift0 >= 5 else "") + (" (Sabche-realigned old-batch book)" if r.sabche_realigned == True else " (old-batch book)"))
for p in test_drift:
    r = drift.loc[p]
    add(p, "drift", f"re-align positions: only {r.start_ok_pct:.0f}% of its {int(r.spans)} tsawa spans start at a line start (clean books ~100%)" + (" (Sabche-realigned old-batch book)" if r.sabche_realigned == True else " (not Sabche-realigned, same pattern)"))

# 1b. Sabche-realigned old-batch books that are dropped from the tsawa dataset (not covered by the train/val drift scan): measure them the same way
SV = pd.read_csv("sabche/data/processed/sabche_book_verdicts.csv").set_index("pecha_id")
def base(p): return open([x for x in glob.glob(f"data/raw_opf/{p}.opf/*") if x.endswith(".opf")][0] + "/base/v001.txt", encoding="utf-8").read()
for p in ALL:
    if split_of(p) == "dropped" and p in SV.index and SV.realigned.get(p) == True:
        b = base(p); g = T[T.pecha_id == p]; ok = float(np.mean([s == 0 or b[s - 1] == "\n" for s in g.start.astype(int)])) * 100
        add(p, "drift", f"re-align positions: Sabche-realigned old-batch book; only {ok:.0f}% of its {len(g)} tsawa span(s) start at a line start (clean books ~100%)" if ok < 60
            else f"check alignment: Sabche-realigned old-batch book; {ok:.0f}% of its {len(g)} tsawa span(s) start at a line start")

# 2. interlinear: found, not assumed. Rule: at least 30 spans and at least 40% of them under 10 characters.
cand = T.groupby("pecha_id").agg(spans=("len", "size"), median_len=("len", "median"), pct_under10=("len", lambda x: 100 * (x < 10).mean())).reset_index()
inter = cand[(cand.spans >= 30) & (cand.pct_under10 >= 40)].sort_values("pct_under10", ascending=False)
for r in inter.itertuples():
    add(r.pecha_id, "interlinear", f"not really tsawa: {r.spans} short word-fragment spans (median {int(r.median_len)} chars, {r.pct_under10:.0f}% under 10 chars) look like interlinear glosses; remove them or move them to the right layer, keep only real root-text verse")


# 1c. old-batch books in train/val where 60-97% of the spans start at a line start: possible partial drift, ask for a check
for p in ALL:
    if split_of(p) in ("train", "eval") and p in drift.index and p not in flag_tv.index:
        r = drift.loc[p]
        if r.batch == "old" and 60 <= r.start_ok_pct < 97:
            add(p, "drift", f"check alignment: old-batch book; {r.start_ok_pct:.0f}% of its {int(r.spans)} tsawa spans start at a line start (clean books ~100%)"
                            + (f" and only {r.end_ok_pct:.0f}% end before a shad or newline" if r.end_ok_pct < 60 else "")
                            + "; re-align the spans that start or end mid-line, part of the book may be shifted")

# 2b. borderline interlinear: not word fragments by the rule above, but short, mid-line spans
for p, why in (("I9CB71958", "median {m} chars, only {s:.0f}% start at a line start"), ("IB522F095", "median {m} chars, {u:.0f}% under 10 chars, only {s:.0f}% start at a line start")):
    g = T[T.pecha_id == p]; r = drift.loc[p]
    add(p, "interlinear", f"possible interlinear (borderline): {len(g)} short spans ({why.format(m=int(g.len.median()), u=100 * (g.len < 10).mean(), s=r.start_ok_pct)}); check whether they are glosses inside commentary lines, not real root text; remove those, keep only real tsawa")

# 4b. train books whose tsawa covers under a fifth of their verse (>= 50 verse runs outside Quotation/Citation); P000219, P000074, P000193 are already listed for drift
for p in ("P000218", "IA2F1ACFA", "IDDA7F69E"):
    r = mv.loc[p]
    add(p, "unlabeled-verse", f"mark missed tsawa: only {int(r.n_tsawa)} tsawa spans ({int(r.tsawa_chars):,} chars) but {int(r.strict_runs_unexplained)} verse runs outside Quotation/Citation ({int(r.strict_chars):,} chars of verse); the unmarked verse teaches the model that verse in this book is not tsawa, annotate the root-text verse")
for p in ("P000219", "P000074", "P000193"):
    r = mv.loc[p]
    add(p, "unlabeled-verse", f"also mark missed tsawa: {int(r.strict_runs_unexplained)} verse runs outside Quotation/Citation for {int(r.n_tsawa)} tsawa spans")
r = drift.loc["P000145"]
add("P000145", "drift", f"check alignment: old-batch book; only {r.start_ok_pct:.0f}% of its {int(r.spans)} tsawa spans start at a line start and {r.end_ok_pct:.0f}% end before a shad or newline (the verse metre itself is fine); re-align the spans that start or end mid-line")

# 3. mislabeled / verify gold
add("I23323023", "mislabeled", "not really tsawa: spans 0, 205 and 206 are a bare mantra, the scribe's note and a dated colophon, remove them; also verify the ~175 numbered-question spans (question-and-answer text) really are tsawa")
add("I85485484", "mislabeled", "possible label swap: only 1 tsawa span but 15 of its 22 Commentary spans are verse quatrains (isverse); check which layer the verse belongs to")
add("I4B1806B4", "mislabeled", "verify gold: tsawa spans are quoted prose mixed with verse (each followed by an explanation); decide whether quoted prose counts as tsawa")
add("I454D2699", "mislabeled", "verify gold: tsawa spans are practice liturgy, mantras and verse quoted before an explanation; decide whether they count as tsawa")

# 4. under-annotated (<= 5 tsawa spans and >= 50 verse runs no layer covers; all are dropped books)
for p, r in under.iterrows():
    add(p, "under-annotated", f"mark missed tsawa: only {int(r.n_tsawa)} tsawa span(s) but {int(r.strict_runs_no_layer)} verse runs that no layer covers (rank {int(r['rank'])} of {len(under)}); annotate the root-text verse")

# 5. weakest test books (per-book F1 on the saved v6 predictions, IoU 0.5)
add("P000269", "weak-test", "verify gold: weakest test book (F1 0.20): model predicts 339 spans against 63 gold; 50 verse runs are uncovered and 171 quoted-verse runs sit in Quotation/Citation spans")
add("ICDC84458", "weak-test", "verify gold: weakest test book (F1 0.25): model predicts 268 spans against 106 gold; 217 quoted-verse runs sit in Quotation/Citation spans, check whether quoted verse should be tsawa")
add("P000242", "weak-test", "verify gold: weakest test book (F1 0.15): see re-align note, labels do not fit the text")
add("P000144", "weak-test", "verify gold: weakest test book (F1 0.29): see re-align note, labels do not fit the text")
add("IE5895799", "weak-test", "verify gold: weakest test book (F1 0.30): model predicts 287 spans against 55 gold; 99 quoted-verse runs sit in Quotation/Citation spans, check whether quoted verse should be tsawa")

# 6. dropped with tsawa (fewer than 10 spans): listed in train.txt
dropped = [p for p in ALL if split_of(p) == "dropped"]
for p in dropped:
    u = int(mv.loc[p].strict_runs_no_layer) if p in mv.index else 0          # in no layer at all
    ua = int(mv.loc[p].strict_runs_unexplained) if p in mv.index else 0      # outside Quotation/Citation (includes verse inside Commentary spans)
    if p in under.index: add(p, "dropped", f"dropped from the dataset (only {nspans[p]} tsawa spans)")
    elif ua < MIN_PLAIN_DROPPED_RUNS and not reasons[p]: continue      # little verse and nothing else flagged: left out of the handoff
    else:
        verse = f"{ua} verse runs outside Quotation/Citation" + (f", {u} of them in no layer at all" if ua else "")
        more = (f"annotate all its root-text verse ({verse}) so it reaches 10 or more spans" if ua >= 20
                else f"annotate any root text it has so it reaches 10 or more spans (only {verse}, so check first whether there is more tsawa)")
        add(p, "dropped", f"mark missed tsawa: dropped book with {nspans[p]} tsawa span(s); {more}")

# ---- assemble
FILE = {"train": "train", "dropped": "train", "eval": "eval", "test": "test"}
lines = {"train": [], "eval": [], "test": []}
for p in ALL:
    if not reasons[p]: continue
    f = FILE[split_of(p)]
    lines[f].append((p, split_of(p), reasons[p]))
def fmt(p, sp, rs):
    cats = list(dict.fromkeys(c for c, _ in rs))
    # merge wording: a dropped book already described by another category keeps only that category's text
    texts = [t for c, t in rs if not (c == "dropped" and len(rs) > 1)]
    return f"{p} [{sp}] ({', '.join(c for c in cats if not (c == 'dropped' and len(cats) > 1))}): " + " | ".join(texts)
print("interlinear picks (>=30 spans, >=40% under 10 chars):"); print(inter.assign(median_len=inter.median_len.astype(int), pct_under10=inter.pct_under10.round(1)).merge(pd.Series({p: split_of(p) for p in inter.pecha_id}, name="split").rename_axis("pecha_id").reset_index(), on="pecha_id").to_string(index=False))
print("\nborderline, not picked:", {p: dict(median_len=int(T[T.pecha_id == p].len.median()), pct_under10=round(100 * (T[T.pecha_id == p].len < 10).mean(), 1), spans=nspans[p]) for p in ["I9CB71958", "IB522F095"]})
print("\nbooks per file:", {k: len(v) for k, v in lines.items()}, "| total", sum(len(v) for v in lines.values()))
cats_by_file = {k: pd.Series([c for _, _, rs in v for c in dict.fromkeys(c for c, _ in rs)]).value_counts().to_dict() for k, v in lines.items()}
for k, v in cats_by_file.items(): print(f"  {k}: by reason (a book can have several): {v}")
print("dropped books in train.txt:", len(dropped), "| flagged train-split books:", sum(1 for _, sp, _ in lines['train'] if sp == 'train'))
if "--write" in sys.argv:
    import os; os.makedirs(OUTDIR, exist_ok=True)
    for k, v in lines.items():
        order = {"drift": 0, "interlinear": 1, "mislabeled": 2, "under-annotated": 3, "unlabeled-verse": 3, "weak-test": 4, "dropped": 5}
        weak = lambda rs: all(t.startswith(('check alignment', 'possible interlinear')) for c, t in rs if c in ('drift', 'interlinear'))   # weaker 'check' asks go after the firm ones
        v = sorted(v, key=lambda x: (min(order[c] for c, _ in x[2]), weak(x[2]), x[0]))
        open(f"{OUTDIR}/{k}.txt", "w", encoding="utf-8").write("\n".join(fmt(*x) for x in v) + "\n")
    print("wrote", OUTDIR)
else:
    print("\n(dry run: nothing written; pass --write)")

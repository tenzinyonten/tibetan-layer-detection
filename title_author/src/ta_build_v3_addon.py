#!/usr/bin/env python3
"""Build gold_dataset_all_v3 = gold_dataset_all + auto-span works from the BDRC v3 release (TRAIN ONLY).

v3 work kept if: TEXT, typed title and author both found exactly once in the text (title = first hit, author = last hit,
same rule as fill_spans_auto.py), not a near-duplicate of any gold work (all splits), one copy per duplicate cluster.
validation and test are copied unchanged from gold_dataset_all. source column = "v3_auto".

    python title_author/src/ta_build_v3_addon.py CLEAN_DIR [--out-dir DIR]
"""
import argparse, csv, hashlib, json, os, re, sys, zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common")); sys.path.insert(0, str(ROOT / "title_author/src"))
from build_tsawa_dataset import IGNORE_LABEL, label_tokens, pack_window, sliding_windows, special_token_ids  # noqa: E402
from ta_build_gold_dataset import LABEL2ID, ID2LABEL, OUT, norm, regions, resolve_overlap, skip_leading_ws, strip_title_marks, strip_title_formula, tib_letters  # noqa: E402
from ta_load_spans import GOLD, load  # noqa: E402

STRIP = " \t\r\n།་༎"


def find(t, p, last):
    for q in (p.strip(), p.strip(STRIP)):
        if not q: continue
        i = t.rfind(q) if last else t.find(q)
        if i == -1: continue
        return None if t.count(q) > 1 else (i, i + len(q))
    return None


def sig(text):
    head = norm(text)[:1500]
    return sorted({zlib.crc32(head[i:i + 8].encode()) for i in range(len(head) - 7)})[:8] if len(head) >= 200 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_dir", type=Path)
    ap.add_argument("--k", type=int, default=3000)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=5120)
    ap.add_argument("--base", default="gold_dataset_all", help="gold dataset folder (under data/processed) to add v3 to")
    ap.add_argument("--single", action="store_true", help="one sequence per work: top + separator + bottom")
    ap.add_argument("--all-groups", action="store_true",
                    help="also keep title-only and author-only works; tokens where the missing side could be get label -100")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tokenizer", default=str(Path.home() / ".cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots/c5955035435e2bf121cde7f3c8863ef52ff35d82"))
    a = ap.parse_args()
    a.out_dir = a.out_dir or OUT / (f"{a.base}_v3_full" if a.all_groups else f"{a.base}_v3")
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    # ---- gold index (every annotated work, all splits) ----
    gidx, gtyped, gfp = defaultdict(set), set(), set()
    n = 0
    for r in load()[0]:
        pass
    byf = defaultdict(list)
    for r in load()[0]: byf[r["file"]].append(r)
    for f, rs in byf.items():
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            text = c[r["span_start"]:r["span_end"]]
            for h in sig(text): gidx[h].add(n)
            t, au = norm(r["title_text"]), norm(r["author_text"])
            if len(t) > 15: gtyped.add((t, au))
            s = norm(text)
            if len(s) >= 200: gfp.add(hashlib.md5(s[:300].encode()).hexdigest())
            n += 1
    print("gold works indexed:", n)

    # ---- v3 candidates ----
    st, cands = Counter(), []
    for d in sorted(os.listdir(a.clean_dir)):
        try:
            m = json.load(open(a.clean_dir / d / "meta.json", encoding="utf-8"))
            text = open(a.clean_dir / d / f"{d}.txt", encoding="utf-8").read()
        except Exception:
            st["unreadable"] += 1; continue
        if m.get("segment_type") != "TEXT": st["non_text"] += 1; continue
        st["readable_text_works"] += 1
        t = find(text, m["title"], False) if m["title"].strip() else None
        au = find(text, m["author"], True) if m["author"].strip() else None
        kind = "both" if t and au else "title_only" if t else "author_only" if au else None
        if kind is None or (kind != "both" and not a.all_groups):
            st["no_usable_span" if kind is None else "skipped_one_sided"] += 1; continue
        if kind == "both":                # same cleaning as the annotated data
            t = strip_title_formula(text, strip_title_marks(text, t))
            T, A = text[t[0]:t[1]], text[au[0]:au[1]]
            def junk(x):
                nt = len(re.sub(r"\s", "", x)); non = sum(1 for ch in x if not ("\u0F00" <= ch <= "\u0FFF") and not ch.isspace())
                return tib_letters(x) == 0 or len(re.sub(r"[་།\s]", "", x)) <= 2 or "\n\n" in x or (nt and non / nt > 0.2)
            if junk(T) or junk(A) or norm(T) == norm(A): st["cleaning_dropped"] += 1; continue
        t_, au_ = norm(m["title"]), norm(m["author"])
        s = norm(text)
        sg = sig(text)
        if (len(t_) > 15 and (t_, au_) in gtyped) or (len(s) >= 200 and hashlib.md5(s[:300].encode()).hexdigest() in gfp):
            st["dup_of_gold"] += 1; continue
        shared = Counter(j for h in sg for j in gidx.get(h, ()) if len(gidx[h]) <= 20)
        if any(v >= 4 for v in shared.values()): st["dup_of_gold"] += 1; continue
        cands.append(dict(id=d, text=text, title=t, author=au, ttl=m["title"], aut=m["author"], sig=sg, typed=(t_, au_), L=len(text), kind=kind))
    print(dict(st), "candidates:", len(cands))

    # ---- dedupe inside v3 (same typed title+author, or MinHash >= 4 shared) ----
    parent = list(range(len(cands)))
    def f_(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    typed = defaultdict(list); idx = defaultdict(list)
    for i, c in enumerate(cands):
        if len(c["typed"][0]) > 15: typed[c["typed"]].append(i)
        for h in c["sig"]: idx[h].append(i)
    for ws in typed.values():
        for w in ws[1:]: parent[f_(w)] = f_(ws[0])
    shared = Counter()
    for ws in idx.values():
        if len(ws) > 20: continue
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)): shared[(ws[i], ws[j])] += 1
    for (i, j), v in shared.items():
        if v >= 4: parent[f_(i)] = f_(j)
    cl = defaultdict(list)
    for i in range(len(cands)): cl[f_(i)].append(i)
    keep = [min(ws, key=lambda w: (cands[w]["kind"] != "both", -cands[w]["L"], cands[w]["id"])) for ws in cl.values()]
    st["dup_inside_v3"] = len(cands) - len(keep)
    cands = [cands[i] for i in sorted(keep)]
    print("after dedupe inside v3:", len(cands), "(dropped", st["dup_inside_v3"], ")")

    # ---- tokenize + label ----
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True); tok.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = special_token_ids(tok)
    content_len = a.max_length - 2
    gold = load_from_disk(str(OUT / a.base))
    feats = gold["train"].features
    rows, counts, nolabel, lost = [], Counter(), 0, Counter()
    for c in cands:
        text = c["text"]
        t, au0 = c["title"], c["author"]
        au, note = (resolve_overlap(t, au0) if t and au0 else (au0, ""))
        if note: lost[note] += 1
        lost["works_" + c["kind"]] += 1
        mid = len(text) / 2
        work_id = f"v3:{c['id']}#0"
        pieces = []
        for rname, rs_, re_ in regions(text, a.k, [t, au]):
            rt = text[rs_:re_]; sp = []
            for kind, s in (("TITLE", t), ("AUTHOR", au)):
                if s is None: continue
                lo, hi = max(s[0], rs_), min(s[1], re_)
                if hi > lo: sp.append((lo - rs_, hi - rs_, work_id, kind))
            sp.sort()
            enc = tok(rt, add_special_tokens=False, return_offsets_mapping=True)
            offs = [tuple(o) for o in enc["offset_mapping"]]
            labels = label_tokens(skip_leading_ws(offs, rt), sp, LABEL2ID)
            if c["kind"] != "both":   # missing side: do not teach "O" where it may be
                for ti, (o0, _) in enumerate(offs):
                    if labels[ti] != 0: continue
                    if (c["kind"] == "title_only") == (rs_ + o0 >= mid): labels[ti] = -100; lost["tokens_masked"] += 1
            pieces.append((rname, rs_, list(enc["input_ids"]), list(labels), offs, rt))
        if a.single and len(pieces) == 2:       # top + <eos> + bottom as ONE sequence
            (_, s1, i1, l1, o1, t1), (_, _, i2, l2, o2, _) = pieces
            shift = len(t1) + 1
            pieces = [("both", s1, i1 + [sep_id] + i2, l1 + [IGNORE_LABEL] + l2,
                       o1 + [(len(t1), len(t1))] + [(x + shift, y + shift) for x, y in o2], None)]
        for rname, rs_, ids_, labels, offs, _rt in pieces:
            for wi, (ws, we) in enumerate(sliding_windows(len(ids_), content_len, a.stride)):
                ex = pack_window(ids_, labels, offs, ws, we, cls_id, sep_id, pad_id, a.max_length)
                ex.update(work_id=work_id, cid=c["id"].split("_")[0], source="v3_auto", region=rname,
                          region_start=rs_, window_index=wi, n_tokens_region=len(ids_))
                counts.update(x for x in ex["labels"] if x != IGNORE_LABEL)
                nolabel += not any(x > 0 for x in ex["labels"])
                rows.append(ex)
    add = Dataset.from_list(rows, features=feats)
    dd = DatasetDict(train=concatenate_datasets([gold["train"], add]), validation=gold["validation"], test=gold["test"])
    dd.save_to_disk(str(a.out_dir))
    stats = dict(v3_filter=dict(st), v3_works_added=len(cands), v3_windows_added=len(rows), v3_windows_no_label=nolabel,
                 v3_label_counts={ID2LABEL[k]: v for k, v in sorted(counts.items())}, notes=dict(lost),
                 windows={s: len(v) for s, v in dd.items()})
    (OUT / f"{a.out_dir.name}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with (OUT / f"{a.out_dir.name}_works.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["v3_id", "kind", "typed_title", "typed_author", "title_start", "title_end", "author_start", "author_end"])
        for c in cands: w.writerow([c["id"], c["kind"], c["ttl"], c["aut"], *(c["title"] or ["", ""]), *(c["author"] or ["", ""])])
    print(json.dumps(stats, indent=2)); print("saved", a.out_dir)


if __name__ == "__main__":
    main()

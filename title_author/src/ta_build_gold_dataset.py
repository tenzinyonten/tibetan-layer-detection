#!/usr/bin/env python3
"""Build the TITLE/AUTHOR BIO DatasetDict from the annotator-span JSONs (no training).

Per work (TEXT segment): keep head [0,K) and tail [L-K,L) of the segment text (whole text if L<=2K).
Spans come from the provided offsets only. Labels: token-start rule via common/build_tsawa_dataset.label_tokens.
Split by group of collection ids (ids linked by shared text / shared title+author are one group).

    python title_author/src/ta_build_gold_dataset.py [--limit N]
"""
import argparse, csv, hashlib, json, random, re, sys, zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common")); sys.path.insert(0, str(ROOT / "title_author/src"))
from build_tsawa_dataset import IGNORE_LABEL, label_tokens, pack_window, sliding_windows, special_token_ids  # noqa: E402
from ta_load_spans import GOLD, TRAIN_SOURCES, load  # noqa: E402

LABEL2ID = {"O": 0, "B-TITLE": 1, "I-TITLE": 2, "B-AUTHOR": 3, "I-AUTHOR": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test", "val_ocr": "validation", "test_ocr": "test"}   # OCR holdouts join the same splits (see `source` column)
OUT = ROOT / "title_author/data/processed"
norm = lambda s: re.sub(r"[\s།་༎]+", "", s)


def skip_leading_ws(offs, text):
    """mmBERT tokens carry the space before a word (" ར" = offsets 2-4 for the letter at 3). The token-start rule would
    call that token outside a span that starts at the letter and put B on the NEXT token (the span starts one letter
    late). Only for labelling: move each token start past leading whitespace; stored offsets are unchanged."""
    out = []
    for s, e in offs:
        if text[s:e].strip():            # a whitespace-ONLY token keeps its offsets: inside a span it stays I (not O)
            while s < e and text[s].isspace(): s += 1
        out.append((s, e))
    return out


MARKS = "༄༅༆༇༈༉༊།༎ \n\t\r"          # opening marks, shad, whitespace
tib_letters = lambda s: len(re.findall(r"[\u0F40-\u0FBC]", s))


def strip_title_marks(text, t):
    """Move a title span's start past leading opening marks / shad / whitespace (the annotators included them only sometimes)."""
    s, e = t
    while s < e and text[s] in MARKS: s += 1
    return (s, e)


END_FORMULA = re.compile(r"(?:ཞེས་བྱ་བ་?)?བཞུགས(?:་སོ)?[།༎་\s]*$|ཞེས་བྱ་བ[།༎་\s]*$")


def strip_title_formula(text, t):
    """Remove a closing formula (ཞེས་བྱ་བ / བཞུགས་སོ) from the END of a title span only; a formula in the middle is part of the title."""
    s, e = t
    m = END_FORMULA.search(text[s:e])
    if m and tib_letters(text[s:s + m.start()]) > 0:
        e = s + m.start()
        while e > s and text[e - 1].isspace(): e -= 1
    return (s, e)


def clean_rows(rows, log_path):
    """Drop works whose title or author span is junk (no Tibetan letters, 1-2 characters, paragraph break inside),
    and strip the leading opening mark from titles. Returns the kept rows; writes the dropped ones to log_path."""
    byf = defaultdict(list)
    for r in rows: byf[r["file"]].append(r)
    keep, dropped, stripped, formula = [], [], 0, 0
    for f, rs in byf.items():
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            why = []
            for name, x in (("title", c[slice(*r["title_span"])]), ("author", c[slice(*r["author_span"])])):
                if tib_letters(x) == 0: why.append(f"{name} has no Tibetan letters")
                elif len(re.sub(r"[་།\s]", "", x)) <= 2: why.append(f"{name} is 1-2 characters")
                if "\n\n" in x: why.append(f"{name} has a paragraph break")
                nt = len(re.sub(r"\s", "", x)); non = sum(1 for ch in x if not ("\u0F00" <= ch <= "\u0FFF") and not ch.isspace())
                if nt and non / nt > 0.2: why.append(f"{name} is over 20% non-Tibetan characters")
            tn = re.sub(r"[\s།་༎༄༅]+", "", c[slice(*r["title_span"])]); an = re.sub(r"[\s།་༎༄༅]+", "", c[slice(*r["author_span"])])
            if tn and tn == an: why.append("title text equals author text")
            if why: dropped.append((f, r["seg"], r["source"], r["cid"], "; ".join(why))); continue
            s, e = r["title_span"]; s2, _ = strip_title_marks(c, (s, e))
            s2, e2 = strip_title_formula(c, (s2, e))
            if s2 != s: r["title_span"] = (s2, e2); stripped += 1
            if e2 != e: r["title_span"] = (s2, e2); formula += 1
            keep.append(r)
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "segment", "source", "cid", "reason"]); w.writerows(sorted(dropped))
    print(f"cleaning: dropped {len(dropped)} works", dict(Counter(x for d in dropped for x in d[4].split('; '))),
          f"| titles with the leading mark stripped: {stripped} | titles with the closing formula stripped: {formula}")
    return keep


def snap_forward(text, pos, limit=600):
    """First line start at/after pos; else just after the next tsek/shad/space; else pos."""
    i = text.find("\n", pos, pos + limit)
    if i >= 0: return i + 1
    for j in range(pos, min(len(text), pos + limit)):
        if text[j] in "་། ": return j + 1
    return pos


def snap_backward(text, pos, limit=600):
    """Last line start at/before pos; else just after the previous tsek/shad/space; else pos."""
    i = text.rfind("\n", max(0, pos - limit), pos + 1)
    if i >= 0: return i + 1
    for j in range(pos, max(0, pos - limit), -1):
        if text[j - 1] in "་། ": return j
    return pos


def paragraphs(text):
    """(start, end) of each blank-line separated paragraph."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S(?:.|\n(?!\s*\n))*", text)]


def regions_paras(text, spans, n=2):
    """First n and last n paragraphs; whole text if there are <= 2n. A span crossing an edge moves the edge."""
    paras = paragraphs(text); L = len(text)
    if len(paras) <= 2 * n: return [("full", 0, L)]
    spans = [sp for sp in spans if sp]
    e, st = paras[n - 1][1], paras[-n][0]
    for a_, b_ in spans:
        if a_ < e < b_: e = next(pe for _, pe in paras if pe >= b_)
        if a_ < st < b_: st = next(ps for ps, _ in reversed(paras) if ps <= a_)
    return [("full", 0, L)] if e >= st else [("head", 0, e), ("tail", st, L)]


def regions(text, K, spans):
    """Head/tail crop that starts/ends on a line (or syllable) boundary and never cuts a span.
    Whole text is one region if it is short or head and tail would meet."""
    L = len(text)
    if L <= 2 * K: return [("full", 0, L)]
    spans = [sp for sp in spans if sp]
    e = snap_forward(text, K)
    for a_, b_ in spans:
        if a_ < e < b_: e = snap_forward(text, b_)
    st = snap_backward(text, L - K)
    for a_, b_ in spans:
        if a_ < st < b_: st = snap_backward(text, a_)
    return [("full", 0, L)] if e >= st else [("head", 0, e), ("tail", st, L)]


def resolve_overlap(t, a):
    """Title wins; author keeps its longest piece outside the title (None if nothing left)."""
    if not (t[0] < a[1] and a[0] < t[1]):
        return a, ""
    pieces = [p for p in ((a[0], min(a[1], t[0])), (max(a[0], t[1]), a[1])) if p[1] > p[0]]
    if not pieces:
        return None, "author_inside_title_dropped"
    return max(pieces, key=lambda p: p[1] - p[0]), "author_trimmed"


def assign_all(rows, frozen_path, k, seed):
    """Place the OCR collections around the FROZEN clean split (clean ids never move).
    Two works are linked if they share: typed (title, author); a long typed title; the exact first 300 chars;
    or a near-identical text start (MinHash, tolerates OCR errors). An OCR collection linked to clean ids of one
    split follows it (train->train, val->val_ocr, test->test_ocr, never into clean val/test); linked to ids of
    several splits -> excluded; unlinked -> stratified 80/10/10 by collection group (train / val_ocr / test_ocr)."""
    fixed = {l["cid"]: l["split"] for l in csv.DictReader(
        x for x in open(frozen_path, encoding="utf-8") if not x.startswith("#"))}
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    first, edges, sig_index, sigs = {}, [], defaultdict(list), []
    byf = defaultdict(list)
    for r in rows: byf[r["file"]].append(r)
    for f, rs in byf.items():
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            find(r["cid"]); cid = r["cid"]
            seg = norm(c[r["span_start"]:r["span_end"]])
            tt, ta_ = norm(r["title_text"]), norm(r["author_text"])
            st_, sa_ = norm(c[slice(*r["title_span"])]), norm(c[slice(*r["author_span"])])
            keys = []
            if len(tt) > 15: keys.append(f"typed:{tt}|{ta_}")
            if len(tt) >= 25: keys.append(f"typedtitle:{tt}")
            if len(st_) > 15: keys.append(f"span:{st_}|{sa_}")
            if len(seg) >= 200: keys.append("fp:" + hashlib.md5(seg[:300].encode()).hexdigest())
            for key in keys:
                if key in first and first[key] != cid: edges.append((first[key], cid, key.split(":")[0]))
                else: first.setdefault(key, cid)
            head = seg[:1500]
            if len(head) >= 200:
                sk = sorted({zlib.crc32(head[i:i + 8].encode()) for i in range(len(head) - 7)})[:8]
                w = len(sigs); sigs.append((cid, sk))
                for h in sk: sig_index[h].append(w)
    shared = Counter()
    for h, ws in sig_index.items():
        if len(ws) > 20: continue   # boilerplate shingle, not evidence of a copy
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if sigs[ws[i]][0] != sigs[ws[j]][0]: shared[(ws[i], ws[j])] += 1
    for (i, j), n in shared.items():
        if n >= 4: edges.append((sigs[i][0], sigs[j][0], "minhash"))
    for x, y, _ in edges: parent[find(x)] = find(y)
    comp = defaultdict(set)
    for r in rows: comp[find(r["cid"])].add(r["cid"])
    works_per_cid = Counter(r["cid"] for r in rows)
    strata = defaultdict(Counter)
    for r in rows:
        L = r["span_end"] - r["span_start"]
        strata[r["cid"]][(r["source"], "short" if L <= 2 * k else "long")] += 1
    split, why, free = {}, {}, []
    for cs in comp.values():
        fs = {fixed[c] for c in cs if c in fixed}
        new = [c for c in cs if c not in fixed]
        for c in cs:
            if c in fixed: split[c] = fixed[c]; why[c] = "frozen clean split"
        if not new: continue
        if len(fs) >= 2:
            for c in new: split[c] = "excluded"; why[c] = "linked to clean ids of several splits"
        elif len(fs) == 1:
            s_ = {"train": "train", "val": "val_ocr", "test": "test_ocr"}[next(iter(fs))]
            for c in new: split[c] = s_; why[c] = "linked to clean " + next(iter(fs))
        else:
            free.append(new)
    gstr = [sum((strata[c] for c in g), Counter()) for g in free]
    linked = {s_: Counter() for s_ in ("val_ocr", "test_ocr")}
    for c, s_ in split.items():
        if s_ in linked: linked[s_].update(strata[c])
    tot = sum(gstr, Counter()) + sum(linked.values(), Counter()) + sum(
        (strata[c] for c, s_ in split.items() if s_ == "train" and c not in fixed), Counter())
    rng = random.Random(seed); left = set(range(len(free)))
    for name, cap in (("test_ocr", 40), ("val_ocr", 100)):
        cur = Counter(linked[name]); want = {d: 0.1 * tot[d] for d in tot}
        dev = lambda c_: sum((c_[d] - want[d]) ** 2 / max(want[d], 1) for d in tot)
        while sum(cur.values()) < 0.97 * 0.1 * sum(tot.values()):
            cands = [i for i in left if sum(gstr[i].values()) <= cap]
            if not cands: break
            best = min(cands, key=lambda i: dev(cur + gstr[i]) - dev(cur) - rng.random() * 0.01)
            cur.update(gstr[best]); left.discard(best)
            for c in free[best]: split[c] = name; why[c] = "unlinked OCR, stratified"
    for i in left:
        for c in free[i]: split[c] = "train"; why[c] = "unlinked OCR, stratified"
    gid_of = {c: find(c) for c in split}
    gid_of = {c: list(dict.fromkeys(gid_of.values())).index(g) for c, g in gid_of.items()}
    # audit: links that cross the train / held-out boundary (the leak we must not have)
    cross = Counter()
    for x, y, kind in edges:
        sx, sy = split.get(x), split.get(y)
        if sx in (None, "excluded") or sy in (None, "excluded"): continue
        if (sx == "train") != (sy == "train"):
            both_clean = x in fixed and y in fixed
            cross[("clean-clean (pre-existing)" if both_clean else "OCR involved", kind)] += 1
    global LAST_EDGES
    LAST_EDGES = edges
    print("links crossing train/held-out:", dict(cross))
    print("link types found:", dict(Counter(e[2] for e in edges)))
    return split, why, gid_of


def heldout_near_dups(rows, split):
    """Work-level audit: validation/test works that have a near-duplicate work in train
    (same typed title+author, or near-identical text start via MinHash). Returns {(file, seg): evidence}."""
    byf = defaultdict(list)
    for r in rows: byf[r["file"]].append(r)
    W, idx, typed = [], defaultdict(list), defaultdict(set)
    for f, rs in byf.items():
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            head = norm(c[r["span_start"]:r["span_end"]])[:1500]
            sk = sorted({zlib.crc32(head[i:i + 8].encode()) for i in range(len(head) - 7)})[:8] if len(head) >= 200 else []
            w = len(W); W.append((r["cid"], split[r["cid"]] == "train", (r["file"], r["seg"])))
            for h in sk: idx[h].append(w)
            t, au = norm(r["title_text"]), norm(r["author_text"])
            if len(t) > 15: typed[(t, au)].add(w)
    flag = defaultdict(set); shared = Counter()
    for ws in idx.values():
        if len(ws) > 20: continue
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if W[ws[i]][0] != W[ws[j]][0]: shared[(ws[i], ws[j])] += 1
    for (i, j), n in shared.items():
        if n >= 4:
            for a_, b_ in ((i, j), (j, i)):
                if W[b_][1] and not W[a_][1]: flag[a_].add("text")
    for ws in typed.values():
        tr = [w for w in ws if W[w][1]]
        for w in ws:
            if not W[w][1] and any(W[x][0] != W[w][0] for x in tr): flag[w].add("typed title+author")
    return {W[w][2]: "+".join(sorted(ev)) for w, ev in flag.items()}


SRC_RANK = {"tei": 0, "google_books": 1, "google_vision": 2, "paddleocr_v2": 3, "ocrv1-ws-ldv1": 4,
            "ocr_qwen_v1": 5, "paddleocr_v1": 6}


def dedupe_within_splits(rows, split):
    """Inside each split, cluster works that are the same work (same typed title+author, same first 300 chars,
    or near-identical text start via MinHash) and keep ONE per cluster: cleaner source first, then longer text.
    Returns {(file, seg): "dup of <file>#<seg> (evidence)"} for the works to drop."""
    byf = defaultdict(list)
    for r in rows: byf[r["file"]].append(r)
    W, idx, keyed = [], defaultdict(list), defaultdict(list)
    for f, rs in byf.items():
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            seg = norm(c[r["span_start"]:r["span_end"]]); head = seg[:1500]
            sk = sorted({zlib.crc32(head[i:i + 8].encode()) for i in range(len(head) - 7)})[:8] if len(head) >= 200 else []
            w = len(W); W.append((r, split[r["cid"]]))
            for h in sk: idx[h].append(w)
            t, au = norm(r["title_text"]), norm(r["author_text"])
            if len(t) > 15: keyed[("typed", W[w][1], t, au)].append(w)
            if len(seg) >= 200: keyed[("fp", W[w][1], hashlib.md5(seg[:300].encode()).hexdigest())].append(w)
    parent = list(range(len(W)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ev = defaultdict(set)
    def join(a_, b_, why):
        if W[a_][1] == W[b_][1] and a_ != b_:
            parent[find(a_)] = find(b_); ev[a_].add(why); ev[b_].add(why)
    for key, ws in keyed.items():
        for w in ws[1:]: join(ws[0], w, key[0])
    shared = Counter()
    for ws in idx.values():
        if len(ws) > 20: continue
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)): shared[(ws[i], ws[j])] += 1
    for (i, j), n in shared.items():
        if n >= 4: join(i, j, "text")
    clusters = defaultdict(list)
    for w in range(len(W)): clusters[find(w)].append(w)
    drop = {}
    for ws in clusters.values():
        if len(ws) < 2: continue
        keep = min(ws, key=lambda w: (SRC_RANK[W[w][0]["source"]], -(W[w][0]["span_end"] - W[w][0]["span_start"]), W[w][0]["file"]))
        for w in ws:
            if w != keep:
                r = W[w][0]; k = W[keep][0]
                drop[(r["file"], r["seg"])] = f"dup of {k['file']}#{k['seg']} ({'+'.join(sorted(ev[w]))}); cluster size {len(ws)}"
    sizes = Counter(len(ws) for ws in clusters.values() if len(ws) > 1)
    print("duplicate clusters inside splits by size:", dict(sorted(sizes.items())))
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3000, help="chars kept at each end when --crop chars")
    ap.add_argument("--crop", choices=["chars", "paras"], default="chars",
                    help="chars: first/last K chars snapped to a line; paras: first 2 / last 2 blank-line paragraphs")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=5120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true",
                    help="include the OCR sources; clean split stays frozen, OCR gets train/validation_ocr/test_ocr")
    ap.add_argument("--split-csv", type=Path, default=None,
                    help="per-work split file (file,seg,cid,split,unit) from ta_restratify.py; replaces the collection-level split")
    ap.add_argument("--single", action="store_true", help="one sequence per work: top + separator + bottom (instead of two windows)")
    ap.add_argument("--clean", action="store_true", help="drop junk-span works and strip the leading opening mark from titles")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. strat -> gold_dataset_all_strat, gold_all_strat_*.csv")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N files")
    ap.add_argument("--tokenizer", default=str(Path.home() / ".cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots/c5955035435e2bf121cde7f3c8863ef52ff35d82"))
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    a.out_dir = a.out_dir or OUT / (("gold_dataset_all" if a.all else "gold_dataset") + (f"_{a.tag}" if a.tag else ""))
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    from transformers import AutoTokenizer

    rows = [r for r in load()[0] if a.all or r["source"] in TRAIN_SOURCES]
    files = sorted({r["file"] for r in rows})[: a.limit] if a.limit else None
    if files is not None:
        fs = set(files); rows = [r for r in rows if r["file"] in fs]
    byf = defaultdict(list)
    for r in rows:
        byf[r["file"]].append(r)

    prefix = ("gold_all_" if a.all else "gold_") + (f"{a.tag}_" if a.tag else "")
    why = {}
    if a.split_csv:
        # per-work split: a collection may sit in several splits, so its works get cid "<cid>~<split>" while we build
        wsplit = {(l["file"], int(l["seg"])): (l["split"], int(l["unit"])) for l in csv.DictReader(
            x for x in open(a.split_csv, encoding="utf-8") if not x.startswith("#"))}
        used = defaultdict(set)
        for r in rows: used[r["cid"]].add(wsplit[(r["file"], r["seg"])][0])
        split, gid_of = {}, {}
        for r in rows:
            sp_, unit = wsplit[(r["file"], r["seg"])]
            r["cid_orig"] = r["cid"]
            if len(used[r["cid"]]) > 1: r["cid"] = f"{r['cid_orig']}~{sp_}"
            split[r["cid"]] = sp_; gid_of[r["cid"]] = unit
        byf = defaultdict(list)
        for r in rows: byf[r["file"]].append(r)
        print("per-work split:", dict(Counter(wsplit[(r["file"], r["seg"])][0] for r in rows)),
              "| collections in >1 split:", sum(len(v) > 1 for v in used.values()))
    elif a.all:
        split, why, gid_of = assign_all(rows, OUT / "gold_split_frozen.csv", a.k, a.seed)
        excluded = {c for c, v in split.items() if v == "excluded"}
        print("excluded collection ids:", len(excluded), "works:", sum(1 for r in rows if r["cid"] in excluded))
        rows = [r for r in rows if r["cid"] not in excluded]
        byf = defaultdict(list)
        for r in rows: byf[r["file"]].append(r)
    else:
        # ---- pass 1: groups (union-find over collection ids) ----
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y): parent[find(x)] = find(y)
        seen = {}
        for f, rs in byf.items():
            c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
            for r in rs:
                find(r["cid"])
                seg = norm(c[r["span_start"]:r["span_end"]])
                keys = []
                if len(seg) >= 200: keys.append("fp:" + hashlib.md5(seg[:300].encode()).hexdigest())
                t, au = norm(c[slice(*r["title_span"])]), norm(c[slice(*r["author_span"])])
                if len(t) > 15: keys.append(f"ta:{t}|{au}")
                for k in keys:
                    if k in seen: union(seen[k], r["cid"])
                    else: seen[k] = r["cid"]
        groups = defaultdict(list)
        for cid in {r["cid"] for r in rows}:
            groups[find(cid)].append(cid)
        works_per_cid = Counter(r["cid"] for r in rows)
        gl = sorted(groups.values(), key=lambda g: sorted(g)[0])
        # Diversity-aware stratified group split (test first = the benchmark, then val, rest = train).
        # Stratum = source x length (short = whole text in one region, long = head+tail). Groups go whole.
        # Test/val only take groups <= GROUP_CAP works (giant linked clusters stay in train, so no series dominates
        # the benchmark) and are rewarded for adding collection families the split does not have yet.
        CAPS, FAMILY_BONUS = {"test": 40, "val": 100}, 3.0
        fam = lambda c: (re.match(r"(W\d?[A-Z]+|IE\d?[A-Z]+)", c) or [None, c[:4]])[1]
        strata = defaultdict(Counter)
        for r in rows:
            L = r["span_end"] - r["span_start"]
            strata[r["cid"]][(r["source"], "short" if L <= 2 * a.k else "long")] += 1
        gstr = [sum((strata[c] for c in g), Counter()) for g in gl]
        gfam = [{fam(c) for c in g} for g in gl]
        tot = sum(gstr, Counter()); rng = random.Random(a.seed)
        split = {}; gid_of = {}; free = set(range(len(gl)))
        for name, frac in (("test", 0.1), ("val", 0.1)):
            GROUP_CAP = CAPS[name]
            cur = Counter(); fams = set(); want = {d: frac * tot[d] for d in tot}
            def dev(c): return sum((c[d] - want[d]) ** 2 / max(want[d], 1) for d in tot)
            while sum(cur.values()) < 0.97 * frac * sum(tot.values()):
                cands = [i for i in free if sum(gstr[i].values()) <= GROUP_CAP]
                def score(i):
                    new = cur + gstr[i]
                    return dev(new) - dev(cur) - FAMILY_BONUS * len(gfam[i] - fams) - rng.random() * 0.01
                if not cands: break
                best = min(cands, key=score)
                cur.update(gstr[best]); fams |= gfam[best]; free.discard(best)
                for c in gl[best]: split[c] = name
            print(name, {f"{k[0]}/{k[1]}": f"{v} ({100*v/tot[k]:.0f}%)" for k, v in sorted(cur.items())}, "families", len(fams))
        for i in free:
            for c in gl[i]: split[c] = "train"
        for i, g in enumerate(gl):
            for c in g: gid_of[c] = i
        print(f"collection ids {len(split)}, groups {len(gl)}, biggest group {max(len(g) for g in gl)} ids / "
              f"{max(sum(works_per_cid[c] for c in g) for g in gl)} works")

    if a.clean:
        rows = clean_rows(rows, OUT / f"{prefix}cleaning_dropped.csv")
    drop = heldout_near_dups(rows, split)
    print("held-out works dropped (near-duplicate of a train work):", len(drop),
          dict(Counter(split[r["cid"]] for r in rows if (r["file"], r["seg"]) in drop)))
    with (OUT / f"{prefix}heldout_dropped.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "segment", "split", "evidence"])
        sp_of = {(r["file"], r["seg"]): split[r["cid"]] for r in rows}
        for (f, sg), ev in sorted(drop.items()): w.writerow([f, sg, sp_of[(f, sg)], ev])
    rows = [r for r in rows if (r["file"], r["seg"]) not in drop]
    dups = dedupe_within_splits(rows, split)
    print("works dropped as duplicates inside a split:", len(dups),
          dict(Counter(split[r["cid"]] for r in rows if (r["file"], r["seg"]) in dups)),
          "| by source:", dict(Counter(r["source"] for r in rows if (r["file"], r["seg"]) in dups)))
    with (OUT / f"{prefix}duplicates_dropped.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "segment", "split", "reason"])
        sp_of = {(r["file"], r["seg"]): split[r["cid"]] for r in rows}
        for (f, sg), why_ in sorted(dups.items()): w.writerow([f, sg, sp_of[(f, sg)], why_])
    rows = [r for r in rows if (r["file"], r["seg"]) not in dups]
    byf = defaultdict(list)
    for r in rows: byf[r["file"]].append(r)

    # ---- pass 2: spans, regions, tokenize, label ----
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    assert tok.is_fast
    tok.model_max_length = int(1e12)
    cls_id, sep_id, pad_id = special_token_ids(tok)
    content_len = a.max_length - 2
    out = {s: [] for s in SPLIT_NAMES.values()}
    counts = {s: Counter() for s in SPLIT_NAMES.values()}
    works = {s: Counter() for s in SPLIT_NAMES.values()}
    lost = Counter()
    side = []
    for n, (f, rs) in enumerate(sorted(byf.items())):
        c = json.load(open(GOLD / f, encoding="utf-8"))["content"]
        for r in rs:
            sname = SPLIT_NAMES[split[r["cid"]]]
            s0 = r["span_start"]; text = c[s0:r["span_end"]]; L = len(text)
            t = (r["title_span"][0] - s0, r["title_span"][1] - s0)
            au = (r["author_span"][0] - s0, r["author_span"][1] - s0)
            au2, note = resolve_overlap(t, au)
            if note: lost[note] += 1
            if t[0] < au[1] and au[0] < t[1]: lost["overlap_pairs"] += 1
            work_id = f"{f[:-5]}#{r['seg']}"
            regs = regions_paras(text, [t, au2]) if a.crop == "paras" else regions(text, a.k, [t, au2])
            works[sname][r["source"]] += 1
            for kind, sp in (("TITLE", t), ("AUTHOR", au2)):
                side.append(dict(work_id=work_id, file=f, cid=r["cid"], source=r["source"], kind=kind,
                                 start=(sp[0] + s0) if sp else "", end=(sp[1] + s0) if sp else "",
                                 note=note if kind == "AUTHOR" else "", split=split[r["cid"]], group=gid_of[r["cid"]]))
            pieces = []
            for rname, rs_, re_ in regs:
                rt = text[rs_:re_]
                sp_list = []
                for kind, sp in (("TITLE", t), ("AUTHOR", au2)):
                    if sp is None: continue
                    lo, hi = max(sp[0], rs_), min(sp[1], re_)
                    if hi > lo:
                        if (lo, hi) != sp: lost[f"{kind}_cut_by_region_edge"] += 1
                        sp_list.append((lo - rs_, hi - rs_, work_id, kind))
                sp_list.sort()
                enc = tok(rt, add_special_tokens=False, return_offsets_mapping=True)
                offs = [tuple(o) for o in enc["offset_mapping"]]
                labels = label_tokens(skip_leading_ws(offs, rt), sp_list, LABEL2ID)
                for kind in ("TITLE", "AUTHOR"):
                    ns = sum(1 for x in sp_list if x[3] == kind)
                    nb = sum(1 for x in labels if x == LABEL2ID["B-" + kind])
                    lost["spans_in_regions"] += ns; lost["spans_without_B_token"] += ns - nb
                pieces.append((rname, rs_, list(enc["input_ids"]), list(labels), offs, rt))
            if a.single and len(pieces) == 2:       # top + <eos> + bottom as ONE sequence; the separator carries no label
                (_, s1, i1, l1, o1, t1), (_, _, i2, l2, o2, _) = pieces
                shift = len(t1) + 1
                pieces = [("both", s1, i1 + [sep_id] + i2, l1 + [IGNORE_LABEL] + l2,
                           o1 + [(len(t1), len(t1))] + [(x + shift, y + shift) for x, y in o2], None)]
            for rname, rs_, ids_, labels, offs, _rt in pieces:
                for wi, (ws, we) in enumerate(sliding_windows(len(ids_), content_len, a.stride)):
                    ex = pack_window(ids_, labels, offs, ws, we, cls_id, sep_id, pad_id, a.max_length)
                    ex.update(work_id=work_id, cid=r.get("cid_orig", r["cid"]), source=r["source"], region=rname,
                              region_start=rs_, window_index=wi, n_tokens_region=len(ids_))
                    counts[sname].update(x for x in ex["labels"] if x != IGNORE_LABEL)
                    ex["_has_label"] = any(x > 0 for x in ex["labels"])
                    out[sname].append(ex)
            for kind, sp in (("TITLE", t), ("AUTHOR", au2)):
                if sp and not any(max(sp[0], x) < min(sp[1], y) for _, x, y in regs):
                    lost[f"{kind}_outside_all_regions"] += 1
        if (n + 1) % 200 == 0: print(f"  {n + 1}/{len(byf)} files", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{prefix}spans_sidecar.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(side[0])); w.writeheader(); w.writerows(side)
    with (OUT / f"{prefix}split_frozen.csv").open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# group-level 80/10/10 by collection id, seed={a.seed}, K={a.k}; linked ids (same text / same title+author) share a split\n")
        w = csv.writer(fh); w.writerow(["cid", "split", "group"] + (["reason"] if a.all else []))
        for cid in sorted(split): w.writerow([cid, split[cid], gid_of[cid]] + ([why.get(cid, "")] if a.all else []))

    nolabel = {s: sum(1 for e in v if not e["_has_label"]) for s, v in out.items()}
    for v in out.values():
        for e in v: e.pop("_has_label")
    feats = Features({
        "input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")), "token_start": Value("int32"), "token_end": Value("int32"),
        "char_start": Value("int32"), "char_end": Value("int32"), "work_id": Value("string"),
        "cid": Value("string"), "source": Value("string"), "region": Value("string"),
        "region_start": Value("int32"), "window_index": Value("int32"), "n_tokens_region": Value("int32"),
    })
    dd = DatasetDict({s: Dataset.from_list(v, features=feats) if v
                      else Dataset.from_dict({k: [] for k in feats}, features=feats) for s, v in out.items()})
    dd.save_to_disk(str(a.out_dir))

    tr = counts["train"]; tt = sum(tr.values())
    stats = {
        "label2id": LABEL2ID, "crop": a.crop, "K": a.k, "max_length": a.max_length, "stride": a.stride, "seed": a.seed,
        "sources": sorted({r["source"] for r in rows}),
        "windows": {s: len(v) for s, v in out.items()},
        "windows_with_no_label": nolabel,
        "works_by_source": {s: dict(c) for s, c in works.items()},
        "label_counts": {s: {ID2LABEL[k]: v for k, v in sorted(c.items())} for s, c in counts.items()},
        "tokens": {s: sum(c.values()) for s, c in counts.items()},
        "class_weights_inverse_freq_train_NOT_APPLIED": {ID2LABEL[k]: round(tt / (5 * tr[k]), 3) for k in sorted(tr)},
        "notes": dict(lost),
    }
    (OUT / f"{a.out_dir.name}_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()

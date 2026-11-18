#!/usr/bin/env python3
"""Turn predict_layer_spans.py output (token spans) into character spans and compare with gold.

Writes, next to the predictions file:
  <name>_spans.csv     one row per gold or predicted span: status, char offsets, text, IoU, model score
  <name>_by_book.csv   precision / recall / F1 (IoU@0.5) per book, with category
  <name>_view.html     the documents with gold vs predicted spans coloured

Usage: python yigchung/src/analyze_predictions.py PREDICTIONS.json [NAME]
  PREDICTIONS.json  output of common/predict_layer_spans.py (token spans per document, splits validation and/or test)
  NAME              genre set whose parquet holds the gold text and spans (default with_lotsawa)
Classified system. Matching is greedy one-to-one on character IoU, a match needs IoU >= 0.5 (the training metric);
a gold span with some overlap but IoU < 0.5 is reported as "boundary", with none as "missed". Token spans are
converted to characters with the mmBERT tokenizer (the token count must equal the one in the predictions file).
Reads:  PREDICTIONS.json and <local>/genre_training/NAME/<split>.parquet.
Writes: the three files above, with the predictions file's name stem; prints P / R / F1 overall and per genre.
"""
import paths as P
import html
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PRED = Path(sys.argv[1])
NAME = sys.argv[2] if len(sys.argv) > 2 else "with_lotsawa"
G = P.GENRE
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
tok.model_max_length = int(1e12)
J = json.loads(PRED.read_text())


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return 0.0 if hi < lo else (hi - lo + 1) / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - (hi - lo + 1))


rows, books, view = [], [], []
for split, S in J["splits"].items():
    texts = pd.read_parquet(G / NAME / f"{split}.parquet").set_index("book_id")
    for bk, D in S["docs"].items():
        r = texts.loc[bk]
        t = r.text
        offs = tok(t, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        assert len(offs) == D["n_tokens"], (bk, len(offs), D["n_tokens"])

        def chars(s, e):
            a, b = offs[s][0], offs[e][1] - 1
            while a < b and t[a] in " \n": a += 1
            while b > a and t[b] in " \n": b -= 1
            return a, b
        gold = [(int(s["start"]), int(s["end"])) for s in r.spans]      # true character gold (not token-snapped)
        pred = [(*chars(s, e), sc) for s, e, sc in D["pred"]]
        # greedy one-to-one matching on IoU, same as the training metric
        cand = sorted(((iou(g, p[:2]), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)
                       if p[0] <= g[1] and p[1] >= g[0]), reverse=True)
        gm, pm = {}, {}
        for v, gi, pi in cand:
            if v >= 0.5 and gi not in gm and pi not in pm:
                gm[gi] = (pi, v); pm[pi] = (gi, v)
        best_g = {gi: max([v for v, g2, _ in cand if g2 == gi], default=0) for gi in range(len(gold))}
        best_p = {pi: max([v for v, _, p2 in cand if p2 == pi], default=0) for pi in range(len(pred))}
        for gi, (a, b) in enumerate(gold):
            st = "matched" if gi in gm else ("boundary (overlap < 0.5)" if best_g[gi] > 0 else "missed")
            rows.append({"split": split, "book_id": bk, "category": r.category, "kind": "gold", "status": st,
                         "start": a, "end": b, "len": b - a + 1, "iou": round(gm[gi][1] if gi in gm else best_g[gi], 3),
                         "score": "", "text": t[a:b + 1].replace("\n", "⏎")[:400]})
        for pi, (a, b, sc) in enumerate(pred):
            st = "matched" if pi in pm else ("boundary (overlap < 0.5)" if best_p[pi] > 0 else "false positive")
            rows.append({"split": split, "book_id": bk, "category": r.category, "kind": "pred", "status": st,
                         "start": a, "end": b, "len": b - a + 1, "iou": round(pm[pi][1] if pi in pm else best_p[pi], 3),
                         "score": sc, "text": t[a:b + 1].replace("\n", "⏎")[:400]})
        tp = len(gm); P = tp / max(len(pred), 1); R = tp / max(len(gold), 1)
        books.append({"split": split, "book_id": bk, "category": r.category, "source": r.source, "gold": len(gold),
                      "pred": len(pred), "matched": tp, "precision": round(P, 3), "recall": round(R, 3),
                      "f1": round(2 * P * R / max(P + R, 1e-9), 3)})
        # character mask for the viewer: 1 gold, 2 pred, 3 both
        m = bytearray(len(t))
        for a, b in gold:
            for k in range(a, b + 1): m[k] |= 1
        for a, b, _ in pred:
            for k in range(a, b + 1): m[k] |= 2
        segs, k = [], 0
        while k < len(t):
            j = k
            while j < len(t) and m[j] == m[k]: j += 1
            segs.append([m[k], t[k:j]]); k = j
        view.append({"id": bk, "cat": r.category, "split": split, "f1": books[-1]["f1"], "segs": segs})

stem = PRED.with_suffix("")
sp = pd.DataFrame(rows); bb = pd.DataFrame(books)
sp.to_csv(f"{stem}_spans.csv", index=False, encoding="utf-8-sig")
bb.to_csv(f"{stem}_by_book.csv", index=False, encoding="utf-8-sig")
tot = bb[["gold", "pred", "matched"]].sum()
P, R = tot.matched / tot.pred, tot.matched / tot.gold
print(f"character-level IoU@0.5 over all books: P {P:.4f} R {R:.4f} F1 {2*P*R/(P+R):.4f}")
cat = bb.groupby("category")[["gold", "pred", "matched"]].sum()
cat["precision"] = (cat.matched / cat.pred).round(3); cat["recall"] = (cat.matched / cat.gold).round(3)
cat["f1"] = (2 * cat.precision * cat.recall / (cat.precision + cat.recall)).round(3)
print(cat.to_string())
print(sp.groupby(["kind", "status"]).size().to_string())
print(bb.sort_values("f1").to_string(index=False))

page = """<!doctype html><html><head><meta charset="utf-8"><title>Yigchung predictions</title>
<style>:root{--bg:#fff;--fg:#1d1d1f;--both:#b7e4c7;--gold:#f8b4b4;--pred:#ffd59a}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--both:#1f5f3a;--gold:#7a2b2b;--pred:#7a5418}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui;margin:0 16px}
header{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid #8884}
#t{font-family:'Noto Serif Tibetan','Jomolhari',serif;font-size:20px;line-height:2;white-space:pre-wrap}
.c3{background:var(--both)}.c1{background:var(--gold)}.c2{background:var(--pred)}
.k{display:inline-block;padding:0 6px;margin-right:6px;border-radius:3px}select{font-size:15px;max-width:100%}</style></head>
<body><header><select id="s"></select>
<span class="k c3">both</span><span class="k c1">gold only (missed)</span><span class="k c2">predicted only (false)</span>
<button id="n">next difference</button></header><div id="t"></div>
<script>const D=__DATA__;const s=document.getElementById('s'),t=document.getElementById('t');
D.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${d.id} · ${d.cat} · ${d.split} · F1 ${d.f1}`;s.appendChild(o)});
function show(i){t.innerHTML='';const f=document.createDocumentFragment();D[i].segs.forEach(([c,x])=>{const e=document.createElement('span');if(c)e.className='c'+c;e.textContent=x;f.appendChild(e)});t.appendChild(f);window.scrollTo(0,0)}
s.onchange=()=>show(s.value);show(0);
document.getElementById('n').onclick=()=>{const els=[...t.querySelectorAll('.c1,.c2')];const y=window.scrollY+120;const nx=els.find(e=>e.getBoundingClientRect().top+window.scrollY>y);if(nx)nx.scrollIntoView({block:'center'})};
</script></body></html>"""
Path(f"{stem}_view.html").write_text(page.replace("__DATA__", json.dumps(view, ensure_ascii=False)), encoding="utf-8")
print("wrote", f"{stem}_spans.csv", f"{stem}_by_book.csv", f"{stem}_view.html")

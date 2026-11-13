"""Review pack for the tsawa spans the predicted-quotation filter would delete although they matched gold.
Reads tsawa_dropped_tp_gold_quote_check.csv (90 window rows = 68 distinct spans, 10 test books), read-only on data.
Writes tsawa_quote_dropped_review.csv and .html here. Run from the repo root."""
import csv, glob, html
from collections import defaultdict

AUD = "common/annotation_audit"
SRC = f"{AUD}/tsawa_dropped_tp_gold_quote_check.csv"
URL = "https://github.com/OpenPecha-Data/{}"
CTX = 350  # chars of context each side

def base(pid):
    return open(glob.glob(f"data/raw_opf/{pid}.opf/**/base/v001.txt", recursive=True)[0], encoding="utf-8").read()

spans = {}
for r in csv.DictReader(open(SRC, encoding="utf-8")):
    k = (r["pecha_id"], int(r["start"]), int(r["end"]))
    cur = spans.get(k)
    cov = float(r["pred_quote_cov"]); g = float(r["gold_cov_of_span[quote+citation]"])
    if cur is None or cov > cur["cov"]:
        spans[k] = dict(cov=cov, gold_q=g, windows=(cur["windows"] if cur else 0) + 1)
    else:
        cur["windows"] += 1
rows = sorted(spans.items(), key=lambda kv: (kv[0][0], kv[0][1]))

texts = {p: base(p) for p in {k[0] for k, _ in rows}}
def bucket(g): return "gold quotation here" if g >= 0.5 else ("partly" if g > 0 else "no gold quotation")

with open(f"{AUD}/tsawa_quote_dropped_review.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pecha_id", "start", "end", "chars", "pred_quote_cov", "gold_quotation_at_span", "text"])
    for (p, s, e), v in rows:
        w.writerow([f'=HYPERLINK("{URL.format(p)}","{p}")', s, e, e - s, round(v["cov"], 2), bucket(v["gold_q"]),
                    texts[p][s:e].replace("\n", " ⏎ ")])

by = defaultdict(list)
for (p, s, e), v in rows: by[p].append((s, e, v))
css = """body{font-family:Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}
h2{margin-top:36px;border-bottom:2px solid #1F4E79;padding-bottom:4px}
.card{border:1px solid #ddd;border-radius:6px;margin:14px 0;padding:10px 14px}
.meta{color:#555;font-size:13px;margin-bottom:6px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:6px;background:#eee}
.tag.real{background:#f8d7da}.tag.part{background:#fff3cd}.tag.none{background:#d4edda}
pre{white-space:pre-wrap;font-family:'Noto Serif Tibetan','Jomolhari','Microsoft Himalaya',serif;font-size:20px;line-height:1.7;margin:0}
mark{background:#ffe08a}"""
out = [f"<html><head><meta charset='utf-8'><title>Tsawa spans dropped by the quotation filter</title><style>{css}</style></head><body>",
       "<h1>Tsawa spans the quotation filter would drop</h1>",
       f"<p>{len(rows)} distinct spans in {len(by)} test books. Each one matched gold tsawa, but the predicted quotation layer covers at least 50% of it, so the filter would delete it. "
       "Highlighted text is the span; the rest is context. Question for review: is this really tsawa, or is it a quotation that was labelled as tsawa?</p>",
       "<p><b>Tag:</b> <span class='tag real'>gold quotation here</span> gold Quotation/Citation covers the span · "
       "<span class='tag part'>partly</span> · <span class='tag none'>no gold quotation</span> the quotation model is probably wrong (filter hurts).</p>"]
for p in sorted(by):
    t = texts[p]
    out.append(f"<h2><a href='{URL.format(p)}'>{p}</a> &nbsp;<small>{len(by[p])} spans</small></h2>")
    for i, (s, e, v) in enumerate(by[p], 1):
        b = bucket(v["gold_q"]); cls = {"gold quotation here": "real", "partly": "part"}.get(b, "none")
        out.append(f"<div class='card'><div class='meta'>{p} #{i} &middot; chars {s}-{e} ({e-s}) &middot; predicted-quote coverage {v['cov']:.2f}"
                   f"<span class='tag {cls}'>{b}</span></div><pre>{html.escape(t[max(0,s-CTX):s])}<mark>{html.escape(t[s:e])}</mark>{html.escape(t[e:e+CTX])}</pre></div>")
out.append("</body></html>")
open(f"{AUD}/tsawa_quote_dropped_review.html", "w", encoding="utf-8").write("\n".join(out))

from collections import Counter
print(len(rows), "spans;", len(by), "books;", Counter(bucket(v["gold_q"]) for _, v in rows))

"""Count predicted test spans that contain no Tibetan character (U+0F00-U+0FFF). Read-only.
Sources (saved test predictions): tsawa scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl, sabche scratch/sabche/analysis/sabche_test_spans.jsonl
and ~/sabche-layer-detection/results/mmbert-sabche-v1/test/spans/, chapter scratch/chapter/chapter_test_spans.jsonl, quotation
~/quotation_eval/analysis/mmbert_spans.jsonl (text from quotation_eval/dataset/test.jsonl). Yigchung: no saved test predictions exist locally.
Token-space window predictions are converted to characters with the mmBERT tokenizer offsets. Writes non_tibetan_predictions.md here."""
import glob, json, re, collections
import pandas as pd
from transformers import AutoTokenizer

R = "data/raw_opf"; OUT = "common/annotation_audit"; HOME = "/Users/tenzinyonten"
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base", use_fast=True)
TIB = re.compile("[ༀ-࿿]")
def base(p): return open([x for x in glob.glob(f"{R}/{p}.opf/*") if x.endswith(".opf")][0] + "/base/v001.txt", encoding="utf-8").read()
TEXT, OFF = {}, {}
def text(p):
    if p not in TEXT: TEXT[p] = base(p)
    return TEXT[p]
def offs(p):
    if p not in OFF: OFF[p] = tok(text(p), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return OFF[p]
def kind(s):
    if re.search("[A-Za-z]", s): return "Latin letters"
    if re.search("[ऀ-ॿ]", s): return "Devanagari"
    if re.search("[0-9]", s): return "digits"
    if not s.strip(): return "whitespace only"
    return "punctuation / other symbols"
def window_spans(path, key):
    """-> (window-level list of (book, s, e), unique set) in characters"""
    win, uniq = [], set()
    for l in open(path, encoding="utf-8"):
        w = json.loads(l); o = offs(w["pecha_id"])
        for a, b in w[key]:
            sp = (w["pecha_id"], o[a][0], o[b][1]); win.append(sp); uniq.add(sp)
    return win, uniq
res = {}
def summarize(name, win, uniq, textof, note=""):
    nt = [sp for sp in win if not TIB.search(textof(sp[0])[sp[1]:sp[2]])]
    nu = [sp for sp in uniq if not TIB.search(textof(sp[0])[sp[1]:sp[2]])]
    kinds = collections.Counter(kind(textof(p)[s:e]) for p, s, e in nu)
    ex = []
    for p, s, e in sorted(nu, key=lambda x: -(x[2] - x[1]))[:0] + sorted(nu)[:0]: pass
    seen = set()
    for p, s, e in sorted(nu):
        t = textof(p)[s:e].replace("\n", "⏎")
        if t not in seen and len(ex) < 6: ex.append(f"`{p}` {s}:{e} {t[:50]!r}"); seen.add(t)
    res[name] = dict(win_total=len(win), win_nt=len(nt), uniq_total=len(uniq), uniq_nt=len(nu), kinds=dict(kinds), examples=ex, note=note)
w, u = window_spans("scratch/tsawa/analysis/v6_nofeat_test_spans_bp4.jsonl", "viterbi_bp4.0_tok"); summarize("tsawa", w, u, text)
w, u = window_spans("scratch/sabche/analysis/sabche_test_spans.jsonl", "viterbi_bp4.0_tok")
whole = set()   # whole-book sabche predictions (characters, inclusive end) from the sabche repo
for f in glob.glob(f"{HOME}/sabche-layer-detection/results/mmbert-sabche-v1/test/spans/*.json"):
    d = json.load(open(f)); whole |= {(d["book"], s, e + 1) for s, e in d["spans"]}
summarize("sabche", w, u, text, note=f"whole-book predictions from the sabche repo: {len(whole)} spans")
sab_whole_nt = sum(1 for p, s, e in whole if not TIB.search(text(p)[s:e])); res["sabche"]["whole_total"] = len(whole); res["sabche"]["whole_nt"] = sab_whole_nt
w, u = window_spans("scratch/chapter/chapter_test_spans.jsonl", "viterbi_bp4.0_tok"); summarize("chapter", w, u, text)
qt = {r["book_id"]: r["text"] for r in map(json.loads, open(f"{HOME}/quotation_eval/dataset/test.jsonl", encoding="utf-8"))}
qs = [(r["book_id"], r["start"], r["end"]) for r in map(json.loads, open(f"{HOME}/quotation_eval/analysis/mmbert_spans.jsonl", encoding="utf-8"))]
summarize("quotation", qs, set(qs), lambda p: qt[p], note="quotation_eval mmBERT spans on the 19-book test set")

rows = []
for n in ("tsawa", "sabche", "chapter", "quotation"):
    r = res[n]
    rows.append({"layer": n, "total predictions (window level)": r["win_total"], "non-Tibetan": r["win_nt"], "%": f"{100 * r['win_nt'] / r['win_total']:.2f}",
                 "unique spans": r["uniq_total"], "non-Tibetan ": r["uniq_nt"], "% ": f"{100 * r['uniq_nt'] / r['uniq_total']:.2f}", "what they are (unique spans)": ", ".join(f"{k} {v}" for k, v in r["kinds"].items()) or "none"})
rows.append({"layer": "yigchung", "total predictions (window level)": "n/a", "non-Tibetan": "n/a", "%": "", "unique spans": "n/a", "non-Tibetan ": "n/a", "% ": "", "what they are (unique spans)": "no saved test predictions"})
tot = {k: sum(res[n][k] for n in res) for k in ("win_total", "win_nt", "uniq_total", "uniq_nt")}
rows.append({"layer": "total (4 layers)", "total predictions (window level)": tot["win_total"], "non-Tibetan": tot["win_nt"], "%": f"{100 * tot['win_nt'] / tot['win_total']:.2f}",
             "unique spans": tot["uniq_total"], "non-Tibetan ": tot["uniq_nt"], "% ": f"{100 * tot['uniq_nt'] / tot['uniq_total']:.2f}", "what they are (unique spans)": ""})
df = pd.DataFrame(rows)
def table(d): return "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in d.itertuples(index=False))
L = ["# Predicted test spans with no Tibetan character\n", "A span counts as non-Tibetan when its text has no character in U+0F00-U+0FFF. Window level counts a span once per window it appears in (neighbouring windows overlap, as in the reported scores); unique spans count each (book, start, end) once.\n", table(df), "\nExamples:\n"]
for n in ("tsawa", "sabche", "chapter", "quotation"):
    L.append(f"- {n}: " + ("; ".join(res[n]["examples"]) if res[n]["examples"] else "none") + (f" ({res[n]['note']})" if res[n]['note'] else ""))
L.append("\nYigchung: the Yigchung model's test windows (`karma689/layer_detection`, config `windowed_w8192_s4916`) have no saved predictions on this machine; producing them means running the model, which was not done here.")
open(f"{OUT}/non_tibetan_predictions.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(df.to_string(index=False)); print({n: res[n]["examples"] for n in res}); print("sabche whole-book:", res["sabche"]["whole_total"], res["sabche"]["whole_nt"])

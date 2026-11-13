#!/usr/bin/env python3
"""Run Gemini with a prompt file on annotator works, score title/author at IoU>=0.5, write the failures for review.
UNTESTED first version. Python 3.9, no extra packages. The key stays in your terminal:  export GEMINI_API_KEY=...   (never in a file)

    python title_author/src/ta_gemini_eval.py make-v0 CURSOR_MD          # build prompts/v0.md from Cursor's file (no network)
    python title_author/src/ta_gemini_eval.py dry PROMPT.md              # show one request, no network
    python title_author/src/ta_gemini_eval.py run PROMPT.md SPLIT N [SEED]   # SPLIT = train | validation; test is refused

Outputs go to title_author/data/gemini_runs/<prompt name>_<split>_<N>/ : results.json (scores) and failures.md."""
import concurrent.futures as cf, json, os, random, re, sys, time, urllib.request, urllib.error
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PQ = Path(os.environ.get("TA_PQ", str(ROOT / "title_author/data/processed/parquet_export_strat_both_notrans_nous")))
RUNS = ROOT / "title_author/data/gemini_runs"; PROMPTS = ROOT / "title_author/data/gemini_prompts"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

def load(split, n, seed):
    if split not in ("train", "validation") and not (split == "test" and os.environ.get("ALLOW_TEST") == "1"):
        sys.exit("only train or validation. For the final comparison on test, set ALLOW_TEST=1 (scores only, no failures file)")
    df = pd.read_parquet(PQ / f"{split}.parquet")
    df = df[df.source != "v3_auto"].dropna(subset=["title", "author"])
    return df.sample(min(n, len(df)), random_state=seed) if n else df

def build_input(r):
    head, tail = r.text_head, ("" if r.whole_work else r.text_tail)
    return "[HEAD]\n" + head + ("\n[TAIL]\n" + tail if tail else "")

_KEY_I = [0]

def call_gemini(prompt, user, key, tries=5):
    body = json.dumps({"systemInstruction": {"parts": [{"text": prompt}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    keys = [x for x in (os.environ.get("GEMINI_API_KEYS") or key).split(",") if x.strip()]   # several keys: rotate on 429
    tries = max(tries, 2 * len(keys) + 3)
    for k in range(tries):
        kk = keys[(_KEY_I[0] + (0 if k == 0 else 0)) % len(keys)]
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent", data=body,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": kk.strip()})
        try:
            with urllib.request.urlopen(req, timeout=120) as f:
                out = json.load(f)
            return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
        except urllib.error.HTTPError as e:
            if e.code in (402, 429) and len(keys) > 1 and k < tries - 1:
                _KEY_I[0] += 1                                   # next key; wait a little only after a full cycle
                if (k + 1) % len(keys) == 0: time.sleep(5 + random.random() * 5)
                continue
            if e.code in (429, 500, 503) and k < tries - 1: time.sleep(min(2 ** k, 30) + random.random()); continue
            return "HTTP_ERROR %s" % e.code
        except Exception as e:
            if k < tries - 1: time.sleep(2 ** k); continue
            return "ERROR %s" % type(e).__name__
    return "ERROR"

def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0

def score(r, ans, which):
    """returns (kind, iou, predicted string). kind: ok | null | no_json | not_substring | wrong_region | low_iou"""
    if ans is None: return "no_json", 0.0, ""
    pred, reg = ans.get(which), ans.get(which + "_region")
    if not pred: return "null", 0.0, ""
    gold_reg = getattr(r, which + "_region"); text = r.text_head if reg == "head" else r.text_tail
    if reg not in ("head", "tail") or not isinstance(text, str) or pred not in text: return "not_substring", 0.0, pred
    i = text.find(pred) if which == "title" else text.rfind(pred)          # title: first match, author: last match
    gs = int(getattr(r, which + "_start")); g = (gs, gs + len(getattr(r, which)))
    if reg != gold_reg: return "wrong_region", 0.0, pred
    v = iou((i, i + len(pred)), g)
    return ("ok" if v >= 0.5 else "low_iou"), v, pred

mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "make-v0":
    src = Path(sys.argv[2]).read_text(encoding="utf-8")
    a, b = src.index("## What Gemini receives"), src.index("## Optimise the Gemini prompt")
    PROMPTS.mkdir(parents=True, exist_ok=True)
    head = "You find the title and the author of one Tibetan work. Follow the sections below exactly.\n\n"
    (PROMPTS / "v0.md").write_text(head + src[a:b].strip() + "\n", encoding="utf-8"); print("wrote", PROMPTS / "v0.md"); sys.exit()
if mode in ("dry", "run"):
    prompt = Path(sys.argv[2]).read_text(encoding="utf-8")
if mode == "dry":
    r = next(load("validation", 1, 0).itertuples()); u = build_input(r)
    print(f"prompt {len(prompt):,} chars | input {len(u):,} chars | model {MODEL}\ngold title: {r.title[:50]!r} | gold author: {r.author[:50]!r}"); sys.exit()
if mode == "run":
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS", "").split(",")[0] or sys.exit("GEMINI_API_KEY is not set in this terminal")
    split, n = sys.argv[3], int(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rows = list(load(split, n, seed).itertuples())
    if os.environ.get("IDS_FILE"):                      # score only these works (for example the ones another run answered)
        keep = set(json.load(open(os.environ["IDS_FILE"]))); rows = [r for r in rows if r.work_id in keep]
    _rows = os.environ.get("ROWS")                       # ROWS=a:b -> only works a..b of the split in fixed order (for chunked runs)
    if _rows: a_, b_ = map(int, _rows.split(":")); rows = rows[a_:b_]
    out_dir = RUNS / f"{Path(sys.argv[2]).stem}_{split}_{n}" ; out_dir = out_dir.with_name(out_dir.name + (f"_rows{_rows.replace(':', '-')}" if _rows else "")); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prev = None
    if os.environ.get("RESUME"):                         # RESUME=<run dir name>: keep the good answers, redo only the failed calls
        pj = json.loads((RUNS / os.environ["RESUME"] / "partial_raw.json").read_text(encoding="utf-8"))
        assert pj["work_ids"] == [r.work_id for r in rows], "works differ from the run being resumed"; prev = pj["raw"]
    raw = list(prev) if prev else [None] * len(rows)
    todo = [i for i in range(len(rows)) if not prev or prev[i].startswith(("HTTP_ERROR", "ERROR"))]
    print(f"{len(todo)} calls to make", flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("THREADS", "8"))) as ex:
        futs = {i: ex.submit(call_gemini, prompt, build_input(rows[i]), key) for i in todo}
        for c, i in enumerate(todo):
            raw[i] = futs[i].result()
            if (c + 1) % 50 == 0 or c + 1 == len(todo):
                (out_dir / "partial_raw.json").write_text(json.dumps({"work_ids": [r.work_id for r in rows], "raw": [x if x is not None else "ERROR pending" for x in raw]}, ensure_ascii=False), encoding="utf-8")
                print(f"{c + 1}/{len(todo)} redone  {time.time() - t0:.0f}s", flush=True)
    acc = {w: [0, 0, 0] for w in ("title", "author")}; kinds = {w: Counter() for w in acc}; fails = []
    for r, t in zip(rows, raw):
        ans = parse(t)
        for w in acc:
            k, v, pred = score(r, ans, w); kinds[w][k] += 1; a = acc[w]
            if k == "ok": a[0] += 1
            else:
                a[2] += 1; a[1] += 0 if k in ("null", "no_json") else 1
                fails.append((w, k, v, r, pred))
    res = {"model": MODEL, "prompt": sys.argv[2], "split": split, "n": len(rows), "seed": seed, "seconds": round(time.time() - t0)}
    for w, (tp, fp, fn) in acc.items():
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0; F = 2 * P * R / (P + R) if P + R else 0.0
        res[w] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(F, 3), "outcomes": dict(kinds[w])}
    res["mean_f1"] = round((res["title"]["f1"] + res["author"]["f1"]) / 2, 3)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Failures: {sys.argv[2]} on {split} ({len(rows)} works), mean F1 {res['mean_f1']}\n"]
    for w, k, v, r, pred in sorted(fails, key=lambda x: (x[0], x[1])):
        reg = getattr(r, w + "_region"); text = r.text_head if reg == "head" else r.text_tail; s = int(getattr(r, w + "_start")); e = int(getattr(r, w + "_end"))
        md.append(f"## {w} | {k} | IoU {v:.2f} | {r.work_id} | {r.source}\n- gold ({reg}): `{getattr(r, w)!r}`\n- gemini: `{pred!r}`\n- context: `{text[max(0, s - 80):s]}`**[{text[s:e]}]**`{text[e:e + 60]}`\n")
    if split != "test": (out_dir / "failures.md").write_text("\n".join(md), encoding="utf-8")   # no failures file for test: never tune on it
    print(json.dumps({k: v for k, v in res.items() if k in ("n", "seconds", "mean_f1")}), "\n", json.dumps({w: res[w] for w in acc}, ensure_ascii=False)); print("wrote", out_dir)

if mode == "partial":      # score whatever has finished: partial RUN_DIR_NAME SPLIT
    p = json.loads((RUNS / sys.argv[2] / "partial_raw.json").read_text(encoding="utf-8")); K = len(p["work_ids"])
    rows = list(load(sys.argv[3], 0, 0).itertuples())[:K]; assert [r.work_id for r in rows] == p["work_ids"], "row order differs"
    acc = {w: [0, 0, 0] for w in ("title", "author")}
    answered = []
    for r, t in zip(rows, p["raw"]):
        if t.startswith(("HTTP_ERROR", "ERROR")): continue          # no answer (for example HTTP 402 = out of credit): not scored
        answered.append(r.work_id); ans = parse(t)
        for w in acc:
            k, v, pred = score(r, ans, w); a = acc[w]
            if k == "ok": a[0] += 1
            else: a[2] += 1; a[1] += 0 if k in ("null", "no_json") else 1
    (RUNS / sys.argv[2] / "answered_ids.json").write_text(json.dumps(answered), encoding="utf-8"); print(f"works with an answer: {len(answered)} of {K}")
    for w, (tp, fp, fn) in acc.items():
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0; print(f"{w:7s} P {P:.3f}  R {R:.3f}  F1 {2 * P * R / (P + R) if P + R else 0:.3f}")

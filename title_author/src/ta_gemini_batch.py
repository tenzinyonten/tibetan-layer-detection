#!/usr/bin/env python3
"""Gemini Batch API runner: half price, asynchronous. The key comes only from the environment (GEMINI_API_KEY).

    python title_author/src/ta_gemini_batch.py submit PROMPT.md SPLIT MODEL [N] [SEED]   # N=0: the whole split
    python title_author/src/ta_gemini_batch.py status TAG
    python title_author/src/ta_gemini_batch.py collect TAG      # downloads, scores like ta_gemini_eval.py, prints the token bill

TAG is printed by submit and is also the name of the folder in data/gemini_runs/ and of the state file in data/gemini_batches/.
SPLIT test needs ALLOW_TEST=1 (scores only). Set TA_PQ to the parquet folder as for ta_gemini_eval.py."""
import os, sys, json, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
BASE = "https://generativelanguage.googleapis.com"
STATE = ROOT / "title_author/data/gemini_batches"; RUNS = ROOT / "title_author/data/gemini_runs"

# reuse the loader, input builder, parser and scorer of the single-call evaluator (everything before its main block)
_src = (HERE / "ta_gemini_eval.py").read_text(encoding="utf-8").split("mode = sys.argv[1]")[0]
_g = {"__name__": "ta_eval", "__file__": str(HERE / "ta_gemini_eval.py")}
exec(_src, _g)
load, build_input, parse, score = _g["load"], _g["build_input"], _g["parse"], _g["score"]

# paid-tier prices per 1M tokens, standard mode (batch is half); from Google's pricing page, check before relying on them
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.8-flash": (0.75, 3.75)}


def key():
    return os.environ.get("GEMINI_API_KEY") or sys.exit("GEMINI_API_KEY is not set in this terminal")


def http(url, data=None, headers=None, method=None, raw=False):
    h = {"x-goog-api-key": key()}; h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as f:
            if raw == "bytes": return f.read()
            return f if raw else json.load(f)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url.split('?')[0]}: {e.read().decode('utf-8', 'replace')[:400]}")


def upload(path):
    size = path.stat().st_size
    r = http(f"{BASE}/upload/v1beta/files", json.dumps({"file": {"display_name": path.name}}).encode(),
             {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
              "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "application/jsonl",
              "Content-Type": "application/json"}, raw=True)
    url = r.headers["X-Goog-Upload-URL"]
    out = http(url, path.read_bytes(), {"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                        "X-Goog-Upload-Command": "upload, finalize"})
    return out["file"]["name"]


def state_of(j):
    return (j.get("metadata", {}) or {}).get("state") or j.get("state")


def submit(prompt_path, split, model, n=0, seed=0):
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    rows = list(load(split, n, seed).itertuples())
    _rows = os.environ.get("ROWS")
    if _rows: a_, b_ = map(int, _rows.split(":")); rows = rows[a_:b_]
    tag = f"batch_{Path(prompt_path).stem}_{model}_{split}_{n or 'all'}" + (f"_think-{os.environ['THINK_LEVEL']}" if os.environ.get("THINK_LEVEL") else "") + (f"_rows{_rows.replace(':', '-')}" if _rows else "")
    STATE.mkdir(parents=True, exist_ok=True); (RUNS / tag).mkdir(parents=True, exist_ok=True)
    jl = STATE / f"{tag}.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in rows:
            req = {"systemInstruction": {"parts": [{"text": prompt}]},
                   "contents": [{"role": "user", "parts": [{"text": build_input(r)}]}],
                   "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            if os.environ.get("THINK_LEVEL"):             # gemini-3.8-flash: low gives 0 thinking tokens on this task
                req["generationConfig"]["thinkingConfig"] = {"thinkingLevel": os.environ["THINK_LEVEL"]}
            f.write(json.dumps({"key": r.work_id, "request": req}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests, file {jl.stat().st_size / 1e6:.1f} MB, uploading ...", flush=True)
    fname = upload(jl)
    job = http(f"{BASE}/v1beta/models/{model}:batchGenerateContent",
               json.dumps({"batch": {"display_name": tag, "input_config": {"file_name": fname}}}).encode(),
               {"Content-Type": "application/json"}, method="POST")
    (STATE / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "batch": job.get("name"), "file": fname, "model": model, "prompt": str(prompt_path),
         "split": split, "n": n, "seed": seed, "work_ids": [r.work_id for r in rows]}, indent=1), encoding="utf-8")
    print("TAG:", tag, "| batch:", job.get("name"), "| state:", state_of(job))


def status(tag):
    st = json.loads((STATE / f"{tag}.json").read_text(encoding="utf-8"))
    j = http(f"{BASE}/v1beta/{st['batch']}")
    print(tag, "->", state_of(j)); return st, j


def collect(tag):
    st, j = status(tag)
    if not str(state_of(j)).endswith("SUCCEEDED"):
        sys.exit("not finished yet")
    resp = j.get("response") or {}
    rf = resp.get("responsesFile") or (j.get("dest") or {}).get("fileName")
    if not rf:
        sys.exit("no responses file in the job: " + json.dumps(j)[:400])
    body = http(f"{BASE}/download/v1beta/{rf}:download?alt=media", raw="bytes")
    text = {}; tok = [0, 0, 0]
    for line in body.decode("utf-8").splitlines():
        if not line.strip(): continue
        o = json.loads(line); k = o.get("key")
        r = o.get("response")
        if r and r.get("candidates"):
            text[k] = "".join(p.get("text", "") for p in r["candidates"][0].get("content", {}).get("parts", []))
            u = r.get("usageMetadata", {}); tok[0] += u.get("promptTokenCount", 0)
            tok[1] += u.get("candidatesTokenCount", 0); tok[2] += u.get("thoughtsTokenCount", 0)
        else:
            text[k] = "ERROR batch"
    split = st["split"]; rows = {r.work_id: r for r in load(split, st["n"], st["seed"]).itertuples()}
    ids = st["work_ids"]; raw = [text.get(i, "ERROR missing") for i in ids]
    out = RUNS / tag; out.mkdir(parents=True, exist_ok=True)
    (out / "partial_raw.json").write_text(json.dumps({"work_ids": ids, "raw": raw}, ensure_ascii=False), encoding="utf-8")
    res = {"model": st["model"], "prompt": st["prompt"], "split": split, "n": len(ids), "mode": "batch"}
    kinds = {w: {} for w in ("title", "author")}
    for w in kinds:
        tp = fp = fn = 0
        for i, t in zip(ids, raw):
            if t.startswith("ERROR"):
                k = "error"
            else:
                k, _, _ = score(rows[i], parse(t), w)
            kinds[w][k] = kinds[w].get(k, 0) + 1
            if k == "ok": tp += 1
            else:
                fn += 1; fp += 0 if k in ("null", "no_json", "error") else 1
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        res[w] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R) if P + R else 0.0, 3),
                  "outcomes": kinds[w]}
    res["mean_f1"] = round((res["title"]["f1"] + res["author"]["f1"]) / 2, 3)
    pin, pout = PRICES.get(st["model"], (0, 0))
    res["tokens"] = {"input": tok[0], "output": tok[1], "thinking": tok[2]}
    res["cost_usd_batch"] = round((tok[0] * pin + (tok[1] + tok[2]) * pout) / 1e6 / 2, 3) if pin else None
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)); print("wrote", out)


def combine(name, tags):
    """Merge several collected chunks (same prompt/model/split) into one scored run folder `name`."""
    ids, raw, tokens, cost = [], [], [0, 0, 0], 0.0
    st0 = None
    for t in tags:
        st = json.loads((STATE / f"{t}.json").read_text(encoding="utf-8")); st0 = st0 or st
        pr = json.loads((RUNS / t / "partial_raw.json").read_text(encoding="utf-8"))
        ids += pr["work_ids"]; raw += pr["raw"]
        rs = json.loads((RUNS / t / "results.json").read_text(encoding="utf-8"))
        tokens = [a + b for a, b in zip(tokens, (rs["tokens"]["input"], rs["tokens"]["output"], rs["tokens"]["thinking"]))]
        cost += rs.get("cost_usd_batch") or 0
    rows = {r.work_id: r for r in load(st0["split"], st0["n"], st0["seed"]).itertuples()}
    res = {"model": st0["model"], "prompt": st0["prompt"], "split": st0["split"], "n": len(ids), "mode": "batch (combined)"}
    for w in ("title", "author"):
        tp = fp = fn = 0; kinds = {}
        for i, t in zip(ids, raw):
            k = "error" if t.startswith("ERROR") else score(rows[i], parse(t), w)[0]
            kinds[k] = kinds.get(k, 0) + 1
            if k == "ok": tp += 1
            else: fn += 1; fp += 0 if k in ("null", "no_json", "error") else 1
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        res[w] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R) if P + R else 0.0, 3), "outcomes": kinds}
    res["mean_f1"] = round((res["title"]["f1"] + res["author"]["f1"]) / 2, 3)
    res["tokens"] = {"input": tokens[0], "output": tokens[1], "thinking": tokens[2]}; res["cost_usd_batch"] = round(cost, 3)
    out = RUNS / name; out.mkdir(parents=True, exist_ok=True)
    (out / "partial_raw.json").write_text(json.dumps({"work_ids": ids, "raw": raw}, ensure_ascii=False), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)); print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "submit":
        submit(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0,
               int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif mode == "status": status(sys.argv[2])
    elif mode == "collect": collect(sys.argv[2])
    elif mode == "combine": combine(sys.argv[2], sys.argv[3:])
    else: sys.exit(__doc__)

#!/usr/bin/env python3
"""Trials F and G on random VALIDATION works (never test), first-pass extraction with the final prompt (gepa_final_full.md).
F = voting: flash-lite once at temperature 0 (baseline) + 4 samples at temperature 0.8; the majority title and author win (tie -> baseline).
G = two models: a second model (default gemini-3.5-flash) answers once; where it disagrees with the baseline on a field, the baseline model
    is asked as a judge to pick between the two candidates (it sees both names with their context).
All answers go through the same rule layer as the skill (repair, role word, fallback). Replies are cached, a rerun resumes.

    GEMINI_API_KEY=... python3 title_author/src/ta_vote_trial.py [N] [MODEL] [MODEL2]"""
import os, sys, json, re, time, random, urllib.request, urllib.error
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "title_author/skill"))
import title_author_skill as S
_src = (HERE / "ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
_g = {"__name__": "cmp", "__file__": str(HERE / "ta_compare_runs.py")}; exec(_src, _g)
load, score = _g["load"], _g["score"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
M1 = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.5-flash-lite"
M2 = sys.argv[3] if len(sys.argv) > 3 else "gemini-3.5-flash"
KEY = os.environ.get("GEMINI_API_KEY") or sys.exit("set GEMINI_API_KEY")
OUT = ROOT / "title_author/data/gemini_runs" / f"vote_trial_{M1}_{M2}"; OUT.mkdir(parents=True, exist_ok=True)
RF = OUT / "results.json"; RES = json.loads(RF.read_text(encoding="utf-8")) if RF.exists() else {}
PAUSE = float(os.environ.get("PAUSE", "3")); SYSTEM = Path(S.DEFAULT_PROMPT).read_text(encoding="utf-8")
JUDGE = """Two candidate answers for the {field} of one Tibetan work are given with the text around each. Pick the one that is the {what}.
Reply ONLY with JSON: {{"choice": 1 or 2}}."""

def post(model, system, user, temp):
    body = json.dumps({"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                       "generationConfig": {"temperature": temp, "responseMimeType": "application/json"}}).encode()
    for wait in (5, 15, 40, 90, 180):
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
            out = json.load(urllib.request.urlopen(req, timeout=180))
            return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503): print(f"  HTTP {e.code} ({model}), waiting {wait}s", flush=True); time.sleep(wait); continue
            raise SystemExit(f"HTTP {e.code} (stopping; results are cached): {e.read()[:200]!r}")
        except Exception as e:
            print("  retry after", type(e).__name__, str(e)[:100], flush=True); time.sleep(wait)
    raise SystemExit("too many retries; rerun later")

def cached(key, fn):
    if key not in RES:
        RES[key] = fn(); RF.write_text(json.dumps(RES, ensure_ascii=False), encoding="utf-8"); time.sleep(PAUSE)
    return RES[key]

norm = lambda s: re.sub(r"[\s་།]+", "", s or "")
def ctx(text, name, w=120):
    i = text.find(name) if name else -1
    return text[max(0, i - w): i + len(name) + w].replace("\n", " ") if i >= 0 else "(not found)"

rows = list(load("validation", 0, 0).itertuples())
random.seed(1); random.shuffle(rows); rows = rows[:N]
stat = Counter(); n = len(rows)
for k, r in enumerate(rows, 1):
    head = r.text_head; tail = "" if r.whole_work else (r.text_tail or ""); whole = bool(r.whole_work); user = S.build_input(head, tail, whole); w = r.work_id
    fin = lambda raw: S.extract(head, tail, whole, call_llm=lambda s, u: raw)
    base = fin(cached(f"{w}|base", lambda: post(M1, SYSTEM, user, 0)))
    samples = [base] + [fin(cached(f"{w}|s{i}", lambda: post(M1, SYSTEM, user, 0.8))) for i in range(1, 5)]
    m2 = fin(cached(f"{w}|m2", lambda: post(M2, SYSTEM, user, 0)))
    res = {"base": base, "m2": m2, "vote": dict(base), "judge": dict(base)}
    for f in ("title", "author"):
        c = Counter(norm(a.get(f)) for a in samples if a.get(f)); top = c.most_common()
        if top and (top[0][1] > 1 or not base.get(f)):
            best = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else (norm(base.get(f)) if norm(base.get(f)) in {t for t, v in top if v == top[0][1]} else top[0][0])
            pick = next(a for a in samples if norm(a.get(f)) == best); res["vote"][f] = pick[f]; res["vote"][f + "_region"] = pick[f + "_region"]
        b, o = base.get(f), m2.get(f)
        if b and o and norm(b) != norm(o):                                  # disagreement -> judge (baseline model)
            what = "title of this work (its name, not a heading or a section)" if f == "title" else "composer of this work (not translator, compiler, copyist, proofreader or requester)"
            text = head + "\n" + tail
            q = JUDGE.format(field=f, what=what) + f"\n\nCandidate 1: {b}\n...{ctx(text, b)}...\n\nCandidate 2: {o}\n...{ctx(text, o)}..."
            raw = cached(f"{w}|j{f}", lambda: post(M1, "You decide between two candidate answers about a Tibetan work.", q, 0))
            m = re.search(r"\{.*\}", raw or "", re.S)
            try: ch = json.loads(m.group(0)).get("choice") if m else 1
            except Exception: ch = 1
            if ch == 2: res["judge"][f] = o; res["judge"][f + "_region"] = m2[f + "_region"]
        elif not b and o: res["judge"][f] = o; res["judge"][f + "_region"] = m2[f + "_region"]
    for name in ("base", "vote", "m2", "judge"):
        for f in ("title", "author"): stat[(name, f)] += score(r, res[name], f)[0] == "ok"
    print(f"{k}/{n}", flush=True)
print(f"\nF and G on {n} random validation works (rule layer on, {M1}; second model {M2})")
print(f"{'':28}{'title':>8}{'author':>8}")
for name, label in (("base", f"{M1}, 1 run (baseline)"), ("vote", "F: vote of 5 runs"), ("m2", f"{M2}, 1 run"), ("judge", "G: two models + judge")):
    print(f"{label:28}{stat[(name,'title')]/n:8.3f}{stat[(name,'author')]/n:8.3f}")

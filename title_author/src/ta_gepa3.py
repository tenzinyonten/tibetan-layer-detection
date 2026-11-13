#!/usr/bin/env python3
"""GEPA (DSPy 3) on the title/author prompt. The Gemini key comes only from the environment (GEMINI_API_KEY).
Run with ~/dspy-env/bin/python.

    ta_gepa3.py run START_PROMPT.md N_TRAIN N_VAL BUDGET   # optimise; writes title_author/data/gemini_prompts/$OUT_NAME
    ta_gepa3.py eval PROMPT.md SPLIT N                      # score a prompt with the same settings (no optimisation)
    ta_gepa3.py demo RUN_FOLDER N                           # print the GEPA feedback for N author mistakes of an existing validation run (no API call)

Settings (environment variables):
  BATCH=4 CROP=2000     BATCH works per call; CROP keeps the first and last CROP characters
  STRATIFY=1            sample train/validation works in proportion to their source
  MINIBATCH=12          how many mistakes GEPA reads per rewrite (default 3)
  RULE_FEEDBACK=1       (default 1) name the TYPE of each mistake and show the gold context, so the rewriter can see patterns
  LOCK_FILE=rules.md    text GEPA can NOT change: it is added to every call next to the prompt GEPA rewrites
  HYBRID=1              after the model answers, add an adjacent role word to the author and fall back to the rule-based
                        script (ta_rule_extract.py) when the model gives no usable author; the score then covers the whole pipeline
  EFFICIENCY=1          pick the final prompt by skill score = validation score - TOKEN_W * (prompt tokens / 1000) (default TOKEN_W 0.004, tokens estimated from
                        characters as in skill/skill_score.py) instead of the plain validation score; all candidates are saved to $OUT_NAME.candidates.json
  EXCLUDE_N=80          validation only: leave out the 80 works a GEPA run used for selection, so an eval sees fresh works
  GEMINI_MODEL, REFLECT_GEMINI, TASK_EFFORT, THREADS, TA_PQ, OUT_NAME
Uses train for reflection and validation for selection. The test split is never opened (eval refuses it unless ALLOW_TEST=1)."""
import os, sys, json, re
from types import SimpleNamespace
from pathlib import Path
import pandas as pd, dspy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]
PQ = Path(os.environ.get("TA_PQ", str(ROOT / "title_author/data/processed/parquet_export_strat_both_notrans_nous")))
MODEL = "gemini/" + os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
REFLECT = "gemini/" + os.environ.get("REFLECT_GEMINI", os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"))

BATCH = int(os.environ.get("BATCH", "1")); CROP = int(os.environ.get("CROP", "0"))
MINIBATCH = int(os.environ.get("MINIBATCH", "3")); STRATIFY = os.environ.get("STRATIFY", "0") == "1"
RULE_FEEDBACK = os.environ.get("RULE_FEEDBACK", "1") == "1"; HYBRID = os.environ.get("HYBRID", "0") == "1"
LOCK = Path(os.environ["LOCK_FILE"]).read_text(encoding="utf-8") if os.environ.get("LOCK_FILE") else ""

import ta_rule_extract as T          # the rule-based script: its role-word list and its fallback predictor
ROLES = sorted(T.ROLES, key=len, reverse=True)
AGENT = re.compile(r"[་\s]*(?:ཀྱིས|གྱིས|གིས|ཡིས|ས)(?=[་།\s]|$)")
VERB = re.compile(r"མཛད|སྦྱར|བྲིས|བཀོད|བརྩམས|བགྱིས|བསྒྲིགས|རྩོམ")
REQUESTER = re.compile(r"བསྐུལ|ངོར|ཞུས|གསོལ་བ|བཀའ་བསྒོ")


def sent(r):
    """The text one work contributes to a call. With CROP, only the first and last CROP characters are sent; scoring still uses the full texts."""
    head, tail = r.text_head, ("" if r.whole_work else (r.text_tail or ""))
    if not CROP:
        return "[HEAD]\n" + head + ("\n[TAIL]\n" + tail if tail else "")
    if tail:
        return "[HEAD]\n" + head[:CROP] + ("\n[...]" if len(head) > CROP else "") + "\n[TAIL]\n" + ("[...]\n" if len(tail) > CROP else "") + tail[-CROP:]
    if len(head) > 2 * CROP:
        return "[HEAD]\n" + head[:CROP] + "\n[...]\n" + head[-CROP:]
    return "[HEAD]\n" + head


def strat_sample(df, n, seed, floor=8):
    """n works spread over the sources in proportion to their size, with at least `floor` per source (when it has that many)."""
    cnt = df.source.value_counts(); n = min(n, len(df))
    take = {s: max(min(floor, c), round(n * c / len(df))) for s, c in cnt.items()}
    while sum(take.values()) > n:                       # trim the largest source until the total fits
        big = max(take, key=take.get); take[big] -= 1
    parts = [df[df.source == s].sample(k, random_state=seed) for s, k in take.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def make_item(r):
    tail = "" if r.whole_work else (r.text_tail or "")
    return SimpleNamespace(text=sent(r), head=r.text_head, tail=tail, work_id=r.work_id,
                           rule_row=SimpleNamespace(text_head=r.text_head, text_tail=r.text_tail, whole_work=r.whole_work),
                           gold={w: (getattr(r, w), getattr(r, w + "_region"), int(getattr(r, w + "_start"))) for w in ("title", "author")})


def group(items):
    out = []
    for k in range(0, len(items), BATCH):
        g = items[k:k + BATCH]
        works = "\n\n".join(f"=== WORK {m + 1} ===\n{x.text}" for m, x in enumerate(g))
        out.append(dspy.Example(works=works, batch=g).with_inputs("works"))
    return out


def load(split, n, seed):
    df = pd.read_parquet(PQ / f"{split}.parquet"); df = df[df.source != "v3_auto"].dropna(subset=["title", "author"])
    if os.environ.get("EXCLUDE_N") and split == "validation":      # fresh works only: drop the EXCLUDE_N works a run (seed 3) used for selection
        df = df.drop(strat_sample(df, int(os.environ["EXCLUDE_N"]), 3).index)
    df = strat_sample(df, n, seed) if STRATIFY else df.sample(n, random_state=seed)
    return group([make_item(r) for r in df.itertuples()])


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


# ---------- hybrid step: runs on the model's answer before it is scored ----------
def postprocess(it, ans):
    """Author only. (1) usable answer: if a role word touches the name on its left, include it (76% of the time the gold does).
    (2) no usable answer (null, not an exact substring): use the rule-based script's author."""
    a = dict(ans); s = (a.get("author") or "").strip(); reg = (a.get("author_region") or "").strip().lower()
    text = it.head if reg == "head" else it.tail if reg == "tail" else ""
    note = None
    # (0) the span is not an exact substring because the model changed a space or newline: find it again on the page
    t0 = (a.get("title") or "").strip()
    if t0 and t0.lower() != "null" and not any(t0 in x for x in (it.head, it.tail) if x):
        f = T.repair_span(t0, it.head, it.tail, (a.get("title_region") or "").strip().lower(), last=False)
        if f: a["title"], a["title_region"] = f
    if s and s.lower() != "null" and not (text and s in text):
        f = T.repair_span(s, it.head, it.tail, reg, last=True)
        if f:
            a["author"], a["author_region"] = f; s, reg = f[0], f[1]; text = it.head if reg == "head" else it.tail
            note = "the author was found again after ignoring spaces and newlines"
    if s and s.lower() != "null" and text and s in text:
        i = text.rfind(s); pre = text[:i].rstrip("་ \n")
        for role in ROLES:
            if pre.endswith(role) and (i - len(pre)) <= 2:
                a["author"] = text[len(pre) - len(role): i + len(s)]; note = f"post-processing added the role word {role!r}"; break
    else:
        try:
            pr = T.predict(it.rule_row)
        except Exception:
            pr = {}
        if pr.get("author"):
            a["author"], a["author_region"] = pr["author"], pr["author_region"]; note = f"the model gave no usable author; the rule script supplied {pr['author']!r}"
    return a, note


# ---------- feedback: name the type of each mistake ----------
def context(text, s, e, before=45, after=35):
    return (text[max(0, s - before):s] + "[[" + text[s:e] + "]]" + text[e:e + after]).replace("\n", "⏎")


def explain(w, it, p, reg, i):
    """One sentence on WHY the span is wrong, from the gold position and the text around it."""
    gstr, greg, gs = it.gold[w]; ge = gs + len(gstr)
    gtext = it.head if greg == "head" else it.tail
    a, b = i, i + len(p)
    ov = min(b, ge) - max(a, gs)
    gctx = context(gtext, gs, ge)
    if ov <= 0:
        pre = ["Different place."]
        if w == "author":
            near = gtext[ge:ge + 60]
            pre.append("The gold author is followed by an agentive and a writing verb." if AGENT.match(near) and VERB.search(near) else "The gold author has no agentive+verb right after it.")
            ptext = it.head if reg == "head" else it.tail
            if REQUESTER.search(ptext[b:b + 70]):
                pre.append("Your name has a requester/patron cue after it (བསྐུལ, ངོར ...), so it is probably the person who asked for the text, not the writer.")
        return " ".join(pre) + f" Gold in context: {gctx}"
    if a > gs:
        miss = gtext[gs:a]
        role = next((r for r in ROLES if r in miss), None)
        why = f"missing role word {role!r}" if role else f"the gold starts earlier, with {miss!r}"
        return f"Too short on the LEFT: {why}. Gold in context: {gctx}"
    if a < gs:
        return f"Too long on the LEFT: you added {gtext[a:gs]!r} before the gold. Gold in context: {gctx}"
    if b > ge:
        extra = gtext[ge:b]
        why = "you included the agentive/verb after the name" if (AGENT.match(extra) or VERB.search(extra)) else "you ran on past the name"
        return f"Too long on the RIGHT: {why} ({extra!r}). Gold in context: {gctx}"
    return f"Too short on the RIGHT: the gold continues with {gtext[b:ge]!r}. Gold in context: {gctx}"


def one(ex, pred, w):
    gstr, greg, gs = ex.gold[w]; p = (getattr(pred, w, "") or "").strip(); reg = (getattr(pred, w + "_region", "") or "").strip().lower()
    if not p or p.lower() == "null": return 0.0, f"{w}: you returned nothing. Gold {w} is {gstr!r} in the {greg}."
    text = ex.head if reg == "head" else ex.tail if reg == "tail" else ""
    if p not in text: return 0.0, f"{w}: {p!r} is not an exact substring of the {reg or '?'} block. Copy exactly. Gold {w} is {gstr!r} in the {greg}."
    i = text.find(p) if w == "title" else text.rfind(p)
    if reg != greg: return 0.0, f"{w}: you returned {p!r} from the {reg}, but the gold {w} {gstr!r} is in the {greg}."
    v = iou((i, i + len(p)), (gs, gs + len(gstr)))
    if v >= 0.5: return 1.0, f"{w}: you returned {p!r}; gold {gstr!r}; IoU {v:.2f} (correct)."
    msg = f"{w}: you returned {p!r}; gold {gstr!r}; IoU {v:.2f} (WRONG)."
    return 0.0, msg + (" " + explain(w, ex, p, reg, i) if RULE_FEEDBACK else "")


def parse_list(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        arr = json.loads(m.group(0)) if m else None
    except Exception:
        arr = None
    if not isinstance(arr, list): arr = []
    arr = [a if isinstance(a, dict) else {} for a in arr][:n]
    return arr + [{}] * (n - len(arr))


def score_item(it, a):
    """Returns (score, feedback text, author score) for one work."""
    note = None
    if HYBRID: a, note = postprocess(it, a)
    p = SimpleNamespace(**{key: a.get(key, "") for key in ("title", "title_region", "author", "author_region")})
    t, ft = one(it, p, "title"); au, fa = one(it, p, "author")
    return (t + au) / 2, ft + " " + fa + (f" ({note})" if note else ""), au


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    arr = parse_list(getattr(pred, "answers", ""), len(gold.batch)); tot = 0.0; fb = []
    for k, (it, a) in enumerate(zip(gold.batch, arr)):
        s, f, _ = score_item(it, a); tot += s; fb.append(f"Work {k + 1}: {f}")
    return dspy.Prediction(score=tot / len(gold.batch), feedback="\n".join(fb))


class Extract(dspy.Signature):
    works: str = dspy.InputField(desc="One or more works, each starting with '=== WORK i ==='. Treat every work separately.")
    answers: str = dspy.OutputField(desc='A JSON list with exactly one object per work, in the same order: [{"title": ..., "title_region": "head" or "tail", "author": ..., "author_region": "head" or "tail"}, ...]')


def program(start):
    sig = Extract.with_instructions(Path(start).read_text(encoding="utf-8"))
    if LOCK:    # GEPA rewrites only the instructions; this text lives in the input field description, so it stays fixed
        base = "One or more works, each starting with '=== WORK i ==='. Treat every work separately."
        sig = sig.with_updated_fields("works", desc=base + "\n\nFIXED RULES AND EXAMPLES (always apply them; they take priority over the instructions):\n" + LOCK)
    return dspy.Predict(sig)


def evaluate(prog, ds):
    return dspy.Evaluate(devset=ds, metric=lambda g, p, trace=None: metric(g, p).score, num_threads=int(os.environ.get("THREADS", "24")), display_progress=False)(prog).score


def tokens_used(lm):
    """Total input/output tokens the task model has used so far (from DSPy's call history)."""
    i = o = 0
    for h in getattr(lm, "history", []):
        u = h.get("usage") or {}
        i += u.get("prompt_tokens", 0) or 0; o += u.get("completion_tokens", 0) or 0
    return i, o


def demo(run, n):
    """Show the feedback GEPA's rewriter would get for n author mistakes of an existing validation run. No API call."""
    p = json.loads((ROOT / "title_author/data/gemini_runs" / run / "partial_raw.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(PQ / "validation.parquet"); df = df[df.source != "v3_auto"].dropna(subset=["title", "author"]).set_index("work_id", drop=False)
    shown = 0
    for wid, raw in zip(p["work_ids"], p["raw"]):
        if wid not in df.index or str(raw).startswith(("HTTP", "ERROR")): continue
        it = make_item(df.loc[wid]); m = re.search(r"\{.*\}", raw, re.S)
        try: a = json.loads(m.group(0)) if m else {}
        except Exception: a = {}
        _, f, au = score_item(it, a)
        if au < 1.0:
            print(f"--- {wid} ---\n{f}\n"); shown += 1
        if shown >= n: break


if __name__ == "__main__":
    mode, start = sys.argv[1], sys.argv[2]
    _kw = {"reasoning_effort": os.environ["TASK_EFFORT"]} if os.environ.get("TASK_EFFORT") else {}      # TASK_EFFORT=low turns the hidden thinking down
    dspy.configure(lm=dspy.LM(MODEL, temperature=0, max_tokens=4000 + 2500 * BATCH, cache=False, num_retries=12, **_kw), adapter=dspy.JSONAdapter())
    if mode == "demo":
        demo(start, int(sys.argv[3]))
    elif mode == "eval":
        split, n = sys.argv[3], int(sys.argv[4])
        if split not in ("train", "validation") and not (split == "test" and os.environ.get("ALLOW_TEST") == "1"):
            sys.exit("only train or validation. For the final comparison on test, set ALLOW_TEST=1 (scores only)")
        print("EVAL %s %d works, BATCH=%d CROP=%d HYBRID=%s: %.3f" % (split, n, BATCH, CROP, HYBRID, evaluate(program(start), load(split, n, 3))), flush=True)
    elif mode == "run":
        ntr, nva, budget = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        train, val = load("train", ntr, 3), load("validation", nva, 3); prog = program(start)
        print(f"SETTINGS: BATCH={BATCH} CROP={CROP} MINIBATCH={MINIBATCH} STRATIFY={STRATIFY} RULE_FEEDBACK={RULE_FEEDBACK} HYBRID={HYBRID} LOCK={'yes' if LOCK else 'no'} train={ntr} val={nva} budget={budget}", flush=True)
        SKIP = os.environ.get("SKIP_EVALS") == "1"       # GEPA prints the same start and end validation scores itself; skipping saves two full passes
        if not SKIP:
            print("BEFORE (validation, %d works): %.3f" % (nva, evaluate(prog, val)), flush=True)
            print("TOKENS so far (task model): input %d, output %d" % tokens_used(dspy.settings.lm), flush=True)
        import threading, time
        def _tick():
            while True:
                time.sleep(60); print("TOKENS NOW (task model): input %d, output %d" % tokens_used(dspy.settings.lm), flush=True)
        threading.Thread(target=_tick, daemon=True).start()
        opt = dspy.GEPA(metric=metric, reflection_lm=dspy.LM(REFLECT, temperature=1.0, max_tokens=8000, cache=False, num_retries=12), max_metric_calls=budget,
                        reflection_minibatch_size=MINIBATCH, num_threads=int(os.environ.get("THREADS", "24")), track_stats=(os.environ.get("EFFICIENCY") == "1"))
        best = opt.compile(prog, trainset=train, valset=val)
        if not SKIP: print("AFTER  (validation, %d works): %.3f" % (nva, evaluate(best, val)), flush=True)
        print("TOKENS total (task model): input %d, output %d" % tokens_used(dspy.settings.lm), flush=True)
        out = ROOT / "title_author/data/gemini_prompts" / os.environ.get("OUT_NAME", "gepa.md")
        if os.environ.get("EFFICIENCY") == "1" and getattr(best, "detailed_results", None):
            dr, W = best.detailed_results, float(os.environ.get("TOKEN_W", "0.004")); cpt = 11529 / 8000      # characters per token, calibrated on the default prompt
            rows = []; need = re.findall(r"^#{2,3} .*$", Path(start).read_text(encoding="utf-8"), re.M)       # every heading of the starting prompt must survive
            for i, (c, v) in enumerate(zip(dr.candidates, dr.val_aggregate_scores)):
                ins = c.signature.instructions; tok = (len(ins) + len(LOCK)) / cpt
                missing = [h for h in need if h not in ins]; cut = ins.count("`") % 2 == 1          # an odd number of backticks = cut off inside a code span
                rows.append({"idx": i, "val": v, "prompt_tokens_est": round(tok), "skill_score": v - W * tok / 1000, "complete": not missing and not cut,
                             "missing_sections": missing, "instructions": ins})
            out.with_suffix(".candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            for r in rows: print("CANDIDATE %d: val %.3f  ~%d tokens  skill score %.4f%s" % (r["idx"], r["val"], r["prompt_tokens_est"], r["skill_score"], "  <- plain best" if r["idx"] == dr.best_idx else "") + ("" if r["complete"] else "  REJECTED: cut off, missing %s" % r["missing_sections"]), flush=True)
            # the token penalty is only a tie-break between COMPLETE prompts; a cut-off prompt can never win
            pick = max((r for r in rows if r["complete"]), key=lambda r: r["skill_score"]); print("PICKED candidate", pick["idx"], "by skill score among complete prompts", flush=True)
            out.write_text(pick["instructions"], encoding="utf-8")
        else:
            out.write_text(best.signature.instructions, encoding="utf-8")
        print("saved", out)

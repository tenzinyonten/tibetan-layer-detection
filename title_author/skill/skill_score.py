#!/usr/bin/env python3
"""skill_score: ONE number for a candidate skill that rewards accuracy and punishes waste. Free (reads saved answers, no network).

    python3 skill_score.py PROMPT.md "RUN_GLOB" [validation]            e.g. ../data/gemini_prompts/archive/gepa_v2.md "gepa_v2_validation_heldout"
    from skill_score import skill_score                                    # for GEPA: skill_score(correct_title, correct_author, prompt_chars, tool_calls)

score = mean(title, author share correct) - TOKEN_W * (prompt tokens / 1000) - CALL_W * (tool calls per work)
- accuracy: same scorer as every other result (IoU >= 0.5, same block), saved answers + the rule layer, one answer per work
- prompt tokens: ESTIMATED as prompt characters / 1.44 (the default prompt is 11,529 characters = about 8,000 input tokens; rough, Tibetan examples cost more tokens than English rules)
- tool calls per work: 0 for the plain skill; pass the measured value for an agent version
Weights are a choice, not a measurement: TOKEN_W=0.004 means 1,000 fewer prompt tokens is worth 0.4 accuracy points. Change with env TOKEN_W / CALL_W.
Only validation answers should be scored while tuning (test needs ALLOW_TEST=1, final comparison only)."""
import os, sys, json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import title_author_skill as S
CHARS_PER_TOKEN = 11529 / 8000      # calibrated on the default prompt (11529 characters, about 8,000 input tokens per README)
TOKEN_W = float(os.environ.get("TOKEN_W", 0.004)); CALL_W = float(os.environ.get("CALL_W", 0.01))


def skill_score(title_ok, author_ok, prompt_chars, tool_calls_per_work=0.0):
    acc = (title_ok + author_ok) / 2; tok = prompt_chars / CHARS_PER_TOKEN
    return {"accuracy": acc, "prompt_tokens_est": tok, "score": acc - TOKEN_W * tok / 1000 - CALL_W * tool_calls_per_work}


def evaluate(prompt_path, run_glob, split="validation", tool_calls=0.0):
    if split == "test" and os.environ.get("ALLOW_TEST") != "1": sys.exit("test needs ALLOW_TEST=1")
    _src = (HERE.parent / "src/ta_compare_runs.py").read_text(encoding="utf-8").split("def main")[0]
    g = {"__name__": "cmp", "__file__": str(HERE.parent / "src/ta_compare_runs.py")}; exec(_src, g)
    rows = {r.work_id: r for r in g["load"](split, 0, 0).itertuples()}
    raw = g["answers"](run_glob); ids = [w for w in raw if w in rows and g["answered"](raw[w])]
    if not ids: sys.exit("no answered works in that run")
    ok = {"title": 0, "author": 0}
    for w in ids:
        r = rows[w]; head, tail, _w = S.normalize_input(r.text_head, "" if r.whole_work else (r.text_tail or ""), bool(r.whole_work))
        ans, _ = S.postprocess(S.parse_answer(raw[w]), head, tail)                     # exactly what extract() does after the model answers
        for f in ok: ok[f] += g["score"](r, ans, f)[0] == "ok"
    t, a = ok["title"] / len(ids), ok["author"] / len(ids)
    out = skill_score(t, a, len(Path(prompt_path).read_text(encoding="utf-8")), tool_calls)
    return {"works": len(ids), "title": t, "author": a, **out}


if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    r = evaluate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "validation")
    print(f"{r['works']} works  title {r['title']:.3f}  author {r['author']:.3f}  accuracy {r['accuracy']:.3f}  prompt ~{r['prompt_tokens_est']:.0f} tokens  SCORE {r['score']:.4f}")

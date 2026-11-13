#!/usr/bin/env python3
"""title_author_fix: a second pass that fixes SHAKY author answers by showing the LLM a short list of candidate names with context.

Why: the first answer is wrong about 20-30% of the time for the author, mostly because of a different person (stanza writer, compiler,
copyist, patron). The rule script can list the names that stand before an agentive + writing verb; the right author is among its
top 5 in about 43% of the wrong cases. The LLM only has to PICK one (a short prompt, no tools, no internet), so it is cheap.

    from title_author_skill import extract, gemini_call
    from title_author_fix import fix
    first = extract(head, tail, whole_work, call_llm=gemini_call())
    final = fix(first, head, tail, whole_work, call_llm=gemini_call())     # changes only the author, only for flagged works

Offline check:  python title_author_fix.py selftest        (no key, no cost)
"""
import re, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ta_rule_extract as T
import title_author_skill as S

REQUESTER = re.compile(r"བསྐུལ|ངོར|ཞུས|གསོལ་བ|བཀའ་བསྒོ")
SIGNATURE = re.compile(r"(?:མཛད|སྦྱར|བྲིས|བཀོད)་?པ")

FIX_SYSTEM = """You choose the AUTHOR of a Tibetan work from a short list of candidate names.
The author is the person who composed the WHOLE work. Do not choose:
- a translator or lotsawa (བསྒྱུར, ལོ་ཙཱ་བ),
- someone who only wrote a stanza, a verse or a prayer inside or at the end of the text,
- someone who only compiled, collected, copied, proofread (དག་པར་ཕབ་པ), edited or printed it,
- a patron or the person who asked for the text (བསྐུལ, ངོར, ཞུས, གསོལ་བ, བཀའ་བསྒོ).
If the only name is the scribe or the compiler, choose that name.
Each candidate is shown with the text before and after it; the name itself is inside [[ ]].
Answer with JSON only: {"choice": N, "reason": "<one short sentence>"} where N is the candidate number, or 0 to keep the current answer."""


def flags(ans, notes, head, tail):
    """Free rule-based reasons to doubt the author. Measured on validation: flagged works are right about 56% of the time, others 78%."""
    out = []
    a = (ans or {}).get("author")
    if not a or any(("again" in n) or ("rule-based" in n) for n in notes or []):
        out.append("author repaired, from the fallback, or missing")
    text = S.region_text((ans or {}).get("author_region"), head, tail)
    i = text.rfind(a) if a and a in text else -1
    if i >= 0 and REQUESTER.search(text[i + len(a): i + len(a) + 70]):
        out.append("requester cue after the author")
    if len(SIGNATURE.findall(text)) >= 2:
        out.append("two or more signature verbs in the region")
    return out


def candidates(head, tail, whole_work, k=5, ctx=110):
    """Top-k author candidates from the rule script (ranked by its fitted weights), each with the text around it."""
    W = T.load_weights(); found = {}
    for region, text in (("head", head), ("tail", "" if whole_work else (tail or ""))):
        if not text:
            continue
        for f, i, name in T.author_candidates_v2(text, region, bool(whole_work or not tail)):
            sc = sum(W.get(key, 0.0) * v for key, v in f.items()) if W else 0.0
            key = (region, name)
            if key not in found or sc > found[key][0]:
                found[key] = (sc, region, i, name, text)
    top = sorted(found.values(), key=lambda x: -x[0])[:k]
    out = []
    for n, (sc, region, i, name, text) in enumerate(top, 1):
        out.append({"n": n, "region": region, "start": i, "name": name,
                    "context": (text[max(0, i - ctx): i] + "[[" + name + "]]" + text[i + len(name): i + len(name) + ctx]).replace("\n", " ")})
    return out


def build_fix_input(ans, cands):
    lines = [f"Current answer: {ans.get('author')!r} (region {ans.get('author_region')})", "", "Candidates:"]
    for c in cands:
        lines.append(f"{c['n']}. ({c['region']}) …{c['context']}…")
    return "\n".join(lines)


def fix(ans, head, tail, whole_work, call_llm, k=5, notes=None):
    """Returns (answer, notes). Changes only the author, only when flags() finds a reason, only to an exact substring."""
    ans = dict(ans); notes = list(notes if notes is not None else ans.get("notes", []))
    why = flags(ans, notes, head, tail)
    if not why:
        return ans, notes
    cands = candidates(head, tail, whole_work, k)
    if not cands:
        return ans, notes
    raw = call_llm(FIX_SYSTEM, build_fix_input(ans, cands))
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        choice = int(json.loads(m.group(0)).get("choice", 0)) if m else 0
    except Exception:
        choice = 0
    if 1 <= choice <= len(cands):
        c = cands[choice - 1]
        ans["author"], ans["author_region"] = c["name"], c["region"]
        ans, n2 = S.apply_rules(ans, head, tail, fallback=False)            # put a touching role word back
        notes += [f"fix pass ({'; '.join(why)}): chose candidate {choice}"] + n2
    else:
        notes.append(f"fix pass ({'; '.join(why)}): kept the current answer")
    ans["notes"] = notes
    return ans, notes


def selftest():
    head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
    wrong = {"title": "x", "title_region": "head", "author": None, "author_region": None, "notes": []}
    assert flags(wrong, [], head, "") , "a missing author must be flagged"
    cs = candidates(head, "", True)
    assert cs and any("གྲགས་པ་རྒྱལ་མཚན" in c["name"] for c in cs), cs
    pick = next(c["n"] for c in cs if "གྲགས་པ་རྒྱལ་མཚན" in c["name"])
    out, notes = fix(wrong, head, "", True, call_llm=lambda s, u: json.dumps({"choice": pick, "reason": "test"}))
    assert out["author"] and out["author"] in head, out
    keep, _ = fix(wrong, head, "", True, call_llm=lambda s, u: "{}")
    assert keep["author"] is None
    print("selftest OK:", out["author"], "|", notes[-2:])


if __name__ == "__main__":
    selftest() if (sys.argv[1:] or ["selftest"])[0] == "selftest" else sys.exit(__doc__)

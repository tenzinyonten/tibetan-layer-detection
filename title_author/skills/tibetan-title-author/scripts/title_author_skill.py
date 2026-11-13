#!/usr/bin/env python3
"""title_author_skill: find the title and the author of a Tibetan work with ANY LLM, plus a rule-based layer around it.

The module does not care which LLM you use. You pass a function `call_llm(system_prompt, user_text) -> str`
(a Gemini helper is included; for another model write a 3-line function). The skill then
  1. builds the input ([HEAD] and [TAIL] blocks),
  2. calls your LLM with the prompt (instructions + 12 examples),
  3. reads the JSON answer and checks that the title and author are exact substrings of the text,
  4. applies the rule-based layer: put back a role word that touches the name, and fall back to the rule-based
     extractor (no LLM) when the LLM gives no usable author.

    from title_author_skill import extract, gemini_call, rule_only
    result = extract(head, tail, whole_work=False, call_llm=gemini_call("gemini-3.8-flash"))
    result = rule_only(head, tail)                       # no LLM at all, much weaker (author F1 about 0.38)

Command line:
    python title_author_skill.py selftest                # offline check with a fake LLM, no key, no cost
    python title_author_skill.py rule-only FILE.txt      # rule-based answer for one text file
    python title_author_skill.py extract FILE.txt [MODEL]   # needs GEMINI_API_KEY in the environment (costs a little)
"""
import os, re, sys, json, threading, urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # ta_rule_extract.py lives next to this file
import ta_rule_extract as T                       # the rule-based extractor (plain Python, no LLM)

REF = HERE.parent / "references"
DEFAULT_PROMPT = Path(os.environ["TA_PROMPT"]) if os.environ.get("TA_PROMPT") else None      # None = references/rules.md + references/examples.md (about 8,000 tokens)
REQUIRED_SECTIONS = ("## Title Extraction Rules", "## Author Extraction Rules", "### Twelve examples")     # a prompt without these is cut off
ROLES = sorted(T.ROLES, key=len, reverse=True)
FIELDS = ("title", "title_region", "author", "author_region")


MAX_BLOCK = 3000
_PROMPTS = {}


def _prompt_text(path=None):
    p = str(path)
    if p not in _PROMPTS:
        t = Path(path).read_text(encoding="utf-8") if path else (REF / "rules.md").read_text(encoding="utf-8") + "\n\n" + (REF / "examples.md").read_text(encoding="utf-8")   # read once per process
        missing = [x for x in REQUIRED_SECTIONS if x not in t]
        if missing and not os.environ.get("TA_ALLOW_INCOMPLETE_PROMPT"):
            raise ValueError(f"prompt {path} is cut off: missing {missing}")          # a truncated prompt once became the default; refuse it
        _PROMPTS[p] = t
    return _PROMPTS[p]


def _review_flags(ans, notes, head, tail):
    try:
        import title_author_fix as F
        return F.flags(ans, notes, head, tail)
    except Exception:
        return []


def build_input(head, tail="", whole_work=False):
    """The user message: one or two labelled blocks. A short work has only [HEAD]."""
    tail = "" if whole_work else (tail or "")
    return "[HEAD]\n" + head + ("\n[TAIL]\n" + tail if tail else "")


def parse_answer(raw):
    """First JSON object in the reply (extra braces or text around it are ignored); None when there is none."""
    raw, dec, i = raw or "", json.JSONDecoder(), 0
    while True:
        i = raw.find("{", i)
        if i < 0: return None
        try:
            d, _ = dec.raw_decode(raw[i:])
            if isinstance(d, dict): return {k: d.get(k) for k in FIELDS}
        except Exception:
            pass
        i += 1


def region_text(region, head, tail):
    return head if region == "head" else tail if region == "tail" else ""


def usable(ans, which, head, tail):
    """True when the answer for `which` is an exact substring of the block it names."""
    s, reg = (ans or {}).get(which), (ans or {}).get(which + "_region")
    return bool(s) and str(s).lower() != "null" and s in region_text(reg, head, tail)


def apply_rules(ans, head, tail, add_role_word=True, fallback=True):
    """The rule-based layer. Returns (answer, notes). Only the author is changed."""
    ans = dict(ans or {k: None for k in FIELDS}); notes = []
    if not usable(ans, "author", head, tail) and ans.get("author"):
        fixed = T.repair_span(ans["author"], head, tail, ans.get("author_region"), last=True)
        if fixed:
            ans["author"], ans["author_region"] = fixed; notes.append("author found again after ignoring spaces and newlines")
    if usable(ans, "author", head, tail):
        if add_role_word:
            s, text = ans["author"], region_text(ans["author_region"], head, tail)
            i = text.rfind(s); pre = text[:i].rstrip("་ \n")
            for role in ROLES:
                if pre.endswith(role) and (i - len(pre)) <= 2:
                    ans["author"] = text[len(pre) - len(role): i + len(s)]; notes.append(f"added role word {role!r}"); break
    elif fallback:
        r = rule_only(head, tail)
        if r.get("author"):
            ans["author"], ans["author_region"] = r["author"], r["author_region"]; notes.append("author from the rule-based extractor")
    return ans, notes


def rule_only(head, tail="", whole_work=False):
    """Rule-based answer, no LLM. Weak on author, but free and instant."""
    row = SimpleNamespace(text_head=head, text_tail=(None if whole_work else tail), whole_work=bool(whole_work or not tail))
    p = T.predict(row)
    return {k: p.get(k) for k in FIELDS}


def normalize_input(head, tail="", whole_work=False):
    """Bound the input without losing the ending. A whole work (<= 6,000 characters) is sent in full as [HEAD]; a longer work is cut to its first
    and last 3,000 characters. A text flagged whole_work (or with no tail) that is longer than 6,000 characters is split into its start and its end."""
    tail = tail or ""
    if whole_work or not tail:
        if len(head) > 2 * MAX_BLOCK: return head[:MAX_BLOCK], head[-MAX_BLOCK:], False
        return head, "", True
    return head[:MAX_BLOCK], tail[-MAX_BLOCK:], False


def postprocess(ans, head, tail, hybrid=True):
    """Everything that happens to the model's answer after it is parsed (also used by skill_score.py, so the score is the skill's own score).
    Returns (answer, notes)."""
    ans = dict(ans or {k: None for k in FIELDS}); notes = []
    if not usable(ans, "title", head, tail):
        fixed = T.repair_span(ans.get("title"), head, tail, ans.get("title_region"), last=False) if ans.get("title") else None
        if fixed:
            ans["title"], ans["title_region"] = fixed; notes.append("title found again after ignoring spaces and newlines")
        else:
            if ans.get("title"): notes.append("title was not an exact substring")
            ans["title"] = ans["title_region"] = None
    if hybrid:
        ans, n2 = apply_rules(ans, head, tail); notes += n2
    return ans, notes


def extract(head, tail="", whole_work=False, call_llm=None, prompt=None, hybrid=True):
    """Title and author of one work. `call_llm(system, user) -> str` is any function that calls your LLM.
    Returns {"title", "title_region", "author", "author_region", "notes", "needs_review", "review_reasons"}."""
    head, tail, whole_work = normalize_input(head, tail, whole_work)
    if call_llm is None:
        r = rule_only(head, tail, whole_work); r["notes"] = ["rule-based only (no LLM given)"]; r["needs_review"] = True; r["review_reasons"] = ["no LLM used"]; return r
    system = _prompt_text(prompt or DEFAULT_PROMPT)
    ans = parse_answer(call_llm(system, build_input(head, tail, whole_work))) or {k: None for k in FIELDS}
    ans, notes = postprocess(ans, head, tail, hybrid)
    ans["notes"] = notes
    why = _review_flags(ans, notes, head, tail)
    ans["needs_review"] = bool(why); ans["review_reasons"] = why            # flagged works were right 56-60% of the time, the others 78-79% (validation)
    return ans


def gemini_call(model="gemini-3.8-flash", thinking="low"):
    """A ready-made call_llm for Gemini over plain REST. The key is read from GEMINI_API_KEY and never stored.
    Token usage of the LAST call made by the calling thread is in call.local.usage (in, out); call.tag names model + thinking (cache key)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key: raise RuntimeError("set GEMINI_API_KEY in the environment")
    def call(system, user):
        gc = {"temperature": 0, "responseMimeType": "application/json"}
        if thinking and model.startswith("gemini-3.8"): gc["thinkingConfig"] = {"thinkingLevel": thinking}
        body = json.dumps({"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
                           "generationConfig": gc}).encode()
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", data=body,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        out = json.load(urllib.request.urlopen(req, timeout=180))
        u = out.get("usageMetadata") or {}
        call.local.usage = (u.get("promptTokenCount", 0), (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)))   # per thread: no race
        return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"] if not p.get("thought"))
    call.local = threading.local(); call.tag = f"{model}|thinking={thinking}"
    return call


def selftest():
    """Offline: a fake LLM drops the role word and, for a second text, returns nothing. No key, no network."""
    head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
    fake = lambda s, u: json.dumps({"title": "ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་", "title_region": "head", "author": "གྲགས་པ་རྒྱལ་མཚན", "author_region": "head"})
    r = extract(head, "", True, call_llm=fake)
    assert r["author"] == "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན", r            # role word put back by the rule layer
    none = lambda s, u: "{}"
    r2 = extract(head, "", True, call_llm=none)
    assert r2["author"], r2                                          # rule-based fallback supplied an author
    # a 4,000-character whole work keeps its ending (the author is at the very end): the old 3,000-character crop dropped it
    long_head = "ཀ" * 3500 + head
    seen = {}
    spy = lambda s, u: (seen.__setitem__("u", u), json.dumps({"title": None, "author": None}))[1]
    extract(long_head, "", True, call_llm=spy)
    assert "སྦྱར་བའོ" in seen["u"] and "[TAIL]" not in seen["u"], "whole work lost its ending"
    # a text over 6,000 characters flagged as whole is split into start + end
    huge = "ཀ" * 7000 + head; extract(huge, "", True, call_llm=spy)
    assert "[TAIL]" in seen["u"] and "སྦྱར་བའོ" in seen["u"]
    # the default prompt is complete (Title + Author sections + examples)
    assert all(x in _prompt_text(DEFAULT_PROMPT) for x in REQUIRED_SECTIONS)
    assert parse_answer('noise {"title": "a"} } extra')["title"] == "a"         # a stray brace does not drop the answer
    import tempfile                                                                # a cut-off prompt (no Title/Author sections) must be refused
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf: tf.write("## What Gemini receives\nIf the input contains no `[TAIL]` block, regions are `")
    try: _prompt_text(tf.name); ok = True
    except ValueError: ok = False
    finally: os.unlink(tf.name)
    assert not ok, "an incomplete prompt must be refused"
    print("selftest OK:", r["notes"], "|", r2["notes"])


if __name__ == "__main__":
    a = sys.argv[1:] or ["selftest"]
    if a[0] == "selftest": selftest()
    elif a[0] == "rule-only":
        t = Path(a[1]).read_text(encoding="utf-8"); print(json.dumps(rule_only(t, "", True), ensure_ascii=False, indent=1))
    elif a[0] == "extract":
        t = Path(a[1]).read_text(encoding="utf-8")
        head, tail = (t, "") if len(t) <= 6000 else (t[:3000], t[-3000:])
        print(json.dumps(extract(head, tail, len(t) <= 6000, call_llm=gemini_call(a[2] if len(a) > 2 else "gemini-3.8-flash")), ensure_ascii=False, indent=1))
    else: sys.exit(__doc__)

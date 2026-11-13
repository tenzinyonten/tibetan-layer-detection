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


def second_look(ans, head, tail, whole_work, call_llm):
    """Second look for a work that needs review: one more call that re-checks the AUTHOR with the first answer, the rule script's name candidates
    and the excerpt in view (prompt: references/second_look.md). Returns (answer, outcome) with outcome "changed", "kept" or "no reply".
    Measured with tools (text search, name candidates, BDRC) on 139 flagged validation works with gemini-3.5-flash-lite: author 0.597 -> 0.669,
    11 fixed and 1 broken. This version has no tools and is NOT measured."""
    import title_author_fix as F
    cands = F.candidates(head, tail, whole_work)
    user = F.build_fix_input(ans, cands) + "\n\n" + build_input(head, tail, whole_work)
    d = parse_answer(call_llm((REF / "second_look.md").read_text(encoding="utf-8"), user))
    cand = (d or {}).get("author")
    if d is None: return ans, "no reply"
    if not cand or cand == ans.get("author"): return ans, "kept"
    fixed = T.repair_span(cand, head, tail, "tail", last=True)                     # the colophon is usually in the tail: look there first
    if not fixed: return ans, "kept"
    new = dict(ans); new["author"], new["author_region"] = fixed
    new, _ = apply_rules(new, head, tail)                                          # same role-word step as the first pass
    return new, "changed"


def extract(head, tail="", whole_work=False, call_llm=None, prompt=None, hybrid=True, second=False):
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
    if second and why:                                                      # only the flagged third of the works gets a second call
        ans2, outcome = second_look(ans, head, tail, whole_work, call_llm)
        ans2["second_look"] = outcome; ans2["needs_review"] = ans["needs_review"]; ans2["review_reasons"] = ans["review_reasons"]
        ans2["notes"] = list(ans["notes"]) + [f"second look: {outcome}"]
        return ans2
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


def anthropic_call(model, max_tokens=4096):
    """call_llm for a Claude model through the official `anthropic` SDK (pip install anthropic; credentials from ANTHROPIC_API_KEY or `ant auth login`).
    No temperature or thinking settings are sent: current Claude models reject non-default sampling values and manage thinking themselves.
    NOT benchmarked: every accuracy number in this skill was measured with Gemini."""
    try: import anthropic
    except ImportError: raise RuntimeError("pip install anthropic to use a Claude model")
    client = anthropic.Anthropic()
    def call(system, user):
        msg = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        call.local.usage = (getattr(msg.usage, "input_tokens", 0), getattr(msg.usage, "output_tokens", 0))
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    call.local = threading.local(); call.tag = f"anthropic:{model}"
    return call


def llm_for(model):
    """call_llm for a model name: claude-* -> Claude API, gemini-* -> Gemini API. For any other provider write your own call_llm(system, user) -> str."""
    m = (model or "").lower()
    if m.startswith("claude"): return anthropic_call(model)
    if m.startswith("gemini"): return gemini_call(model)
    raise ValueError(f"unknown provider for model {model!r}: pass your own call_llm(system, user) -> str")


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
    # second look: a work whose author is doubtful gets one more call; only an exact substring can replace the author
    wrong = {"title": None, "title_region": None, "author": "ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ", "author_region": "head"}
    good = lambda s_, u_: json.dumps({"author": "གྲགས་པ་རྒྱལ་མཚན", "reason": "colophon"})
    new, out = second_look(wrong, head, "", True, good); assert out == "changed" and new["author"] == "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན", (new, out)
    _, out = second_look(wrong, head, "", True, lambda s_, u_: json.dumps({"author": "not in the text"})); assert out == "kept"
    _, out = second_look(wrong, head, "", True, lambda s_, u_: "no json"); assert out == "no reply"
    two = extract(head, "", True, call_llm=lambda s_, u_: json.dumps({"title": None, "author": None}), second=True)
    assert two.get("second_look") in ("changed", "kept", "no reply") and two["needs_review"], two
    # provider routing + the Claude adapter, with a fake SDK (no key, no network)
    import types
    sent = {}
    class _Msgs:
        def create(self, **kw):
            sent.update(kw); return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text='{"title": null}')], usage=types.SimpleNamespace(input_tokens=7, output_tokens=3))
    fake_sdk = types.ModuleType("anthropic"); fake_sdk.Anthropic = lambda: types.SimpleNamespace(messages=_Msgs())
    sys.modules["anthropic"] = fake_sdk
    try:
        c = llm_for("claude-sonnet-5-5"); assert c("SYS", "USER") == '{"title": null}' and c.local.usage == (7, 3)
        assert sent["system"] == "SYS" and sent["messages"][0]["content"] == "USER" and "temperature" not in sent and "thinking" not in sent
        assert extract(head, "", True, call_llm=c)["author"], "extract works with a Claude adapter"
    finally: del sys.modules["anthropic"]
    try: llm_for("some-other-model"); ok = True
    except ValueError: ok = False
    assert not ok
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

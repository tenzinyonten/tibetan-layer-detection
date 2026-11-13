#!/usr/bin/env python3
"""ta.py: the one command an agent (or a person) runs for the Tibetan title/author skill. Prints JSON. No key and no network unless you use `llm`.

    python3 scripts/ta.py rules  WORK.txt                                  # free rule-based candidate (title ~0.64, author ~0.36: a starting point, not an answer)
    python3 scripts/ta.py verify WORK.txt --title "..." --author "..."     # check/repair YOUR answer against the text, add needs_review
    python3 scripts/ta.py verify WORK.txt --json '{"title": "...", "title_region": "head", "author": "...", "author_region": "tail"}'
    GEMINI_API_KEY=... python3 scripts/ta.py llm WORK.txt [MODEL]          # whole pipeline with Gemini (batch use; costs money)
    python3 scripts/ta.py selftest

WORK.txt is the work's text. A work of 6,000 characters or fewer is used whole; a longer one is cut to its first and last 3,000 characters.
Output keys: title, title_region, author, author_region, notes, needs_review, review_reasons.
`verify` makes both strings exact substrings (finds them again if a space or newline differs), puts back a role word that touches the author's name,
and falls back to the rule script's author when yours cannot be found."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import title_author_skill as S

MAX = S.MAX_BLOCK


def split_work(text):
    return (text, "", True) if len(text) <= 2 * MAX else (text[:MAX], text[-MAX:], False)


def tibetan_share(text):
    """Share of Tibetan-script characters (U+0F00-U+0FFF) among the non-space characters."""
    chars = [c for c in text if not c.isspace()]
    return sum("\u0f00" <= c <= "\u0fff" for c in chars) / len(chars) if chars else 0.0


def locate(s, head, tail, last):
    """Block that contains the string: the author is searched in the tail first (it is usually the last match), the title in the head first."""
    order = ("tail", "head") if last else ("head", "tail")
    return next((r for r in order if s and s in S.region_text(r, head, tail)), None)


def verify(head, tail, whole, title=None, author=None, answer=None):
    ans = {k: None for k in S.FIELDS}
    if answer: ans.update({k: answer.get(k) for k in S.FIELDS})
    if title is not None: ans["title"], ans["title_region"] = title, ans.get("title_region") or locate(title, head, tail, False)
    if author is not None: ans["author"], ans["author_region"] = author, ans.get("author_region") or locate(author, head, tail, True)
    ans, notes = S.postprocess(ans, head, tail)
    ans["notes"] = notes; why = S._review_flags(ans, notes, head, tail)
    ans["needs_review"] = bool(why); ans["review_reasons"] = why
    return ans


def selftest():
    head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
    r = verify(head, "", True, title="ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་", author="གྲགས་པ་རྒྱལ་མཚན")
    assert r["author"] == "དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན" and r["title_region"] == "head" and r["author_region"] == "head", r      # role word restored, regions found
    r2 = verify(head, "", True, title="ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་", author="not in the text")
    assert r2["author"] and r2["author"] != "not in the text" and r2["needs_review"], r2                                       # rule fallback, flagged
    assert tibetan_share(head) > 0.8 and tibetan_share("hello world") == 0 and tibetan_share("") == 0
    assert split_work("ཀ" * 7000)[2] is False and split_work("ཀ" * 100)[2] is True
    print("ta.py selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rules", "verify", "llm", "selftest"]); ap.add_argument("file", nargs="?"); ap.add_argument("model", nargs="?", default="gemini-3.8-flash")
    ap.add_argument("--title"); ap.add_argument("--author"); ap.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest(); sys.exit()
    if not a.file: sys.exit("give the work's text file")
    text = Path(a.file).read_text(encoding="utf-8")
    if tibetan_share(text) < 0.2: sys.exit("this does not look like Tibetan text (under 20% Tibetan characters); the skill is for Tibetan works only")
    head, tail, whole = split_work(text)
    if a.cmd == "rules": out = S.extract(head, tail, whole, call_llm=None)
    elif a.cmd == "verify":
        if not (a.title or a.author or a.json): sys.exit("give --title/--author or --json")
        out = verify(head, tail, whole, a.title, a.author, json.loads(a.json) if a.json else None)
    else: out = S.extract(head, tail, whole, call_llm=S.gemini_call(a.model))
    print(json.dumps(out, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""ta.py: the one command for the Tibetan title/author skill. You only need the BOOK or WHOLE TEXT as a file; you do NOT need a title or an author.
The skill takes the first 3,000 and the last 3,000 characters of the text (a text of 6,000 characters or fewer is used whole), finds the title and
the author there, repairs the answer and flags doubt. Prints JSON.

    python3 scripts/ta.py llm     BOOK.txt MODEL        # text in, answer out: the model reads the excerpt, the scripts check it. MODEL = gemini-* (GEMINI_API_KEY) or claude-* (ANTHROPIC_API_KEY). Costs money.
    python3 scripts/ta.py rules   BOOK.txt              # free, no model: rule-based guess (title ~0.64, author ~0.36; a hint, not an answer)
    python3 scripts/ta.py excerpt BOOK.txt              # prints only the [HEAD] and [TAIL] the model would read (for an agent: read this, not the whole book)
    python3 scripts/ta.py verify  BOOK.txt --title "..." --author "..."    # only if a draft already exists (an agent's own answer): check and repair it
    python3 scripts/ta.py selftest
Options: --second-look re-checks the author with one more model call on works that need review (about a third of the works; see references/second_look.md).
--chars N sets the block size (default 3,000, the size everything was measured with; 2,000 is untested).
Output keys: title, title_region, author, author_region, notes, needs_review, review_reasons.
`verify` makes both strings exact substrings (finds them again if a space or newline differs), puts back a role word that touches the author's name,
and falls back to the rule script's author when the draft's cannot be found. It never compares against labels."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import title_author_skill as S

def split_work(text):
    m = S.MAX_BLOCK                                                     # block size (set by --chars)
    return (text, "", True) if len(text) <= 2 * m else (text[:m], text[-m:], False)


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
    book = "ཀ" * 20000 + head                                  # a "book": the colophon is at the very end
    h, t, w = split_work(book); assert w is False and len(h) == 3000 and len(t) == 3000 and "སྦྱར་བའོ" in t
    S.MAX_BLOCK = 2000; h, t, w = split_work(book); assert len(h) == 2000 and len(t) == 2000; S.MAX_BLOCK = 3000
    assert split_work("ཀ" * 7000)[2] is False and split_work("ཀ" * 100)[2] is True
    print("ta.py selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["llm", "rules", "excerpt", "verify", "selftest"]); ap.add_argument("file", nargs="?"); ap.add_argument("model", nargs="?", default="gemini-3.8-flash")
    ap.add_argument("--title"); ap.add_argument("--author"); ap.add_argument("--json"); ap.add_argument("--chars", type=int); ap.add_argument("--second-look", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest(); sys.exit()
    if a.chars: S.MAX_BLOCK = a.chars
    if not a.file: sys.exit("give the work's text file")
    text = Path(a.file).read_text(encoding="utf-8")
    if tibetan_share(text) < 0.2: sys.exit("this does not look like Tibetan text (under 20% Tibetan characters); the skill is for Tibetan works only")
    head, tail, whole = split_work(text)
    print(f"[ta] {len(text):,} characters in the text: " + (f"short, using the whole text" if whole else f"long, using the first {len(head):,} and the last {len(tail):,} characters (the middle is skipped)"), file=sys.stderr)
    if a.cmd == "excerpt":
        print(S.build_input(head, tail, whole)); sys.exit()
    if a.cmd == "rules": out = S.extract(head, tail, whole, call_llm=None)
    elif a.cmd == "verify":
        if not (a.title or a.author or a.json): sys.exit("give --title/--author or --json")
        out = verify(head, tail, whole, a.title, a.author, json.loads(a.json) if a.json else None)
    else: out = S.extract(head, tail, whole, call_llm=S.llm_for(a.model), second=a.second_look)
    print(json.dumps(out, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""ta.py: the one command for the Tibetan title/author skill. You only need the BOOK or WHOLE TEXT as a file; you do NOT need a title or an author.
The skill takes the first 3,000 and the last 3,000 characters of the text (a text of 6,000 characters or fewer is used whole), finds the title and
the author there, repairs the answer and flags doubt. Prints JSON.

    python3 scripts/ta.py BOOK.txt                      # AUTO: the whole pipeline in one command. Uses the model your key allows (GEMINI_API_KEY -> gemini-3.8-flash, ANTHROPIC_API_KEY -> claude-sonnet-5-5), else the free rule-based guess.
    python3 scripts/ta.py FOLDER/                       # AUTO on every .txt file of a folder: one JSON line per book
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
import os, sys, json, argparse
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
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        assert pick_model(None) is None and pick_model("claude-x") == "claude-x"
        os.environ["GEMINI_API_KEY"] = "x"; assert pick_model(None) == "gemini-3.8-flash"
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    print("ta.py selftest OK")


COMMANDS = ["auto", "llm", "rules", "excerpt", "verify", "selftest"]


def pick_model(explicit):
    """auto mode: the model you name, else the one your key allows, else None (rules only)."""
    if explicit: return explicit
    if os.environ.get("GEMINI_API_KEY"): return "gemini-3.8-flash"
    if os.environ.get("ANTHROPIC_API_KEY"): return "claude-sonnet-5-5"          # not benchmarked
    return None


def process(path, a, model=None):
    """One text file -> one result dict (or an {"error": ...} dict). Prints the one-line note about the cut to stderr."""
    text = Path(path).read_text(encoding="utf-8")
    if tibetan_share(text) < 0.2: return {"error": "this does not look like Tibetan text (under 20% Tibetan characters); the skill is for Tibetan works only"}
    head, tail, whole = split_work(text)
    print(f"[ta] {Path(path).name}: {len(text):,} characters: " + ("short, using the whole text" if whole else f"long, using the first {len(head):,} and the last {len(tail):,} characters (the middle is skipped)"), file=sys.stderr)
    if a.cmd == "excerpt": return {"excerpt": S.build_input(head, tail, whole)}
    if a.cmd == "rules" or (a.cmd == "auto" and not model): return S.extract(head, tail, whole, call_llm=None)
    if a.cmd == "verify":
        if not (a.title or a.author or a.json): sys.exit("give --title/--author or --json")
        return verify(head, tail, whole, a.title, a.author, json.loads(a.json) if a.json else None)
    return S.extract(head, tail, whole, call_llm=LLM[model], second=a.second_look)


class _Llm(dict):
    def __missing__(self, model): self[model] = S.llm_for(model); return self[model]
LLM = _Llm()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in COMMANDS and not sys.argv[1].startswith("-"): sys.argv.insert(1, "auto")     # `ta.py book.txt` = auto mode
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=COMMANDS); ap.add_argument("file", nargs="?"); ap.add_argument("model", nargs="?")
    ap.add_argument("--title"); ap.add_argument("--author"); ap.add_argument("--json"); ap.add_argument("--chars", type=int); ap.add_argument("--second-look", action="store_true")
    ap.add_argument("--model", dest="model_opt")
    a = ap.parse_args()
    if a.cmd == "selftest": selftest(); sys.exit()
    if a.chars: S.MAX_BLOCK = a.chars
    if not a.file: sys.exit("give the work's text file (or, in auto mode, a folder of .txt files)")
    explicit = a.model_opt or a.model
    model = pick_model(explicit) if a.cmd == "auto" else (explicit or "gemini-3.8-flash")
    if a.cmd == "auto":
        print(f"[ta] auto mode: " + (f"using the model {model}" if model else "no key found (GEMINI_API_KEY or ANTHROPIC_API_KEY): free rule-based guess only, weak on the author"), file=sys.stderr)
    target = Path(a.file)
    if target.is_dir():
        files = sorted(target.glob("*.txt"))
        if not files: sys.exit(f"no .txt files in {target}")
        for f in files:                                                         # a folder: one JSON line per book
            try: out = process(f, a, model)
            except Exception as e: out = {"error": f"{type(e).__name__}: {e}"}
            print(json.dumps({"file": f.name, **out}, ensure_ascii=False), flush=True)
    else:
        out = process(target, a, model)
        if "error" in out: sys.exit(out["error"])
        print(out["excerpt"] if a.cmd == "excerpt" else json.dumps(out, ensure_ascii=False, indent=1))

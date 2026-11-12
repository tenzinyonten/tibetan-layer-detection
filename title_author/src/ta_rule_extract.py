#!/usr/bin/env python3
"""Rule-based title + author spans. Read this file as a grammar of the page.

A Tibetan work usually has two *lines*:

  TITLE LINE (opening, sometimes only at the close)
    [yig mgo] [homage] [Sanskrit རྒྱ་གར་སྐད་དུ …] [བོད་སྐད་དུ]
    TITLE   ཞེས་བྱ་བ / བཞུགས་སོ
    The gold span is TITLE. Formula words stay outside.

  AUTHOR LINE (colophon; last quarter of a short work, tail of a long work)
    [role] NAME  ས / ཀྱིས / གྱིས   མཛད / སྦྱར / བྲིས / བཀོད
    The gold span is [role] NAME. Agentive and verb stay outside.
    If མཛད and བསྒྱུར both appear, take མཛད (the composer, not the lotsawa).

Gold is character offsets in text_head / text_tail. A hit is IoU >= 0.5.

    python title_author/src/ta_rule_extract.py score validation
    python title_author/src/ta_rule_extract.py score train --n 2000 --seed 0
    python title_author/src/ta_rule_extract.py show WORK_ID

Uses parquet_export_strat_both_notrans_nous (lotsawas dropped, underscores removed). Does not
open test unless ALLOW_TEST=1. Does not write over the gold parquet.
"""
from __future__ import annotations

import json, os, re, sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "title_author/data/processed/parquet_export_strat_both_notrans_nous"
OUT = ROOT / "title_author/data/rule_runs"

# --- the title line ----------------------------------------------------------
# Skip page furniture until Tibetan content starts.
YIG = r"[༄༅༆༈༔༼༽།༎༏༐༑༔\s\u0F0B]*"
HOMAGE = re.compile(
    r"^(?:" + YIG + r")*(?:ན་མོ|ཨོཾ|ༀ|སྭསྟི|བཛྲ|ཧཱུྃ|ཧཱུཾ)[^\n]{0,80}\n?",
    re.M,
)
SANSKRIT = re.compile(r"རྒྱ་གར་སྐད་དུ[^\n]{0,400}?བོད་སྐད་དུ[།\s]*")
BOD_SKAD = re.compile(r"བོད་སྐད་དུ[།\s]*")
# Formula that *closes* the title. Keep it outside the span.
TITLE_STOP = re.compile(r"(?:ཞེས|ཅེས)་བྱ་བ|བཞུགས་སོ|བཞུགས།|བཞུགས༔")
# Closing title only, no opening heading:  X ཞེས་བྱ་བ་འདི་ནི
CLOSE_TITLE = re.compile(
    r"([^\n།]{8,120}?)(?:ཞེས|ཅེས)་བྱ་བ་འདི་ནི"
)

# --- the author line ---------------------------------------------------------
# Role words that sit *in* the gold span ~76% of the time they touch the name.
ROLES = (
    "ཤཱཀྱའི་དགེ་སློང་མ", "ཤཱཀྱའི་དགེ་སློང", "དགེ་སློང་མ", "དགེ་སློང",
    "སློབ་དཔོན་ཆེན་པོ", "སློབ་དཔོན", "རྗེ་བཙུན་མ", "རྗེ་བཙུན",
    "མཁས་པ་ཆེན་པོ", "མཁན་ཆེན", "མཁན་པོ", "བཙུན་པ", "བནྡེ",
    "རིག་འཛིན", "པཎྜི་ཏ", "པཎ་ཆེན", "གཏེར་སྟོན", "དགེ་བསྙེན",
    "དགེ་བཤེས", "སྤྲུལ་སྐུ", "ཆོས་རྗེ", "བླ་མ",
)
ROLE_ALT = "|".join(sorted(ROLES, key=len, reverse=True))
# Composer *line* verbs (not the word མཛད inside a title like མཛད་པའི་).
# བསྒྱུར་བ is last and only used if no composer verb is on the page.
VERB_FORMS = (
    ("མཛད", re.compile(r"མཛད་པ(?!འི)")),
    ("སྦྱར", re.compile(r"སྦྱར་བ")),
    ("བྲིས", re.compile(r"བྲིས་པ|བྲིས།|བྲིས༔")),
    ("བཀོད", re.compile(r"བཀོད་པ")),
    ("བསྒྱུར", re.compile(r"བསྒྱུར་བ|བསྒྱུར།")),
)
# Explicit agentives. Do *not* treat every final ས as agentive (ཤེས, གྲོས).
AGENT_WORD = re.compile(r"(?:ཀྱིས|གྱིས|གིས|ཡིས)$")
VOWEL_S = re.compile(r"([ཱིེོུ])ས(?:་)?$")
HONOR = re.compile(r"འི་(?:ཞབས|ཞལ)(?:་)?$")
LOTSAWA = re.compile(r"ལོ་ཙཱ|ལོ་ཙྪ|ལོཙཱ|ལོཙྪ|ལོ་ཙ་བ")
OPEN_AUTHOR = re.compile(
    rf"((?:{ROLE_ALT})?(?:[\u0F40-\u0F6C][\u0F71-\u0FBC]*\u0F0B?){{2,10}})"
    r"(?:ཀྱིས|གྱིས|གིས|ཡིས|(?<=[ཱིེོུ])ས)མཛད་པ་བཞུགས"
)


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def region_text(r):
    head = r.text_head or ""
    tail = "" if r.whole_work else (r.text_tail or "")
    return head, tail


HOMAGE_START = re.compile(r"(?:ན་མོ|ན་མཿ|ན་མཽ|ན་མ་|ན་མ[ཿ་]|མ་མོ|ཨོཾ|ཨཱཿ|ཧྲཱིཿ|ཧཱུྃ|ཧཱུཾ|ༀ|སྭསྟི|བཛྲ|ཨ་ཧོ|ཨེ་མ་ཧོ|ཀྱེ་མ|ཛཿ|ཧྲཱིཿ|ཨ་རྒྱ)")
NON_TIBETAN = re.compile(r"[^\u0F00-\u0FFF\s\d\W]")
SANSKRIT_MARKS = re.compile(r"[ཱཿྃཾྀྭྱྲླྷཥཊྚྜྞྟྡྫ]")


def sanskrit_like(line):
    """A line is Sanskrit / mantra / OCR junk when marks are dense or another script appears."""
    if NON_TIBETAN.search(line):
        return True
    n = len(line.strip())
    return n >= 6 and len(SANSKRIT_MARKS.findall(line)) / n > 0.28


def skip_opening(text):
    """Walk past yig mgo, homage, mantra and Sanskrit lines. Stop at བོད་སྐད་དུ if present."""
    m = SANSKRIT.search(text)
    if m:
        return m.end()
    m = BOD_SKAD.search(text[:800])
    if m:
        return m.end()
    i = 0
    for _ in range(5):
        y = re.match(YIG, text[i:])
        i += y.end() if y else 0
        line_end = text.find("\n", i)
        line_end = len(text) if line_end < 0 else line_end
        # clause = up to the first shad that follows some content, but not past the line
        sh = re.search(r"[།༎༔]", text[i + 3: line_end])
        clause_end = i + 3 + sh.end() if sh else line_end
        clause = text[i:clause_end]
        if HOMAGE_START.match(clause) or sanskrit_like(clause):
            i = clause_end
            continue
        break
    y = re.match(YIG, text[i:])
    return i + (y.end() if y else 0)


def cut_title(chunk):
    """Take letters from the start of chunk until a title-stop formula."""
    m = TITLE_STOP.search(chunk)
    if m:
        end = m.start()
    else:
        # no formula: first shad after a real heading, else cap
        m2 = re.search(r"།", chunk[12:])
        end = 12 + m2.start() if m2 else min(len(chunk), 90)
    span = chunk[:end]
    span = re.sub(r"[།༔\s\u0F0B]+$", "", span)
    span = span.strip()
    return span if len(span) >= 3 else ""


def find_title_v0(head, tail):
    """Single-line heuristic: first heading after skip_opening, else ཞེས་བྱ་བ་འདི་ནི."""
    start = skip_opening(head)
    chunk = head[start: start + 400]
    title = cut_title(chunk)
    if title and not TITLE_STOP.search(head[start: start + 12]):
        return title, "head", head.find(title, max(0, start - 5))
    for region, text in (("tail", tail), ("head", head)):
        if not text:
            continue
        m = CLOSE_TITLE.search(text)
        if m:
            s = m.group(1)
            s = re.sub(r"^[།\s]+", "", s)
            s = re.sub(r"[།\s\u0F0B]+$", "", s)
            if len(s) >= 8:
                return s, region, text.rfind(s)
    if title:
        return title, "head", head.find(title)
    return None, None, None


def tidy_title(span):
    span = re.sub(r"^[༄༅༆༈༼༽།༎\s]+", "", span)
    span = re.sub(r"[།༔\s\u0F0B]+$", "", span)
    return span.strip()


def title_from_stop(text, stop_at):
    """Text immediately before a title-stop formula, after Sanskrit / yig mgo / homage."""
    win = text[max(0, stop_at - 160): stop_at]
    bm = list(BOD_SKAD.finditer(win))
    if bm:
        win = win[bm[-1].end():]
    ym = list(re.finditer(r"[༄༅༈][^\n།]{0,12}[།\s]*", win))
    if ym:
        win = win[ym[-1].end():]
    win = re.sub(r"^(?:[\s།༔༄༅]+)", "", win)
    hm = HOMAGE_START.match(win)
    if hm:
        nl = win.find("\n", hm.end())
        win = win[nl + 1:] if 0 <= nl < 80 else win[hm.end():]
    win = re.split(r"[།༎]\s*", win)[-1] if len(win) > 90 else win
    return tidy_title(win)


TITLE_WEIGHTS_FILE = Path(__file__).with_name("ta_rule_title_weights.json")
_TITLE_W = None
_AUTHOR_W = None


def title_features(text, start, span, region, kind, stop_i, n_stop):
    """Readable features of one title candidate. Weights from ta_rule_fit.py title."""
    before = text[max(0, start - 40): start]
    after = text[start + len(span): start + len(span) + 28]
    f = {"bias": 1.0, "region_tail": float(region == "tail"),
         "kind_formula": float(kind == "formula"), "kind_close": float(kind == "close"),
         "kind_open": float(kind == "open"), "kind_shad": float(kind == "shad")}
    f["pos"] = start / max(1, len(text))
    f["early"] = float(start <= 40)
    f["in_first_400"] = float(start < 400)
    f["after_bod"] = float(bool(BOD_SKAD.search(before)))
    f["yig_before"] = float(bool(re.search(r"[༄༅༈]", before)))
    f["zhes_after"] = float(bool(re.search(r"(?:ཞེས|ཅེས)་བྱ་བ", after)))
    f["bzhugs_after"] = float(bool(re.search(r"བཞུགས", after)))
    f["adi_after"] = float("འདི་ནི" in after)
    f["first_stop"] = float(kind == "formula" and n_stop == 0)
    f["homage"] = float(bool(HOMAGE_START.match(span)))
    f["sanskrit"] = float(sanskrit_like(span))
    f["formula_in"] = float(bool(TITLE_STOP.search(span)))
    n = len(span)
    f["len_short"] = float(n < 12)
    f["len_mid"] = float(12 <= n <= 80)
    f["len_long"] = float(n > 80)
    f["has_newline"] = float("\n" in span)
    f["has_latin"] = float(bool(re.search(r"[A-Za-z]", span)))
    f["has_mdzad"] = float("མཛད" in span)
    return f


def title_candidates(head, tail):
    """All (features, start, span, region) title guesses in head and tail."""
    out, seen = [], set()

    def add(text, region, span, start, kind, stop_i=0, n_stop=0):
        span = tidy_title(span)
        if not span or len(span) < 3 or len(span) > 180:
            return
        if start < 0 or text[start: start + len(span)] != span:
            start = text.find(span) if region == "head" else text.find(span)  # title: first match
            if start < 0:
                return
        key = (region, start, span)
        if key in seen:
            return
        seen.add(key)
        out.append((title_features(text, start, span, region, kind, stop_i, n_stop),
                    start, span, region))

    for region, text in (("head", head), ("tail", tail)):
        if not text:
            continue
        # formula-closed headings (the usual title line)
        stops = list(TITLE_STOP.finditer(text[:1800] if region == "head" else text))
        for k, m in enumerate(stops[:12]):
            span = title_from_stop(text, m.start())
            i = text.rfind(span, max(0, m.start() - 180), m.start()) if span else -1
            if i < 0 and span:
                i = text.find(span)
            if span and i >= 0:
                add(text, region, span, i, "formula", m.start(), k)
            if "འདི་ནི" in text[m.end(): m.end() + 12]:
                add(text, region, span, i, "close", m.start(), k)
        for m in CLOSE_TITLE.finditer(text):
            add(text, region, m.group(1), m.start(1), "close")
        # opening cuts with no formula: first clauses after skip_opening (head only)
        if region != "head":
            continue
        start = skip_opening(text)
        chunk = text[start: start + 400]
        t0 = cut_title(chunk)
        if t0:
            add(text, region, t0, text.find(t0, max(0, start - 5)), "open")
        # first few shad clauses (wrong-line ranker fuel)
        pos = start
        for _ in range(4):
            sh = re.search(r"[།༎]", text[pos + 4: pos + 120])
            end = pos + 4 + sh.start() if sh else min(len(text), pos + 80)
            add(text, region, text[pos:end], pos, "shad")
            if not sh:
                break
            pos = pos + 4 + sh.end()
            if pos - start > 300:
                break
    return out


def find_title(head, tail):
    W = load_title_weights()
    if not W:
        return find_title_v0(head, tail)
    best = None
    for f, i, span, region in title_candidates(head, tail):
        score = sum(W.get(k, 0.0) * v for k, v in f.items())
        better = best is None or score > best[0]
        if not better and score == best[0]:
            better = (region == "head" and best[3] != "head") or (region == best[3] and i < best[1])
        if better:
            best = (score, i, span, region)
    if not best:
        return find_title_v0(head, tail)
    return best[2], best[3], best[1]


def strip_agentive(name):
    name = HONOR.sub("", name)
    name = AGENT_WORD.sub("", name)
    name = VOWEL_S.sub(r"\1", name)
    name = re.sub(r"[།༔\s\u0F0B]+$", "", name)
    return name.strip()


def take_name(left):
    """Clause before the agentive is already the name (+ optional role).

    Start at a role if one is in this clause. Otherwise keep the clause when
    it is short (a name), or the last 32 characters when it still has junk
    before the name. Do not split on Unicode stacks: གྲགས is one word.
    """
    left = re.split(r"[།༎]", left)[-1]
    left = HONOR.sub("", left).strip()
    if len(left) < 3:
        return ""
    for role in ROLES:
        i = left.rfind(role)
        if i >= 0 and len(left) - i <= 80:
            return strip_agentive(left[i:])
    if len(left) <= 36:
        return strip_agentive(left)
    return strip_agentive(left[-32:])


def cut_at_agentive(left):
    """Drop the last agentive and anything after it (the place phrase).

    གྱིས is one word; do not treat the ིས inside it as a glued ས.
    Glued ས is `vowel + ས` (`རྡོ་རྗེས་`, `པོས་`, `མོས་`).
    """
    left = re.sub(r"[\s\u0F0B]+$", "", left)
    window = left[-120:]
    base = len(left) - len(window)
    cut = None
    for m in re.finditer(r"ཀྱིས|གྱིས|གིས|ཡིས", window):
        cut = base + m.start()
    for m in re.finditer(r"[ཱིེོུ]ས(?:་)?", window):
        a = base + m.start()
        ctx = left[max(0, a - 3): a + 3]
        if any(w in ctx for w in ("ཀྱིས", "གྱིས", "གིས", "ཡིས")):
            continue
        cut = a + 1  # keep the vowel, drop ས
    if cut is None:
        return left
    return left[:cut]


def name_before(text, verb_at):
    """Name governed by this verb: [role] NAME. Agentive and place stay out.

    Line:  [role] NAME + ས/གྱིས + (place) + མཛད་པ/སྦྱར་བ/…
    Place can be `དཔལ་ས་སྐྱའི་གཙུག་ལག་ཁང་དུ`. Name is just before the agentive.
    """
    left = text[max(0, verb_at - 160): verb_at]
    if LOTSAWA.search(left[-60:]):
        return ""
    name = take_name(cut_at_agentive(left))
    name = re.sub(r"^[།\s\u0F0B]+", "", name)
    if not name or len(name) < 3 or LOTSAWA.search(name):
        return ""
    return name


def author_candidates(text):
    """(priority, start, name, verb). Lower priority wins; last hit among equals."""
    out = []
    has_mdzad = bool(VERB_FORMS[0][1].search(text))
    for pri, (verb, cre) in enumerate(VERB_FORMS):
        if verb == "བསྒྱུར" and has_mdzad:
            continue
        for m in cre.finditer(text):
            ctx = text[max(0, m.start() - 80): m.start() + 20]
            if LOTSAWA.search(ctx) and verb == "བསྒྱུར":
                continue
            name = name_before(text, m.start())
            if not name or name not in text:
                continue
            i = text.rfind(name, 0, m.start())
            if i < 0:
                continue
            out.append((pri, i, name, verb))
    return out


# --- author v2: anchor on the agentive, score the candidates -------------------
# Train check (4,000 works): the gold author is followed by a glued ས in 42%, by
# ཀྱིས/གྱིས/གིས/ཡིས in 24%. A colophon verb sits within 160 characters in only 64%.
# So the agentive is the anchor and the verb adds to the score; it is not required.
_AG = r"(ཀྱིས|གྱིས|གིས|ཡིས|ནས)(?=[་།\s]|$)|(?<=[^\s།་ས])ས(?=[་།\s]|$)"
_ALT = (r"|(?P<gen>(?:འི|ཀྱི|གྱི|གི|ཡི)(?=་?(?:གསུང|ཞལ་སྔ|ཞལ་ནས|བཞེད|དགོངས|གདམས|མཛད་པ|གཏེར|མིང་གིས|མིང་ཅན|སྡེས|སྨྲས)))"
        r"|(?P<named>(?<=[^\s།་])(?=་?(?:ཞེས|ཅེས)་བྱ་བ))")
AGENT_RE = re.compile(_AG + (_ALT if os.environ.get("GEN", "0") == "1" else r"|(?P<gen>(?!x)x)|(?P<named>(?!x)x)"))
VERB_W = (  # (regex, weight): how strongly the verb says 'this is the maker of the text'
    (re.compile(r"མཛད"), 3.0), (re.compile(r"སྦྱར"), 2.6), (re.compile(r"བྲིས"), 2.6),
    (re.compile(r"བཀོད|བརྩམས|རྩོམ|བསྒྲིགས|སྒྲིག"), 2.0), (re.compile(r"བྱས|ཞུས|གསུངས"), 1.2),
)
CLAUSE_STOP = {"ནི", "ངོར", "ཡང", "འདི", "ཞེས", "ཅེས", "ནས", "ཏེ", "དུ", "ལ", "འོ"}
CLAUSE_BREAK = re.compile(r"[།༎༔]")
NAME_SYLLABLES = int(os.environ.get("NAME_SYL", "6"))


PCUT = os.environ.get("PCUT", "0") == "1"
PARTICLE_END = re.compile(r"(?:འི|ཀྱི|གྱི|གི|ཡི|ལས|སུ|དུ|ཏུ|ཕྱིར|ནི)$")


def name_left_span(text, end):
    """[role] NAME: walk left over syllables from the agentive to a clause break.
    Returns (start, end) inside text, so line breaks in the name are kept."""
    base = max(0, end - 120)
    seg = text[base: end]
    cut0 = max((m.end() for m in CLAUSE_BREAK.finditer(seg)), default=0)
    toks = [(base + cut0 + m.start(), base + cut0 + m.end(), m.group())
            for m in re.finditer(r"[^\s\u0F0B]+\u0F0B?", seg[cut0:])]
    keep = []
    for t in reversed(toks):
        if t[2].strip("\u0F0B") in CLAUSE_STOP:
            break
        keep.append(t)
        if len(keep) >= 9:
            break
    keep.reverse()
    if not keep:
        return None
    first = None
    for k, t in enumerate(keep):
        pass
    name = text[keep[0][0]: keep[-1][1]]
    cut = None
    for role in ROLES:
        i = name.rfind(role)
        if i >= 0 and (cut is None or i < cut):
            cut = i
    if cut is not None:
        a = keep[0][0] + cut
    else:
        use = keep[-NAME_SYLLABLES:]
        for k in range(len(use) - 2, -1, -1):  # name starts after a genitive/terminative particle
            if PCUT and PARTICLE_END.search(use[k][2].strip("\u0F0B")):
                use = use[k + 1:]
                break
        a = use[0][0]
    b = keep[-1][1]
    return a, b


def name_left(text, end):
    sp = name_left_span(text, end)
    return text[sp[0]: sp[1]].strip("\u0F0B \n") if sp else ""


WEIGHTS_FILE = Path(__file__).with_name("ta_rule_weights.json")


def before_cue(text, name, m):
    j = max(0, m.start() - len(name))
    return text[max(0, j - 16): j]


def cand_features(text, m, name, region, whole):
    """Readable features of one author candidate. The weights come from ta_rule_fit.py."""
    right = text[m.end(): m.end() + 160]
    f = {"bias": 1.0, "glued": float(m.group(1) is None and m.group("gen") is None and m.group("named") is None), "gen": float(m.group("gen") is not None), "named": float(m.group("named") is not None), "nas": float(m.group(1) == "ནས"),
         "region_tail": float(region == "tail"), "whole": float(bool(whole))}
    near = 0.0
    for key, vre in (("mdzad", re.compile(r"མཛད")), ("sbyar", re.compile(r"སྦྱར")),
                     ("bris", re.compile(r"བྲིས")), ("bkod", re.compile(r"བཀོད|བརྩམས|རྩོམ|བསྒྲིགས|སྒྲིག")),
                     ("other", re.compile(r"བྱས|ཞུས|གསུངས|བསྒྱུར|གསུང|བཞེད|དགོངས")),
                     ("say", re.compile(r"སྨྲས|བགྱིས|བརྗོད|གསུངས")),
                     ("misc", re.compile(r"ཕབ|བཏབ|བཏོན|ཕུལ|ཕྱུང|གཏེར་ནས|སྤེལ|བསྡུས"))):
        vm = vre.search(right)
        f["v_" + key] = (1.0 - vm.start() / 160.0) if vm else 0.0
        if vm and vm.start() < 12:
            near = 1.0
    f["verb_near"] = near
    f["end_after"] = float(bool(re.match(r"[་\s]*(?:སོ|པའོ|བའོ|དགེའོ|ལོ|པ|བ)(?=[་།༎༔\s]|$)", right)))
    nxt = text[m.end(): m.end() + 16]
    f["alt_gsung"] = float(bool(re.match(r"་?(?:གསུང|ཞལ|བཞེད|དགོངས|གདམས)", nxt)) and m.group("gen") is not None)
    f["alt_ming"] = float(bool(re.match(r"་?མིང་", nxt)) and m.group("gen") is not None)
    f["alt_dge"] = float(bool(re.match(r"་?(?:མཛད་པ|སྡེས|གཏེར|སྨྲས)", nxt)) and m.group("gen") is not None)
    # nearest closing verb: how many other agentives stand between this one and the verb
    any_v = re.search(r"མཛད|སྦྱར|བྲིས|བཀོད|བརྩམས|རྩོམ|བསྒྲིགས|སྒྲིག|བྱས|ཞུས|གསུངས|བསྒྱུར", right)
    if any_v:
        between = len(AGENT_RE.findall(right[:any_v.start()]))
        f["agent_between_0"] = float(between == 0)
        f["agent_between_2p"] = float(between >= 2)
        f["verb_close"] = float(any_v.start() < 60)
    else:
        f["agent_between_0"] = f["agent_between_2p"] = f["verb_close"] = 0.0
    # glued ས after a letter that can end a syllable by itself (ལེགས, ནམས, དབངས): often part of a name
    pre = text[m.start() - 1: m.start()] if m.group(1) is None and m.start() > 0 else ""
    f["glued_native_cluster"] = float(pre in ("ག", "ང", "བ", "མ"))
    f["cue_wide"] = float(bool(re.search(r"(?:ཞེས་པ་|འདི་ཡང་|འདི་ནི་|པའི་ངོར་|ཅེས་)\s*$", before_cue(text, name, m))))
    f["name_chars"] = min(len(name), 60) / 60.0
    f["role_in"] = float(any(r in name for r in ROLES))
    f["role_start"] = float(any(name.startswith(r) for r in ROLES))
    syl = len([x for x in name.split("\u0F0B") if x.strip()])
    f["syl_1"], f["syl_2"], f["syl_3_5"], f["syl_6p"] = float(syl <= 1), float(syl == 2), float(3 <= syl <= 5), float(syl >= 6)
    f["pos_end"] = m.start() / max(1, len(text))
    before = text[max(0, m.start() - len(name) - 16): max(0, m.start() - len(name))]
    f["cue_before"] = float(bool(re.search(r"ཡི་གེ་པ་ནི་\s*$|ངོར་\s*$|བདག་\s*$", before)))
    f["has_digit"] = float(bool(re.search(r"[0-9\u0F20-\u0F33]", name)))
    f["has_newline"] = float("\n" in name)
    f["sentence_start"] = float(bool(re.search(r"[།༎]\s*$", before)) or before.strip() == "")
    return f


def load_weights():
    global _AUTHOR_W
    if _AUTHOR_W is None:
        _AUTHOR_W = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8")) if WEIGHTS_FILE.exists() else False
    return _AUTHOR_W or None


def load_title_weights():
    global _TITLE_W
    if _TITLE_W is None:
        _TITLE_W = json.loads(TITLE_WEIGHTS_FILE.read_text(encoding="utf-8")) if TITLE_WEIGHTS_FILE.exists() else False
    return _TITLE_W or None


def author_candidates_v2(text, region, whole):
    """All (features, start, name) candidates in one block."""
    out = []
    for m in AGENT_RE.finditer(text):
        sp = name_left_span(text, m.start())
        if not sp:
            continue
        raw = text[sp[0]: sp[1]]
        name = raw.strip("\u0F0B \n")
        if len(name) < 3 or LOTSAWA.search(name):
            continue
        i = sp[0] + (len(raw) - len(raw.lstrip("\u0F0B \n")))
        out.append((cand_features(text, m, name, region, whole), i, name))
    return out


def author_in(text, region="tail", whole=False):
    """Best (score, start, name) in one block."""
    W = load_weights()
    best = None
    for f, i, name in author_candidates_v2(text, region, whole):
        score = sum(W.get(k, 0.0) * v for k, v in f.items()) if W else f["v_mdzad"] + f["v_sbyar"] + f["v_bris"]
        if best is None or score > best[0] or (score == best[0] and i > best[1]):
            best = (score, i, name)
    return best


def find_author(head, tail, whole):
    m = OPEN_AUTHOR.search(head[:500])
    if m:
        name = strip_agentive(m.group(1))
        if name and len(name) >= 3 and not LOTSAWA.search(name):
            i = head.find(name)
            if i >= 0:
                return name, "head", i
    blocks = [("head", head)] + ([("tail", tail)] if tail else [])
    best = None
    for region, text in blocks:
        hit = author_in(text, region, whole)
        if hit and (best is None or hit[0] > best[0]):
            best = (hit[0], hit[1], hit[2], region)
    if not best:
        return None, None, None
    return best[2], best[3], best[1]


def predict(r):
    head, tail = region_text(r)
    t, tr, ts = find_title(head, tail)
    a, ar, as_ = find_author(head, tail, r.whole_work)
    return {
        "title": t, "title_region": tr, "title_start": ts,
        "author": a, "author_region": ar, "author_start": as_,
    }


def repair_span(pred, head, tail, region=None, last=False):
    """The model's span is not an exact substring (it changed a space or a newline). Find where it really is when spaces and
    newlines are ignored, and return (exact text, region) from the page, or None. Looks in the stated region first, then the other.
    last=True for the author (last match), False for the title (first match)."""
    if not isinstance(pred, str) or not pred.strip():
        return None
    p = "".join(c for c in pred if not c.isspace())
    seen = []
    for rg in (region, "head", "tail"):
        if rg not in ("head", "tail") or rg in seen:
            continue
        seen.append(rg)
        text = head if rg == "head" else tail
        if not isinstance(text, str) or not text:
            continue
        idx = [i for i, c in enumerate(text) if not c.isspace()]
        t = "".join(text[i] for i in idx)
        j = t.rfind(p) if last else t.find(p)
        if j >= 0:
            return text[idx[j]: idx[j + len(p) - 1] + 1], rg
    return None


def gold_span(r, which):
    s = int(getattr(r, which + "_start"))
    return getattr(r, which + "_region"), (s, s + len(getattr(r, which)))


def score_one(r, pred, which):
    g_reg, g = gold_span(r, which)
    p = pred[which]
    if not p:
        return "null", 0.0, ""
    reg = pred[which + "_region"]
    text = r.text_head if reg == "head" else r.text_tail
    if reg not in ("head", "tail") or not isinstance(text, str) or p not in text:
        return "not_substring", 0.0, p
    i = pred[which + "_start"]
    if i is None or text[i: i + len(p)] != p:
        i = text.find(p) if which == "title" else text.rfind(p)
    if reg != g_reg:
        return "wrong_region", 0.0, p
    v = iou((i, i + len(p)), g)
    return ("ok" if v >= 0.5 else "low_iou"), v, p


def load(split, n=0, seed=0):
    if split not in ("train", "validation") and not (split == "test" and os.environ.get("ALLOW_TEST") == "1"):
        sys.exit("only train or validation (set ALLOW_TEST=1 for test scores, no miss dump)")
    df = pd.read_parquet(PQ / f"{split}.parquet")
    df = df[df.source != "v3_auto"].dropna(subset=["title", "author"])
    if n:
        df = df.sample(min(n, len(df)), random_state=seed)
    return df


def run_score(split, n=0, seed=0):
    rows = list(load(split, n, seed).itertuples())
    acc = {w: [0, 0, 0] for w in ("title", "author")}
    kinds = {w: Counter() for w in acc}
    fails = []
    for r in rows:
        pred = predict(r)
        for w in acc:
            k, v, p = score_one(r, pred, w)
            kinds[w][k] += 1
            if k == "ok":
                acc[w][0] += 1
            else:
                acc[w][2] += 1
                acc[w][1] += 0 if k in ("null",) else 1
                fails.append((w, k, v, r, p, pred))
    res = {"split": split, "n": len(rows), "seed": seed, "parquet": str(PQ)}
    for w, (tp, fp, fn) in acc.items():
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        res[w] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(F, 3),
                  "outcomes": dict(kinds[w])}
    res["mean_f1"] = round((res["title"]["f1"] + res["author"]["f1"]) / 2, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"rule_{split}_{n or 'all'}"
    dest = OUT / tag
    dest.mkdir(exist_ok=True)
    (dest / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if split != "test":
        md = [f"# Rule misses: {split} ({len(rows)} works), mean F1 {res['mean_f1']}\n"]
        for w, k, v, r, p, pred in sorted(fails, key=lambda x: (x[0], x[1]))[:400]:
            reg = getattr(r, w + "_region")
            text = r.text_head if reg == "head" else r.text_tail
            s, e = int(getattr(r, w + "_start")), int(getattr(r, w + "_end"))
            md.append(
                f"## {w} | {k} | IoU {v:.2f} | {r.work_id} | {r.source}\n"
                f"- gold ({reg}): `{getattr(r, w)!r}`\n"
                f"- rule: `{p!r}`\n"
                f"- context: `{text[max(0, s - 60):s]}`**[{text[s:e]}]**`{text[e:e + 40]}`\n"
            )
        (dest / "failures.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", dest)
    return res


def show(work_id):
    for split in ("train", "validation"):
        df = pd.read_parquet(PQ / f"{split}.parquet")
        hit = df[df.work_id == work_id]
        if len(hit):
            r = next(hit.itertuples())
            pred = predict(r)
            print("split", split)
            print("PRED", json.dumps(pred, ensure_ascii=False, indent=1))
            print("GOLD title", r.title_region, r.title)
            print("GOLD author", r.author_region, r.author)
            return
    sys.exit("work_id not in train/validation notrans parquet")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    if mode == "show":
        show(sys.argv[2])
    elif mode == "score":
        split = sys.argv[2] if len(sys.argv) > 2 else "validation"
        n = 0
        seed = 0
        args = sys.argv[3:]
        if "--n" in args:
            n = int(args[args.index("--n") + 1])
        if "--seed" in args:
            seed = int(args[args.index("--seed") + 1])
        run_score(split, n, seed)
    else:
        sys.exit("score | show")

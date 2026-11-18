"""Fill missing title/author spans by finding the title/author text inside its segment.

Same rule the Cataloger uses (findPhraseDocSpan / segment_title_author_auto.py):
  title  -> FIRST occurrence of the title text in the segment
  author -> LAST occurrence of the author text in the segment (colophon at the end)
Spans are DOCUMENT offsets into content, like the annotator spans.
Match: exact text first; if not found, retry with trailing shad/tsek/space stripped
(typed titles often end in "།" where the text has "་" or a line break).

Local only - reads json_files_all_approved_spans_*/ , never touches the DB.

  python title_author/src/fill_ta_spans.py eval  SRC [N]   # test against annotator spans on N random docs
  python title_author/src/fill_ta_spans.py build SRC OUT   # write docs whose segments have all four spans,
                                            # each pair marked title_span_source/author_span_source
                                            # = "annotator" | "auto"
"""

import json
import os
import random
import sys
from collections import Counter

STRIP = " \t\r\n།་༎"


def find(seg_text, seg_start, phrase, last):
    """Return (start, end, how) in document offsets, None if not found, "multi" if ambiguous.

    Only a text that occurs exactly ONCE in the segment is accepted: tested against
    annotator spans, unique matches land on the right place ~97% of the time,
    repeated ones only ~70%.
    """
    for how, p in (("exact", phrase.strip()), ("trimmed", phrase.strip(STRIP))):
        if not p:
            continue
        idx = seg_text.rfind(p) if last else seg_text.find(p)
        if idx == -1:
            continue
        if seg_text.count(p) > 1:
            return "multi"
        return seg_start + idx, seg_start + idx + len(p), how
    return None


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate(src, n):
    files = sorted(os.listdir(src))
    random.seed(0)
    sample = random.sample(files, min(n, len(files)))
    st = Counter()
    for p in sample:
        d = load(os.path.join(src, p))
        c = d["content"]
        for s in d["segments"]:
            seg = c[s["span_start"]:s["span_end"]]
            for kind, last in (("title", False), ("author", True)):
                if f"{kind}_span_start" not in s or not s[kind].strip():
                    continue
                st[f"{kind}_total"] += 1
                hit = find(seg, s["span_start"], s[kind], last)
                if hit == "multi":
                    st[f"{kind}_repeated_skipped"] += 1
                    continue
                if not hit:
                    st[f"{kind}_not_found"] += 1
                    continue
                a, b, how = hit
                gs, ge = s[f"{kind}_span_start"], s[f"{kind}_span_end"]
                if (a, b) == (gs, ge):
                    st[f"{kind}_same"] += 1
                elif a < ge and gs < b:
                    st[f"{kind}_overlap"] += 1   # same place, slightly different edges
                else:
                    st[f"{kind}_elsewhere"] += 1  # found the text, but at a different place
    print(f"eval on {len(sample)} docs (segments that already have annotator spans):")
    for kind in ("title", "author"):
        t = st[f"{kind}_total"] or 1
        print(f"  {kind}: {st[f'{kind}_total']} spans | same {st[f'{kind}_same']} ({100*st[f'{kind}_same']/t:.1f}%)"
              f" | overlapping {st[f'{kind}_overlap']} ({100*st[f'{kind}_overlap']/t:.1f}%)"
              f" | elsewhere {st[f'{kind}_elsewhere']} ({100*st[f'{kind}_elsewhere']/t:.1f}%)"
              f" | not found {st[f'{kind}_not_found']} ({100*st[f'{kind}_not_found']/t:.1f}%)")


def build(src, out):
    os.makedirs(out, exist_ok=True)
    st = Counter()
    for p in sorted(os.listdir(src)):
        d = load(os.path.join(src, p))
        c = d["content"]
        keep = []
        for s in d["segments"]:
            seg = c[s["span_start"]:s["span_end"]]
            for kind, last in (("title", False), ("author", True)):
                if not s[kind].strip():
                    continue
                st[f"{kind}_with_text"] += 1
                if f"{kind}_span_start" in s:
                    s[f"{kind}_span_source"] = "annotator"
                    st[f"{kind}_annotator"] += 1
                    continue
                hit = find(seg, s["span_start"], s[kind], last)
                if hit == "multi":
                    st[f"{kind}_repeated_skipped"] += 1
                elif hit:
                    s[f"{kind}_span_start"], s[f"{kind}_span_end"], how = hit
                    s[f"{kind}_span_source"] = "auto"
                    st[f"{kind}_auto_{how}"] += 1
                else:
                    st[f"{kind}_not_found"] += 1
            # An author name is often part of the title (".. སྙིང་པོའི་མཛད་པ"); an auto span sitting
            # inside the other span is that, not the real author/title - drop the auto one.
            if ("title_span_start" in s and "author_span_start" in s
                    and s["title_span_start"] < s["author_span_end"]
                    and s["author_span_start"] < s["title_span_end"]):
                drop = "author" if s["author_span_source"] == "auto" else "title"
                if s[f"{drop}_span_source"] == "auto":
                    for key in ("span_start", "span_end", "span_source"):
                        del s[f"{drop}_{key}"]
                    st[f"{drop}_auto_overlap_dropped"] += 1
            if "title_span_start" in s and "author_span_start" in s:
                keep.append(s)
                st["segs_both_" + ("annotator" if s["title_span_source"] == s["author_span_source"] == "annotator"
                                   else "with_auto")] += 1
        if keep:
            with open(os.path.join(out, p), "w", encoding="utf-8") as fh:
                json.dump({"filename": d["filename"], "content": c, "segments": keep},
                          fh, ensure_ascii=False, indent=2)
            st["docs"] += 1
    print(json.dumps(st, indent=1, sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
    else:
        build(sys.argv[2], sys.argv[3])

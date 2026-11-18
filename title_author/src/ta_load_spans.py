"""Load gold (annotator) title/author span JSONs -> one row per TEXT segment.

Offsets are in `content`. Only the provided offsets are used (no string matching).

Restored on 2026-11-18 from the command that first wrote it (2026-10-29); it had never been
committed. The only change is the TA_GOLD_DIR environment variable for the folder. The folder
itself (one JSON per volume, named like IE..._google_books.json, with `content` and `segments`)
is not in this repo.
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path

GOLD = Path(os.environ.get("TA_GOLD_DIR") or Path.home() / "Downloads/json_files_title_author_spans_2026-10-29")   # folder of annotator JSONs; set TA_GOLD_DIR to use another
SOURCES = ["tei", "google_books", "google_vision", "paddleocr_v1", "paddleocr_v2", "ocr_qwen_v1", "ocrv1-ws-ldv1"]
TRAIN_SOURCES = {"tei", "google_books", "google_vision"}


def parse_name(fn):
    stem = fn[:-5]
    src = next((s for s in SOURCES if stem.endswith("_" + s)), None)
    m = re.match(r"(IE[0-9A-Z]+|W[0-9A-Z]+)", stem)
    return (m.group(1) if m else None), src


def load(folder=GOLD):
    rows, skipped, bad, files = [], Counter(), [], 0
    for p in sorted(Path(folder).glob("*.json")):
        files += 1
        d = json.load(open(p, encoding="utf-8"))
        cid, src = parse_name(p.name)
        c = d["content"]
        for i, s in enumerate(d["segments"]):
            if s.get("label") != "TEXT":
                skipped[s.get("label")] += 1
                continue
            r = dict(file=p.name, cid=cid, source=src, seg=i, content_len=len(c),
                     span_start=s["span_start"], span_end=s["span_end"])
            for k in ("title", "author"):
                a, b = s.get(f"{k}_span_start"), s.get(f"{k}_span_end")
                txt = s.get(k) or ""
                r[k + "_text"] = txt
                if a is None or b is None or a == b:
                    r[k + "_span"] = None
                    continue
                r[k + "_span"] = (a, b)
                if not (s["span_start"] <= a < b <= s["span_end"]):
                    bad.append((p.name, i, k, a, b, s["span_start"], s["span_end"]))
            rows.append(r)
    return rows, skipped, bad, files


if __name__ == "__main__":
    rows, skipped, bad, files = load()
    print("files", files, "TEXT works", len(rows), "non-TEXT skipped", dict(skipped), "out-of-segment spans", len(bad))
    for b in bad[:10]: print(b)
    t = {s: Counter() for s in SOURCES + [None]}
    vols = {s: set() for s in SOURCES + [None]}
    nf = Counter()
    for r in rows:
        t[r["source"]]["works"] += 1
        t[r["source"]]["title"] += r["title_span"] is not None
        t[r["source"]]["author"] += r["author_span"] is not None
        t[r["source"]]["both"] += r["title_span"] is not None and r["author_span"] is not None
        vols[r["source"]].add(r["file"])
    for s in SOURCES + [None]:
        if t[s]["works"]: print(f"{str(s):15} vols={len(vols[s]):5} {dict(t[s])}")
    print("cid None:", sum(1 for r in rows if r["cid"] is None))

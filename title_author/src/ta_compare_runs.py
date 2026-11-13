#!/usr/bin/env python3
"""Compare two Gemini runs on the works both of them answered (no network, no cost).

    python title_author/src/ta_compare_runs.py SPLIT "RUN_A_GLOB" "RUN_B_GLOB" [ROWS=a:b]

RUN_x_GLOB is a folder name (or a glob such as "gepa_v2_validation_0_rows*") inside title_author/data/gemini_runs/.
Each folder needs partial_raw.json ({"work_ids": [...], "raw": [...]}). SPLIT is validation or test (test needs ALLOW_TEST=1).
Set TA_PQ to the parquet folder, as for ta_gemini_eval.py.
Prints title/author/mean share-correct (IoU >= 0.5, same region) for both runs on the shared works,
and a paired bootstrap 95% range for the mean difference (A minus B)."""
import os, sys, json, glob, random
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE.parents[1] / "title_author/data/gemini_runs"
_src = (HERE / "ta_gemini_eval.py").read_text(encoding="utf-8").split("mode = sys.argv[1]")[0]
_g = {"__name__": "ta_eval", "__file__": str(HERE / "ta_gemini_eval.py")}
exec(_src, _g)
load, parse, score = _g["load"], _g["parse"], _g["score"]


def answers(pattern):
    out = {}
    for d in sorted(glob.glob(str(RUNS / pattern))):
        f = Path(d) / "partial_raw.json"
        if f.exists():
            p = json.loads(f.read_text(encoding="utf-8")); out.update(zip(p["work_ids"], p["raw"]))
    return out


def answered(t):
    return not str(t).startswith(("HTTP_ERROR", "ERROR"))


def main(split, pa, pb):
    rows = {r.work_id: r for r in load(split, 0, 0).itertuples()}
    A, B = answers(pa), answers(pb)
    ids = [w for w in A if w in B and w in rows and answered(A[w]) and answered(B[w])]
    if not ids: sys.exit("no works answered by both runs")
    print(f"works answered by both: {len(ids)}  (A answered {sum(map(answered, A.values()))} of {len(A)}, B {sum(map(answered, B.values()))} of {len(B)})")
    per = {}
    for name, M in (("A", A), ("B", B)):
        ok = {w: [score(rows[i], parse(M[i]), w)[0] == "ok" for i in ids] for w in ("title", "author")}
        t, a = sum(ok["title"]) / len(ids), sum(ok["author"]) / len(ids)
        print(f"{name}: title {t:.3f}  author {a:.3f}  mean {(t + a) / 2:.3f}")
        per[name] = [(ok["title"][k] + ok["author"][k]) / 2 for k in range(len(ids))]
    diff = [x - y for x, y in zip(per["A"], per["B"])]
    random.seed(0)
    bs = sorted(sum(random.choice(diff) for _ in diff) / len(diff) for _ in range(2000))
    print(f"mean difference A - B: {sum(diff) / len(diff):+.4f}   95% range [{bs[50]:+.4f}, {bs[1950]:+.4f}]  (a range that includes 0 means a tie)")


if __name__ == "__main__":
    if len(sys.argv) < 4: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

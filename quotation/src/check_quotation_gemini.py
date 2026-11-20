#!/usr/bin/env python3
"""Fuzzy-locate Gemini quote dumps in a book and score them against gold.

Reads the human ``[Quote N]`` dump, JSON ``{"spans":[{"text":...}]}``, or the
anchor format ``{"spans":[{"head":...,"tail":...}]}`` where only the first and
last ~20 characters of each span are emitted. Never trusts model offsets — finds
the emitted text in the book (exact, then fuzzy), then reports IoU@0.5 vs the
gold Quotation layer.

Anchors that short are not unique in a whole book, so quotes are located with a
cursor that only moves forward, in the order the model emitted them. Pass
``--unordered`` to search the whole book for each quote independently.

Examples:
  python quotation/src/check_quotation_gemini.py --book P000152 --quotes dump.txt
  python quotation/src/check_quotation_gemini.py --book P000152 --quotes anchors.json --out hits.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quotation_lib.audit_utils import load_book  # noqa: E402
from quotation_lib.dataset_utils import DEFAULT_REPOS_DIR, book_spans  # noqa: E402
from quotation_lib.gemini_locate import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    ParsedQuote,
    canonical_label,
    gold_tuples,
    locate_parsed_quote,
    parse_quote_dump,
)
from quotation_lib.span_metrics import inclusive_iou, match_iou  # noqa: E402

PREVIEW = 72


def preview(text: str, start: int, end: int) -> str:
    snippet = text[start : end + 1].replace("\n", "⏎")
    if len(snippet) <= PREVIEW:
        return snippet
    return snippet[: PREVIEW - 1] + "…"


def hint(quote: ParsedQuote) -> str:
    """Shortest human handle on a quote: frame, else head anchor, else text."""
    for field in (quote.frame, quote.start_text, quote.text):
        if field.strip():
            return field.strip()[:40]
    return "?"


def load_book_row(repos: Path, book_id: str, layer: str) -> tuple[str, list[dict]]:
    rec = load_book(repos / book_id)
    if not rec.base_path or not rec.base_path.is_file():
        raise FileNotFoundError(f"No base text for {book_id} under {repos}")
    text = rec.base_path.read_text(encoding="utf-8")
    return text, book_spans(rec, layer)


def stitch_adjacent_spans(text: str, pred: list[tuple[str, int, int]], max_gap_len: int = 15) -> list[tuple[str, int, int]]:
    if not pred:
        return []
    sorted_pred = sorted(pred, key=lambda x: x[1])
    stitched = [sorted_pred[0]]
    for current in sorted_pred[1:]:
        last_label, last_start, last_end = stitched[-1]
        curr_label, curr_start, curr_end = current
        if curr_label == last_label and curr_start > last_end:
            gap = text[last_end + 1 : curr_start]
            gap_cleaned = gap.strip().replace("།", "").replace("༎", "").replace("༑", "").replace("་", "")
            if not gap_cleaned and len(gap) <= max_gap_len:
                stitched[-1] = (last_label, last_start, curr_end)
                continue
        stitched.append(current)
    return stitched


def score_quotes(
    book_id: str,
    quotes_path: Path,
    repos: Path = DEFAULT_REPOS_DIR,
    layer: str = "Quotation",
    min_score: float = DEFAULT_MIN_SCORE,
    unordered: bool = False,
    max_len: int = 2000,
    out: Path | None = None,
    stitch: bool = True,
) -> dict:
    text, gold_spans = load_book_row(repos, book_id, layer)
    try:
        quotes = parse_quote_dump(quotes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{quotes_path}: unparseable JSON ({exc}) — truncated response?", file=sys.stderr)
        return {
            "error": "JSONDecodeError",
            "message": str(exc)
        }

    gold_label = canonical_label(layer)
    gold = gold_tuples(gold_spans, gold_label)
    pred: list[tuple[str, int, int]] = []
    rows: list[dict] = []

    mode = "unordered" if unordered else "ordered"
    print(
        f"book={book_id}  chars={len(text):,}  gold_{layer}={len(gold)}  "
        f"parsed={len(quotes)}  search={mode}"
    )
    print()

    max_span = max_len if max_len > 0 else None
    cursor = 0
    for quote in quotes:
        from_index = 0 if unordered else cursor
        hit = locate_parsed_quote(
            text,
            quote,
            min_score=min_score,
            from_index=from_index,
            max_span=max_span,
        )
        rewound = False
        if hit is None and from_index:
            # a quote out of emission order, or a bad hit earlier that ran the cursor past it
            hit = locate_parsed_quote(
                text, quote, min_score=min_score, max_span=max_span
            )
            rewound = hit is not None
        row: dict = {
            "quote_id": quote.quote_id,
            "frame": quote.frame,
            "closer": quote.closer,
            "head": quote.start_text,
            "tail": quote.end_text,
            "located": hit is not None,
        }
        if hit is None:
            row["status"] = "match_failure"
            why = ""
            if max_span is not None:
                # was it the cap, or is the anchor simply not in the book?
                loose = locate_parsed_quote(text, quote, min_score=min_score)
                if loose is not None:
                    row["status"] = "tail_beyond_max_len"
                    row["unbounded_len"] = loose.length
                    row["unbounded_start"] = loose.start
                    why = f"  tail is {loose.length} chars past the head"
            print(f"[{quote.quote_id:>4}] MATCH FAIL  {hint(quote)!r}{why}")
        else:
            if not unordered and hit.end >= cursor:
                cursor = hit.end + 1
            pred.append((gold_label, hit.start, hit.end))
            best_iou = 0.0
            best_gold = None
            for g in gold:
                value = inclusive_iou(g, (gold_label, hit.start, hit.end))
                if value > best_iou:
                    best_iou = value
                    best_gold = g
            row.update(
                {
                    "status": "located",
                    "start": hit.start,
                    "end": hit.end,
                    "len": hit.length,
                    "score": round(hit.score, 4),
                    "method": hit.method,
                    "n_hits": hit.n_hits,
                    "ambiguous": hit.n_hits > 1,
                    "rewound": rewound,
                    "overshoot": bool(max_len) and hit.length > max_len,
                    "best_gold_iou": round(best_iou, 4),
                    "preview": preview(text, hit.start, hit.end),
                }
            )
            if best_gold is not None:
                row["gold_start"] = best_gold[1]
                row["gold_end"] = best_gold[2]
                row["gold_len"] = best_gold[2] - best_gold[1] + 1
            mark = "OK " if best_iou >= 0.5 else "LOW"
            flags = "".join(
                (
                    "!" if row["overshoot"] else "",
                    "?" if row["ambiguous"] else "",
                    "<" if rewound else "",
                )
            )
            print(
                f"[{quote.quote_id:>4}] {mark} {hit.method:14s} "
                f"[{hit.start}:{hit.end}] len={hit.length:<6d} "
                f"loc={hit.score:.2f} iou={best_iou:.3f} {flags:<3s} {row['preview']}"
            )
        rows.append(row)

    # Perform optional adjacent stitching
    pred_for_scoring = stitch_adjacent_spans(text, pred) if stitch else pred
    if stitch and len(pred) != len(pred_for_scoring):
        print(f"Stitched adjacent predicted spans: {len(pred)} -> {len(pred_for_scoring)}")

    matches = match_iou(gold, pred_for_scoring, threshold=0.5)
    n_pred = len(pred_for_scoring)
    n_gold = len(gold)
    n_match = len(matches)

    if n_gold == 0 and n_pred == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = n_match / n_pred if n_pred else 0.0
        recall = n_match / n_gold if n_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    n_fail = sum(1 for r in rows if not r["located"])
    n_too_long = sum(1 for r in rows if r.get("status") == "tail_beyond_max_len")
    n_ambiguous = sum(1 for r in rows if r.get("ambiguous"))
    n_overshoot = sum(1 for r in rows if r.get("overshoot"))
    n_rewound = sum(1 for r in rows if r.get("rewound"))

    print()
    print(
        f"located {len(pred)}/{len(quotes)}  match_failures {n_fail} "
        f"(of which tail past --max-len {n_too_long})  "
        f"ambiguous {n_ambiguous}  overshoot {n_overshoot}  rewound {n_rewound}  "
        f"IoU@0.5 vs all gold {layer}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
        f"(matched {n_match} / pred {n_pred} / gold {n_gold})"
    )
    print(
        "Recall is against the whole book gold set, not only quotes in this dump. "
        "Per-quote `iou` is the best overlapping gold span. "
        "Flags: ! span over --max-len, ? anchor matched more than once, "
        "< found only by rewinding the cursor."
    )

    result_dict = {
        "book_id": book_id,
        "layer": layer,
        "n_gold": n_gold,
        "n_parsed": len(quotes),
        "n_located": len(pred),
        "n_pred": n_pred,
        "n_match": n_match,
        "n_match_failure": n_fail,
        "n_tail_beyond_max_len": n_too_long,
        "n_ambiguous": n_ambiguous,
        "n_overshoot": n_overshoot,
        "n_rewound": n_rewound,
        "search": mode,
        "iou50_precision": precision,
        "iou50_recall": recall,
        "iou50_f1": f1,
        "quotes": rows,
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {out}")

    return result_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, help="OpenPecha book id, e.g. P000152")
    parser.add_argument("--quotes", type=Path, required=True, help="Gemini dump or JSON spans")
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS_DIR)
    parser.add_argument("--layer", default="Quotation", help="Gold layer name (OpenPecha stem)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument(
        "--unordered",
        action="store_true",
        help="Search the whole book per quote instead of a forward-only cursor",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=2000,
        help="Cap head-to-tail distance for anchor input; longer spans are flagged (0 = off)",
    )
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Do not stitch adjacent predicted spans",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON of per-quote hits")
    args = parser.parse_args(argv)

    score_quotes(
        args.book,
        args.quotes,
        repos=args.repos,
        layer=args.layer,
        min_score=args.min_score,
        unordered=args.unordered,
        max_len=args.max_len,
        out=args.out,
        stitch=not args.no_stitch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

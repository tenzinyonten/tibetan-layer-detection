#!/usr/bin/env python3
"""Run Gemini anchor extract + locate over a Hub split and write metrics.

Loads book ids from HuggingFace (revision from the experiment config), extracts
whole-book JSON with the anchor prompt, then scores with IoU@0.5.

Resume: skip the API call when a dump already exists; always re-score so
locator changes stay consistent. Reuses the existing P000069 dump instead
of calling Gemini again.

Examples:
  python quotation/src/eval_quotation_gemini.py --split test
  python quotation/src/eval_quotation_gemini.py --split test --books P000069
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402

import check_quotation_gemini  # noqa: E402
import extract_quotation_gemini  # noqa: E402
from quotation_lib.dataset_utils import DEFAULT_REPOS_DIR  # noqa: E402

CONFIG_PATH = ROOT / "results" / "zeroshot_single_book" / "config.yaml"
DEFAULT_PROMPT = ROOT / "prompts" / "quotation_anchor_prompt.md"
DEFAULT_OUT_DIR = ROOT / "results" / "zeroshot_single_book" / "test"
SEED_DUMPS = {
    "P000069": ROOT / "results" / "zeroshot_single_book" / "p000069_anchor.json",
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_split_book_ids(repo: str, revision: str, split: str) -> list[str]:
    ds = load_dataset(repo, split=split, revision=revision)
    ids = []
    seen: set[str] = set()
    for row in ds:
        book_id = str(row["book_id"])
        if book_id not in seen:
            seen.add(book_id)
            ids.append(book_id)
    return ids


def seed_dump_if_needed(book_id: str, dest: Path) -> None:
    src = SEED_DUMPS.get(book_id)
    if src is None or dest.is_file() or not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"reused {src} -> {dest}", flush=True)


def prf1(n_match: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    if n_gold == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def aggregate(rows: list[dict]) -> dict:
    n_gold = sum(int(r.get("n_gold") or 0) for r in rows)
    n_parsed = sum(int(r.get("n_parsed") or 0) for r in rows)
    n_located = sum(int(r.get("n_located") or 0) for r in rows)
    n_pred = sum(int(r.get("n_pred") or 0) for r in rows)
    n_match = sum(int(r.get("n_match") or 0) for r in rows)
    n_fail = sum(int(r.get("n_match_failure") or 0) for r in rows)
    n_too_long = sum(int(r.get("n_tail_beyond_max_len") or 0) for r in rows)
    n_ambiguous = sum(int(r.get("n_ambiguous") or 0) for r in rows)
    precision, recall, f1 = prf1(n_match, n_pred, n_gold)
    return {
        "n_books": len(rows),
        "n_gold": n_gold,
        "n_parsed": n_parsed,
        "n_located": n_located,
        "n_pred": n_pred,
        "n_match": n_match,
        "n_match_failure": n_fail,
        "n_tail_beyond_max_len": n_too_long,
        "n_ambiguous": n_ambiguous,
        "locate_failure_rate": (n_fail / n_parsed) if n_parsed else 0.0,
        "ambiguous_rate": (n_ambiguous / n_located) if n_located else 0.0,
        "iou50_precision": precision,
        "iou50_recall": recall,
        "iou50_f1": f1,
    }


def write_scoreboard(metrics: dict, path: Path) -> None:
    micro = metrics["micro"]
    lines = [
        "# Gemini anchors — Hub test split",
        "",
        f"**Prompt:** `{metrics['prompt']}`",
        f"**Model:** `{metrics['model']}`, temperature {metrics['temperature']}",
        f"**Mode:** {metrics['mode']}",
        f"**Locator:** `--max-len {metrics['max_len']}`, stitch adjacent for scoring",
        f"**Dataset:** `{metrics['dataset_repo']}` revision `{metrics['dataset_revision']}` split `{metrics['split']}`",
        "",
        "Micro P/R/F1 is sum of IoU@0.5 matches / predicted (after stitch) / gold.",
        "",
        "## Micro",
        "",
        f"- books: {micro['n_books']}",
        f"- gold: {micro['n_gold']}",
        f"- parsed: {micro['n_parsed']}",
        f"- located: {micro['n_located']}",
        f"- pred (stitched): {micro['n_pred']}",
        f"- match: {micro['n_match']}",
        f"- locate-failure rate: {micro['locate_failure_rate']:.3f}",
        f"- ambiguous rate: {micro['ambiguous_rate']:.3f}",
        f"- P={micro['iou50_precision']:.3f} R={micro['iou50_recall']:.3f} F1={micro['iou50_f1']:.3f}",
        "",
        "## Per book",
        "",
        "| Book | Gold | Parsed | Located | Fail | P | R | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["books"]:
        lines.append(
            f"| {row['book_id']} | {row.get('n_gold', '')} | {row.get('n_parsed', '')} | "
            f"{row.get('n_located', '')} | {row.get('n_match_failure', '')} | "
            f"{row.get('iou50_precision', 0):.3f} | {row.get('iou50_recall', 0):.3f} | "
            f"{row.get('iou50_f1', 0):.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_book(
    book_id: str,
    *,
    out_dir: Path,
    prompt: Path,
    model: str,
    temperature: float,
    max_len: int,
    repos: Path,
    skip_existing: bool,
    stitch: bool,
) -> dict:
    dump_path = out_dir / f"{book_id.lower()}_anchors.json"
    check_path = out_dir / f"{book_id.lower()}_check.json"
    seed_dump_if_needed(book_id, dump_path)
    argv = [
        "--book",
        book_id,
        "--prompt",
        str(prompt),
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--repos",
        str(repos),
        "--out",
        str(dump_path),
    ]
    if skip_existing:
        argv.append("--skip-existing")
    rc = extract_quotation_gemini.main(argv)
    if rc != 0:
        raise SystemExit(f"extract failed for {book_id} (exit {rc})")
    if not extract_quotation_gemini.dump_is_valid(dump_path):
        raise SystemExit(f"missing or invalid dump after extract: {dump_path}")
    return check_quotation_gemini.score_quotes(
        book_id,
        dump_path,
        repos=repos,
        max_len=max_len,
        out=check_path,
        stitch=stitch,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="Hub split name (default: test)")
    parser.add_argument("--books", nargs="*", help="Restrict to these book ids")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-call Gemini even when a dump already exists",
    )
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    repo = cfg.get("dataset_repo") or "karma689/layer_detection"
    revision = cfg.get("dataset_revision") or "v1.1"
    model = args.model or (cfg.get("models") or {}).get("primary") or "gemini-3.8-flash"

    book_ids = load_split_book_ids(repo, revision, args.split)
    if args.books:
        wanted = set(args.books)
        book_ids = [b for b in book_ids if b in wanted]
        missing = wanted - set(book_ids)
        if missing:
            print(f"warning: not in {args.split}: {sorted(missing)}", file=sys.stderr)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"split={args.split}  books={len(book_ids)}  model={model}  "
        f"prompt={args.prompt.name}  out={out_dir}"
    )

    rows: list[dict] = []
    for i, book_id in enumerate(book_ids, start=1):
        print(f"\n=== [{i}/{len(book_ids)}] {book_id} ===", flush=True)
        row = eval_book(
            book_id,
            out_dir=out_dir,
            prompt=args.prompt,
            model=model,
            temperature=args.temperature,
            max_len=args.max_len,
            repos=args.repos,
            skip_existing=not args.no_skip_existing,
            stitch=not args.no_stitch,
        )
        slim = {k: v for k, v in row.items() if k != "quotes"}
        rows.append(slim)

    micro = aggregate(rows)
    payload = {
        "split": args.split,
        "dataset_repo": repo,
        "dataset_revision": revision,
        "prompt": str(args.prompt),
        "model": model,
        "temperature": args.temperature,
        "mode": "whole-book",
        "max_len": args.max_len,
        "stitch": not args.no_stitch,
        "books": rows,
        "micro": micro,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {metrics_path}")
    write_scoreboard(payload, out_dir / "test.md")
    print(f"wrote {out_dir / 'test.md'}")
    print(
        f"micro  books={micro['n_books']}  "
        f"P={micro['iou50_precision']:.3f} R={micro['iou50_recall']:.3f} "
        f"F1={micro['iou50_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

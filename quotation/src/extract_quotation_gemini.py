#!/usr/bin/env python3
"""Send a book (chunked or whole) through a Gemini prompt and write JSON spans.

Needs a Google AI Studio API key. Set one of:

  export GEMINI_API_KEY=...
  export GOOGLE_API_KEY=...

Get a key at https://aistudio.google.com/apikey — there is no unauthenticated
path. ``genai.Client()`` with no args only works if one of those env vars is
set (or you pass ``--api-key``).

The prompt files in ``quotation/prompts/`` are written for one window with
``«chunk»`` replaced. By default the whole book goes in a single call; pass
``--chunked`` to split it into overlapping windows (W/S from the experiment
config) instead.

Rate limits are handled by waiting: on 429/503/500 the script sleeps for the
delay Google asks for (or an exponential backoff) and retries the same request.

Examples:
  python quotation/src/extract_quotation_gemini.py --book P000152
  python quotation/src/extract_quotation_gemini.py --book P000152 --prompt quotation/prompts/quotation_plain_prompt.md
  python quotation/src/extract_quotation_gemini.py --book P000152 --chunked --out quotes.json
  python quotation/src/extract_quotation_gemini.py --text path/to/book.txt --chunked --max-chunks 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quotation_lib.audit_utils import load_book  # noqa: E402
from quotation_lib.dataset_utils import DEFAULT_REPOS_DIR  # noqa: E402
from quotation_lib.gemini_chunks import make_chunks  # noqa: E402

CONFIG_PATH = ROOT / "results" / "zeroshot_single_book" / "config.yaml"
DEFAULT_PROMPT = ROOT / "prompts" / "quotation_plain_prompt.md"
RULER_RE = re.compile(r"^---\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
PLACEHOLDER = "«chunk»"
RETRY_DELAY_RE = re.compile(r"(?:retryDelay'?:\s*'?|retry in )(\d+(?:\.\d+)?)s")
RETRY_STATUSES = (429, 500, 502, 503, 504)
TAIL_CHARS = 200
CONTINUE_SUFFIX = (
    "\n\nYour previous reply was cut off by the output limit. "
    "The last quote already recorded ends with:\n{tail}\n"
    "Resume from immediately after that point in the text and return only the quotes "
    'that come later. Do not repeat any earlier one. Same JSON, nothing else: '
    '{{"spans":[{{"label":"QUOTE","text":"..."}}]}}. '
    'If nothing is left, reply {{"spans": []}}.'
)
CONTINUE_ANCHOR_SUFFIX = (
    "\n\nYour previous reply was cut off by the output limit. "
    "The last quote already recorded ends with:\n{tail}\n"
    "Resume from immediately after that point in the text and return only the quotes "
    "that come later. Do not repeat any earlier one. Same JSON, nothing else: "
    '{{"spans":[{{"label":"QUOTE","frame":"...","head":"...","tail":"..."}}]}}. '
    'If nothing is left, reply {{"spans": []}}.'
)


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_api_key(explicit: str | None) -> str:
    key = (explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "No API key. Create one at https://aistudio.google.com/apikey and set "
            "GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api-key."
        )
    return key


def load_prompt_body(path: Path) -> str:
    """Use the turn between markdown rulers, or the whole file."""
    raw = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in RULER_RE.split(raw)]
    middles = [p for p in parts[1:-1] if p] if len(parts) >= 3 else []
    for body in middles:
        if PLACEHOLDER in body:
            return body
    if middles:
        return middles[-1]
    return raw.strip()


def fill_prompt(template: str, chunk: str) -> str:
    if PLACEHOLDER in template:
        return template.replace(PLACEHOLDER, chunk)
    return f"{template.rstrip()}\n\nText: {chunk}"


def load_book_text(book_id: str | None, text_path: Path | None, repos: Path) -> tuple[str, str]:
    if text_path is not None:
        if not text_path.is_file():
            raise FileNotFoundError(text_path)
        name = book_id or text_path.stem
        return name, text_path.read_text(encoding="utf-8")
    if not book_id:
        raise SystemExit("Pass --book P000152 or --text some.txt")
    rec = load_book(repos / book_id)
    if rec.base_path is None or not rec.base_path.is_file():
        raise FileNotFoundError(f"No base text for {book_id} under {repos}")
    return book_id, rec.base_path.read_text(encoding="utf-8")


def parse_spans_json(raw: str) -> dict:
    """Strip fences / chatter and return a {\"spans\": [...]} object."""
    text = FENCE_RE.sub("", (raw or "").strip()).strip()
    if not text:
        return {"spans": []}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if isinstance(payload, list):
        return {"spans": payload}
    if isinstance(payload, dict):
        if "spans" not in payload:
            payload = {"spans": payload.get("quotes", [])}
        return payload
    raise ValueError(f"Unexpected JSON type: {type(payload).__name__}")


def salvage_spans(raw: str) -> list[dict]:
    """Pull whole span objects out of a reply that was cut off mid-JSON."""
    start = raw.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    spans: list[dict] = []
    i = raw.find("{", start)
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and (obj.get("text") or obj.get("head")):
            spans.append(obj)
        i = raw.find("{", end)
    return spans


def read_usage(response) -> dict[str, int]:
    """Split the reply's token spend into prompt / thinking / answer."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    fields = {
        "prompt": "prompt_token_count",
        "thoughts": "thoughts_token_count",
        "candidates": "candidates_token_count",
        "total": "total_token_count",
    }
    return {name: int(getattr(meta, attr, 0) or 0) for name, attr in fields.items()}


def format_usage(usage: dict[str, int]) -> str:
    if not usage:
        return "no usage metadata"
    return (
        f"in {usage.get('prompt', 0):,} / thinking {usage.get('thoughts', 0):,} "
        f"/ answer {usage.get('candidates', 0):,} tokens"
    )


def generate_json(
    client,
    model: str,
    contents: str,
    max_output_tokens: int,
    temperature: float,
    thinking_level: str | None = None,
) -> tuple[str, dict, bool, dict[str, int]]:
    """Return (raw reply, parsed payload, hit_output_cap, token usage)."""
    from google.genai import types

    config = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        response_mime_type="application/json",
    )
    if thinking_level:
        config.thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

    response = client.models.generate_content(model=model, contents=contents, config=config)
    usage = read_usage(response)
    candidates = response.candidates or []
    reason = str(getattr(candidates[0], "finish_reason", "") if candidates else "")
    truncated = "MAX_TOKENS" in reason.upper()
    raw = (response.text or "").strip()
    if truncated:
        return raw, {"spans": salvage_spans(raw)}, True, usage
    return raw, parse_spans_json(raw), False, usage


def retry_after(exc: Exception) -> tuple[bool, float | None]:
    """Say whether the API asked us to wait, and for how long it suggested."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc)
    if code not in RETRY_STATUSES and not any(f"{s} " in text for s in RETRY_STATUSES):
        return False, None
    match = RETRY_DELAY_RE.search(text)
    return True, float(match.group(1)) if match else None


def generate_json_with_wait(
    client,
    model: str,
    contents: str,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    max_wait: float,
    thinking_level: str | None = None,
) -> tuple[str, dict, bool, dict[str, int]]:
    """Call the model, sleeping through rate limits instead of giving up."""
    attempt = 0
    while True:
        try:
            return generate_json(
                client, model, contents, max_output_tokens, temperature, thinking_level
            )
        except Exception as exc:
            retryable, suggested = retry_after(exc)
            if not retryable or attempt >= max_retries:
                raise
            wait = min(suggested if suggested is not None else 2.0 * (2**attempt), max_wait)
            attempt += 1
            print(
                f"    rate limited ({str(exc).split('{', 1)[0].strip()}); "
                f"waiting {wait:.0f}s, retry {attempt}/{max_retries}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)


def _span_key(s: dict) -> str:
    """Identity for dedup: full text when present, else (head, tail)."""
    text = s.get("text") or ""
    if text:
        return f"text:{text}"
    return f"head:{s.get('head', '')}|tail:{s.get('tail', '')}"


def dump_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "spans" in payload


def generate_all_spans(
    client,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    max_wait: float,
    max_continues: int,
    thinking_level: str | None = None,
    anchor_continue: bool = False,
) -> tuple[list[dict], list[str], list[dict[str, int]]]:
    """Collect spans across as many replies as the output cap forces."""
    spans: list[dict] = []
    raws: list[str] = []
    usages: list[dict[str, int]] = []
    seen: set[str] = set()
    contents = prompt
    for round_no in range(max_continues + 1):
        raw, payload, truncated, usage = generate_json_with_wait(
            client,
            model,
            contents,
            max_output_tokens,
            temperature,
            max_retries,
            max_wait,
            thinking_level,
        )
        raws.append(raw)
        usages.append(usage)
        print(f"    {format_usage(usage)}", flush=True)
        new = [s for s in (payload.get("spans") or []) if isinstance(s, dict)]
        fresh = [s for s in new if _span_key(s) not in seen]
        seen.update(_span_key(s) for s in fresh)
        spans.extend(fresh)
        if not truncated:
            break
        if not fresh:
            print(
                "    output cap reached but nothing new came back; stopping",
                file=sys.stderr,
                flush=True,
            )
            break
        last_span = spans[-1]
        tail_text = ""
        if last_span.get("text"):
            tail_text = str(last_span.get("text"))[-TAIL_CHARS:]
        elif last_span.get("tail"):
            tail_text = str(last_span.get("tail"))
        else:
            tail_text = str(last_span.get("head", ""))

        suffix = CONTINUE_ANCHOR_SUFFIX if anchor_continue else CONTINUE_SUFFIX

        print(
            f"    output cap reached after {len(spans)} span(s); "
            f"resuming ({round_no + 1}/{max_continues})",
            flush=True,
        )
        contents = prompt + suffix.format(tail=tail_text)
    return spans, raws, usages


def count_input_tokens(client, model: str, contents: str) -> int | None:
    try:
        return int(client.models.count_tokens(model=model, contents=contents).total_tokens)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", help="OpenPecha book id, e.g. P000152")
    parser.add_argument("--text", type=Path, help="Plain-text file instead of a local repo")
    parser.add_argument("--repos", type=Path, default=DEFAULT_REPOS_DIR)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Markdown prompt (default: quotation/prompts/quotation_plain_prompt.md)",
    )
    parser.add_argument("--model", default=None, help="Override model id (default: config primary)")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Split into overlapping windows instead of one request for the full text",
    )
    parser.add_argument("--whole-book", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after N windows (debug)")
    parser.add_argument("--start-chunk", type=int, default=0, help="Skip the first N windows")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="How many times to wait out a rate limit per request (default: 10)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=120.0,
        help="Cap on a single wait, in seconds (default: 120)",
    )
    parser.add_argument(
        "--thinking-level",
        default=None,
        help="Gemini thinking level, e.g. low — thinking shares the output token budget",
    )
    parser.add_argument(
        "--max-continues",
        type=int,
        default=30,
        help="How many times to resume after hitting the output cap (default: 30)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests, e.g. 13 to stay under 5 RPM",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dump-raw", type=Path, default=None, help="Write each model reply as-is")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction if the output JSON already exists and is valid",
    )
    args = parser.parse_args(argv)

    cfg = load_config(CONFIG_PATH)
    model = args.model or (cfg.get("models") or {}).get("primary") or "gemini-3.8-flash"
    window = int(cfg.get("W") or 2156)
    stride = int(cfg.get("S") or 1078)

    if not args.prompt.is_file():
        raise FileNotFoundError(f"Prompt not found: {args.prompt}")
    template = load_prompt_body(args.prompt)
    book_id_hint = args.book or (args.text.stem if args.text else "unknown")
    out_path = args.out or (
        ROOT / "results" / "zeroshot_single_book" / f"{book_id_hint.lower()}_gemini_spans.json"
    )
    if args.skip_existing and dump_is_valid(out_path):
        print(f"File {out_path} already exists and is valid. Skipping API extraction.", flush=True)
        return 0

    book_id, text = load_book_text(args.book, args.text, args.repos)

    api_key = resolve_api_key(args.api_key)
    try:
        from google import genai
    except ImportError:
        raise SystemExit("Install the client: pip install google-genai") from None
    client = genai.Client(api_key=api_key)

    whole_book = args.whole_book or not args.chunked
    if whole_book:
        chunks = [
            {
                "chunk_id": f"{book_id}_full",
                "book_id": book_id,
                "start": 0,
                "end": len(text),
                "text": text,
            }
        ]
    else:
        chunks = make_chunks(text, book_id, window, stride)

    chunks = chunks[args.start_chunk :]
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    raw_dir = args.dump_raw
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"book={book_id}  chars={len(text):,}  chunks={len(chunks)}  "
        f"model={model}  prompt={args.prompt.name}  mode={'whole-book' if whole_book else 'chunked'}"
    )

    if whole_book:
        n_tokens = count_input_tokens(client, model, fill_prompt(template, text))
        if n_tokens is not None:
            print(f"  input tokens: {n_tokens:,}")

    chunk_rows: list[dict] = []
    all_spans: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = fill_prompt(template, chunk["text"])
        print(
            f"  [{i}/{len(chunks)}] {chunk['chunk_id']} "
            f"[{chunk['start']}:{chunk['end']}] {len(chunk['text'])} chars …",
            flush=True,
        )
        if i > 1 and args.pause > 0:
            time.sleep(args.pause)
        try:
            spans, raws, usages = generate_all_spans(
                client,
                model,
                prompt,
                args.max_output_tokens,
                args.temperature,
                args.max_retries,
                args.max_wait,
                args.max_continues,
                args.thinking_level,
                anchor_continue='"head"' in template and '"tail"' in template,
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            chunk_rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "error": str(exc),
                    "spans": [],
                }
            )
            continue
        print(f"    {len(spans)} span(s)")
        if raw_dir is not None:
            for part, raw in enumerate(raws):
                suffix = "" if part == 0 else f".{part}"
                (raw_dir / f"{chunk['chunk_id']}{suffix}.json").write_text(raw, encoding="utf-8")
        chunk_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "n_spans": len(spans),
                "n_calls": len(raws),
                "usage": usages,
                "spans": spans,
            }
        )
        all_spans.extend(spans)

    if chunks and all("error" in row for row in chunk_rows):
        print(
            f"every request failed; leaving {out_path} untouched",
            file=sys.stderr,
        )
        return 1

    result = {
        "book_id": book_id,
        "model": model,
        "prompt": str(args.prompt),
        "whole_book": whole_book,
        "n_chunks": len(chunks),
        "n_spans": len(all_spans),
        "spans": all_spans,
        "chunks": chunk_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  ({len(all_spans)} spans from {len(chunks)} chunk(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

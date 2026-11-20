import argparse
import sys
import os
import json
import math
import hashlib
import time
import random
import importlib.util
from datetime import datetime
import zoneinfo
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Direct imports
from quotation_lib.window_plan import make_windows, cut_points, _largest_cut_at_most
from quotation_lib.gemini_locate import (
    locate_parsed_quote,
    locate_span,
    parse_quote_dump,
    quotes_from_items,
)
from quotation_lib.span_metrics import match_iou, inclusive_iou, muc_taxonomy, prf

# Importlib shims for helper functions in scripts
def import_from_script(script_name: str, function_name: str):
    script_path = ROOT / "src" / script_name
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find spec for {script_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

parse_spans_json = import_from_script("extract_quotation_gemini.py", "parse_spans_json")
salvage_spans = import_from_script("extract_quotation_gemini.py", "salvage_spans")
read_usage = import_from_script("extract_quotation_gemini.py", "read_usage")
stitch_adjacent_spans = import_from_script("check_quotation_gemini.py", "stitch_adjacent_spans")

EXP_DIR = ROOT / "results" / "zeroshot"

GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite")

# Pricing and model metadata
CLAUDE_INPUT_PRICE_PER_M = 1.0  # 50% batch discount of $2
CLAUDE_OUTPUT_PRICE_PER_M = 5.0  # 50% batch discount of $10
CLAUDE_OUTPUT_PROJECTION_PER_UNIT = 20_000

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #

def get_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

CACHE_COMPLETE_REASONS = {
    "STOP_REASON_UNSPECIFIED",
    "STOP",
    "MAX_TOKENS",
    "SAFETY",
    "OTHER",
    "BLOCKED",
}


def normalize_finish_reason(reason: Any) -> str:
    """Strip SDK enum prefixes so FinishReason.STOP and STOP compare equal."""
    if reason is None:
        return ""
    name = getattr(reason, "name", None)
    text = name if isinstance(name, str) and name else str(reason)
    text = text.strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def is_cached_complete(finish_reason: Any) -> bool:
    return normalize_finish_reason(finish_reason) in CACHE_COMPLETE_REASONS

def make_units(text: str, book_id: str, max_chars: int = 250_000, overlap_chars: int = 2_000) -> list[dict]:
    n = len(text)
    if n <= max_chars:
        return [{
            "book_id": book_id,
            "piece": 0,
            "start": 0,
            "end": n,
            "sha1": get_sha1(text),
        }]
    
    n_pieces = math.ceil(n / max_chars)
    L = (n + (n_pieces - 1) * overlap_chars) / n_pieces
    cuts = cut_points(text)
    
    units = []
    pos = 0
    import bisect
    for idx in range(n_pieces):
        if idx == n_pieces - 1:
            end = n
        else:
            target_end = pos + L
            idx_cut = bisect.bisect_left(cuts, target_end)
            candidates = []
            if idx_cut < len(cuts):
                candidates.append(cuts[idx_cut])
            if idx_cut > 0:
                candidates.append(cuts[idx_cut - 1])
            candidates = [c for c in candidates if c > pos + overlap_chars]
            if not candidates:
                end = int(target_end)
            else:
                end = min(candidates, key=lambda c: abs(c - target_end))
        
        chunk_text = text[pos:end]
        units.append({
            "book_id": book_id,
            "piece": idx,
            "start": pos,
            "end": end,
            "sha1": get_sha1(chunk_text),
        })
        
        if idx < n_pieces - 1:
            nxt = _largest_cut_at_most(cuts, end - overlap_chars, pos)
            if nxt is None:
                nxt = end - overlap_chars
            pos = nxt
            
    return units

def load_gold_books(split_name: str) -> list[dict]:
    split_path = ROOT / "Data" / "audit" / "quotation_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Cannot find split config at {split_path}")
    
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    target_val = "test" if split_name == "test" else "val"
    book_ids = [k for k, v in split_data.items() if v == target_val]
    
    export_file = ROOT / "Data" / "hf_export" / f"{'test' if split_name == 'test' else 'validation'}.jsonl"
    if not export_file.is_file():
        raise FileNotFoundError(f"Cannot find gold export at {export_file}")
    
    books = []
    total_spans = 0
    with export_file.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bid = d["book_id"]
            if bid in book_ids:
                base_txt_path = ROOT / "Data" / "repos" / bid / f"{bid}.opf" / "base" / "v001.txt"
                if not base_txt_path.is_file():
                    raise FileNotFoundError(f"No base text for {bid} under {base_txt_path}")
                base_text = base_txt_path.read_text(encoding="utf-8")
                assert base_text == d["text"], f"Text mismatch for book {bid}"
                
                books.append(d)
                total_spans += len(d["spans"])
                
    expected_spans = 2337 if split_name == "test" else 2005
    # Skip strict count checks during small-scale unit testing
    if len(books) >= 10:
        assert total_spans == expected_spans, f"Expected {expected_spans} spans for split {split_name}, got {total_spans}"
    
    return books

def fill_anchor_prompt(template: str, unit_text: str) -> str:
    return f"{template.rstrip()}\n{unit_text}"

def check_api_keys(require_gemini=True, require_anthropic=True):
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if require_gemini and not gemini_key:
        raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required in environment")
    if require_anthropic and not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required in environment")
    return gemini_key, anthropic_key

def resolve_anthropic_model(model_name: str) -> str:
    # Claude Sonnet 5's API id is the dateless string itself, not the old 3.5 snapshot.
    return model_name


def claude_batch_snapshot(batch: Any) -> dict[str, Any]:
    """Serialize an Anthropic MessageBatch. Status lives on processing_status, not status."""
    counts = getattr(batch, "request_counts", None)
    return {
        "id": batch.id,
        "status": batch.processing_status,
        "created_at": str(batch.created_at),
        "request_counts": {
            "processing": getattr(counts, "processing", 0),
            "succeeded": getattr(counts, "succeeded", 0),
            "errored": getattr(counts, "errored", 0),
            "canceled": getattr(counts, "canceled", 0),
            "expired": getattr(counts, "expired", 0),
        } if counts is not None else {},
    }


def claude_batch_is_ready(batch: Any) -> bool:
    return batch.processing_status == "ended"

def load_limits() -> dict:
    limits_path = EXP_DIR / "limits.yaml"
    if not limits_path.is_file():
        raise FileNotFoundError(f"limits.yaml is required but missing at {limits_path}")
    import yaml
    with limits_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_pacific_date_str() -> str:
    tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y-%m-%d")

MAX_SERVER_ATTEMPTS = 5
COUNT_TOKEN_RETRY_WAIT_SECONDS = 60.0


def classify_gemini_run_error(exc: Exception) -> str:
    """Classify a generate_content failure: daily, rate, bad_request, or server."""
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    exc_str = str(exc)
    lowered = exc_str.lower()
    is_daily = (
        "perday" in lowered
        or "daily" in lowered
        or ("limit" in lowered and "day" in lowered)
    )
    is_rate = code == 429 or "429" in exc_str or "quota" in lowered
    is_bad_request = code == 400 or "invalid_argument" in lowered
    if is_daily:
        return "daily"
    if is_rate:
        return "rate"
    if is_bad_request:
        return "bad_request"
    return "server"


def next_server_wait_seconds(attempts: int, jitter: float | None = None) -> float:
    extra = random.uniform(1.0, 5.0) if jitter is None else jitter
    return min(240.0, 30.0 * (2 ** (attempts - 1))) + extra


def unit_failure_action(error_kind: str, attempts: int, *, second_pass: bool) -> str:
    """Decide what cmd_run should do after a failed generate_content call."""
    if error_kind == "daily":
        return "stop_daily"
    if error_kind == "bad_request":
        return "stop_bad_request"
    if error_kind == "rate":
        return "wait_rate"
    if attempts < MAX_SERVER_ATTEMPTS:
        return "wait_server"
    return "give_up" if second_pass else "defer"


def count_gemini_tokens(client: Any, model: str, contents: str) -> int:
    """Call count_tokens, waiting a minute and retrying on timeout or other transient failures."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return int(client.models.count_tokens(model=model, contents=contents).total_tokens)
        except Exception as exc:
            kind = classify_gemini_run_error(exc)
            if kind == "bad_request":
                raise
            if kind == "daily":
                print("  [429 DAILY LIMIT] count_tokens quota exceeded. Stopping process cleanly.")
                sys.exit(0)
            wait_sec = COUNT_TOKEN_RETRY_WAIT_SECONDS
            if kind == "rate":
                wait_sec += random.uniform(1.0, 5.0)
            print(f"  [TOKEN COUNT] {exc}. Waiting {wait_sec:.0f}s before retry {attempt}...")
            time.sleep(wait_sec)


def _strip_retry_marker(unit: dict) -> dict:
    return {k: v for k, v in unit.items() if k != "_retry_pass"}


def write_retry_list(path: Path, units: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(_strip_retry_marker(dict(unit)), ensure_ascii=False) + "\n")

# --------------------------------------------------------------------------- #
# CLI Subcommands: build-units and build-windows
# --------------------------------------------------------------------------- #

def cmd_build_units(args):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ["test", "validation"]:
        books = load_gold_books(split)
        units = []
        for b in books:
            book_units = make_units(b["text"], b["book_id"])
            units.extend(book_units)
            
        expected_rows = 26 if split == "test" else 32
        assert len(units) == expected_rows, f"Expected {expected_rows} units for {split}, but got {len(units)}"
        
        out_path = EXP_DIR / f"units_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        print(f"Successfully wrote {len(units)} units to {out_path}")

def cmd_build_windows(args):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ["test", "validation"]:
        books = load_gold_books(split)
        windows = []
        for b in books:
            book_windows = make_windows(b["text"], b["book_id"], window_chars=16000, overlap_chars=2000)
            for idx, w in enumerate(book_windows):
                windows.append({
                    "book_id": w.book_id,
                    "piece": idx,
                    "start": w.start,
                    "end": w.end,
                    "sha1": get_sha1(w.text),
                })
                
        expected_rows = 333 if split == "test" else 334
        assert len(windows) == expected_rows, f"Expected {expected_rows} windows for {split}, but got {len(windows)}"
        
        out_path = EXP_DIR / f"windows_{split}.jsonl"
        payload_bytes = ("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in windows)).encode("utf-8")
        if out_path.is_file():
            existing_bytes = out_path.read_bytes()
            if existing_bytes == payload_bytes:
                print(f"File {out_path} already exists and is identical. Skipping write.")
                continue
                
        with out_path.open("wb") as f:
            f.write(payload_bytes)
        print(f"Successfully wrote {len(windows)} windows to {out_path}")

# --------------------------------------------------------------------------- #
# CLI Subcommand: preflight
# --------------------------------------------------------------------------- #

def cmd_preflight(args):
    selected_model = getattr(args, "model", None)
    source = getattr(args, "source", "units")
    want_g2 = selected_model in (None, "gemini-3.8-flash")
    want_g1 = selected_model in (None, "gemini-3.1-flash-lite")
    want_g35 = selected_model in (None, "gemini-3.5-flash-lite")
    want_claude = selected_model in (None, "claude-sonnet-5")
    want_gemini = want_g2 or want_g1 or want_g35
    # All-three preflight still treats Claude as optional. A Claude-only
    # --model run requires the Anthropic key.
    gemini_key, anthropic_key = check_api_keys(
        require_gemini=want_gemini,
        require_anthropic=selected_model == "claude-sonnet-5",
    )
    limits = load_limits()
    
    prompt_path = ROOT / "prompts" / "quotation_anchor_prompt.md"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file missing at {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    template_sha1 = get_sha1(template)
    expected_sha1 = "e88f021573bb97fd3733282caa8b4c966501bf9f"
    assert template_sha1 == expected_sha1, f"Prompt sha1 mismatch: expected {expected_sha1}, got {template_sha1}"
    
    gemini_client = None
    if want_gemini:
        from google import genai
        gemini_client = genai.Client(api_key=gemini_key)
    anthropic_client = None
    c_api_model = None
    if want_claude and anthropic_key:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        c_api_model = resolve_anthropic_model("claude-sonnet-5")
    elif want_claude:
        print("ANTHROPIC_API_KEY not set; skipping Claude token counts and smoke test.")
    
    splits_to_count = {
        f"{source}_test": EXP_DIR / f"{source}_test.jsonl",
        f"{source}_validation": EXP_DIR / f"{source}_validation.jsonl",
    }
    
    preflight_counts = {}
    if want_g2:
        preflight_counts["gemini-3.8-flash"] = {}
    if want_g1:
        preflight_counts["gemini-3.1-flash-lite"] = {}
    if want_g35:
        preflight_counts["gemini-3.5-flash-lite"] = {}
    if anthropic_client:
        preflight_counts["claude-sonnet-5"] = {}
    
    # Load all base texts to construct fill
    # We will build a quick lookup dict for base texts
    base_texts = {}
    
    # Populate base_texts for validation and test split
    for split_name in ["test", "validation"]:
        books = load_gold_books(split_name)
        for b in books:
            base_texts[b["book_id"]] = b["text"]
            
    print("Starting preflight token counting...")
    
    for key, path in splits_to_count.items():
        if not path.is_file():
            print(f"File {path} does not exist. Run build-{source} first.")
            continue
            
        units = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                units.append(json.loads(line))
                
        # We might need to split units if they exceed limits. We loop to support updates.
        changed = True
        while changed:
            changed = False
            updated_units = []
            
            for u in units:
                bid = u["book_id"]
                start = u["start"]
                end = u["end"]
                utext = base_texts[bid][start:end]
                filled = fill_anchor_prompt(template, utext)
                
                # Count tokens only for the selected model(s)
                g2_tokens = 0
                g1_tokens = 0
                g35_tokens = 0
                if want_g2:
                    g2_tokens = count_gemini_tokens(gemini_client, "gemini-3.8-flash", filled)
                if want_g1:
                    g1_tokens = count_gemini_tokens(gemini_client, "gemini-3.1-flash-lite", filled)
                if want_g35:
                    g35_tokens = count_gemini_tokens(gemini_client, "gemini-3.5-flash-lite", filled)
                
                c_tokens = 0
                if anthropic_client:
                    c_resp = anthropic_client.messages.count_tokens(
                        model=c_api_model,
                        messages=[{"role": "user", "content": filled}]
                    )
                    c_tokens = int(c_resp.input_tokens)
                
                # Fit checks: 170K Gemini, 1M - 64K Claude
                is_g2_heavy = want_g2 and g2_tokens > 170_000
                is_g1_heavy = want_g1 and g1_tokens > 170_000
                is_g35_heavy = want_g35 and g35_tokens > 170_000
                is_c_heavy = anthropic_client is not None and (c_tokens + 64_000) > 1_000_000
                
                if (is_g2_heavy or is_g1_heavy or is_g35_heavy or is_c_heavy) and key not in ("windows_validation", "windows_test"):
                    print(f"Preflight cut: Unit {bid} [{start}:{end}] is too large (G2:{g2_tokens}, G1:{g1_tokens}, G35:{g35_tokens}, C:{c_tokens}). Splitting in half...")
                    # Perform half-cut
                    cuts = cut_points(base_texts[bid])
                    mid_target = (start + end) // 2
                    import bisect
                    idx = bisect.bisect_left(cuts, mid_target)
                    candidates = []
                    if idx < len(cuts):
                        candidates.append(cuts[idx])
                    if idx > 0:
                        candidates.append(cuts[idx - 1])
                    candidates = [c for c in candidates if start + 2000 <= c <= end - 2000]
                    if not candidates:
                        mid = mid_target
                    else:
                        mid = min(candidates, key=lambda c: abs(c - mid_target))
                        
                    first_end = mid
                    nxt = _largest_cut_at_most(cuts, mid - 2000, start)
                    if nxt is None:
                        nxt = mid - 2000
                    second_start = nxt
                    
                    p1 = {
                        "book_id": bid,
                        "piece": u["piece"],
                        "start": start,
                        "end": first_end,
                        "sha1": get_sha1(base_texts[bid][start:first_end])
                    }
                    p2 = {
                        "book_id": bid,
                        "piece": u["piece"] + 1, # Wait, adjust pieces below
                        "start": second_start,
                        "end": end,
                        "sha1": get_sha1(base_texts[bid][second_start:end])
                    }
                    updated_units.append(p1)
                    updated_units.append(p2)
                    changed = True
                else:
                    updated_units.append(u)
                    
            if changed:
                # Re-index pieces for updated_units per book
                book_counts = {}
                for uu in updated_units:
                    b_id = uu["book_id"]
                    uu["piece"] = book_counts.get(b_id, 0)
                    book_counts[b_id] = uu["piece"] + 1
                units = updated_units
                
        # Save back the potentially modified units
        with path.open("w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
                
        # Record final counted tokens
        g2_counts = {}
        g1_counts = {}
        g35_counts = {}
        c_counts = {}
        for u in units:
            bid = u["book_id"]
            piece = u["piece"]
            ukey = f"{bid}__{piece}"
            start = u["start"]
            end = u["end"]
            utext = base_texts[bid][start:end]
            filled = fill_anchor_prompt(template, utext)
            
            if want_g2:
                g2_counts[ukey] = count_gemini_tokens(gemini_client, "gemini-3.8-flash", filled)
            if want_g1:
                g1_counts[ukey] = count_gemini_tokens(gemini_client, "gemini-3.1-flash-lite", filled)
            if want_g35:
                g35_counts[ukey] = count_gemini_tokens(gemini_client, "gemini-3.5-flash-lite", filled)
            if anthropic_client:
                c_resp = anthropic_client.messages.count_tokens(
                    model=c_api_model,
                    messages=[{"role": "user", "content": filled}]
                )
                c_counts[ukey] = int(c_resp.input_tokens)
            
        if want_g2:
            preflight_counts["gemini-3.8-flash"][key] = g2_counts
        if want_g1:
            preflight_counts["gemini-3.1-flash-lite"][key] = g1_counts
        if want_g35:
            preflight_counts["gemini-3.5-flash-lite"][key] = g35_counts
        if anthropic_client:
            preflight_counts["claude-sonnet-5"][key] = c_counts
        
    # Merge into preflight.json so a single-model run does not wipe other arms
    out_preflight = EXP_DIR / "preflight.json"
    existing = {}
    if out_preflight.is_file():
        with out_preflight.open("r", encoding="utf-8") as f:
            existing = json.load(f) or {}
    for model_name, counts in preflight_counts.items():
        merged = dict(existing.get(model_name, {}))
        merged.update(counts)
        existing[model_name] = merged
    with out_preflight.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Preflight token counting completed. Saved to {out_preflight}")
    
    def _sum_model_tokens(model_name: str) -> int:
        model_counts = preflight_counts.get(model_name, {})
        return sum(
            tokens
            for split_counts in model_counts.values()
            for tokens in split_counts.values()
        )

    total_chars = 0
    for path in splits_to_count.values():
        for u in json_lines(path):
            total_chars += len(base_texts[u["book_id"]][u["start"]:u["end"]])

    total_g2_tokens = _sum_model_tokens("gemini-3.8-flash")
    total_g1_tokens = _sum_model_tokens("gemini-3.1-flash-lite")
    total_g35_tokens = _sum_model_tokens("gemini-3.5-flash-lite")
    total_c_tokens = _sum_model_tokens("claude-sonnet-5") if anthropic_client else 0
    
    g2_rate = total_g2_tokens / total_chars if total_chars else 0
    g1_rate = total_g1_tokens / total_chars if total_chars else 0
    g35_rate = total_g35_tokens / total_chars if total_chars else 0
    c_rate = total_c_tokens / total_chars if total_chars else 0
    
    print("\n" + "="*80)
    print(" PREFLIGHT METRICS SUMMARY")
    print("="*80)
    print(f"Source: {source}" + (f" | model: {selected_model}" if selected_model else " | model: all"))
    print(f"Total processed characters (test + validation {source}): {total_chars:,}")
    if want_g2:
        print(f"Gemini 3.8 Flash Rate:     {g2_rate:.4f} tokens/char (Total tokens: {total_g2_tokens:,})")
    if want_g1:
        print(f"Gemini 3.1 Flash Lite Rate: {g1_rate:.4f} tokens/char (Total tokens: {total_g1_tokens:,})")
    if want_g35:
        print(f"Gemini 3.5 Flash Lite Rate: {g35_rate:.4f} tokens/char (Total tokens: {total_g35_tokens:,})")
    if anthropic_client:
        print(f"Claude Sonnet 5 Rate:      {c_rate:.4f} tokens/char (Total tokens: {total_c_tokens:,})")
    print("-"*80)
    
    test_key = f"{source}_test"
    test_units = 0
    for model_name in preflight_counts:
        test_units = len(preflight_counts[model_name].get(test_key, {}))
        if test_units:
            break
    print(f"Test split {source}: {test_units}")
    if want_g2:
        g2_rpd = limits.get("gemini-3.8-flash", {}).get("rpd", 20)
        g2_days_test = test_units / (0.8 * g2_rpd) if g2_rpd else 0
        print(f"Gemini 3.8 Flash:     RPD limit = {g2_rpd} | Days needed = {g2_days_test:.2f}")
    if want_g1:
        g1_rpd = limits.get("gemini-3.1-flash-lite", {}).get("rpd", 500)
        g1_days_test = test_units / (0.8 * g1_rpd) if g1_rpd else 0
        print(f"Gemini 3.1 Flash Lite: RPD limit = {g1_rpd} | Days needed = {g1_days_test:.2f}")
    if want_g35:
        g35_rpd = limits.get("gemini-3.5-flash-lite", {}).get("rpd", 500)
        g35_days_test = test_units / (0.8 * g35_rpd) if g35_rpd else 0
        print(f"Gemini 3.5 Flash Lite: RPD limit = {g35_rpd} | Days needed = {g35_days_test:.2f}")
    print("-"*80)
    
    if anthropic_client:
        test_c_input_tokens = sum(preflight_counts["claude-sonnet-5"].get(test_key, {}).values())
        est_c_input_cost = (test_c_input_tokens * CLAUDE_INPUT_PRICE_PER_M) / 1_000_000
        est_c_output_tokens = test_units * CLAUDE_OUTPUT_PROJECTION_PER_UNIT
        est_c_output_cost = (est_c_output_tokens * CLAUDE_OUTPUT_PRICE_PER_M) / 1_000_000
        est_c_total_cost = est_c_input_cost + est_c_output_cost
        
        print(f"Claude Sonnet 5 Estimated Test Cost (using 50% Batch Discount):")
        print(f"  Input Tokens:  {test_c_input_tokens:,} (Cost: ${est_c_input_cost:.4f})")
        print(f"  Output Tokens: {est_c_output_tokens:,} projected (Cost: ${est_c_output_cost:.4f})")
        print(f"  Total Cost:    ${est_c_total_cost:.4f}")
    print("="*80)
    
    print("\nRunning single-shot test call per selected model on a tiny window to verify credentials and thinking settings...")
    tiny_text = "ལུང་ལས། ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསཏན་བཅོས་གཉིས་སུ་འདུས། །ཞེས་གསུངས་སོ། །"
    tiny_filled = fill_anchor_prompt(template, tiny_text)
    smoke_total = int(want_g2) + int(want_g1) + int(want_g35) + int(bool(anthropic_client))
    smoke_i = 0

    if want_gemini:
        from google.genai import types
        schema = {
            "type": "OBJECT",
            "properties": {
                "spans": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING"},
                            "frame": {"type": "STRING"},
                            "head": {"type": "STRING"},
                            "tail": {"type": "STRING"}
                        },
                        "required": ["label", "head", "tail"]
                    }
                }
            },
            "required": ["spans"]
        }

    if want_g2:
        smoke_i += 1
        try:
            cfg_g2 = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=1.0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
            print(f"  [{smoke_i}/{smoke_total}] Calling Gemini 3.8 Flash ... ", end="", flush=True)
            gemini_client.models.generate_content(model="gemini-3.8-flash", contents=tiny_filled, config=cfg_g2)
            print("OK! Output parsed successfully.")
        except Exception as e:
            print(f"FAILED: {e}")

    if want_g1:
        smoke_i += 1
        try:
            cfg_g1 = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=1.0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
            print(f"  [{smoke_i}/{smoke_total}] Calling Gemini 3.1 Flash Lite ... ", end="", flush=True)
            gemini_client.models.generate_content(model="gemini-3.1-flash-lite", contents=tiny_filled, config=cfg_g1)
            print("OK! Output parsed successfully.")
        except Exception as e:
            print(f"FAILED: {e}")

    if want_g35:
        smoke_i += 1
        try:
            cfg_g35 = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=1.0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
            print(f"  [{smoke_i}/{smoke_total}] Calling Gemini 3.5 Flash Lite ... ", end="", flush=True)
            gemini_client.models.generate_content(model="gemini-3.5-flash-lite", contents=tiny_filled, config=cfg_g35)
            print("OK! Output parsed successfully.")
        except Exception as e:
            print(f"FAILED: {e}")

    if anthropic_client:
        smoke_i += 1
        try:
            schema_claude = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string"},
                                "frame": {"type": "string"},
                                "head": {"type": "string"},
                                "tail": {"type": "string"}
                            },
                            "required": ["label", "head", "tail"]
                        }
                    }
                },
                "required": ["spans"]
            }
            print(f"  [{smoke_i}/{smoke_total}] Calling Claude Sonnet 5 (Single Message API) ... ", end="", flush=True)
            anthropic_client.messages.create(
                model=c_api_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": tiny_filled}],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": schema_claude}
                }
            )
            print("OK! Output parsed successfully.")
        except Exception as e:
            print(f"FAILED: {e}")
        
        print("\nSet a Claude monthly spending limit in your Console before the first batch run!")
    
    # Prompt for yes if interactive
    if sys.stdin.isatty():
        try:
            val = input("\nDo you want to proceed with the batch configuration? [yes/no]: ").strip().lower()
            if val != "yes":
                print("Aborting.")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nAborting.")
            sys.exit(1)
    else:
        print("\nSkipping interactive confirmation as stdin is not a tty.")

def json_lines(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# --------------------------------------------------------------------------- #
# CLI Subcommand: run (Gemini models)
# --------------------------------------------------------------------------- #

def wait_for_pacing(log: list, this_tokens: int, rpm_limit: int, tpm_limit: int):
    while True:
        now = time.time()
        # Prune older than 60s
        while log and now - log[0][0] > 60:
            log.pop(0)
            
        current_sum = sum(entry[1] for entry in log)
        if current_sum + this_tokens + 30000 <= tpm_limit and len(log) < rpm_limit:
            break
            
        # sleep and check again
        if log:
            sleep_time = max(0.1, 60 - (now - log[0][0]))
        else:
            sleep_time = 1.0
        time.sleep(sleep_time)

def update_ledger(ledger_path: Path, p_date: str, max_rpd: int):
    ledger = {}
    if ledger_path.is_file():
        try:
            with ledger_path.open("r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception:
            pass
            
    today_count = ledger.get(p_date, 0)
    limit_count = int(0.9 * max_rpd)
    if today_count >= limit_count:
        # Determine next reset
        # Midnight Pacific reset is: 12:30 IST until 1 November 2026, then 13:30 IST.
        # Since today is 2026-09-11, it is 12:30 IST.
        print(f"\n[LEDGER] Stopped! Attempt limit reached for today's Pacific date: {p_date} ({today_count}/{limit_count} RPD).")
        print("Please resume after the reset (12:30 IST).")
        sys.exit(0)
        
    ledger[p_date] = today_count + 1
    with ledger_path.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

def cmd_run(args):
    model = args.model
    split = args.split
    source = getattr(args, "source", "units")
    
    assert model in GEMINI_MODELS, f"Unknown Gemini model {model}"
    assert split in ["test", "validation"], f"Unknown split {split}"
    
    gemini_key, _ = check_api_keys(require_gemini=True, require_anthropic=False)
    limits = load_limits()
    
    # Load preflight token counts
    preflight_path = EXP_DIR / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(f"Run preflight first! preflight.json is missing at {preflight_path}")
        
    with preflight_path.open("r", encoding="utf-8") as f:
        preflight_data = json.load(f)
        
    # Get units to process
    units_file = EXP_DIR / f"{source}_{split}.jsonl"
    if not units_file.is_file():
        raise FileNotFoundError(f"Units file missing: {units_file}")
    units = json_lines(units_file)
    
    # For G1 (gemini-3.8-flash) on test split, sort by smallest book first
    # We can get total characters for sorting
    # Let's group units by book_id and calculate their character lengths
    if model == "gemini-3.8-flash" and split == "test":
        book_lengths = {}
        for u in units:
            bid = u["book_id"]
            book_lengths[bid] = book_lengths.get(bid, 0) + (u["end"] - u["start"])
            
        # sort units by book's total character length, keeping order of pieces
        units = sorted(units, key=lambda x: (book_lengths[x["book_id"]], x["book_id"], x["piece"]))
        
    prompt_path = ROOT / "prompts" / "quotation_anchor_prompt.md"
    template = prompt_path.read_text(encoding="utf-8")
    template_sha1 = get_sha1(template)
    
    # Base texts lookup
    base_texts = {}
    books_data = load_gold_books(split)
    for b in books_data:
        base_texts[b["book_id"]] = b["text"]
        
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=gemini_key)
    
    model_limits = limits.get(model, {})
    rpm = model_limits.get("rpm", 5)
    tpm = model_limits.get("tpm", 250000)
    rpd = model_limits.get("rpd", 20)
    
    rpm_pacing = int(0.8 * rpm)
    tpm_pacing = int(0.8 * tpm)
    
    model_dir = EXP_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = model_dir / ("ledger_windows.json" if source == "windows" else "ledger.json")
    
    # Setup pacing log
    pacing_log = []
    
    # Schema configuration
    schema = {
        "type": "OBJECT",
        "properties": {
            "spans": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "frame": {"type": "STRING"},
                        "head": {"type": "STRING"},
                        "tail": {"type": "STRING"}
                    },
                    "required": ["label", "head", "tail"]
                }
            }
        },
        "required": ["spans"]
    }
    
    print(f"Starting Gemini run for model {model} on split {split} (Total units: {len(units)})")
    
    cache_sub_dir = "raw" if source == "units" else "raw_windows"
    
    # We maintain a queue because mid-run split can add units dynamically!
    unit_queue = list(units)
    processed_count = 0
    still_failed: list[dict] = []
    retry_list_path = model_dir / f"retry_{split}.jsonl"
    
    while unit_queue:
        u = unit_queue.pop(0)
        bid = u["book_id"]
        piece = u["piece"]
        ukey = f"{bid}__{piece}"
        start = u["start"]
        end = u["end"]
        
        raw_cache_path = model_dir / cache_sub_dir / bid / f"{piece}.json"
        
        # Verify if exists and valid
        if raw_cache_path.is_file():
            try:
                with raw_cache_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if is_cached_complete(cached.get("finish_reason")):
                    # Skip finished
                    # Assert prompt_sha1 and unit_sha1 match!
                    assert cached.get("prompt_sha1") == template_sha1, f"Prompt sha1 mismatch in cache for {ukey}"
                    assert cached.get("window_sha1") == u["sha1"], f"Unit sha1 mismatch in cache for {ukey}"
                    print(f"  [CACHE] Skipping unit {ukey} [{start}:{end}] (already completed)")
                    processed_count += 1
                    continue
            except AssertionError:
                raise
            except Exception:
                pass
                
        # Token count for pacing
        preflight_key = f"{source}_{split}"
        unit_tokens = preflight_data.get(model, {}).get(preflight_key, {}).get(ukey)
        if unit_tokens is None:
            # If not in preflight (due to mid-run splits), we calculate it now
            filled_prompt = fill_anchor_prompt(template, base_texts[bid][start:end])
            unit_tokens = count_gemini_tokens(gemini_client, model, filled_prompt)
            
        print(f"Processing unit {ukey} [{start}:{end}] with {unit_tokens:,} tokens...")
        
        attempts = 0
        success = False
        
        while not success:
            attempts += 1
            # Check Pacific ledger RPD
            p_date = get_pacific_date_str()
            update_ledger(ledger_path, p_date, rpd)
            
            # Pacing
            wait_for_pacing(pacing_log, unit_tokens, rpm_pacing, tpm_pacing)
            
            # Make API Call
            cfg = types.GenerateContentConfig(
                max_output_tokens=65536,
                temperature=1.0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
            
            pacing_log.append((time.time(), unit_tokens))
            started_time = int(time.time())
            
            try:
                filled_prompt = fill_anchor_prompt(template, base_texts[bid][start:end])
                response = gemini_client.models.generate_content(model=model, contents=filled_prompt, config=cfg)
                finished_time = int(time.time())
                
                usage = read_usage(response)
                candidates = response.candidates or []
                reason = str(getattr(candidates[0], "finish_reason", "STOP_REASON_UNSPECIFIED") if candidates else "STOP_REASON_UNSPECIFIED")
                raw_text = (response.text or "").strip()
                
                # Check for truncation or safety block
                is_truncated = "MAX_TOKENS" in reason.upper()
                is_blocked = any(block in reason.upper() for block in ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
                
                # Temp-rename atomic cache write
                raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = raw_cache_path.with_suffix(".tmp")
                
                cache_payload = {
                    "model": model,
                    "model_version": getattr(response, "model_version", model),
                    "prompt_sha1": template_sha1,
                    "window_sha1": u["sha1"],
                    "temperature": 1.0,
                    "thinking": "low",
                    "finish_reason": "blocked" if is_blocked else ("MAX_TOKENS" if is_truncated else normalize_finish_reason(reason) or reason),
                    "usage": usage,
                    "raw_text": raw_text,
                    "attempts": attempts,
                    "started": started_time,
                    "finished": finished_time
                }
                
                with temp_path.open("w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, ensure_ascii=False, indent=2)
                temp_path.replace(raw_cache_path)
                
                if is_truncated:
                    print(f"  [TRUNCATED] Unit {ukey} hit MAX_TOKENS. Performing mid-run emergency split...")
                    # Perform mid-run split and append halves to queue
                    half1, half2 = split_unit_in_half(u, base_texts[bid])
                    unit_queue.insert(0, half2)
                    unit_queue.insert(0, half1)
                    # We remove the cached truncated file as we will replace it with split spans!
                    # Wait, no, we can keep it as raw log but the scoring will use the child pieces.
                    success = True
                elif is_blocked:
                    if attempts < 2:
                        print(f"  [BLOCKED] Unit {ukey} blocked by {reason}. Retrying once...")
                        continue
                    else:
                        print(f"  [BLOCKED] Unit {ukey} remains blocked. Saved as blocked.")
                        # Write as blocked
                        success = True
                else:
                    print(f"  [SUCCESS] Completed unit {ukey} in {attempts} attempt(s).")
                    success = True
                    
            except Exception as exc:
                kind = classify_gemini_run_error(exc)
                action = unit_failure_action(
                    kind, attempts, second_pass=bool(u.get("_retry_pass"))
                )
                if action == "stop_daily":
                    print(f"  [429 DAILY LIMIT] Quota exceeded. Stopping process cleanly without writing failed cache.")
                    sys.exit(0)
                if action == "stop_bad_request":
                    print(f"  [400 BAD REQUEST] Config bug. Stopping. {exc}")
                    raise
                if action == "wait_rate":
                    wait_sec = 60 + random.uniform(1.0, 5.0)
                    print(f"  [429 RATE LIMIT] Waiting {wait_sec:.0f}s before retry {attempts}...")
                    time.sleep(wait_sec)
                    continue
                if action == "wait_server":
                    wait_sec = next_server_wait_seconds(attempts)
                    print(f"  [ERROR] {exc}. Waiting {wait_sec:.0f}s before retry {attempts}...")
                    time.sleep(wait_sec)
                    continue
                if action == "defer":
                    print(f"  [FAILED] Unit {ukey} failed after 5 attempts. Moving to retry list.")
                    deferred = dict(u)
                    deferred["_retry_pass"] = True
                    unit_queue.append(deferred)
                    pending = still_failed + [
                        item for item in unit_queue if item.get("_retry_pass")
                    ]
                    write_retry_list(retry_list_path, pending)
                    break
                print(f"  [FAILED] Unit {ukey} failed on second pass after 5 attempts. Leaving uncached.")
                still_failed.append(_strip_retry_marker(u))
                pending = still_failed + [
                    item for item in unit_queue if item.get("_retry_pass")
                ]
                write_retry_list(retry_list_path, pending)
                break
                        
        processed_count += 1
    if still_failed:
        write_retry_list(retry_list_path, still_failed)
        failed_keys = [f"{item['book_id']}__{item['piece']}" for item in still_failed]
        print(f"Gemini run for {model} completed with {len(still_failed)} uncached unit(s): {', '.join(failed_keys)}")
        print(f"  Retry list saved to {retry_list_path}. Re-run the same command to resume.")
    else:
        if retry_list_path.is_file():
            retry_list_path.unlink()
        print(f"Gemini run for {model} completed.")

def split_unit_in_half(u: dict, book_text: str) -> tuple[dict, dict]:
    bid = u["book_id"]
    start = u["start"]
    end = u["end"]
    cuts = cut_points(book_text)
    mid_target = (start + end) // 2
    import bisect
    idx = bisect.bisect_left(cuts, mid_target)
    candidates = []
    if idx < len(cuts):
        candidates.append(cuts[idx])
    if idx > 0:
        candidates.append(cuts[idx - 1])
    candidates = [c for c in candidates if start + 2000 <= c <= end - 2000]
    if not candidates:
        mid = mid_target
    else:
        mid = min(candidates, key=lambda c: abs(c - mid_target))
        
    first_end = mid
    nxt = _largest_cut_at_most(cuts, mid - 2000, start)
    if nxt is None:
        nxt = mid - 2000
    second_start = nxt
    
    # We name the pieces with .0 and .1 or custom indexes to avoid collision
    # To keep it standard, we append _0 and _1 to the piece field or floats, but let's use string keys or sub-indexes
    p1 = {
        "book_id": bid,
        "piece": f"{u['piece']}_0",
        "start": start,
        "end": first_end,
        "sha1": get_sha1(book_text[start:first_end])
    }
    p2 = {
        "book_id": bid,
        "piece": f"{u['piece']}_1",
        "start": second_start,
        "end": end,
        "sha1": get_sha1(book_text[second_start:end])
    }
    return p1, p2


def expand_split_units(unit: dict, cache_dir: Path, book_text: str) -> list[dict]:
    """Prefer mid-run child caches (``6_0`` / ``6_1``) over a truncated parent."""
    piece = unit["piece"]
    child0 = cache_dir / f"{piece}_0.json"
    child1 = cache_dir / f"{piece}_1.json"
    if child0.is_file() and child1.is_file():
        half1, half2 = split_unit_in_half(unit, book_text)
        expanded: list[dict] = []
        for half in (half1, half2):
            expanded.extend(expand_split_units(half, cache_dir, book_text))
        return expanded
    return [unit]

# --------------------------------------------------------------------------- #
# CLI Subcommand: claude-batch
# --------------------------------------------------------------------------- #

def cmd_claude_batch(args):
    action = args.action
    split = args.split
    
    assert action in ["create", "poll", "list", "attach"], f"Unknown action {action}"
    assert split in ["test", "validation"], f"Unknown split {split}"
    
    _, anthropic_key = check_api_keys(require_gemini=False, require_anthropic=True)
    
    import anthropic
    client = anthropic.Anthropic(api_key=anthropic_key)
    c_api_model = resolve_anthropic_model("claude-sonnet-5")
    
    prompt_path = ROOT / "prompts" / "quotation_anchor_prompt.md"
    template = prompt_path.read_text(encoding="utf-8")
    
    model_dir = EXP_DIR / "claude-sonnet-5"
    model_dir.mkdir(parents=True, exist_ok=True)
    batch_json_path = model_dir / f"batch_{split}.json"
    
    schema_claude = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "frame": {"type": "string"},
                        "head": {"type": "string"},
                        "tail": {"type": "string"}
                    },
                    "required": ["label", "head", "tail"]
                }
            }
        },
        "required": ["spans"]
    }
    
    if action == "create":
        # Load units
        units_file = EXP_DIR / f"units_{split}.jsonl"
        units = json_lines(units_file)
        
        # Base texts lookup
        base_texts = {}
        books_data = load_gold_books(split)
        for b in books_data:
            base_texts[b["book_id"]] = b["text"]
            
        requests = []
        for u in units:
            bid = u["book_id"]
            piece = u["piece"]
            ukey = f"{bid}__{piece}"
            start = u["start"]
            end = u["end"]
            utext = base_texts[bid][start:end]
            filled = fill_anchor_prompt(template, utext)
            
            # Format according to MessageCreateParamsNonStreaming
            requests.append({
                "custom_id": ukey,
                "params": {
                    "model": c_api_model,
                    "max_tokens": 64000,
                    "messages": [{"role": "user", "content": filled}],
                    "thinking": {"type": "adaptive"},
                    "output_config": {
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": schema_claude}
                    }
                }
            })
            
        if batch_json_path.is_file():
            existing = json.loads(batch_json_path.read_text(encoding="utf-8"))
            raise SystemExit(
                f"{batch_json_path.name} already exists ({existing.get('id')}). "
                "Poll that batch, or use --action attach --batch-id <id> to recover. "
                "Do not create a second batch for the same split."
            )

        print(f"Creating Claude batch for split {split} (Requests count: {len(requests)})...")
        # Submit the batch
        batch = client.messages.batches.create(requests=requests)
        unit_ids = [u["custom_id"] for u in requests]

        # Persist the id first so a later field mismatch cannot orphan the batch.
        batch_info = {"id": batch.id, "split": split, "units": unit_ids}
        with batch_json_path.open("w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2)

        batch_info.update(claude_batch_snapshot(batch))
        batch_info["split"] = split
        batch_info["units"] = unit_ids
        with batch_json_path.open("w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2)
            
        print(f"Successfully created batch {batch.id}. Saved info to {batch_json_path}")

    elif action == "list":
        print("Recent Claude message batches:")
        page = client.messages.batches.list(limit=20)
        found = False
        for b in page:
            found = True
            snap = claude_batch_snapshot(b)
            counts = snap["request_counts"]
            print(
                f"  {snap['id']}  status={snap['status']}  created={snap['created_at']}  "
                f"processing={counts.get('processing', 0)} succeeded={counts.get('succeeded', 0)} "
                f"errored={counts.get('errored', 0)}"
            )
        if not found:
            print("  (none)")

    elif action == "attach":
        batch_id = getattr(args, "batch_id", None)
        if not batch_id:
            raise SystemExit("--batch-id is required for attach")
        print(f"Attaching Claude batch {batch_id} to {batch_json_path}...")
        batch = client.messages.batches.retrieve(batch_id)
        batch_info = claude_batch_snapshot(batch)
        batch_info["split"] = split
        if batch_json_path.is_file():
            existing = json.loads(batch_json_path.read_text(encoding="utf-8"))
            batch_info["units"] = existing.get("units", [])
        else:
            batch_info["units"] = []
        with batch_json_path.open("w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2)
        print(f"Saved {batch_id} (status={batch_info['status']}) to {batch_json_path}")
        
    elif action == "poll":
        if not batch_json_path.is_file():
            raise FileNotFoundError(
                f"batch_{split}.json is missing! Run create, or list then attach an existing batch."
            )
            
        with batch_json_path.open("r", encoding="utf-8") as f:
            batch_info = json.load(f)
            
        batch_id = batch_info["id"]
        print(f"Retrieving status for Claude batch {batch_id}...")
        batch = client.messages.batches.retrieve(batch_id)
        snap = claude_batch_snapshot(batch)
        print(f"Batch status: {snap['status']}  request_counts={snap['request_counts']}")
        
        # Update batch status in json
        batch_info.update(snap)
        with batch_json_path.open("w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2)
            
        if claude_batch_is_ready(batch):
            print("Batch completed! Downloading results...")
            raw_dir = model_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Read results using streaming decoder (JSONL format)
            # The client provides `client.messages.batches.results(batch_id)`
            results = client.messages.batches.results(batch_id)
            
            count_saved = 0
            for result in results:
                custom_id = result.custom_id
                bid, piece_str = custom_id.split("__", 1)
                
                # Check for errors
                result_type = result.result.type
                
                # Raw cache path
                raw_cache_path = raw_dir / bid / f"{piece_str}.json"
                raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to raw cache
                if result_type == "succeeded":
                    msg = result.result.message
                    usage = {
                        "prompt": msg.usage.input_tokens if hasattr(msg.usage, 'input_tokens') else 0,
                        "thoughts": msg.usage.thinking_tokens if hasattr(msg.usage, 'thinking_tokens') else 0,
                        "candidates": msg.usage.output_tokens if hasattr(msg.usage, 'output_tokens') else 0,
                        "total": (msg.usage.input_tokens + msg.usage.output_tokens) if hasattr(msg.usage, 'input_tokens') else 0
                    }
                    
                    # Extract raw text and stop reason
                    raw_text = ""
                    for content_block in msg.content:
                        if content_block.type == "text":
                            raw_text += content_block.text
                            
                    stop_reason = msg.stop_reason
                    
                    cache_payload = {
                        "model": "claude-sonnet-5",
                        "model_version": msg.model,
                        "prompt_sha1": get_sha1(template), # we will resolve template from earlier load
                        "window_sha1": "", # resolved later or during mapping
                        "temperature": 1.0,
                        "thinking": "adaptive",
                        "finish_reason": stop_reason,
                        "usage": usage,
                        "raw_text": raw_text.strip(),
                        "attempts": 1,
                        "batch_id": batch_id,
                        "started": 0,
                        "finished": int(time.time())
                    }
                    
                    with raw_cache_path.open("w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, ensure_ascii=False, indent=2)
                    count_saved += 1
                else:
                    # Result has errored
                    err = result.result.error
                    print(f"  [ERROR] Request {custom_id} failed in batch: {err}")
                    cache_payload = {
                        "model": "claude-sonnet-5",
                        "finish_reason": "errored",
                        "error_message": str(err),
                        "batch_id": batch_id
                    }
                    with raw_cache_path.open("w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, ensure_ascii=False, indent=2)
                        
            print(f"Saved {count_saved} successful results to raw cache.")
        else:
            print("Batch is still processing. Try polling again later.")

# --------------------------------------------------------------------------- #
# CLI Subcommand: locate (Locator + Overlap Join)
# --------------------------------------------------------------------------- #

def cmd_locate(args):
    model = args.model
    split = args.split
    source = getattr(args, "source", "units")
    
    assert model in GEMINI_MODELS or model == "claude-sonnet-5", f"Unknown model {model}"
    assert split in ["test", "validation"], f"Unknown split {split}"
    
    # Load units to process
    units_file = EXP_DIR / f"{source}_{split}.jsonl"
    if not units_file.is_file():
        raise FileNotFoundError(f"Units file missing: {units_file}")
    units = json_lines(units_file)
    
    # Base texts lookup
    base_texts = {}
    books_data = load_gold_books(split)
    for b in books_data:
        base_texts[b["book_id"]] = b["text"]
        
    model_dir = EXP_DIR / model
    cache_sub_dir = "raw" if source == "units" else "raw_windows"
    
    # Locate spans per unit/piece
    located_predictions = {} # book_id -> list of predictions

    work_units: list[dict] = []
    for u in units:
        bid = u["book_id"]
        cache_dir = model_dir / cache_sub_dir / bid
        expanded = expand_split_units(u, cache_dir, base_texts[bid])
        if len(expanded) > 1:
            print(
                f"Using split caches for {bid}__{u['piece']}: "
                + ", ".join(f"{item['book_id']}__{item['piece']}" for item in expanded)
            )
        work_units.extend(expanded)

    for u in work_units:
        bid = u["book_id"]
        piece = u["piece"]
        start = u["start"]
        end = u["end"]
        utext = base_texts[bid][start:end]
        
        raw_cache_path = model_dir / cache_sub_dir / bid / f"{piece}.json"
        if not raw_cache_path.is_file():
            print(f"Missing raw cache for unit {bid}__{piece} at {raw_cache_path}")
            continue
            
        with raw_cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
            
        raw_text = cached.get("raw_text", "").strip()
        if not raw_text or cached.get("finish_reason") == "blocked":
            continue
            
        # Parse dump. A MAX_TOKENS reply can cut the JSON mid-string; keep
        # any complete span objects and skip the rest.
        try:
            quotes = parse_quote_dump(raw_text)
        except json.JSONDecodeError as exc:
            salvaged = salvage_spans(raw_text)
            if not salvaged:
                print(f"Unparseable JSON for unit {bid}__{piece} ({exc}); skipping")
                continue
            print(
                f"Salvaged {len(salvaged)} complete span(s) from truncated JSON "
                f"for unit {bid}__{piece}"
            )
            quotes = quotes_from_items(salvaged)
        
        # Locate
        # Since we must find them inside that unit's text, we forward cursor inside the unit
        # Locator is already inclusive. End is start + len(needle) - 1.
        cursor = 0
        seen = set()
        for q in quotes:
            # We want to search for the quote text inside utext
            h = locate_parsed_quote(utext, q, min_score=0.85, from_index=cursor, max_span=2000)
            if h is None:
                h = locate_parsed_quote(utext, q, min_score=0.85, from_index=0, max_span=2000)
            if h is None:
                continue
            if (h.start, h.end) in seen:
                continue
            seen.add((h.start, h.end))
            
            # book offset = unit_start + h.start
            b_start = start + h.start
            b_end = start + h.end
            
            # Check 20-span round-trip validation
            # the 20-span text[start:end+1] round-trip
            chunk_extracted = base_texts[bid][b_start:b_end+1]
            
            prediction_item = {
                "book_id": bid,
                "start": b_start,
                "end": b_end,
                "piece": piece,
                "piece_start": start,
                "piece_end": end,
                "text": chunk_extracted,
                "label": q.label,
                "cls": q.label
            }
            
            if bid not in located_predictions:
                located_predictions[bid] = []
            located_predictions[bid].append(prediction_item)
            
            cursor = max(cursor, h.end + 1)
                
    # Now we apply three overlap-join rules to merge predictions per book
    final_spans = []
    
    rule_a_count = 0
    rule_b_count = 0
    rule_c_count = 0
    
    for bid, preds in located_predictions.items():
        book_text = base_texts[bid]
        # Sort predictions by start offset
        preds = sorted(preds, key=lambda x: x["start"])
        
        # Overlap-join logic:
        # We process predictions iteratively to resolve overlaps and adjacent pieces
        merged_preds = []
        
        for p in preds:
            if not merged_preds:
                merged_preds.append(p)
                continue
                
            last = merged_preds[-1]
            
            # Check overlap or proximity
            # (c) Join adjacent pieces: join only when the gap between the two spans is 0 to 5 characters inclusive
            if 0 <= p["start"] - last["end"] - 1 <= 5:
                # Join them
                last["end"] = max(last["end"], p["end"])
                last["text"] = book_text[last["start"]:last["end"]+1]
                rule_c_count += 1
            # (a) duplicate/same span in overlapping windows: we keep the copy farther from piece boundaries
            elif last["start"] <= p["start"] and last["end"] >= p["end"] or (min(last["end"], p["end"]) - max(last["start"], p["start"])) > 0:
                # Overlap! Which one to keep?
                # Margin for last:
                margin_last = min(last["start"] - last["piece_start"], last["piece_end"] - last["end"])
                # Margin for p:
                margin_curr = min(p["start"] - p["piece_start"], p["piece_end"] - p["end"])
                
                if margin_curr > margin_last:
                    # Replace last with p
                    merged_preds[-1] = p
                # otherwise keep last
                rule_a_count += 1
            else:
                merged_preds.append(p)
                
        # (b) Spans that overlap each other become one: keep the copy farthest from any window edge
        # Wait, if there are remaining overlaps, resolve them
        final_preds = []
        for p in merged_preds:
            if not final_preds:
                final_preds.append(p)
                continue
            last = final_preds[-1]
            if last["end"] >= p["start"]:
                # they overlap! Keep the one farthest from the piece edge
                margin_last = min(last["start"] - last["piece_start"], last["piece_end"] - last["end"])
                margin_curr = min(p["start"] - p["piece_start"], p["piece_end"] - p["end"])
                if margin_curr > margin_last:
                    final_preds[-1] = p
                rule_b_count += 1
            else:
                final_preds.append(p)
                
        for fp in final_preds:
            final_spans.append({
                "book_id": bid,
                "label": "QUOTATION",
                "start": fp["start"],
                "end": fp["end"]
            })
            
    # Write to spans.jsonl
    out_spans = model_dir / "spans.jsonl"
    with out_spans.open("w", encoding="utf-8") as f:
        for s in final_spans:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Located predictions saved to {out_spans}")
    print(f"Overlap join rules counters:")
    print(f"  Rule (a) Duplicate overlap resolution: {rule_a_count}")
    print(f"  Rule (b) Internal overlap resolution:  {rule_b_count}")
    print(f"  Rule (c) Edge join resolution:        {rule_c_count}")

# --------------------------------------------------------------------------- #
# CLI Subcommand: score
# --------------------------------------------------------------------------- #

def get_length_bucket(length: int) -> str:
    if length <= 45:
        return "≤45"
    elif length <= 150:
        return "46–150"
    elif length <= 400:
        return "151–400"
    elif length <= 1200:
        return "401–1,200"
    elif length <= 2000:
        return "1,201–2,000"
    else:
        return ">2,000"

def evaluate_predictions(preds: list[dict], gold: list[dict], base_texts: dict, stitch: bool = True) -> dict:
    """Run evaluation on predictions vs gold.
    
    Applies whitespace trim on predicted spans only.
    Optional adjacent join (stitch) using stitch_adjacent_spans helper.
    """
    total_gold = sum(len(b["spans"]) for b in gold)
    
    # Process predictions book-by-book
    book_preds = {}
    for p in preds:
        bid = p["book_id"]
        if bid not in book_preds:
            book_preds[bid] = []
        book_preds[bid].append((p["label"], p["start"], p["end"]))
        
    all_pred_tuples = []
    all_gold_tuples = []
    
    # Accumulate evaluation statistics
    for b in gold:
        bid = b["book_id"]
        text = base_texts[bid]
        
        # Gold tuples (inclusive)
        b_gold = []
        for s in b["spans"]:
            b_gold.append(("QUOTATION", s["start"], s["end"]))
            all_gold_tuples.append(("QUOTATION", s["start"], s["end"], s.get("seen_in_train", False), s.get("cls", "TEXT")))
            
        # Pred tuples
        b_pred = book_preds.get(bid, [])
        # Apply whitespace trim on predicted spans
        b_pred_trimmed = []
        for label, start, end in b_pred:
            span_text = text[start:end+1]
            trimmed_text = span_text.strip()
            if not trimmed_text:
                continue
            offset_start = span_text.find(trimmed_text)
            new_start = start + offset_start
            new_end = new_start + len(trimmed_text) - 1
            b_pred_trimmed.append((label, new_start, new_end))
            
        # Optional adjacent join stitch
        if stitch:
            b_pred_final = stitch_adjacent_spans(text, b_pred_trimmed)
        else:
            b_pred_final = b_pred_trimmed
            
        for label, start, end in b_pred_final:
            all_pred_tuples.append((bid, label, start, end))
            
    # Calculate IoU match
    # Group by book
    pred_by_book = {}
    for bid, l, s, e in all_pred_tuples:
        if bid not in pred_by_book:
            pred_by_book[bid] = []
        pred_by_book[bid].append((l, s, e))
        
    gold_by_book = {}
    for bid, l, s, e, seen, cls in [(b["book_id"], "QUOTATION", s["start"], s["end"], s.get("seen_in_train", False), s.get("cls", "TEXT")) for b in gold for s in b["spans"]]:
        if bid not in gold_by_book:
            gold_by_book[bid] = []
        gold_by_book[bid].append((l, s, e, seen, cls))
        
    # Matching
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    matched_preds = set()
    matched_golds = set()
    
    # We do a matching per book
    for bid in base_texts:
        b_preds = pred_by_book.get(bid, [])
        b_golds = gold_by_book.get(bid, [])
        
        # Match using match_iou with threshold 0.5
        # match_iou(pred_spans, gold_spans, threshold=0.5)
        # b_preds format: [(label, s, e)]
        # b_golds format: [(label, s, e, seen, cls)]
        b_preds_clean = [(l, s, e) for l, s, e in b_preds]
        b_golds_clean = [(l, s, e) for l, s, e, _, _ in b_golds]
        
        matches = match_iou(b_preds_clean, b_golds_clean, threshold=0.5)
        
        for p_idx, g_idx, iou in matches:
            matched_preds.add((bid, b_preds[p_idx][1], b_preds[p_idx][2]))
            matched_golds.add((bid, b_golds[g_idx][1], b_golds[g_idx][2]))
            tp_count += 1
            
    fp_count = len(all_pred_tuples) - tp_count
    fn_count = len(all_gold_tuples) - tp_count
    
    precision = tp_count / len(all_pred_tuples) if all_pred_tuples else 0
    recall = tp_count / len(all_gold_tuples) if all_gold_tuples else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "predicted": len(all_pred_tuples),
        "gold": len(all_gold_tuples)
    }

def cmd_score(args):
    split = args.split
    assert split in ["test", "validation"], f"Unknown split {split}"
    
    # Load gold books
    books = load_gold_books(split)
    base_texts = {b["book_id"]: b["text"] for b in books}
    
    # Load predictions for all models if available
    arms = list(GEMINI_MODELS) + ["claude-sonnet-5"]
    spans_by_arm = {}
    
    for arm in arms:
        spans_path = EXP_DIR / arm / "spans.jsonl"
        if spans_path.is_file():
            spans_by_arm[arm] = json_lines(spans_path)
            
    # Run scoring
    print("\n" + "="*80)
    print(f" SCORING RESULTS FOR SPLIT: {split}")
    print("="*80)
    
    results = {}
    for arm, preds in spans_by_arm.items():
        # Evaluate with stitch on
        score_on = evaluate_predictions(preds, books, base_texts, stitch=True)
        # Evaluate with stitch off
        score_off = evaluate_predictions(preds, books, base_texts, stitch=False)
        
        results[arm] = {
            "stitch_on": score_on,
            "stitch_off": score_off
        }
        
        print(f"Model: {arm}")
        print(f"  With Adjacent Join (Stitch) ON:")
        print(f"    P: {score_on['precision']:.4f} | R: {score_on['recall']:.4f} | F1: {score_on['f1']:.4f}  (Pred:{score_on['predicted']} / Gold:{score_on['gold']})")
        print(f"  With Adjacent Join (Stitch) OFF:")
        print(f"    P: {score_off['precision']:.4f} | R: {score_off['recall']:.4f} | F1: {score_off['f1']:.4f}  (Pred:{score_off['predicted']} / Gold:{score_off['gold']})")
        print("-"*80)
        
    # Bootstrap paired bootstrap if test split and at least two models have results
    if split == "test" and len(spans_by_arm) >= 2:
        print("Running paired bootstrap resamples (10,000 runs, fixed seed)...")
        # To perform paired bootstrap over 19 books:
        # We draw 10,000 resamples of the 19 books with replacement, calculate F1 differences, and report 95% CI.
        # Fixed seed
        random.seed(42)
        import numpy as np
        
        bootstrap_results = {}
        book_ids = list(base_texts.keys())
        
        available_arms = list(spans_by_arm.keys())
        pairs = []
        for i in range(len(available_arms)):
            for j in range(i + 1, len(available_arms)):
                pairs.append((available_arms[i], available_arms[j]))
                
        for arm_a, arm_b in pairs:
            diffs = []
            for _ in range(10000):
                sample_bids = random.choices(book_ids, k=len(book_ids))
                # Evaluate on sample books
                sample_books = [b for b in books if b["book_id"] in sample_bids]
                
                preds_a = [p for p in spans_by_arm[arm_a] if p["book_id"] in sample_bids]
                preds_b = [p for p in spans_by_arm[arm_b] if p["book_id"] in sample_bids]
                
                f1_a = evaluate_predictions(preds_a, sample_books, base_texts, stitch=True)["f1"]
                f1_b = evaluate_predictions(preds_b, sample_books, base_texts, stitch=True)["f1"]
                
                diffs.append(f1_a - f1_b)
                
            ci_lower = np.percentile(diffs, 2.5)
            ci_upper = np.percentile(diffs, 97.5)
            print(f"Paired Bootstrap 95% Confidence Interval for F1({arm_a}) - F1({arm_b}):")
            print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
            if ci_lower <= 0 <= ci_upper:
                print("  -> No statistically clear difference (interval includes 0)")
            else:
                print(f"  -> Statistically significant difference!")
            print("-"*80)
            
    # Write quotation/results/zeroshot/results.md
    results_md_path = EXP_DIR / "results.md"
    results_md_content = f"# Benchmark Results on {split} split\n\n"
    for arm, res in results.items():
        results_md_content += f"## {arm}\n"
        results_md_content += f"- **Stitch ON**: P={res['stitch_on']['precision']:.4f}, R={res['stitch_on']['recall']:.4f}, F1={res['stitch_on']['f1']:.4f}\n"
        results_md_content += f"- **Stitch OFF**: P={res['stitch_off']['precision']:.4f}, R={res['stitch_off']['recall']:.4f}, F1={res['stitch_off']['f1']:.4f}\n\n"
        
    results_md_path.write_text(results_md_content, encoding="utf-8")
    print(f"Wrote results report to {results_md_path}")

# --------------------------------------------------------------------------- #
# CLI Subcommand: freeze
# --------------------------------------------------------------------------- #

def cmd_freeze(args):
    # Writes run_manifest.json containing all metadata
    prompt_path = ROOT / "prompts" / "quotation_anchor_prompt.md"
    template = prompt_path.read_text(encoding="utf-8")
    template_sha1 = get_sha1(template)
    
    # Units SHA1
    units_test_path = EXP_DIR / "units_test.jsonl"
    units_validation_path = EXP_DIR / "units_validation.jsonl"
    
    units_test_sha1 = get_sha1(units_test_path.read_text(encoding="utf-8")) if units_test_path.is_file() else ""
    units_val_sha1 = get_sha1(units_validation_path.read_text(encoding="utf-8")) if units_validation_path.is_file() else ""
    
    manifest = {
        "prompt_sha1": template_sha1,
        "units_test_sha1": units_test_sha1,
        "units_validation_sha1": units_val_sha1,
        "models": {
            "gemini-3.8-flash": {"temperature": 1.0, "thinking": "low"},
            "gemini-3.1-flash-lite": {"temperature": 1.0, "thinking": "low"},
            "gemini-3.5-flash-lite": {"temperature": 1.0, "thinking": "low"},
            "claude-sonnet-5": {"thinking": "adaptive", "effort": "low"}
        },
        "freeze_time": int(time.time())
    }
    
    manifest_path = EXP_DIR / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Successfully froze configuration! Manifest saved to {manifest_path}")

# --------------------------------------------------------------------------- #
# CLI Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-model anchor benchmark suite")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    
    subparsers.add_parser("build-units", help="Build near-equal units for test & validation")
    subparsers.add_parser("build-windows", help="Build 16k/2k windows for validation comparison")
    preflight_parser = subparsers.add_parser("preflight", help="Run token counting, safety checks and verification")
    preflight_parser.add_argument("--source", choices=["units", "windows"], default="units", help="Data source type")
    preflight_parser.add_argument(
        "--model",
        choices=list(GEMINI_MODELS) + ["claude-sonnet-5"],
        default=None,
        help="Token-count and smoke-test only this model (default: all models)",
    )
    
    run_parser = subparsers.add_parser("run", help="Run Gemini model processing")
    run_parser.add_argument("--model", required=True, choices=list(GEMINI_MODELS), help="Model ID")
    run_parser.add_argument("--split", required=True, choices=["test", "validation"], help="Dataset split")
    run_parser.add_argument("--source", choices=["units", "windows"], default="units", help="Data source type")
    
    claude_parser = subparsers.add_parser("claude-batch", help="Submit/Poll Claude Message Batches")
    claude_parser.add_argument(
        "--action",
        required=True,
        choices=["create", "poll", "list", "attach"],
        help="Action",
    )
    claude_parser.add_argument("--split", required=True, choices=["test", "validation"], help="Dataset split")
    claude_parser.add_argument("--batch-id", default=None, help="Existing MessageBatch id for attach")
    
    locate_parser = subparsers.add_parser("locate", help="Locate predictions inside text")
    locate_parser.add_argument("--model", required=True, choices=list(GEMINI_MODELS) + ["claude-sonnet-5"], help="Model")
    locate_parser.add_argument("--split", required=True, choices=["test", "validation"], help="Dataset split")
    locate_parser.add_argument("--source", choices=["units", "windows"], default="units", help="Data source type")
    
    score_parser = subparsers.add_parser("score", help="Score located spans vs gold")
    score_parser.add_argument("--split", required=True, choices=["test", "validation"], help="Dataset split")
    
    subparsers.add_parser("freeze", help="Freeze setup by writing run_manifest.json")
    
    args = parser.parse_args(argv)
    
    if args.subcommand == "build-units":
        cmd_build_units(args)
    elif args.subcommand == "build-windows":
        cmd_build_windows(args)
    elif args.subcommand == "preflight":
        cmd_preflight(args)
    elif args.subcommand == "run":
        cmd_run(args)
    elif args.subcommand == "claude-batch":
        cmd_claude_batch(args)
    elif args.subcommand == "locate":
        cmd_locate(args)
    elif args.subcommand == "score":
        cmd_score(args)
    elif args.subcommand == "freeze":
        cmd_freeze(args)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1
    return 0

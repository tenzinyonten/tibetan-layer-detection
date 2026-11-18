"""Stage 3 JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from .review import needs_review


def _span_payload(s: dict, name: str, review_threshold: float | None,
                  ta_threshold: float | None = None) -> dict:
    """Builds one span's output dict. review_needed is true if its confidence
    is below that layer's threshold (threshold_for), OR -- when this span
    came from a --dual-window run -- the two window sizes disagreed on it
    (dual_window_agreement != "both"): disagreement is itself a reason to
    flag a span for review, independent of confidence."""
    out = {"start": s["start"], "end": s["end"],
          "confidence": None if s["confidence"] is None else round(s["confidence"], 4),
          "window_votes": s.get("window_votes", 1)}
    agreement = s.get("dual_window_agreement")
    if agreement is not None:
        out["dual_window_agreement"] = agreement
    for key in ("block", "model"):  # title/author extras
        if key in s:
            out[key] = s[key]
    out["review_needed"] = needs_review(s, name, review_threshold, ta_threshold)
    return out


def write_json(book_id: str, text_length: int, layers: dict[str, list[dict]],
              errors: dict[str, str], out_dir: Path, review_threshold: float | None,
              ta_threshold: float | None = None) -> Path:
    """review_threshold: a single value applied to every layer, or None to use
    each layer's own LAYERS[name].default_review_threshold instead (see
    threshold_for)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}.json"
    payload = {
        "book_id": book_id,
        "text_length": text_length,
        "layers": {name: [_span_payload(s, name, review_threshold, ta_threshold) for s in spans]
                  for name, spans in layers.items()},
    }
    if errors:
        payload["errors"] = errors
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

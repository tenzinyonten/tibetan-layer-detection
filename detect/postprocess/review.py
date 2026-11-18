"""Stage 3 review flags: which review threshold applies to a layer, and whether a span needs review."""

from __future__ import annotations

from ..layer_config import LAYERS, is_title_author


def threshold_for(layer_name: str, review_threshold: float | None,
                  ta_threshold: float | None = None) -> float:
    """review_threshold is the --review-threshold CLI value: a specific
    number applies to every layer alike (the pre-existing behavior), None
    (its new default) falls back to that layer's own
    LAYERS[name].default_review_threshold. For the title/author layers,
    ta_threshold (--title-author-review-threshold) wins over both."""
    if ta_threshold is not None and is_title_author(layer_name):
        return ta_threshold
    return LAYERS[layer_name].default_review_threshold if review_threshold is None else review_threshold


def needs_review(s: dict, name: str, review_threshold: float | None,
                 ta_threshold: float | None = None) -> bool:
    """A span is flagged when its confidence is below the layer's threshold, or
    (--dual-window) the two window sizes disagreed on it. A span with no
    confidence is flagged only if it carries review_needed itself."""
    if s.get("confidence") is None:
        return bool(s.get("review_needed"))
    return (s["confidence"] < threshold_for(name, review_threshold, ta_threshold)
            or s.get("dual_window_agreement") not in (None, "both"))

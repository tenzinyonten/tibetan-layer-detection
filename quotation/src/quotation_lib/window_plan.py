"""Overlap-first windowing for LLM span extraction.

Two ideas, both different from a fixed-stride chunker:

1.  The **overlap** is the parameter you set, not the stride. You choose the
    longest span you need to see whole (`overlap_chars`); the chunker then
    guarantees, by construction, that consecutive windows share at least that
    many characters. The stride is whatever falls out. A snap-to-shad step can
    never erode the guarantee, because the next window's start is *chosen from*
    the already-snapped end.

2.  Every cut lands just after a shad, and a window is never longer than
    `window_chars`, so the output-token budget is a hard ceiling you can assert
    on rather than an average you hope holds.

Coordinates are book-relative, half-open `[start, end)`.
Gold spans in `karma689/layer_detection` are inclusive `[start, end]`, so a gold
span sits whole inside a window iff `w.start <= g.start and g.end + 1 <= w.end`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# Tibetan clause/sentence terminators. All of these end a unit of text.
SHADS = "།༎༏༐༑༔"
_TRAILING = " \t\n\r་"  # spaces, newlines, tsheg


# --------------------------------------------------------------------------- #
# cut points
# --------------------------------------------------------------------------- #

def cut_points(text: str) -> list[int]:
    """Every legal place to cut: just after a shad (and any trailing space).

    Always includes 0 and len(text). Strictly increasing.
    """
    cuts = [0]
    n = len(text)
    i = 0
    while i < n:
        if text[i] in SHADS:
            j = i + 1
            while j < n and text[j] in _TRAILING:
                j += 1
            if j > cuts[-1]:
                cuts.append(j)
            i = j
        else:
            i += 1
    if cuts[-1] != n:
        cuts.append(n)
    return cuts


def _largest_cut_at_most(cuts: Sequence[int], limit: int, lo_exclusive: int) -> int | None:
    """Largest cut c with lo_exclusive < c <= limit, or None."""
    import bisect
    i = bisect.bisect_right(cuts, limit) - 1
    if i >= 0 and cuts[i] > lo_exclusive:
        return cuts[i]
    return None


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Window:
    chunk_id: str
    book_id: str
    start: int          # book offset, inclusive
    end: int            # book offset, exclusive
    text: str

    @property
    def n_chars(self) -> int:
        return self.end - self.start


def make_windows(
    text: str,
    book_id: str,
    window_chars: int,
    overlap_chars: int,
) -> list[Window]:
    """Cut a book into shad-aligned windows with a guaranteed minimum overlap.

    window_chars   hard ceiling on window length (drives the token budget)
    overlap_chars  guaranteed shared characters between consecutive windows
                   -> every span up to this length appears whole somewhere

    Raises ValueError if overlap_chars >= window_chars.
    """
    if overlap_chars >= window_chars:
        raise ValueError("overlap_chars must be < window_chars")

    n = len(text)
    if n == 0:
        return []
    cuts = cut_points(text)

    windows: list[Window] = []
    pos = 0
    idx = 0
    while True:
        # End of this window: the last shad at or before pos + window_chars.
        if pos + window_chars >= n:
            end = n
        else:
            end = _largest_cut_at_most(cuts, pos + window_chars, pos)
            # If the only shad in range sits inside the overlap zone, snapping
            # to it would make a window too short to advance. A shad-free run
            # that long is rare; take the hard cut instead.
            if end is None or end - pos <= overlap_chars:
                end = pos + window_chars

        windows.append(Window(
            chunk_id=f"{book_id}_{idx:05d}",
            book_id=book_id,
            start=pos,
            end=end,
            text=text[pos:end],
        ))
        if end >= n:
            break

        # Next window starts at the last shad that still leaves `overlap_chars`
        # of shared text. Choosing it FROM the snapped end is what makes the
        # overlap guarantee survive snapping.
        target = end - overlap_chars              # > pos, guaranteed above
        nxt = _largest_cut_at_most(cuts, target, pos)
        if nxt is None:                           # no shad -> hard cut at exactly
            nxt = target                          # the required overlap
        pos = nxt
        idx += 1

    return windows


# --------------------------------------------------------------------------- #
# sizing from an output-token budget
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WindowPlan:
    window_chars: int
    overlap_chars: int
    binding_constraint: str
    budget_ceiling_chars: int
    worst_case_output_tokens: int


def plan_windows(
    *,
    containment_chars: int,
    max_output_tokens: int = 65_000,
    tokens_per_char: float = 2.0,
    thinking_tokens: int = 8_000,
    json_overhead_tokens_per_span: int = 20,
    max_spans_per_window: int = 60,
    safety_factor: float = 0.7,
    density: float = 1.0,
) -> WindowPlan:
    """Pick window and overlap sizes that satisfy BOTH constraints.

    containment_chars   longest span you require to be seen whole (e.g. joint p99)
    density             fraction of a window that may be span text in the worst
                        case. Leave at 1.0 -- windows that are 100% quotation
                        exist in this corpus. The corpus *mean* is ~6%, which is
                        not a safety bound.
    safety_factor       how much of the budget you are willing to spend.
    """
    reserved = thinking_tokens + json_overhead_tokens_per_span * max_spans_per_window
    usable = max_output_tokens * safety_factor - reserved
    if usable <= 0:
        raise ValueError("reserved tokens exceed the budget")
    budget_ceiling = int(usable / (tokens_per_char * density))

    quality_window = 2 * containment_chars      # 50% overlap at the target length
    window = min(quality_window, budget_ceiling)
    binding = "output-token budget" if budget_ceiling < quality_window else "span containment"

    worst_case = int(window * density * tokens_per_char) + reserved
    return WindowPlan(
        window_chars=window,
        overlap_chars=containment_chars,
        binding_constraint=binding,
        budget_ceiling_chars=budget_ceiling,
        worst_case_output_tokens=worst_case,
    )


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def check_windows(
    text: str,
    windows: Sequence[Window],
    spans: Iterable[Mapping[str, Any]] = (),
    layers: Sequence[str] = ("QUOTATION", "TSAWA", "YIGCHUNG", "SABCHE"),
) -> dict[str, Any]:
    """Structural checks plus per-layer whole-span containment."""
    n = len(text)
    problems: list[str] = []

    # identity, coverage, monotonicity, minimum overlap
    reach = 0
    min_overlap = None
    for i, w in enumerate(windows):
        if text[w.start:w.end] != w.text:
            problems.append(f"{w.chunk_id}: text != book slice")
        if w.start > reach:
            problems.append(f"{w.chunk_id}: gap before this window")
        if i:
            ov = windows[i - 1].end - w.start
            min_overlap = ov if min_overlap is None else min(min_overlap, ov)
            if w.start <= windows[i - 1].start:
                problems.append(f"{w.chunk_id}: no forward progress")
        reach = max(reach, w.end)
    if reach < n:
        problems.append("book not fully covered")

    starts = [w.start for w in windows]
    ends = [w.end for w in windows]

    per_layer: dict[str, dict[str, Any]] = {}
    for lab in layers:
        whole = cut = 0
        longest_cut = 0
        for g in spans:
            if str(g["label"]).upper() != lab:
                continue
            gs, ge = int(g["start"]), int(g["end"]) + 1
            ok = any(s <= gs and ge <= e for s, e in zip(starts, ends))
            if ok:
                whole += 1
            else:
                cut += 1
                longest_cut = max(longest_cut, ge - gs)
        per_layer[lab] = {"whole": whole, "cut": cut, "longest_cut": longest_cut}

    return {
        "n_windows": len(windows),
        "max_window_chars": max((w.n_chars for w in windows), default=0),
        "min_overlap_chars": min_overlap,
        "problems": problems,
        "layers": per_layer,
    }

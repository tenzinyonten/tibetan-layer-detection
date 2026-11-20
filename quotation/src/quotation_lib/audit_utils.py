"""Audit / clean helpers for OpenPecha annotation layers."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Modeled sequence-labeling targets (workflow §0 / §1) — 7 types, 15 BIO classes.
MODEL_LAYERS = (
    "Quotation",
    "Sabche",
    "Tsawa",
    "Yigchung",
    "Chapter",
    "Author",
    "BookTitle",
)
EXCLUDED_LAYERS = ("Citation",)

# Token-label priority when two modeled layers overlap (workflow §2).
# Lower index wins. Author/BookTitle have zero observed overlap with others.
LAYER_PRIORITY = (
    "Quotation",
    "Sabche",
    "Tsawa",
    "Yigchung",
    "Chapter",
    "Author",
    "BookTitle",
)
PRIORITY_RANK = {L: i for i, L in enumerate(LAYER_PRIORITY)}

# meta.yml uses snake_case (and rare TitleCase) names.
META_TO_LAYER: dict[str, str] = {
    "quotation": "Quotation",
    "Quotation": "Quotation",
    "sabche": "Sabche",
    "Sabche": "Sabche",
    "tsawa": "Tsawa",
    "Tsawa": "Tsawa",
    "yigchung": "Yigchung",
    "Yigchung": "Yigchung",
    "chapter_title": "Chapter",
    "Chapter": "Chapter",
    "citation": "Citation",
    "Citation": "Citation",
    "author": "Author",
    "Author": "Author",
    "book_title": "BookTitle",
    "BookTitle": "BookTitle",
}

# UUID-keyed annotations (most Quotation/Sabche/… layers).
ANNOTATION_BLOCK_RE = re.compile(
    r"^  (?P<ann_id>[a-f0-9]+):\n"
    r"    span:\n"
    r"      start: (?P<start>-?\d+)\n"
    r"      end: (?P<end>-?\d+)",
    re.MULTILINE,
)
# List-style annotations (most Author / BookTitle layers).
LIST_SPAN_RE = re.compile(
    r"^- span:\n"
    r"    start: (?P<start>-?\d+)\n"
    r"    end: (?P<end>-?\d+)",
    re.MULTILINE,
)

# How much border text to show in the report / CSV.
BORDER_RADIUS = 60
# Cap gap text so inverted-empty gaps with distant neighbors stay readable.
GAP_CAP = 120
# How many representative drop cases to embed per pattern in the markdown report.
REPORT_SAMPLES_PER_PATTERN = 8


def span_length(start: int, end: int) -> int:
    """Inclusive OpenPecha [start, end] character count (= end - start + 1)."""
    return end - start + 1


@dataclass
class SpanHit:
    book: str
    layer: str
    ann_id: str
    start: int
    end: int
    line: int  # 1-based line of the annotation id in the YAML
    layer_path: Path
    base_path: Path | None
    doc_len: int
    reasons: list[str] = field(default_factory=list)
    action: str = "keep"  # keep | drop | fix | spot_check | note | exclude
    fixed_start: int | None = None
    fixed_end: int | None = None
    note: str = ""
    category: str = "invalid"  # invalid | long_outlier | meta_mismatch | excluded
    # Neighbor / border context for spot-checking drop decisions.
    prev_start: int | None = None
    prev_end: int | None = None
    next_start: int | None = None
    next_end: int | None = None
    neighbor_gap: int | None = None  # chars between prev.end+1 and next.start
    border_before: str = ""
    border_after: str = ""
    gap_text: str = ""
    start_border: str = ""  # for long outliers: text around start
    end_border: str = ""  # for long outliers: text around end

    @property
    def length(self) -> int:
        return span_length(self.start, self.end)

    @property
    def pattern(self) -> str:
        """Stable key for grouping edge cases in the report."""
        if self.category == "long_outlier":
            return "long_outlier"
        if self.note:
            return self.note
        return "|".join(self.reasons) or self.category


@dataclass
class OverlapHit:
    """One pair of valid modeled spans that share at least one character."""

    book: str
    layer_a: str
    ann_id_a: str
    start_a: int
    end_a: int
    line_a: int
    layer_b: str
    ann_id_b: str
    start_b: int
    end_b: int
    line_b: int
    overlap_start: int
    overlap_end: int
    overlap_chars: int
    relation: str  # exact | a_contains_b | b_contains_a | partial
    priority_winner: str
    text_a: str = ""  # full span A text
    text_b: str = ""  # full span B text
    overlap_text: str = ""  # full shared region
    border_before: str = ""
    border_after: str = ""

    @property
    def pair_key(self) -> str:
        return f"{self.layer_a}|{self.layer_b}"


@dataclass
class BookRecord:
    book: str
    doc_len: int
    meta_path: Path | None
    base_path: Path | None
    declared_layers: set[str] = field(default_factory=set)
    layer_files: dict[str, Path] = field(default_factory=dict)
    span_counts: dict[str, int] = field(default_factory=dict)  # valid kept after clean
    raw_span_counts: dict[str, int] = field(default_factory=dict)
    invalid_counts: dict[str, int] = field(default_factory=dict)
    # Cleaned spans used for tokenization: layer → [(ann_id, start, end, line), ...]
    cleaned_spans: dict[str, list[tuple[str, int, int, int]]] = field(default_factory=dict)


def find_books(repos_dir: Path) -> list[Path]:
    return sorted(p for p in repos_dir.iterdir() if p.is_dir() and re.fullmatch(r"P\d+", p.name))


def parse_meta_layers(meta_path: Path) -> set[str]:
    """Return modeled+known layer names declared in meta.yml."""
    text = meta_path.read_text(encoding="utf-8")
    if "layers:" not in text:
        return set()
    after = text.split("layers:", 1)[1]
    declared: set[str] = set()
    for line in after.splitlines()[1:]:
        if line.startswith("  - "):
            raw = line.strip()[2:].strip()
            if raw in META_TO_LAYER:
                declared.add(META_TO_LAYER[raw])
            continue
        if not line.strip():
            continue
        if line.startswith("    "):
            continue
        if line.startswith("  ") and not line.startswith("  -"):
            break
        if line and not line.startswith(" "):
            break
    return declared


def parse_layer_spans(path: Path) -> list[tuple[str, int, int, int]]:
    """Return list of (ann_id, start, end, line_number).

    Supports UUID-keyed blocks (typical for Quotation/…) and list-style
    ``- span:`` blocks (typical for Author / BookTitle).
    """
    text = path.read_text(encoding="utf-8")
    hits: list[tuple[str, int, int, int]] = []
    for m in ANNOTATION_BLOCK_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        hits.append(
            (
                m.group("ann_id"),
                int(m.group("start")),
                int(m.group("end")),
                line,
            )
        )
    if hits:
        return hits
    for i, m in enumerate(LIST_SPAN_RE.finditer(text)):
        line = text.count("\n", 0, m.start()) + 1
        hits.append((f"idx{i}", int(m.group("start")), int(m.group("end")), line))
    return hits


def _clip(text: str, lo: int, hi: int) -> str:
    return text[max(0, lo) : min(len(text), hi)].replace("\n", "⏎")


def _truncate_middle(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    half = (cap - 1) // 2
    return s[:half] + "…" + s[-half:]


def enrich_border_context(
    hit: SpanHit,
    base_text: str,
    prev: tuple[int, int] | None,
    nxt: tuple[int, int] | None,
) -> None:
    """Attach neighbor offsets and readable border/gap text for spot-checks."""
    if prev is not None:
        hit.prev_start, hit.prev_end = prev
    if nxt is not None:
        hit.next_start, hit.next_end = nxt

    if hit.category == "long_outlier":
        hit.start_border = _clip(base_text, hit.start - BORDER_RADIUS, hit.start + BORDER_RADIUS)
        hit.end_border = _clip(base_text, hit.end - BORDER_RADIUS, hit.end + BORDER_RADIUS)
        return

    # Invalid / drop cases: focus on the border where content would sit.
    # For inverted empty (start == end + 1), the "point" is between end and start.
    point = min(hit.start, hit.end) if hit.start >= 0 and hit.end >= 0 else 0
    if "start_gt_end" in hit.reasons:
        # Boundary sits between end and start (empty/inverted).
        point = hit.end

    hit.border_before = _clip(base_text, point - BORDER_RADIUS, point + 1)
    hit.border_after = _clip(base_text, point + 1, point + 1 + BORDER_RADIUS)

    if prev is not None and nxt is not None:
        gap_lo = prev[1] + 1
        gap_hi = nxt[0]
        hit.neighbor_gap = max(0, gap_hi - gap_lo)
        raw_gap = _clip(base_text, gap_lo, gap_hi)
        hit.gap_text = _truncate_middle(raw_gap, GAP_CAP)
    elif prev is not None:
        gap_lo = prev[1] + 1
        gap_hi = min(len(base_text), gap_lo + GAP_CAP)
        hit.neighbor_gap = None
        hit.gap_text = _clip(base_text, gap_lo, gap_hi)
    elif nxt is not None:
        gap_hi = nxt[0]
        gap_lo = max(0, gap_hi - GAP_CAP)
        hit.neighbor_gap = None
        hit.gap_text = _clip(base_text, gap_lo, gap_hi)


def classify_span(
    start: int,
    end: int,
    doc_len: int,
) -> tuple[list[str], str, int | None, int | None, str]:
    """Return (reasons, action, fixed_start, fixed_end, note)."""
    reasons: list[str] = []
    if start < 0 or end < 0:
        reasons.append("negative_offset")
    if start > end:
        reasons.append("start_gt_end")
    if doc_len >= 0 and (start >= doc_len or end >= doc_len):
        if "oob" not in reasons:
            reasons.append("oob")

    if not reasons:
        return [], "keep", None, None, ""

    if reasons == ["start_gt_end"] and start == end + 1:
        # Inverted empty annotation — no recoverable quoted content between neighbors.
        return reasons, "drop", None, None, "inverted_empty_start_eq_end_plus_1"
    if "negative_offset" in reasons and start < 0 and end < 0:
        return reasons, "drop", None, None, "both_offsets_negative"
    if reasons == ["oob"] and 0 <= start < end < doc_len + 1:
        if end >= doc_len and start < doc_len:
            return reasons, "fix", start, doc_len - 1, "clamp_end_to_doc_len_minus_1"

    if "start_gt_end" in reasons and start > end + 1:
        fs, fe = end, start
        if 0 <= fs < fe < doc_len:
            return reasons, "fix", fs, fe, "swap_start_end"
        return reasons, "drop", None, None, "inverted_unrecoverable"

    return reasons, "drop", None, None, "unrecoverable"


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """True when inclusive spans share ≥1 interior-overlapping character.

    OpenPecha offsets are inclusive on both ends. Adjacent spans that only
    meet at a boundary index (``a.end == b.start``) are NOT an overlap —
    e.g. A=[10,20] and B=[20,30] do not count. Nested / partial ranges do.
    """
    return start_a < end_b and start_b < end_a


def overlap_relation(start_a: int, end_a: int, start_b: int, end_b: int) -> str:
    if start_a == start_b and end_a == end_b:
        return "exact"
    if start_a <= start_b and end_a >= end_b:
        return "a_contains_b"
    if start_b <= start_a and end_b >= end_a:
        return "b_contains_a"
    return "partial"


def priority_winner(layer_a: str, layer_b: str) -> str:
    ra = PRIORITY_RANK.get(layer_a, 99)
    rb = PRIORITY_RANK.get(layer_b, 99)
    return layer_a if ra <= rb else layer_b


def find_book_overlaps(
    book: str,
    valid_by_layer: dict[str, list[tuple[str, int, int, int]]],
    base_text: str,
) -> list[OverlapHit]:
    """Find all cross-layer overlaps among valid spans for one book."""
    hits: list[OverlapHit] = []
    layers_present = [L for L in MODEL_LAYERS if valid_by_layer.get(L)]
    for i, layer_a in enumerate(layers_present):
        a_spans = valid_by_layer[layer_a]
        for layer_b in layers_present[i + 1 :]:
            b_spans = valid_by_layer[layer_b]
            # Sweep: sort both by start.
            a_sorted = sorted(a_spans, key=lambda s: (s[1], s[2]))
            b_sorted = sorted(b_spans, key=lambda s: (s[1], s[2]))
            j = 0
            for ann_a, start_a, end_a, line_a in a_sorted:
                while j < len(b_sorted) and b_sorted[j][2] <= start_a:
                    j += 1
                k = j
                while k < len(b_sorted) and b_sorted[k][1] < end_a:
                    ann_b, start_b, end_b, line_b = b_sorted[k]
                    if spans_overlap(start_a, end_a, start_b, end_b):
                        ov_start = max(start_a, start_b)
                        ov_end = min(end_a, end_b)
                        ov_chars = span_length(ov_start, ov_end)
                        hit = OverlapHit(
                            book=book,
                            layer_a=layer_a,
                            ann_id_a=ann_a,
                            start_a=start_a,
                            end_a=end_a,
                            line_a=line_a,
                            layer_b=layer_b,
                            ann_id_b=ann_b,
                            start_b=start_b,
                            end_b=end_b,
                            line_b=line_b,
                            overlap_start=ov_start,
                            overlap_end=ov_end,
                            overlap_chars=ov_chars,
                            relation=overlap_relation(start_a, end_a, start_b, end_b),
                            priority_winner=priority_winner(layer_a, layer_b),
                        )
                        if base_text:
                            # Inclusive end → slice through end+1. No truncation.
                            hit.text_a = _clip(base_text, start_a, end_a + 1)
                            hit.text_b = _clip(base_text, start_b, end_b + 1)
                            hit.overlap_text = _clip(base_text, ov_start, ov_end + 1)
                            hit.border_before = _clip(
                                base_text, ov_start - BORDER_RADIUS, ov_start
                            )
                            hit.border_after = _clip(
                                base_text, ov_end + 1, ov_end + 1 + BORDER_RADIUS
                            )
                        hits.append(hit)
                    k += 1
    return hits


def load_book(book_dir: Path) -> BookRecord:
    book = book_dir.name
    opf = next(book_dir.glob("*.opf"), None)
    meta_path = (opf / "meta.yml") if opf and (opf / "meta.yml").is_file() else None
    base_candidates = list((opf / "base").glob("*.txt")) if opf else []
    base_path = base_candidates[0] if base_candidates else None
    doc_len = len(base_path.read_text(encoding="utf-8")) if base_path else -1
    declared = parse_meta_layers(meta_path) if meta_path else set()

    layer_files: dict[str, Path] = {}
    if opf:
        for yml in (opf / "layers").glob("*/*.yml"):
            layer_files[yml.stem] = yml

    return BookRecord(
        book=book,
        doc_len=doc_len,
        meta_path=meta_path,
        base_path=base_path,
        declared_layers=declared,
        layer_files=layer_files,
    )


def audit_corpus(
    repos_dir: Path,
    long_quote_min: int,
) -> tuple[list[BookRecord], list[SpanHit], list[OverlapHit]]:
    books: list[BookRecord] = []
    flagged: list[SpanHit] = []
    overlaps: list[OverlapHit] = []

    for book_dir in find_books(repos_dir):
        rec = load_book(book_dir)
        books.append(rec)
        base_text = (
            rec.base_path.read_text(encoding="utf-8") if rec.base_path and rec.base_path.is_file() else ""
        )
        valid_by_layer: dict[str, list[tuple[str, int, int, int]]] = {}

        for layer in MODEL_LAYERS:
            has_file = layer in rec.layer_files
            declared = layer in rec.declared_layers
            if declared and not has_file:
                flagged.append(
                    SpanHit(
                        book=rec.book,
                        layer=layer,
                        ann_id="",
                        start=-1,
                        end=-1,
                        line=1,
                        layer_path=rec.meta_path or book_dir / "missing",
                        base_path=rec.base_path,
                        doc_len=rec.doc_len,
                        reasons=["meta_declares_missing_file"],
                        action="note",
                        note="meta.yml lists layer but no YAML file on disk",
                        category="meta_mismatch",
                    )
                )
            if has_file and not declared:
                flagged.append(
                    SpanHit(
                        book=rec.book,
                        layer=layer,
                        ann_id="",
                        start=-1,
                        end=-1,
                        line=1,
                        layer_path=rec.layer_files[layer],
                        base_path=rec.base_path,
                        doc_len=rec.doc_len,
                        reasons=["file_without_meta_declaration"],
                        action="note",
                        note="layer YAML present but meta.yml does not declare it",
                        category="meta_mismatch",
                    )
                )

        for layer in MODEL_LAYERS:
            path = rec.layer_files.get(layer)
            if not path:
                rec.raw_span_counts[layer] = 0
                rec.span_counts[layer] = 0
                rec.invalid_counts[layer] = 0
                continue

            spans = parse_layer_spans(path)
            rec.raw_span_counts[layer] = len(spans)
            kept = 0
            invalid = 0
            valid_spans: list[tuple[str, int, int, int]] = []
            for i, (ann_id, start, end, line) in enumerate(spans):
                reasons, action, fs, fe, note = classify_span(start, end, rec.doc_len)
                is_long = (
                    layer == "Quotation"
                    and not reasons
                    and span_length(start, end) >= long_quote_min
                )

                prev = (spans[i - 1][1], spans[i - 1][2]) if i > 0 else None
                nxt = (spans[i + 1][1], spans[i + 1][2]) if i + 1 < len(spans) else None

                if reasons:
                    invalid += 1
                    hit = SpanHit(
                        book=rec.book,
                        layer=layer,
                        ann_id=ann_id,
                        start=start,
                        end=end,
                        line=line,
                        layer_path=path,
                        base_path=rec.base_path,
                        doc_len=rec.doc_len,
                        reasons=reasons,
                        action=action,
                        fixed_start=fs,
                        fixed_end=fe,
                        note=note,
                        category="invalid",
                    )
                    if base_text:
                        enrich_border_context(hit, base_text, prev, nxt)
                    flagged.append(hit)
                    if action == "fix" and fs is not None and fe is not None:
                        kept += 1
                        valid_spans.append((ann_id, fs, fe, line))
                else:
                    kept += 1
                    valid_spans.append((ann_id, start, end, line))
                    if is_long:
                        hit = SpanHit(
                            book=rec.book,
                            layer=layer,
                            ann_id=ann_id,
                            start=start,
                            end=end,
                            line=line,
                            layer_path=path,
                            base_path=rec.base_path,
                            doc_len=rec.doc_len,
                            reasons=["long_outlier"],
                            action="spot_check",
                            note=f"length={span_length(start, end)}; confirm real quote not merge",
                            category="long_outlier",
                        )
                        if base_text:
                            enrich_border_context(hit, base_text, prev, nxt)
                        flagged.append(hit)

            rec.span_counts[layer] = kept
            rec.invalid_counts[layer] = invalid
            if valid_spans:
                valid_by_layer[layer] = valid_spans
                rec.cleaned_spans[layer] = valid_spans

        overlaps.extend(find_book_overlaps(rec.book, valid_by_layer, base_text))

        if "Citation" in rec.layer_files:
            path = rec.layer_files["Citation"]
            n = len(parse_layer_spans(path))
            flagged.append(
                SpanHit(
                    book=rec.book,
                    layer="Citation",
                    ann_id="",
                    start=-1,
                    end=-1,
                    line=1,
                    layer_path=path,
                    base_path=rec.base_path,
                    doc_len=rec.doc_len,
                    reasons=["excluded_layer"],
                    action="exclude",
                    note=f"{n} spans excluded from modeling (Citation)",
                    category="excluded",
                )
            )

    return books, flagged, overlaps


def in_scope_layers(rec: BookRecord, mask_source: str) -> set[str]:
    """Layers for which this book provides valid negative (O) evidence.

    ``confirmed`` implements the corpus owner's stated semantics: an
    undeclared layer was assessed as not applicable and is therefore a valid
    negative. Only declared-but-missing files are excluded because those
    mismatches do not prove that assessment happened.

    Declared-but-missing-file mismatches (e.g. P000010 Quotation, P000100 Tsawa)
    are never in-scope — there is no evidence the layer was actually checked.
    """
    declared = rec.declared_layers & set(MODEL_LAYERS)
    present = set(rec.layer_files) & set(MODEL_LAYERS)
    if mask_source == "confirmed":
        return set(MODEL_LAYERS) - (declared - present)
    if mask_source == "meta":
        return declared & present
    if mask_source == "file":
        return present
    return declared | present


def write_cleaned_layers(
    books: list[BookRecord],
    flagged: list[SpanHit],
    repos_dir: Path,
    cleaned_dir: Path,
) -> int:
    """Write cleaned MODEL layer YAMLs; drop/fix per flagged actions. Return files written."""
    drop_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    fixes: dict[tuple[str, str], dict[str, tuple[int, int]]] = defaultdict(dict)
    for hit in flagged:
        if hit.category != "invalid" or not hit.ann_id:
            continue
        key = (hit.book, hit.layer)
        if hit.action == "drop":
            drop_ids[key].add(hit.ann_id)
        elif hit.action == "fix" and hit.fixed_start is not None and hit.fixed_end is not None:
            fixes[key][hit.ann_id] = (hit.fixed_start, hit.fixed_end)

    written = 0
    for rec in books:
        for layer in MODEL_LAYERS:
            src = rec.layer_files.get(layer)
            if not src:
                continue
            key = (rec.book, layer)
            to_drop = drop_ids.get(key, set())
            to_fix = fixes.get(key, {})
            text = src.read_text(encoding="utf-8")
            if to_drop or to_fix:
                if ANNOTATION_BLOCK_RE.search(text):

                    def uuid_replacer(m: re.Match[str]) -> str:
                        ann_id = m.group("ann_id")
                        if ann_id in to_drop:
                            return ""
                        if ann_id in to_fix:
                            fs, fe = to_fix[ann_id]
                            return (
                                f"  {ann_id}:\n"
                                f"    span:\n"
                                f"      start: {fs}\n"
                                f"      end: {fe}"
                            )
                        return m.group(0)

                    text = ANNOTATION_BLOCK_RE.sub(uuid_replacer, text)
                else:
                    # List-style: drop/fix by synthetic idx{i} ids from parse_layer_spans.
                    pieces: list[str] = []
                    last = 0
                    for i, m in enumerate(LIST_SPAN_RE.finditer(text)):
                        ann_id = f"idx{i}"
                        pieces.append(text[last : m.start()])
                        if ann_id in to_drop:
                            pass
                        elif ann_id in to_fix:
                            fs, fe = to_fix[ann_id]
                            pieces.append(f"- span:\n    start: {fs}\n    end: {fe}")
                        else:
                            pieces.append(m.group(0))
                        last = m.end()
                    pieces.append(text[last:])
                    text = "".join(pieces)
                text = re.sub(r"\n{3,}", "\n\n", text)

            rel = src.relative_to(repos_dir)
            dest = cleaned_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            written += 1
    return written


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_neighbor(hit: SpanHit) -> str:
    parts = []
    if hit.prev_start is not None and hit.prev_end is not None:
        parts.append(f"prev `[{hit.prev_start}:{hit.prev_end}]`")
    else:
        parts.append("prev —")
    if hit.next_start is not None and hit.next_end is not None:
        parts.append(f"next `[{hit.next_start}:{hit.next_end}]`")
    else:
        parts.append("next —")
    if hit.neighbor_gap is not None:
        parts.append(f"gap={hit.neighbor_gap} chars")
    return " · ".join(parts)


def write_report(
    out_dir: Path,
    books: list[BookRecord],
    flagged: list[SpanHit],
    overlaps: list[OverlapHit],
    masks: dict[str, set[str]],
    long_quote_min: int,
) -> None:
    invalid = [h for h in flagged if h.category == "invalid"]
    longs = [h for h in flagged if h.category == "long_outlier"]
    by_action = Counter(h.action for h in invalid)
    by_pattern = Counter(h.pattern for h in invalid)
    by_pair = Counter(h.pair_key for h in overlaps)
    by_relation = Counter(h.relation for h in overlaps)

    lines: list[str] = []
    lines.append("# Annotation audit report\n")
    lines.append(
        "Spot-check the **border context** below — not the empty/inverted span itself. "
        "For drops, confirm the gap between neighbors has no recoverable quoted content. "
        "For long outliers, confirm start/end borders look like a real quote, not a merge. "
        "For overlaps, confirm which layer should win at the shared characters.\n"
    )
    lines.append(f"- Books scanned: **{len(books)}**")
    lines.append(f"- Invalid modeled spans flagged: **{len(invalid)}**")
    lines.append(f"- Long Quotation outliers (≥{long_quote_min} chars): **{len(longs)}**")
    lines.append(f"- Cross-layer overlapping pairs: **{len(overlaps)}**")
    lines.append(f"- Actions overall: `{dict(by_action)}`")
    lines.append("")

    lines.append("## Invalid spans by layer\n")
    lines.append("| Layer | Invalid | Drop | Fix |")
    lines.append("|---|---:|---:|---:|")
    for layer in MODEL_LAYERS:
        subset = [h for h in invalid if h.layer == layer]
        n_drop = sum(1 for h in subset if h.action == "drop")
        n_fix = sum(1 for h in subset if h.action == "fix")
        note = " already clean" if not subset and layer in ("Sabche", "Tsawa") else ""
        lines.append(f"| {layer} | {len(subset)} | {n_drop} | {n_fix} |{note}")
    lines.append("")

    lines.append("## Edge-case patterns (what to verify)\n")
    lines.append("| Pattern | Count | Proposed action | What to look at |")
    lines.append("|---|---:|---|---|")
    pattern_hints = {
        "inverted_empty_start_eq_end_plus_1": (
            "drop",
            "Neighbors leave a gap; inverted span marks nothing. Confirm gap is commentary/glue, not a missed quote.",
        ),
        "both_offsets_negative": ("drop", "Both offsets negative — no recoverable span."),
        "inverted_unrecoverable": ("drop", "start>end by >1 and swap still OOB."),
        "swap_start_end": ("fix", "Swapped offsets land in-bounds — confirm that range is the intended span."),
        "clamp_end_to_doc_len_minus_1": ("fix", "End clamped to last char — check end border."),
        "unrecoverable": ("drop", "No clear recovery — inspect border_before / border_after."),
    }
    for pattern, count in by_pattern.most_common():
        action, hint = pattern_hints.get(pattern, ("?", "Inspect border context."))
        lines.append(f"| `{pattern}` | {count} | {action} | {hint} |")
    lines.append("")

    # --- Representative drop/fix spot-checks with border context ---
    lines.append("## Spot-check: proposed drops / fixes\n")
    lines.append(
        f"Up to {REPORT_SAMPLES_PER_PATTERN} examples per pattern. "
        f"`⟦⟧` marks the decision point (between inverted end/start). "
        f"Full table: `flagged_spans.csv`.\n"
    )

    for pattern, _count in by_pattern.most_common():
        subset = [h for h in invalid if h.pattern == pattern]
        # Round-robin across layers so Chapter-first book ids don't dominate.
        by_layer: dict[str, list[SpanHit]] = defaultdict(list)
        for h in subset:
            by_layer[h.layer].append(h)
        samples: list[SpanHit] = []
        layers_cycle = [L for L in MODEL_LAYERS if L in by_layer]
        idx = {L: 0 for L in layers_cycle}
        while len(samples) < REPORT_SAMPLES_PER_PATTERN and layers_cycle:
            progressed = False
            for L in list(layers_cycle):
                i = idx[L]
                if i < len(by_layer[L]):
                    samples.append(by_layer[L][i])
                    idx[L] = i + 1
                    progressed = True
                    if len(samples) >= REPORT_SAMPLES_PER_PATTERN:
                        break
                else:
                    layers_cycle.remove(L)
            if not progressed:
                break

        lines.append(f"### `{pattern}` ({len(subset)} total)\n")
        for h in samples:
            lines.append(
                f"**{h.book}** `{h.layer}` offsets `{h.start}:{h.end}` → **{h.action}**  \n"
                f"{_fmt_neighbor(h)}  \n"
                f"- border: `…{h.border_before}⟦⟧{h.border_after}…`  \n"
                f"- gap between neighbors: `{h.gap_text or '—'}`"
            )
            lines.append("")
        if len(subset) > len(samples):
            lines.append(
                f"_… {len(subset) - len(samples)} more in "
                f"`flagged_spans.csv` (filter `note={pattern}`)._\n"
            )

    # --- Long quotation borders ---
    lines.append("## Spot-check: long Quotation borders\n")
    lines.append(
        "Only start/end borders are shown (not the full span). "
        "Ask: does this look like one continuous quote, or a merge of several?\n"
    )
    for h in sorted(longs, key=lambda x: -(x.end - x.start)):
        lines.append(
            f"**{h.book}** len={span_length(h.start, h.end)} `[{h.start}:{h.end}]`  \n"
            f"- start border: `…{h.start_border}…`  \n"
            f"- end border: `…{h.end_border}…`"
        )
        lines.append("")

    # --- Cross-layer overlaps ---
    lines.append("## Spot-check: cross-layer overlaps\n")
    lines.append(
        "Valid modeled spans that share ≥1 character. Full text of both spans and the "
        "shared region are shown (no truncation). At label time only one BIO tag can win — "
        f"proposed priority `{' > '.join(LAYER_PRIORITY)}`. "
        "Ask: is the shared text really both types, or a mis-annotation? "
        "Full table: `overlaps.csv`.\n"
    )
    lines.append("| Layer pair | Pairs | Relations |")
    lines.append("|---|---:|---|")
    for pair, count in by_pair.most_common():
        rels = Counter(h.relation for h in overlaps if h.pair_key == pair)
        rel_str = ", ".join(f"{r}={n}" for r, n in rels.most_common())
        lines.append(f"| `{pair}` | {count} | {rel_str} |")
    if not by_pair:
        lines.append("| — | 0 | — |")
    lines.append("")
    lines.append(f"Relation totals: `{dict(by_relation)}`\n")

    # Group by pair for readable spot-checks; list all (corpus has ~23).
    for pair, _count in by_pair.most_common():
        subset = [h for h in overlaps if h.pair_key == pair]
        lines.append(f"### `{pair}` ({len(subset)})\n")
        for h in sorted(subset, key=lambda x: (x.book, x.overlap_start)):
            len_a = span_length(h.start_a, h.end_a)
            len_b = span_length(h.start_b, h.end_b)
            lines.append(
                f"**{h.book}** `{h.relation}` · overlap={h.overlap_chars} chars "
                f"`[{h.overlap_start}:{h.overlap_end}]` · "
                f"priority → **{h.priority_winner}**\n"
            )
            lines.append(f"- `{h.layer_a}` `[{h.start_a}:{h.end_a}]` (len={len_a}):")
            lines.append(f"  ```")
            lines.append(f"  {h.text_a}")
            lines.append(f"  ```")
            lines.append(f"- `{h.layer_b}` `[{h.start_b}:{h.end_b}]` (len={len_b}):")
            lines.append(f"  ```")
            lines.append(f"  {h.text_b}")
            lines.append(f"  ```")
            lines.append(f"- shared `[{h.overlap_start}:{h.overlap_end}]`:")
            lines.append(f"  ```")
            lines.append(f"  {h.overlap_text}")
            lines.append(f"  ```")
            lines.append(
                f"- context: `…{h.border_before}⟦SHARED⟧{h.border_after}…`"
            )
            lines.append("")

    # --- Layer masks (kept; needed for later training steps) ---
    lines.append("## Layer masks\n")
    lines.append(
        "A book is valid negative (O) evidence for a layer only when that layer is in-scope "
        "(see `layer_masks.csv`). Undeclared layers are excluded from that type's loss, "
        "not treated as confirmed absence.\n"
    )
    scope_counts = Counter()
    for _book, scope in masks.items():
        for L in scope:
            scope_counts[L] += 1
    for L in MODEL_LAYERS:
        lines.append(f"- {L}: {scope_counts[L]} books in-scope")
    lines.append("")
    lines.append(
        "Citation excluded. Author and BookTitle are modeled targets "
        "(BIO labels `AUTHOR` / `BOOKTITLE`).\n"
    )

    (out_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_outputs(
    out_dir: Path,
    books: list[BookRecord],
    flagged: list[SpanHit],
    overlaps: list[OverlapHit],
    masks: dict[str, set[str]],
    long_quote_min: int,
) -> tuple[list[dict], list[dict]]:
    """Write all audit CSVs + markdown report. Return (flagged_rows, overlap_rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    flagged_fields = [
        "category",
        "book",
        "layer",
        "ann_id",
        "start",
        "end",
        "length",
        "doc_len",
        "yaml_line",
        "reasons",
        "action",
        "fixed_start",
        "fixed_end",
        "note",
        "prev_start",
        "prev_end",
        "next_start",
        "next_end",
        "neighbor_gap",
        "border_before",
        "border_after",
        "gap_text",
        "start_border",
        "end_border",
    ]
    flagged_rows = []
    for h in flagged:
        flagged_rows.append(
            {
                "category": h.category,
                "book": h.book,
                "layer": h.layer,
                "ann_id": h.ann_id,
                "start": h.start,
                "end": h.end,
                "length": span_length(h.start, h.end) if h.start >= 0 and h.end >= h.start else "",
                "doc_len": h.doc_len,
                "yaml_line": h.line,
                "reasons": "|".join(h.reasons),
                "action": h.action,
                "fixed_start": h.fixed_start if h.fixed_start is not None else "",
                "fixed_end": h.fixed_end if h.fixed_end is not None else "",
                "note": h.note,
                "prev_start": h.prev_start if h.prev_start is not None else "",
                "prev_end": h.prev_end if h.prev_end is not None else "",
                "next_start": h.next_start if h.next_start is not None else "",
                "next_end": h.next_end if h.next_end is not None else "",
                "neighbor_gap": h.neighbor_gap if h.neighbor_gap is not None else "",
                "border_before": h.border_before,
                "border_after": h.border_after,
                "gap_text": h.gap_text,
                "start_border": h.start_border,
                "end_border": h.end_border,
            }
        )
    write_csv(out_dir / "flagged_spans.csv", flagged_rows, flagged_fields)

    long_rows = [r for r in flagged_rows if r["category"] == "long_outlier"]
    long_rows.sort(key=lambda r: -(r["length"] or 0))
    write_csv(out_dir / "long_quotations.csv", long_rows, flagged_fields)

    overlap_fields = [
        "book",
        "layer_a",
        "ann_id_a",
        "start_a",
        "end_a",
        "len_a",
        "yaml_line_a",
        "text_a",
        "layer_b",
        "ann_id_b",
        "start_b",
        "end_b",
        "len_b",
        "yaml_line_b",
        "text_b",
        "overlap_start",
        "overlap_end",
        "overlap_chars",
        "relation",
        "priority_winner",
        "overlap_text",
        "border_before",
        "border_after",
    ]
    overlap_rows = [
        {
            "book": h.book,
            "layer_a": h.layer_a,
            "ann_id_a": h.ann_id_a,
            "start_a": h.start_a,
            "end_a": h.end_a,
            "len_a": span_length(h.start_a, h.end_a),
            "yaml_line_a": h.line_a,
            "text_a": h.text_a,
            "layer_b": h.layer_b,
            "ann_id_b": h.ann_id_b,
            "start_b": h.start_b,
            "end_b": h.end_b,
            "len_b": span_length(h.start_b, h.end_b),
            "yaml_line_b": h.line_b,
            "text_b": h.text_b,
            "overlap_start": h.overlap_start,
            "overlap_end": h.overlap_end,
            "overlap_chars": h.overlap_chars,
            "relation": h.relation,
            "priority_winner": h.priority_winner,
            "overlap_text": h.overlap_text,
            "border_before": h.border_before,
            "border_after": h.border_after,
        }
        for h in overlaps
    ]
    write_csv(out_dir / "overlaps.csv", overlap_rows, overlap_fields)

    mask_rows = []
    for rec in books:
        scope = masks[rec.book]
        row = {
            "book": rec.book,
            "doc_len": rec.doc_len,
            "meta_path": str(rec.meta_path) if rec.meta_path else "",
        }
        for L in MODEL_LAYERS:
            row[f"in_scope_{L}"] = int(L in scope)
            row[f"spans_{L}"] = rec.span_counts.get(L, 0)
            row[f"raw_spans_{L}"] = rec.raw_span_counts.get(L, 0)
            row[f"invalid_{L}"] = rec.invalid_counts.get(L, 0)
            row[f"declared_{L}"] = int(L in rec.declared_layers)
            row[f"has_file_{L}"] = int(L in rec.layer_files)
        row["has_Citation"] = int("Citation" in rec.layer_files)
        mask_rows.append(row)
    mask_fields = ["book", "doc_len", "meta_path"]
    for L in MODEL_LAYERS:
        mask_fields += [
            f"in_scope_{L}",
            f"spans_{L}",
            f"raw_spans_{L}",
            f"invalid_{L}",
            f"declared_{L}",
            f"has_file_{L}",
        ]
    mask_fields += ["has_Citation"]
    write_csv(out_dir / "layer_masks.csv", mask_rows, mask_fields)

    summary_rows = []
    for layer in MODEL_LAYERS:
        subset = [h for h in flagged if h.category == "invalid" and h.layer == layer]
        summary_rows.append(
            {
                "layer": layer,
                "invalid": len(subset),
                "drop": sum(1 for h in subset if h.action == "drop"),
                "fix": sum(1 for h in subset if h.action == "fix"),
                "kept_spans_after_clean": sum(b.span_counts.get(layer, 0) for b in books),
                "books_in_scope": sum(1 for b in books if layer in masks[b.book]),
                "books_with_positive": sum(
                    1 for b in books if layer in masks[b.book] and b.span_counts.get(layer, 0) > 0
                ),
            }
        )
    write_csv(
        out_dir / "clean_summary.csv",
        summary_rows,
        [
            "layer",
            "invalid",
            "drop",
            "fix",
            "kept_spans_after_clean",
            "books_in_scope",
            "books_with_positive",
        ],
    )

    write_report(out_dir, books, flagged, overlaps, masks, long_quote_min)
    return flagged_rows, overlap_rows

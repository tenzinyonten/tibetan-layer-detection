#!/usr/bin/env python3
"""Build tsawa train/val/test book catalog (xlsx)."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.filters import AutoFilter

ROOT = Path("/Users/tenzinyonten/layer_detection")
SPLIT = ROOT / "tsawa/data/processed/split_v3_frozen.csv"
AUDIT = ROOT / "tsawa/data/processed/tsawa_audit.csv"
MERGED = ROOT / "tsawa/data/processed/tsawa_spans_merged.csv"
OUT = ROOT / "tsawa/data/processed/tsawa_book_catalog.xlsx"

AUTHOR_PREFIX = re.compile(r"^མཛད་པ་པོ།\s*")


def clean(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        return " | ".join(clean(x) for x in s if clean(x))
    return re.sub(r"\s+", " ", str(s)).strip()


def author_name(s: str) -> str:
    return AUTHOR_PREFIX.sub("", clean(s)).strip(" །")


def pick_meta(m: dict) -> dict:
    src = m.get("source_metadata") or {}
    web = m.get("web_metadata") or {}
    ebook = m.get("ebook_metadata") or {}
    authors = src.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author_bo = author_name(src.get("author")) or " | ".join(author_name(a) for a in authors)
    layers = src.get("layers") or m.get("layers") or []
    if isinstance(layers, str):
        layers = [layers]
    title_en = ""
    author_en = ""
    collection = ""
    publisher = ""
    if isinstance(web.get("title"), dict):
        title_en = clean(web["title"].get("en"))
    elif web.get("title"):
        title_en = clean(web.get("title"))
    if isinstance(web.get("author"), dict):
        author_en = clean(web["author"].get("en"))
    elif web.get("author"):
        author_en = clean(web.get("author"))
    if isinstance(web.get("collection"), dict):
        collection = clean(web["collection"].get("en") or web["collection"].get("bo"))
    if isinstance(web.get("pusblisher") or web.get("publisher"), dict):
        pub = web.get("pusblisher") or web.get("publisher")
        publisher = clean(pub.get("en") or pub.get("bo"))
    return {
        "title_bo": clean(src.get("title") or ebook.get("title")),
        "title_en": title_en,
        "author_bo": author_bo,
        "author_en": author_en,
        "authors_raw": " | ".join(clean(a) for a in authors),
        "sku": clean(src.get("sku") or src.get("id") or ebook.get("sku")),
        "collection": collection,
        "publisher": publisher,
        "layers": ", ".join(clean(x) for x in layers),
        "opf_id": clean(m.get("id")),
    }


def load_split():
    lines = [ln for ln in SPLIT.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(lines))


def main():
    split_rows = [r for r in load_split() if r["split"] in ("train", "val")]
    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    n_keep = Counter()
    n_drop = Counter()
    for r in csv.DictReader(MERGED.open(encoding="utf-8")):
        if str(r.get("dropped", "")).lower() == "true":
            n_drop[r["pecha_id"]] += 1
        else:
            n_keep[r["pecha_id"]] += 1

    books = []
    for r in split_rows:
        pid = r["pecha_id"]
        a = audit.get(pid, {})
        opf = a.get("opf_root") or ""
        meta_path = Path(opf) / "meta.yml" if opf else None
        meta = {}
        if meta_path and meta_path.is_file():
            meta = pick_meta(yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {})
        batch = a.get("source_batch") or ("old" if pid.startswith("P") else "new")
        books.append(
            {
                "pecha_id": pid,
                "split": r["split"],
                "group_id": r["group_id"],
                "batch": batch,
                "title_bo": meta.get("title_bo", ""),
                "title_en": meta.get("title_en", ""),
                "author_bo": meta.get("author_bo", ""),
                "author_en": meta.get("author_en", ""),
                "authors_raw": meta.get("authors_raw", ""),
                "sku": meta.get("sku", ""),
                "collection": meta.get("collection", ""),
                "publisher": meta.get("publisher", ""),
                "n_tsawa_spans": n_keep[pid],
                "n_spans_dropped": n_drop[pid],
                "base_chars": int(float(a["base_chars"])) if a.get("base_chars") else "",
                "tsawa_chars": int(float(a["tsawa_chars_union"])) if a.get("tsawa_chars_union") else "",
                "coverage_pct": float(a["coverage_pct"]) if a.get("coverage_pct") not in (None, "") else "",
                "layers": meta.get("layers", ""),
                "has_tsawa_layer": a.get("has_tsawa_layer", ""),
                "opf_path": opf,
            }
        )
    order = {"train": 0, "val": 1, "test": 2}
    books.sort(key=lambda b: (order.get(b["split"], 9), b["batch"], b["pecha_id"]))

    headers = [
        "pecha_id",
        "split",
        "group_id",
        "batch",
        "title_bo",
        "title_en",
        "author_bo",
        "author_en",
        "collection",
        "publisher",
    ]

    wb = Workbook()
    font = Font(name="Arial", size=11)
    head_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Arial", size=16, bold=True)
    thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    fills = {
        "train": PatternFill("solid", fgColor="1F4E79"),
        "val": PatternFill("solid", fgColor="2E75B6"),
        "test": PatternFill("solid", fgColor="548235"),
        "alt": PatternFill("solid", fgColor="F2F2F2"),
        "head": PatternFill("solid", fgColor="1F4E79"),
    }
    split_fill = {
        "train": PatternFill("solid", fgColor="D6EAF8"),
        "val": PatternFill("solid", fgColor="FCF3CF"),
        "test": PatternFill("solid", fgColor="D5F5E3"),
    }

    # --- Summary ---
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Tsawa book catalog"
    ws["A1"].font = title_font
    ws["A2"] = "Frozen split v3 train + val only (test omitted). Titles and authors from each book's OpenPecha meta.yml."
    ws["A2"].font = Font(name="Arial", size=11, italic=True, color="666666")
    ws["A3"] = "Source: tsawa/data/processed/split_v3_frozen.csv + tsawa_spans_merged.csv + tsawa_audit.csv + data/raw_opf/*/meta.yml"
    ws["A3"].font = Font(name="Arial", size=10, color="666666")

    ws["A5"] = "Split"
    ws["B5"] = "Books"
    ws["C5"] = "Old-batch books"
    ws["D5"] = "New-batch books"
    for col in range(1, 5):
        c = ws.cell(5, col)
        c.font = head_font
        c.fill = fills["head"]
        c.alignment = Alignment(horizontal="center")

    # data starts on Books!A2 (header row 1)
    n_books = len(books)
    last = n_books + 1  # Books last data row
    ws["A6"] = "train"
    ws["B6"] = f'=COUNTIF(Books!B:B,"train")'
    ws["C6"] = f'=COUNTIFS(Books!B:B,"train",Books!D:D,"old")'
    ws["D6"] = f'=COUNTIFS(Books!B:B,"train",Books!D:D,"new")'
    ws["A7"] = "val"
    ws["B7"] = f'=COUNTIF(Books!B:B,"val")'
    ws["C7"] = f'=COUNTIFS(Books!B:B,"val",Books!D:D,"old")'
    ws["D7"] = f'=COUNTIFS(Books!B:B,"val",Books!D:D,"new")'
    ws["A8"] = "all"
    ws["A8"].font = Font(name="Arial", size=11, bold=True)
    ws["B8"] = "=B6+B7"
    ws["C8"] = "=C6+C7"
    ws["D8"] = "=D6+D7"
    for row in range(6, 9):
        for col in range(1, 5):
            ws.cell(row, col).font = Font(name="Arial", size=11, bold=(row == 8))
            ws.cell(row, col).border = thin
            ws.cell(row, col).alignment = Alignment(horizontal="center")
        ws.cell(row, 1).alignment = Alignment(horizontal="left")
        if ws.cell(row, 1).value in split_fill:
            ws.cell(row, 1).fill = split_fill[ws.cell(row, 1).value]

    ws["A11"] = "How to use"
    ws["A11"].font = Font(name="Arial", size=12, bold=True)
    ws["A12"] = "Open the Books tab. Filter by split (train / val), batch, or author. Title/author come from meta.yml; English fields are only filled when web_metadata exists (mostly old-batch / P-books). Missing English is blank, not an error."
    ws["A12"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A12:D14")
    ws["A15"] = "Upload this file to Google Drive → Open with Google Sheets if you want a live sheet."
    ws["A15"].font = Font(name="Arial", size=11, italic=True)
    for col, w in enumerate([18, 18, 24, 20, 20], 1):
        ws.column_dimensions[get_column_letter(col)].width = w
    ws.row_dimensions[12].height = 48
    ws.freeze_panes = "A6"

    # --- Books ---
    sh = wb.create_sheet("Books")
    for i, h in enumerate(headers, 1):
        cell = sh.cell(1, i, h)
        cell.font = head_font
        cell.fill = fills["head"]
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = thin
    for ri, b in enumerate(books, 2):
        for ci, h in enumerate(headers, 1):
            val = b[h]
            cell = sh.cell(ri, ci, val if val != "" else None)
            cell.font = font
            cell.border = thin
            cell.alignment = Alignment(vertical="center", wrap_text=h.startswith("title") or h.startswith("author") or h in ("collection", "publisher"))
            if ri % 2 == 0:
                cell.fill = fills["alt"]
            if h == "split" and val in split_fill:
                cell.fill = split_fill[val]
    sh.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(books)+1}"
    sh.freeze_panes = "B2"
    sh.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{1+len(books)}"
    widths = {
        "A": 14, "B": 10, "C": 14, "D": 10, "E": 48, "F": 36, "G": 32, "H": 28,
        "I": 28, "J": 22,
    }
    for col, w in widths.items():
        sh.column_dimensions[col].width = w
    sh.row_dimensions[1].height = 22
    sh.sheet_properties.tabColor = "1F4E79"

    # --- Notes ---
    nt = wb.create_sheet("Notes")
    nt["A1"] = "Column notes"
    nt["A1"].font = title_font
    notes = [
        ("pecha_id", "OpenPecha / BDRC-style book id (I* = new batch, P* = old batch)."),
        ("split", "Frozen v3 assignment. Test books are omitted from this catalog."),
        ("group_id", "Must-link reprint / root-text group. Books with the same group_id stay in one split."),
        ("batch", "old = P-books (earlier annotation); new = I-books."),
        ("title_bo / author_bo", "From meta.yml source_metadata. The མཛད་པ་པོ། prefix is stripped from author_bo."),
        ("title_en / author_en", "From web_metadata when present. Often empty on new-batch books."),
        ("collection / publisher", "From web_metadata when present."),
    ]
    nt["A3"] = "Field"
    nt["B3"] = "Meaning"
    nt["A3"].font = head_font
    nt["B3"].font = head_font
    nt["A3"].fill = fills["head"]
    nt["B3"].fill = fills["head"]
    for i, (k, v) in enumerate(notes, 4):
        nt.cell(i, 1, k).font = Font(name="Arial", size=11, bold=True)
        nt.cell(i, 2, v).font = font
        nt.cell(i, 2).alignment = Alignment(wrap_text=True)
        nt.row_dimensions[i].height = 32
    nt.column_dimensions["A"].width = 24
    nt.column_dimensions["B"].width = 110
    nt.freeze_panes = "A4"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  books={len(books)}  by split={dict(Counter(b['split'] for b in books))}")
    print(f"missing title_bo: {sum(1 for b in books if not b['title_bo'])}")
    print(f"missing author_bo: {sum(1 for b in books if not b['author_bo'])}")


if __name__ == "__main__":
    main()

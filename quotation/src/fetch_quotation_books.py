#!/usr/bin/env python3
"""Download OpenPecha-Data repos from the Tsadra list."""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ORG = "OpenPecha-Data"
ZIP_URLS = (
    "https://github.com/{org}/{repo}/archive/refs/heads/master.zip",
    "https://github.com/{org}/{repo}/archive/refs/heads/main.zip",
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "Data" / "tsadra_repo_list.txt"
DEFAULT_OUT = ROOT / "Data" / "repos"


def fetch_bytes(url: str, timeout: float = 120.0) -> bytes | None:
    request = Request(url, headers={"User-Agent": "quotation-detection/fetch_quotation_books"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def read_repo_ids(list_path: Path) -> list[str]:
    ids: list[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        repo_id = line.strip()
        if repo_id and not repo_id.startswith("#"):
            ids.append(repo_id)
    return ids


def download_repo(repo_id: str, dest_dir: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            payload = None
            for template in ZIP_URLS:
                payload = fetch_bytes(template.format(org=ORG, repo=repo_id))
                if payload is not None:
                    break
            if payload is None:
                raise FileNotFoundError(f"zip not found for {repo_id}")

            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                members = zf.namelist()
                if not members:
                    raise RuntimeError(f"empty zip for {repo_id}")
                root_prefix = members[0].split("/")[0] + "/"
                for info in zf.infolist():
                    name = info.filename
                    if not name.startswith(root_prefix):
                        continue
                    rel = name[len(root_prefix) :]
                    if not rel:
                        continue
                    target = dest_dir / rel
                    if info.is_dir() or name.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
            return
        except (HTTPError, URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {repo_id}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all OpenPecha-Data repositories from the Tsadra list."
    )
    parser.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_LIST,
        help=f"Repo id list (default: {DEFAULT_LIST})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory for repo folders (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which repos would be downloaded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the repo folder already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N repo ids from the list",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.list.is_file():
        print(f"Repo list not found: {args.list}", file=sys.stderr)
        return 1

    repo_ids = read_repo_ids(args.list)
    if args.limit is not None:
        repo_ids = repo_ids[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0

    print(f"Downloading {len(repo_ids)} repos from {args.list}")
    print(f"Output directory: {args.out}")
    if args.dry_run:
        print("Dry run — no downloads")

    for i, repo_id in enumerate(repo_ids, start=1):
        prefix = f"[{i}/{len(repo_ids)}] {repo_id}"
        dest = args.out / repo_id

        if dest.exists() and not args.force:
            print(f"{prefix}: already present, skip")
            skipped += 1
            continue

        if args.dry_run:
            print(f"{prefix}: would download")
            continue

        try:
            print(f"{prefix}: downloading…")
            download_repo(repo_id, dest)
            downloaded += 1
            print(f"{prefix}: saved to {dest}")
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"{prefix}: download failed ({exc})")
            errors += 1

    print(f"\nDone. downloaded={downloaded} skipped={skipped} errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

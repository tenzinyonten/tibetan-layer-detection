"""Upload / download tokenized window caches to a private Hugging Face dataset repo.

Training stores per-book HF ``save_to_disk`` trees under::

    <window_cache_dir>/<book_id>/<fingerprint>/

Fingerprint inputs (tokenizer, max_length, stride, masked_layers, text, spans,
scope) are defined by ``training_utils.window_cache_key``. Upload that tree once,
then pull it into the next experiment's ``window_cache_dir`` so tokenization is
skipped on cache hits.

Examples::

    # Push a window cache (private dataset repo)
    python quotation/src/upload_quotation_window_cache.py upload \\
        --config quotation/configs/train_quotation_model.yaml \\
        --repo karma689/layer-detection-window-cache-mmbert-base-8192 \\
        --private

    # Pull into a new experiment before training
    python quotation/src/upload_quotation_window_cache.py download \\
        --config quotation/configs/train_quotation_model.yaml \\
        --repo karma689/layer-detection-window-cache-mmbert-base-8192
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


CACHE_MANIFEST_NAME = "cache_manifest.json"
README_NAME = "README.md"


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and validate an experiment YAML mapping."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {path}")
    return config


def resolve_cache_dir(
    config_path: Path,
    config: dict[str, Any],
    cache_dir: Path | None = None,
) -> Path:
    """Return the absolute window-cache directory for an experiment."""
    if cache_dir is not None:
        path = cache_dir if cache_dir.is_absolute() else Path.cwd() / cache_dir
        return path.resolve()
    experiment_dir = config_path.resolve().parent
    configured = Path(str(config.get("window_cache_dir", "window_cache")))
    if not configured.is_absolute():
        configured = experiment_dir / configured
    return configured.resolve()


def scan_window_cache(cache_dir: Path) -> list[dict[str, str]]:
    """List ``{book_id, fingerprint}`` entries under a window cache root."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Window cache directory does not exist: {cache_dir}")

    entries: list[dict[str, str]] = []
    for book_dir in sorted(cache_dir.iterdir(), key=lambda p: p.name):
        if not book_dir.is_dir() or book_dir.name.startswith("."):
            continue
        if book_dir.name in {CACHE_MANIFEST_NAME, README_NAME}:
            continue
        fingerprints = sorted(
            path.name
            for path in book_dir.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and (path / "dataset_info.json").is_file()
        )
        if not fingerprints:
            continue
        for fingerprint in fingerprints:
            entries.append({"book_id": book_dir.name, "fingerprint": fingerprint})
    return entries


def build_manifest(
    config: dict[str, Any],
    entries: list[dict[str, str]],
    *,
    source_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Build a JSON manifest documenting cache params and book fingerprints."""
    return {
        "experiment_name": config.get("experiment_name"),
        "dataset_repo": config.get("dataset_repo"),
        "dataset_revision": config.get("dataset_revision"),
        "scope_policy": config.get("scope_policy"),
        "tokenizer_name": config.get("tokenizer_checkpoint") or config.get("checkpoint"),
        "max_length": config.get("max_length"),
        "stride": config.get("stride"),
        "masked_layers": list(config.get("masked_layers") or []),
        "book_count": len({entry["book_id"] for entry in entries}),
        "entry_count": len(entries),
        "source_cache_dir": source_cache_dir,
        "books": entries,
        "notes": (
            "Each book lives at <book_id>/<fingerprint>/ as a datasets "
            "save_to_disk tree. Fingerprints come from "
            "quotation_lib.training_utils.window_cache_key — re-tokenize "
            "when tokenizer, window settings, text, spans, or scope change."
        ),
    }


def build_dataset_card(repo_id: str, manifest: dict[str, Any]) -> str:
    """Render a Hub README for the window-cache dataset repo."""
    tokenizer = manifest.get("tokenizer_name") or "unknown"
    frontmatter = {
        "language": ["bo"],
        "pretty_name": "Layer detection tokenized window cache",
        "tags": ["tibetan", "token-classification", "tokenized-cache"],
        "license": "other",
    }
    books = "\n".join(
        f"| `{entry['book_id']}` | `{entry['fingerprint']}` |"
        for entry in manifest.get("books", [])[:20]
    )
    more = ""
    total = int(manifest.get("entry_count") or 0)
    if total > 20:
        more = f"\n\n…and {total - 20} more entries in `{CACHE_MANIFEST_NAME}`."

    return f"""---
{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()}
---

# Tokenized window cache

Private cache of per-book sliding windows for Tibetan layer detection.
Layout matches local training caches consumed by
`quotation_lib.training_utils.build_windowed_dataset`.

## Cache parameters

| Parameter | Value |
|---|---|
| Experiment | `{manifest.get("experiment_name")}` |
| Source dataset | `{manifest.get("dataset_repo")}` |
| Dataset revision | `{manifest.get("dataset_revision")}` |
| Tokenizer | `{tokenizer}` |
| max_length | `{manifest.get("max_length")}` |
| stride | `{manifest.get("stride")}` |
| masked_layers | `{json.dumps(manifest.get("masked_layers") or [], ensure_ascii=False)}` |
| Books | `{manifest.get("book_count")}` |

## Layout

```
<book_id>/<fingerprint>/
  data-00000-of-00001.arrow
  dataset_info.json
  state.json
```

## Download into an experiment

```bash
python quotation/src/upload_quotation_window_cache.py download \\
    --config experiments/<experiment>/config.yaml \\
    --repo {repo_id}
```

Or manually:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="experiments/<experiment>/window_cache",
)
```

After download, point `window_cache_dir` at that folder and train as usual —
matching fingerprints are cache hits and skip tokenization.

## Sample books

| book_id | fingerprint |
|---|---|
{books}{more}
"""


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    """Hard-link large artifacts when possible, otherwise copy them."""
    try:
        os.link(source, destination)
        return str(destination)
    except OSError:
        return shutil.copy2(source, destination)


def _stage_tree(source: Path, destination: Path) -> None:
    """Recursively hard-link (or copy) a directory tree."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            _stage_tree(item, target)
        else:
            _link_or_copy(item, target)


def prepare_upload_folder(
    destination: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    repo_id: str,
) -> None:
    """Stage cache trees plus manifest/README for ``HfApi.upload_folder``."""
    destination.mkdir(parents=True, exist_ok=True)
    for book_dir in sorted(cache_dir.iterdir(), key=lambda p: p.name):
        if not book_dir.is_dir() or book_dir.name.startswith("."):
            continue
        _stage_tree(book_dir, destination / book_dir.name)

    with (destination / CACHE_MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (destination / README_NAME).open("w", encoding="utf-8") as handle:
        handle.write(build_dataset_card(repo_id, manifest))


def upload_window_cache(
    *,
    config_path: Path,
    repo_id: str,
    cache_dir: Path | None = None,
    private: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate, stage, and optionally push a window cache to the Hub."""
    config = load_experiment_config(config_path)
    resolved_cache = resolve_cache_dir(config_path, config, cache_dir)
    entries = scan_window_cache(resolved_cache)
    if not entries:
        raise FileNotFoundError(
            f"No book caches found under {resolved_cache} "
            "(expected <book_id>/<fingerprint>/dataset_info.json)"
        )

    manifest = build_manifest(
        config,
        entries,
        source_cache_dir=str(resolved_cache),
    )

    with tempfile.TemporaryDirectory(prefix="layer-window-cache-upload-") as temporary:
        upload_dir = Path(temporary)
        prepare_upload_folder(upload_dir, resolved_cache, manifest, repo_id)
        print(f"Cache dir: {resolved_cache}")
        print(f"Books: {manifest['book_count']}  fingerprint entries: {manifest['entry_count']}")
        print(
            "Params: "
            f"tokenizer={manifest['tokenizer_name']} "
            f"max_length={manifest['max_length']} "
            f"stride={manifest['stride']} "
            f"masked_layers={manifest['masked_layers']}"
        )
        if dry_run:
            print("Dry run complete — validated and staged upload artifacts.")
            return manifest

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=upload_dir,
            commit_message=(
                f"Upload window cache "
                f"({manifest.get('experiment_name')}, "
                f"{manifest['book_count']} books)"
            ),
        )

    print(f"Pushed window cache to https://huggingface.co/datasets/{repo_id}")
    return manifest


def download_window_cache(
    *,
    config_path: Path,
    repo_id: str,
    cache_dir: Path | None = None,
    revision: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Download a Hub window cache into the experiment's local cache dir."""
    config = load_experiment_config(config_path)
    resolved_cache = resolve_cache_dir(config_path, config, cache_dir)
    print(f"Repo: {repo_id}")
    print(f"Local cache dir: {resolved_cache}")
    if dry_run:
        print("Dry run complete — would download into the path above.")
        return resolved_cache

    from huggingface_hub import snapshot_download

    resolved_cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(resolved_cache),
    )
    entries = scan_window_cache(resolved_cache)
    print(f"Downloaded {len({e['book_id'] for e in entries})} books "
          f"({len(entries)} fingerprint entries) into {resolved_cache}")
    return resolved_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser(
        "upload",
        help="Push a local window_cache tree to a (private) Hub dataset repo",
    )
    upload_parser.add_argument(
        "--config", type=Path, required=True, help="Experiment config.yaml"
    )
    upload_parser.add_argument(
        "--repo",
        required=True,
        help="Hub dataset repo id (owner/name), e.g. karma689/layer-detection-window-cache-mmbert-base-8192",
    )
    upload_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Override window_cache_dir from config",
    )
    visibility = upload_parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo (the default; accepted for parity with the other upload CLIs)",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo (default: private)",
    )
    upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage locally without uploading",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Pull a Hub window cache into an experiment's window_cache_dir",
    )
    download_parser.add_argument(
        "--config", type=Path, required=True, help="Experiment config.yaml"
    )
    download_parser.add_argument(
        "--repo",
        required=True,
        help="Hub dataset repo id that holds the window cache",
    )
    download_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Override window_cache_dir from config",
    )
    download_parser.add_argument(
        "--revision",
        help="Optional Hub revision / commit / tag to pin",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the destination path without downloading",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "upload":
            upload_window_cache(
                config_path=args.config.resolve(),
                repo_id=args.repo,
                cache_dir=args.cache_dir,
                private=not args.public,
                dry_run=args.dry_run,
            )
        else:
            download_window_cache(
                config_path=args.config.resolve(),
                repo_id=args.repo,
                cache_dir=args.cache_dir,
                revision=args.revision,
                dry_run=args.dry_run,
            )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"upload_window_cache: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

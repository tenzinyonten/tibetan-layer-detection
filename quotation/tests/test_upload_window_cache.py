from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quotation_lib.upload_window_cache import (
    build_dataset_card,
    build_manifest,
    prepare_upload_folder,
    resolve_cache_dir,
    scan_window_cache,
)


class UploadWindowCacheTests(unittest.TestCase):
    def _write_fake_book(self, cache_dir: Path, book_id: str, fingerprint: str) -> None:
        book_path = cache_dir / book_id / fingerprint
        book_path.mkdir(parents=True)
        (book_path / "dataset_info.json").write_text("{}", encoding="utf-8")
        (book_path / "state.json").write_text("{}", encoding="utf-8")
        (book_path / "data-00000-of-00001.arrow").write_bytes(b"arrow")

    def test_resolve_cache_dir_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "exp"
            experiment.mkdir()
            config_path = experiment / "config.yaml"
            config_path.write_text("window_cache_dir: window_cache\n", encoding="utf-8")
            resolved = resolve_cache_dir(
                config_path, {"window_cache_dir": "window_cache"}
            )
            self.assertEqual(resolved, (experiment / "window_cache").resolve())

    def test_scan_window_cache_lists_valid_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self._write_fake_book(cache_dir, "P000010", "abc123")
            self._write_fake_book(cache_dir, "P000012", "def456")
            (cache_dir / "P000099").mkdir()
            (cache_dir / "README.md").write_text("skip", encoding="utf-8")

            entries = scan_window_cache(cache_dir)
            self.assertEqual(
                entries,
                [
                    {"book_id": "P000010", "fingerprint": "abc123"},
                    {"book_id": "P000012", "fingerprint": "def456"},
                ],
            )

    def test_build_manifest_and_card(self):
        config = {
            "experiment_name": "mmbert_base_v1.2",
            "dataset_repo": "karma689/layer_detection",
            "dataset_revision": None,
            "checkpoint": "jhu-clsp/mmBERT-base",
            "max_length": 8192,
            "stride": 5120,
            "masked_layers": ["BookTitle"],
        }
        entries = [{"book_id": "P000010", "fingerprint": "abc123"}]
        manifest = build_manifest(config, entries, source_cache_dir="/tmp/cache")
        self.assertEqual(manifest["book_count"], 1)
        self.assertEqual(manifest["tokenizer_name"], "jhu-clsp/mmBERT-base")
        self.assertEqual(manifest["masked_layers"], ["BookTitle"])

        card = build_dataset_card("karma689/cache", manifest)
        self.assertIn("max_length", card)
        self.assertIn("P000010", card)
        self.assertIn("snapshot_download", card)

    def test_prepare_upload_folder_stages_books_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            staging = root / "staging"
            self._write_fake_book(cache_dir, "P000010", "abc123")
            manifest = build_manifest(
                {
                    "experiment_name": "test",
                    "checkpoint": "tok",
                    "max_length": 8,
                    "stride": 4,
                    "masked_layers": [],
                },
                [{"book_id": "P000010", "fingerprint": "abc123"}],
            )
            prepare_upload_folder(staging, cache_dir, manifest, "owner/cache")

            self.assertTrue(
                (staging / "P000010" / "abc123" / "dataset_info.json").is_file()
            )
            written = json.loads((staging / "cache_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(written["book_count"], 1)
            self.assertIn("# Tokenized window cache", (staging / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

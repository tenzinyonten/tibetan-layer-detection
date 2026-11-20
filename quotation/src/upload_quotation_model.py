#!/usr/bin/env python3
"""Thin CLI wrapper — implementation lives in ``quotation_lib.upload_model``."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Prefer an editable install, but fall back to src/ on sys.path so a checkout
# without `pip install -e .` can still run this script directly.
if importlib.util.find_spec("quotation_lib") is None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quotation_lib.upload_model import main

if __name__ == "__main__":
    raise SystemExit(main())

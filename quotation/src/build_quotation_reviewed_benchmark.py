#!/usr/bin/env python3
"""Thin CLI wrapper — implementation lives in ``quotation_lib.reviewed_benchmark``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quotation_lib.reviewed_benchmark import BuildError, main  # noqa: E402

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

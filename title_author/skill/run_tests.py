#!/usr/bin/env python3
"""Run every offline selftest of the skill (no key, no network, no cost):  python3 run_tests.py"""
import subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; bad = 0
for f in ("title_author_skill.py", "title_author_fix.py", "tool_handler.py", "efficiency.py", "server.py"):
    r = subprocess.run([sys.executable, str(H / f), "selftest"], capture_output=True, text=True); ok = r.returncode == 0
    print(("PASS " if ok else "FAIL ") + f + ("" if ok else "\n" + r.stdout + r.stderr)); bad += not ok
sys.exit(1 if bad else 0)

#!/usr/bin/env python3
"""Check which Gemini keys still have free quota. Keys are read from YOUR file, one per line, and never printed (only the last 4 characters).

    1. Make a file outside the repo:   ~/.gemini_keys.txt     (one key per line; then run: chmod 600 ~/.gemini_keys.txt)
    2. python3 title_author/src/ta_check_keys.py [MODEL ...]      default models: gemini-3.5-flash-lite gemini-3.5-flash
Each key gets one 8-token call per model (about 0 cost on a free-tier key). 429 = daily quota used up (reset about midnight Pacific time)."""
import sys, re, json, urllib.request, urllib.error
from pathlib import Path
f = Path("~/.gemini_keys.txt").expanduser()
if not f.exists(): sys.exit(f"make {f} first (one key per line)")
keys = [l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")]
models = sys.argv[1:] or ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
for k in keys:
    out = []
    for m in models:
        body = json.dumps({"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 8}}).encode()
        r = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent", data=body, headers={"Content-Type": "application/json", "x-goog-api-key": k})
        try: urllib.request.urlopen(r, timeout=30); out.append(f"{m}: OK")
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:600]; why = "quota used up" if e.code == 429 else "invalid or blocked" if e.code in (400, 403) else "busy" if e.code == 503 else f"HTTP {e.code}"
            w = re.search(r"retry in ([0-9hms.]+)", msg); out.append(f"{m}: {why}" + (f" (retry in {w.group(1)})" if w else ""))
        except Exception as e: out.append(f"{m}: {type(e).__name__}")
    print("key ..." + k[-4:], "|", " | ".join(out))

#!/usr/bin/env python3
"""Tiny HTTP server around the skill (standard library only). One process, any client.

    GEMINI_API_KEY=... python3 server.py [PORT=8765] [MODEL=gemini-3.8-flash]     # LLM mode (costs per request)
    python3 server.py 8765 rules                                                  # rule-based only: free, instant, weak on author

    POST /extract   {"text_head": "...", "text_tail": "...", "whole_work": false}  ->  {"title","title_region","author","author_region","notes","needs_review","review_reasons"}
    GET  /tool      the tool definition (JSON schema)         GET /stats   calls, cache hits, seconds, tokens        GET /health

Answers are cached on disk (~/.cache/ta_skill), failed calls are retried, and the key is read only from the environment.
Binds to 127.0.0.1 only. Selftest (rules mode, no key, no network):  python3 server.py selftest"""
import json, sys, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import title_author_skill as S, tool_handler as H, efficiency as E


def make_handler(llm):
    class Hd(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/health": self._send(200, {"ok": True})
            elif self.path == "/tool": self._send(200, H.SCHEMA)
            elif self.path == "/stats": self._send(200, llm.meter.report() if llm else {"mode": "rules"})
            else: self._send(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/extract": return self._send(404, {"error": "not found"})
            try: n = int(self.headers.get("Content-Length", 0)); args = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
            except Exception: return self._send(400, {"error": "body is not valid JSON"})
            out = json.loads(H.handle_tool_call(H.NAME, args, call_llm=llm))
            self._send(400 if "error" in out else 200, out)
        def log_message(self, *a): pass
    return Hd


def selftest():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(None)); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
    def post(path, obj):
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        try: return json.load(urllib.request.urlopen(r, timeout=10))
        except urllib.error.HTTPError as e: return json.load(e)
    assert post("/extract", {"text_head": head, "whole_work": True})["author"]
    assert "error" in post("/extract", {})
    assert json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))["ok"]
    srv.shutdown(); print("server selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]: selftest(); sys.exit()
    port = int(a[0]) if a else 8765; model = a[1] if len(a) > 1 else "gemini-3.8-flash"
    llm = None if model == "rules" else E.metered(E.cached(E.retrying(S.gemini_call(model)), "~/.cache/ta_skill", tag=model))
    print(f"serving on 127.0.0.1:{port} ({'rules only' if llm is None else model})"); ThreadingHTTPServer(("127.0.0.1", port), make_handler(llm)).serve_forever()

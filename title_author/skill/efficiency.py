#!/usr/bin/env python3
"""efficiency: wrappers that make the skill cheaper and faster. Each wrapper takes a `call_llm(system, user) -> str` and returns one.

    from title_author_skill import extract, gemini_call
    from efficiency import cached, retrying, metered, extract_many
    llm = metered(cached(retrying(gemini_call("gemini-3.8-flash")), "~/.cache/ta_skill"))
    results = extract_many(works, llm, workers=8)         # works = [{"head":..., "tail":..., "whole_work":...}, ...]
    print(llm.meter.report())                              # calls, cache hits, seconds, tokens (if the LLM function reports them)

- cached:   same (system prompt, text, model tag) is answered from disk, so a repeat costs nothing and no network.
- retrying: waits and retries on 429 / 5xx / network errors (exponential backoff, at most `tries`).
- metered:  counts calls, cache hits, wall seconds and tokens. Tokens are read from `call.usage` (a list of (in, out) pairs) when the LLM
            function fills it (gemini_call does); otherwise only calls and seconds are counted.
- extract_many: runs works in parallel threads, keeps the input order, returns an error entry instead of raising for one bad work.
No network and no key are needed for the selftest:  python3 efficiency.py selftest"""
import json, sys, time, uuid, hashlib, threading, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def cached(call_llm, cache_dir, tag=""):
    d = Path(cache_dir).expanduser(); d.mkdir(parents=True, exist_ok=True)
    def call(system, user):
        k = hashlib.sha256(json.dumps([tag, system, user], ensure_ascii=False).encode()).hexdigest()
        f = d / f"{k}.json"
        if f.exists():
            call.hits += 1
            try: return json.loads(f.read_text(encoding="utf-8"))["raw"]
            except Exception: pass                      # a broken cache file is just a miss
        raw = call_llm(system, user); call.misses += 1
        tmp = f.with_name(f"{f.stem}.{uuid.uuid4().hex}.tmp"); tmp.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8"); tmp.replace(f)
        return raw
    call.hits = call.misses = 0; call.inner = call_llm
    return call


def retrying(call_llm, tries=6, base=2.0, cap=60.0):
    def call(system, user):
        for n in range(tries):
            try: return call_llm(system, user)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or n == tries - 1: raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if n == tries - 1: raise
            time.sleep(min(cap, base * 2 ** n))
    call.inner = call_llm
    return call


class Meter:
    def __init__(self): self.calls = self.hits = 0; self.seconds = 0.0; self.t_in = self.t_out = 0; self.errors = 0; self._lock = threading.Lock()
    def report(self, price_in_per_m=None, price_out_per_m=None):
        r = {"calls": self.calls, "cache_hits": self.hits, "errors": self.errors, "seconds_per_call": round(self.seconds / max(1, self.calls), 2),
             "input_tokens": self.t_in, "output_tokens": self.t_out}
        if price_in_per_m is not None and price_out_per_m is not None and self.calls:
            cost = (self.t_in * price_in_per_m + self.t_out * price_out_per_m) / 1e6
            r["cost"] = round(cost, 4); r["cost_per_work"] = round(cost / self.calls, 5)       # same currency as the prices you pass
        return r


def metered(call_llm):
    m = Meter()
    def call(system, user):
        t0 = time.time(); inner = call_llm
        while not hasattr(inner, "usage") and hasattr(inner, "inner"): inner = inner.inner
        before = len(getattr(inner, "usage", []))
        h0 = getattr(call_llm, "hits", 0)
        try: raw = call_llm(system, user)
        except Exception:
            with m._lock: m.errors += 1
            raise
        with m._lock:
            m.calls += 1; m.seconds += time.time() - t0; m.hits += getattr(call_llm, "hits", 0) - h0
            for a, b in getattr(inner, "usage", [])[before:]: m.t_in += a; m.t_out += b
        return raw
    call.meter = m; call.inner = call_llm
    return call


def extract_many(works, call_llm, workers=8, **kw):
    """works: list of dicts {head, tail, whole_work}. Returns a list of result dicts in the same order (an {"error": ...} entry for a failed work)."""
    import title_author_skill as S
    def one(w):
        try: return S.extract(w["head"], w.get("tail", ""), bool(w.get("whole_work", False)), call_llm=call_llm, **kw)
        except Exception as e: return {"error": f"{type(e).__name__}: {e}"}
    with ThreadPoolExecutor(max_workers=workers) as ex: return list(ex.map(one, works))


def selftest():
    import tempfile, urllib.error as ue
    n = {"c": 0}
    def flaky(s, u):
        n["c"] += 1
        if n["c"] < 3: raise ue.HTTPError("u", 429, "busy", {}, None)
        return '{"title": null}'
    flaky.usage = []
    import time as _t; real = _t.sleep; _t.sleep = lambda x: None
    try:
        with tempfile.TemporaryDirectory() as d:
            llm = metered(cached(retrying(flaky), d))
            assert llm("s", "u") == '{"title": null}' and n["c"] == 3         # two 429s, then success
            assert llm("s", "u") == '{"title": null}' and n["c"] == 3         # second time: from the cache, no call
            r = llm.meter.report(); assert r["calls"] == 2 and r["cache_hits"] == 1, r
            import title_author_skill as S
            head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
            out = extract_many([{"head": head, "whole_work": True}] * 3 + [{"nohead": 1}], llm, workers=2)
            assert len(out) == 4 and "error" in out[3] and out[0].get("author"), out
    finally: _t.sleep = real
    print("efficiency selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest()
    else: sys.exit(__doc__)

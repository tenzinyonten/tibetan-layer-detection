#!/usr/bin/env python3
"""efficiency: wrappers that make the skill cheaper and faster. Each wrapper takes a `call_llm(system, user) -> str` and returns one.

    from title_author_skill import extract, gemini_call
    from efficiency import cached, retrying, metered, extract_many
    llm = metered(cached(retrying(gemini_call("gemini-3.8-flash")), "~/.cache/ta_skill"))
    results = extract_many(works, llm, workers=4)         # works = [{"head":..., "tail":..., "whole_work":...}, ...]
    print(llm.meter.report(price_in_per_m, price_out_per_m))

- cached:   the same (model tag, system prompt, text) is answered from disk: a repeat costs nothing and needs no network. The tag is taken from the
            wrapped LLM function (gemini_call sets it to model + thinking level); pass tag="..." for your own function. Only helps when the SAME work is
            sent again; a benchmark of different works never hits it. Provider-side prompt caching is a separate thing.
- retrying: waits and retries on 429 / 5xx / timeouts / network errors (exponential backoff, at most `tries`).
- metered:  counts calls, cache hits, seconds and tokens. Thread-safe: each thread reads the token usage of ITS OWN last call (call.local.usage, set by
            gemini_call), so parallel workers do not mix counts up. A function that does not set `local.usage` is counted without tokens.
- extract_many: parallel threads, input order kept, an error entry instead of an exception for one bad work. Use few workers on a live key: 429 retries
            eat the gain. For thousands of works use the Batch API instead.
Selftest (no key, no network):  python3 efficiency.py selftest"""
import json, sys, time, uuid, socket, hashlib, threading, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def _chain(f):
    while f is not None:
        yield f; f = getattr(f, "inner", None)


def _find(f, attr):
    return next((getattr(o, attr) for o in _chain(f) if hasattr(o, attr)), None)


def cached(call_llm, cache_dir, tag=""):
    d = Path(cache_dir).expanduser(); d.mkdir(parents=True, exist_ok=True); lock = threading.Lock()
    def call(system, user):
        t = tag or _find(call_llm, "tag") or ""                                  # model + thinking level when the LLM function says so
        k = hashlib.sha256(json.dumps([t, system, user], ensure_ascii=False).encode()).hexdigest()
        f = d / f"{k}.json"
        if f.exists():
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))["raw"]
                call.local.hit = True
                with lock: call.hits += 1
                return raw
            except Exception: pass                                               # a broken cache file is just a miss
        call.local.hit = False
        raw = call_llm(system, user)
        with lock: call.misses += 1
        tmp = f.with_name(f"{f.stem}.{uuid.uuid4().hex}.tmp"); tmp.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8"); tmp.replace(f)
        return raw
    call.hits = call.misses = 0; call.inner = call_llm; call.local = threading.local()
    return call


def retrying(call_llm, tries=6, base=2.0, cap=60.0):
    def call(system, user):
        for n in range(tries):
            try: return call_llm(system, user)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or n == tries - 1: raise
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):      # socket.timeout is not a TimeoutError before Python 3.10
                if n == tries - 1: raise
            time.sleep(min(cap, base * 2 ** n))
    call.inner = call_llm
    return call


class Meter:
    def __init__(self):
        self.calls = self.hits = self.errors = self.t_in = self.t_out = 0; self.seconds = 0.0; self._lock = threading.Lock()
    def add(self, seconds, hit, usage, error=False):
        with self._lock:
            if error: self.errors += 1; return
            self.calls += 1; self.seconds += seconds; self.hits += bool(hit)
            if usage: self.t_in += usage[0]; self.t_out += usage[1]
    def report(self, price_in_per_m=None, price_out_per_m=None):
        paid = self.calls - self.hits
        r = {"calls": self.calls, "cache_hits": self.hits, "paid_calls": paid, "errors": self.errors, "seconds_per_call": round(self.seconds / max(1, self.calls), 2),
             "input_tokens": self.t_in, "output_tokens": self.t_out}
        if price_in_per_m is not None and price_out_per_m is not None and self.calls:
            cost = (self.t_in * price_in_per_m + self.t_out * price_out_per_m) / 1e6
            r["cost"] = round(cost, 4); r["cost_per_paid_call"] = round(cost / max(1, paid), 5); r["cost_per_work"] = round(cost / self.calls, 5)   # per work is lower when the cache hits
        return r


def metered(call_llm):
    m = Meter(); layers = [o for o in _chain(call_llm) if hasattr(o, "local")]
    def call(system, user):
        for o in layers: o.local.__dict__.clear()                                 # this thread only: forget the previous call's numbers
        t0 = time.time()
        try: raw = call_llm(system, user)
        except Exception:
            m.add(0, False, None, error=True); raise
        hit = any(getattr(o.local, "hit", False) for o in layers)
        usage = next((o.local.usage for o in layers if getattr(o.local, "usage", None)), None) if not hit else None
        m.add(time.time() - t0, hit, usage)
        return raw
    call.meter = m; call.inner = call_llm
    return call


def extract_many(works, call_llm, workers=4, **kw):
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
    flaky.local = threading.local(); flaky.tag = "model-A|thinking=low"
    _real = time.sleep; time.sleep = lambda x: None
    try:
        with tempfile.TemporaryDirectory() as d:
            llm = metered(cached(retrying(flaky), d))
            assert llm("s", "u") == '{"title": null}' and n["c"] == 3             # two 429s, then success
            assert llm("s", "u") == '{"title": null}' and n["c"] == 3             # second time: from the cache, no call
            r = llm.meter.report(); assert r["calls"] == 2 and r["cache_hits"] == 1 and r["paid_calls"] == 1, r
            # the cache key includes the model tag: another model does not read model A's file
            seen = []
            def other(s, u): seen.append(1); return "{}"
            other.local = threading.local(); other.tag = "model-B|thinking=low"
            assert metered(cached(other, d))("s", "u") == "{}" and seen, "model B must not share model A's cache"
            # threads: 16 workers, every call reports (10, 5) tokens; counts must be exact (the old shared-list meter double counted)
            def fake(s, u): fake.local.usage = (10, 5); return "{}"
            fake.local = threading.local(); fake.tag = "t"
            with tempfile.TemporaryDirectory() as d2:
                m2 = metered(cached(fake, d2))
                with ThreadPoolExecutor(max_workers=16) as ex: list(ex.map(lambda i: m2("s", f"text {i % 40}"), range(160)))
                rep = m2.meter.report()
                assert rep["calls"] == 160 and rep["cache_hits"] == 160 - m2.inner.misses and rep["cache_hits"] + m2.inner.misses == 160, rep
                assert rep["input_tokens"] == 10 * m2.inner.misses and rep["output_tokens"] == 5 * m2.inner.misses, rep
            # socket.timeout is retried (Python 3.9)
            k = {"c": 0}
            def slow(s, u):
                k["c"] += 1
                if k["c"] < 2: raise socket.timeout("t")
                return "{}"
            assert retrying(slow)("s", "u") == "{}" and k["c"] == 2
            import title_author_skill as S
            head = "༄༅། །ཚེས་བཅུའི་དུས་ཀྱི་མཆོད་པ་བཞུགས།། ཚེས་བཅུའི་མཆོད་པའི་ཡི་གེ་དགེ་སློང་གྲགས་པ་རྒྱལ་མཚན་གྱིས་སྦྱར་བའོ།།"
            out = extract_many([{"head": head, "whole_work": True}] * 3 + [{"nohead": 1}], llm, workers=2)
            assert len(out) == 4 and "error" in out[3] and out[0].get("author"), out
    finally: time.sleep = _real
    print("efficiency selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest()
    else: sys.exit(__doc__)

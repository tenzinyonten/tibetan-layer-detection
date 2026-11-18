"""Build a single-book HTML check page: full text with tsawa spans highlighted (read-only on data/)."""
import glob, html, re, sys, yaml
pid = sys.argv[1] if len(sys.argv) > 1 else "P000078"
SH = "།༎༏༐༑༔"; ISO = {7, 9, 11, 13}
d = [x for x in glob.glob(f"data/raw_opf/{pid}.opf/*") if x.endswith(".opf")][0]
b = open(d + "/base/v001.txt", encoding="utf-8").read()
T = sorted((a["span"]["start"], a["span"]["end"]) for a in yaml.safe_load(open(d + "/layers/v001/Tsawa.yml"))["annotations"].values())
layers = sorted(x.split("/")[-1][:-4] for x in glob.glob(d + "/layers/v001/*.yml"))
def syl(s): return len([x for x in s.split("་") if x.strip()])
def clauses(t):
    o, s = [], 0
    for m in re.finditer(f"[{SH}]", t):
        n = syl(t[s:m.start()]); s = m.end()
        if n: o.append(n)
    n = syl(t[s:])
    if n: o.append(n)
    return o
def iso(t):
    c = clauses(t); return (100 * sum(n in ISO for n in c) / len(c)) if c else 0.0
kind = lambda p: "v" if p >= 60 else ("p" if p < 40 else "m")
names = {"v": "verse-like", "m": "mixed", "p": "prose-like"}
info = []
for i, (s, e) in enumerate(T):
    p = iso(b[s:e]); info.append((i, s, e, p, kind(p)))
cnt = {k: sum(1 for x in info if x[4] == k) for k in "vmp"}
tot_iso = 100 * sum(1 for x in info for n in clauses(b[x[1]:x[2]]) if n in ISO) / max(1, sum(len(clauses(b[x[1]:x[2]])) for x in info))
# runs of >= 4 consecutive metrical clauses (7/9/11/13 syllables) that lie wholly outside every tsawa span
mask = bytearray(len(b))
for _s, _e in T: mask[_s:_e] = b"\x01" * (_e - _s)
cl = []; _p = 0
for m in re.finditer(f"[{SH}]", b):
    n = syl(b[_p:m.start()])
    if n: cl.append((_p, m.start(), n))
    _p = m.end()
U, cur = [], []
for cs, ce, n in cl:
    ok = n in ISO and not any(mask[cs:ce])
    if ok: cur.append((cs, ce))
    else:
        if len(cur) >= 4: U.append((cur[0][0], cur[-1][1]))
        cur = []
if len(cur) >= 4: U.append((cur[0][0], cur[-1][1]))
ulen = sum(e - s for s, e in U)
events = sorted([(s, e, "t", i) for i, s, e, p, k in info] + [(s, e, "u", j) for j, (s, e) in enumerate(U)])
parts, pos = [], 0
for s, e, typ, idx in events:
    if typ == "u":
        parts.append(html.escape(b[pos:s]))
        parts.append(f'<span class="u" id="u{idx}" title="unlabelled metrical run {idx}: {e-s} chars, 4+ consecutive 7/9/11/13-syllable clauses outside tsawa">{html.escape(b[s:e])}</span>')
        pos = e
        continue
    i = idx; p = info[i][3]; k = info[i][4]
    parts.append(html.escape(b[pos:s]))
    parts.append(f'<mark id="s{i}" class="{k}" title="span {i} [{s}:{e}] {e-s} chars, {p:.0f}% verse-metre clauses ({names[k]})">{html.escape(b[s:e])}</mark>')
    pos = e
parts.append(html.escape(b[pos:]))
body = "".join(parts)
def short(i, s, e): return html.escape(b[s:e].replace("\n", " ")[:46])
rows = "".join(f'<li class="{k}" data-p="{p:.0f}"><a href="#s{i}"><b>{i}</b> <span class="t">{short(i,s,e)}</span> <em>{p:.0f}%</em></a></li>' for i, s, e, p, k in sorted(info, key=lambda x: (x[4] == "v", x[3])))
doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{pid} tsawa check</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#d9d9de;--v:#bfe3c0;--m:#ffe3a3;--p:#ffb8a8;--panel:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#e8e8ea;--mut:#9a9aa2;--line:#33333a;--v:#2f5a33;--m:#6a5420;--p:#7a3326;--panel:#1f1f25}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{padding:16px;border-bottom:1px solid var(--line)}}h1{{font-size:18px;margin:0 0 6px}}
.stats{{color:var(--mut);font-size:13px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:13px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 300px}}
@media(max-width:800px){{.layout{{grid-template-columns:1fr}}aside{{max-height:40vh;position:static!important}}}}
main{{padding:16px}}
.text{{white-space:pre-wrap;word-break:break-word;font-family:"Noto Serif Tibetan","Jomolhari","Microsoft Himalaya",serif;font-size:19px;line-height:2.1}}
mark{{color:inherit;border-radius:3px;padding:1px 0}}mark.v{{background:var(--v)}}mark.m{{background:var(--m)}}mark.p{{background:var(--p)}}
.u{{border-bottom:2px dotted #3a7bd5;background:rgba(58,123,213,.12)}}.no-u .u{{border-bottom:none;background:none}}
.hide-v mark.v{{background:none}}.only-p mark.v,.only-p mark.m{{background:none}}
aside{{border-left:1px solid var(--line);background:var(--panel);position:sticky;top:0;height:100vh;overflow:auto;padding:10px;font-size:12px}}
aside ol{{list-style:none;margin:0;padding:0}}aside li a{{display:block;padding:3px 6px;border-left:4px solid;color:inherit;text-decoration:none;border-radius:2px}}
aside li.v a{{border-color:var(--v)}}aside li.m a{{border-color:var(--m)}}aside li.p a{{border-color:var(--p)}}aside li a:hover{{background:var(--line)}}
aside .t{{font-family:"Noto Serif Tibetan","Jomolhari",serif;font-size:13px}}aside em{{color:var(--mut);font-style:normal;float:right}}
label{{margin-right:12px;font-size:13px}}
</style></head><body>
<header><h1>{pid}: tsawa spans highlighted in the full text</h1>
<div class="stats">{len(b):,} characters &middot; {len(T)} tsawa spans covering {sum(e-s for s,e in T):,} chars ({100*sum(e-s for s,e in T)/len(b):.1f}%) &middot; layers in this book: {", ".join(layers)} (no Commentary) &middot; {tot_iso:.0f}% of tsawa clauses have a 7/9/11/13-syllable verse metre (a normal tsawa book is about 88%)</div>
<div class="legend"><span><i class="sw" style="background:var(--v)"></i>verse-like span, {cnt["v"]} (&ge;60% metrical clauses)</span><span><i class="sw" style="background:var(--m)"></i>mixed, {cnt["m"]} (40&ndash;60%)</span><span><i class="sw" style="background:var(--p)"></i>prose-like, {cnt["p"]} (&lt;40%)</span><span><i class="sw" style="border-bottom:2px dotted #3a7bd5;background:rgba(58,123,213,.2)"></i>not labelled tsawa, but 4+ consecutive metrical clauses: {len(U)} runs, {ulen:,} chars (possible missed tsawa)</span></div>
<p style="margin:8px 0 0"><label><input type="checkbox" id="hv"> hide verse-like highlights</label><label><input type="checkbox" id="op"> show only prose-like</label><label><input type="checkbox" id="nu"> hide unlabelled-verse markers</label></p></header>
<div class="layout"><main><div class="text" id="txt">{body}</div></main>
<aside><b>All {len(T)} spans, least verse-like first</b> (click to jump)<ol>{rows}</ol></aside></div>
<script>
var t=document.getElementById('txt');
document.getElementById('hv').onchange=function(){{t.classList.toggle('hide-v',this.checked)}};
document.getElementById('nu').onchange=function(){{t.classList.toggle('no-u',this.checked)}};
document.getElementById('op').onchange=function(){{t.classList.toggle('only-p',this.checked)}};
</script></body></html>'''
out = f"scratch/annotation_audit/{pid}_check.html"  # scratch/ is gitignored; the pages are ~1 MB each
open(out, "w", encoding="utf-8").write(doc)
print(out, cnt, f"{tot_iso:.0f}% metrical", "unlabelled runs", len(U), ulen, "chars")

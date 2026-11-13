"""Build a single-book HTML check page: full text with SABCHE spans
highlighted (read-only on data/). A separate script from make_book_check.py,
not a --flag on it: that script's whole analysis (verse-metre clause
classification, unlabelled-metrical-run detection) is specific to Tsawa
spans being verse, which Sabche spans aren't, so there's no shared logic to
branch on beyond "load the book and highlight some spans" -- and this is
also why it never touches Tsawa.yml at all, so it can't crash on a book
that doesn't have one (e.g. IA736981A).

Spans whose text matches the known "Return to the table of contents"
annotation artifact (an English UI string mistakenly tagged as a Sabche
span in several books) are flagged in a visibly different color and listed
first in the sidebar, since finding those is the point of this page.

Usage: python common/annotation_audit/make_sabche_check.py <pid> [<pid> ...]
"""
import glob
import html
import re
import sys

import yaml

ARTIFACT_RE = re.compile(r"return to|table of contents", re.IGNORECASE)


def build(pid: str) -> None:
    matches = [x for x in glob.glob(f"data/raw_opf/{pid}.opf/*") if x.endswith(".opf")]
    if not matches:
        print(f"{pid}: no data/raw_opf/{pid}.opf/*.opf checkout found, skipping")
        return
    d = matches[0]
    b = open(d + "/base/v001.txt", encoding="utf-8").read()

    sabche_path = d + "/layers/v001/Sabche.yml"
    try:
        anns = yaml.safe_load(open(sabche_path, encoding="utf-8"))["annotations"]
    except FileNotFoundError:
        print(f"{pid}: no Sabche.yml, skipping")
        return
    S = sorted((a["span"]["start"], a["span"]["end"]) for a in anns.values())

    layers = sorted(x.split("/")[-1][:-4] for x in glob.glob(d + "/layers/v001/*.yml"))
    has_tsawa = "Tsawa" in layers

    info = []  # (index, start, end, is_artifact)
    for i, (s, e) in enumerate(S):
        info.append((i, s, e, bool(ARTIFACT_RE.search(b[s:e]))))
    n_artifact = sum(1 for *_, art in info if art)

    parts, pos = [], 0
    for i, s, e, art in info:
        parts.append(html.escape(b[pos:s]))
        cls = "artifact" if art else "sabche"
        title = ("ARTIFACT: matches 'return to' / 'table of contents'" if art
                 else f"Sabche span {i}")
        parts.append(f'<mark id="s{i}" class="{cls}" title="{html.escape(title)} [{s}:{e}]">'
                     f'{html.escape(b[s:e])}</mark>')
        pos = e
    parts.append(html.escape(b[pos:]))
    body = "".join(parts)

    def short(i, s, e):
        return html.escape(b[s:e].replace("\n", " ")[:60])

    # artifact spans first, so they're the first thing you see in the sidebar
    ordered = sorted(info, key=lambda x: (not x[3], x[0]))
    rows = "".join(
        f'<li class="{"artifact" if art else "sabche"}"><a href="#s{i}">'
        f'<b>{i}</b> <span class="t">{short(i, s, e)}</span>'
        f'{" <em>ARTIFACT</em>" if art else ""}</a></li>'
        for i, s, e, art in ordered
    )

    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{pid} sabche check</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#d9d9de;--sabche:#4d9df5;--artifact:#f6635a;--panel:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#e8e8ea;--mut:#9a9aa2;--line:#33333a;--sabche:#2d5f94;--artifact:#8a2e27;--panel:#1f1f25}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{padding:16px;border-bottom:1px solid var(--line)}}h1{{font-size:18px;margin:0 0 6px}}
.stats{{color:var(--mut);font-size:13px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:13px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 320px}}
@media(max-width:800px){{.layout{{grid-template-columns:1fr}}aside{{max-height:40vh;position:static!important}}}}
main{{padding:16px}}
.text{{white-space:pre-wrap;word-break:break-word;font-family:"Noto Serif Tibetan","Jomolhari","Microsoft Himalaya",serif;font-size:19px;line-height:2.1}}
mark{{color:inherit;border-radius:3px;padding:1px 0}}
mark.sabche{{background:var(--sabche)}}
mark.artifact{{background:var(--artifact);box-shadow:0 0 0 2px var(--artifact)}}
aside{{border-left:1px solid var(--line);background:var(--panel);position:sticky;top:0;height:100vh;overflow:auto;padding:10px;font-size:12px}}
aside ol{{list-style:none;margin:0;padding:0}}aside li a{{display:block;padding:3px 6px;border-left:4px solid;color:inherit;text-decoration:none;border-radius:2px}}
aside li.sabche a{{border-color:var(--sabche)}}aside li.artifact a{{border-color:var(--artifact);background:rgba(246,99,90,.12)}}
aside li a:hover{{background:var(--line)}}
aside .t{{font-family:"Noto Serif Tibetan","Jomolhari",serif;font-size:13px}}aside em{{color:var(--artifact);font-style:normal;float:right;font-weight:600}}
</style></head><body>
<header><h1>{pid}: Sabche spans highlighted in the full text</h1>
<div class="stats">{len(b):,} characters &middot; {len(S)} Sabche spans
&middot; {n_artifact} flagged as the "return to the table of contents" artifact
&middot; layers in this book: {", ".join(layers)}{" (no Tsawa)" if not has_tsawa else ""}</div>
<div class="legend"><span><i class="sw" style="background:var(--sabche)"></i>Sabche span, {len(S) - n_artifact}</span>
<span><i class="sw" style="background:var(--artifact)"></i>artifact span (English nav text), {n_artifact}</span></div>
</header>
<div class="layout"><main><div class="text" id="txt">{body}</div></main>
<aside><b>All {len(S)} spans, artifacts first</b> (click to jump)<ol>{rows}</ol></aside></div>
</body></html>'''
    out = f"scratch/annotation_audit/{pid}_sabche_check.html"
    open(out, "w", encoding="utf-8").write(doc)
    print(out, f"{len(S)} spans, {n_artifact} artifact, has_tsawa={has_tsawa}")


if __name__ == "__main__":
    pids = sys.argv[1:] or ["P000078"]
    for pid in pids:
        build(pid)

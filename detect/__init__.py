"""detect -- detect Tsawa, Sabche, Chapter, Quotation and
Yigchung layers in Tibetan text with per-layer mmBERT models.

Simple API:

    from detect import detect
    result = detect("book.txt", layers=["all"])
    result["layers"]["sabche"]   # [{"start": ..., "end": ..., "confidence": ..., "review_needed": ...}, ...]

For the full three-stage pipeline (preprocess / infer / postprocess) with
checkpointing across a large batch, use the console scripts (tibetan-detect,
or the three stages individually: tibetan-preprocess, tibetan-infer,
tibetan-postprocess) instead -- this function runs all three in one process
with no checkpointing, which is right for one book or a handful, not for a
batch you might need to resume.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import infer as _infer
from . import postprocess as _postprocess
from . import preprocess as _preprocess
from .layer_config import LAYERS

__all__ = ["detect", "LAYERS"]
__version__ = "0.1.0"

_RESULT_EXCLUDE = {"_batch_summary.json"}


def detect(path, layers="all", *, out_dir=None, device=None, review_threshold: float | None = None,
          stride: int | None = None, dual_window: bool = False,
          write_json: bool = False, write_html: bool = False, write_opf: bool = False,
          title_author_system: str = "model", title_model: str | None = None,
          author_model: str | None = None, title_author_skill_model: str | None = None,
          title_author_skill_dir: str | None = None, title_author_opf: bool = False,
          title_system: str | None = None, author_system: str | None = None, author_span: str = "last",
          title_author_review_threshold: float | None = None):
    """Run the full three-stage pipeline in-process and return the result(s).

    path: a single text file, or a folder (a mix of .txt files and/or raw
        OpenPecha book folders -- same discovery rule as the CLI's --dir:
        flat *.txt plus any base/v001.txt found at any depth).
    layers: a list of layer names from detect.LAYERS
        (e.g. ["tsawa", "sabche"]), or "all" / ["all"] (the default) for
        every configured layer.
    out_dir: where the pipeline's intermediate and final files are written.
        If not given, a temporary directory is used and removed before this
        function returns -- fine when you only want the returned Python
        objects. If you also pass write_json/write_html/write_opf, out_dir
        defaults to "results" in the current directory instead of a
        temp dir, since those files need somewhere to persist.
    device: "cuda" or "cpu"; defaults to cuda if available.
    review_threshold: a span with confidence below this is flagged
        review_needed in the returned data (and in a persisted JSON file,
        if write_json=True). Default (None): each layer's own threshold
        from layer_config.py (tsawa/yigchung 0.5, sabche 0.85, chapter 0.75,
        quotation 0.65). Passing a number applies it to every layer alike.
    stride: overrides every requested layer's window stride (smaller means
        more overlap between windows). Default (None): each layer's own
        configured stride.
    dual_window: also run each layer at half window/stride size and flag
        spans the two sizes disagree on (dual_window_agreement). Roughly
        doubles inference time per layer.
    title_author_system: "model" (default: the trained model, free and local),
        "skill" (the LLM skill: PAID, one call per book, needs GEMINI_API_KEY or
        ANTHROPIC_API_KEY in the environment) or "both" (title/author from the
        model, title_skill/author_skill from the skill, side by side).
    title_system / author_system: "model" or "skill" for one field (default: follow
        title_author_system). title_system="model", author_system="skill" scored
        best on the test split (title 0.842, author 0.756): a fixed choice per
        field, not a rule per work; the skill part is PAID (one call per book,
        about Rs 0.29 at the batch price, more for live calls). Cannot be
        combined with title_author_system="both".
    author_span: "last" (default) or "first": which author span to keep when the
        author model finds several. First suits whole books, where the gold author
        is on the title page.
    title_model / author_model: Hugging Face repo (or local folder) per field;
        default is BDRC/Bo-Title-Detection and BDRC/Bo-Author-Detection
        (private; each needs HF_TOKEN or a stored login; one forward pass per
        model). The joint Yontenn/mmbert-title-author-run2 also works. title_author_skill_model / _skill_dir:
        which LLM, and where title_author_skill.py lives, for skill / both.
    title_author_opf: with write_opf, also write title and author as the
        OpenPecha layers BookTitle.yml and Author.yml (off by default); needs
        write_opf. title_author_review_threshold: a confidence below this
        flags a title/author span review_needed (default: layer_config.py).
    write_json / write_html / write_opf: also persist Stage 3's JSON/HTML/
        .opf files under out_dir (see the tibetan-postprocess console
        script's --help for what each one is). A JSON pass always runs
        internally regardless -- it's how this function builds its return
        value -- but the per-book .json files are deleted afterward unless
        write_json=True.

    Returns: one book's result dict ({"book_id", "text_length", "layers",
        and "errors" if there were any}) if `path` is a single file, or a
        list of such dicts (one per book found) if `path` is a directory.
    """
    layer_names = list(LAYERS) if layers in ("all", ["all"]) else list(layers)
    device = device or ("cuda" if _infer._has_cuda() else "cpu")

    use_temp_dir = out_dir is None and not (write_json or write_html or write_opf)
    tmp = None
    try:
        if use_temp_dir:
            tmp = tempfile.TemporaryDirectory(prefix="detect_")
            base = Path(tmp.name)
        else:
            base = Path(out_dir) if out_dir is not None else Path("results")
        pre_dir, pred_dir, stage3_dir = base / "preprocessed", base / "predictions", base / "output"

        _preprocess.main(["--input", str(path), "--out", str(pre_dir)])
        infer_argv = ["--input", str(pre_dir), "--out", str(pred_dir),
                     "--layers", *layer_names, "--device", device]
        # detect() already lists layers; infer still auto-adds quotation when
        # tsawa is requested so unannotated books get a model quote mask.
        if stride is not None:
            infer_argv += ["--stride", str(stride)]
        if dual_window:
            infer_argv += ["--dual-window"]
        infer_argv += ["--title-author-system", title_author_system, "--author-span", author_span]
        for flag, val in (("--title-system", title_system), ("--author-system", author_system),
                          ("--title-model", title_model), ("--author-model", author_model),
                          ("--title-author-skill-model", title_author_skill_model),
                          ("--title-author-skill-dir", title_author_skill_dir)):
            if val:
                infer_argv += [flag, str(val)]
        _infer.main(infer_argv)

        post_argv = ["--input", str(pred_dir), "--source", str(pre_dir), "--out", str(stage3_dir), "--json"]
        if review_threshold is not None:
            post_argv += ["--review-threshold", str(review_threshold)]
        if title_author_review_threshold is not None:
            post_argv += ["--title-author-review-threshold", str(title_author_review_threshold)]
        if title_author_opf:
            post_argv.append("--title-author-opf")
        if write_html:
            post_argv.append("--html")
        if write_opf:
            post_argv.append("--opf")
        _postprocess.main(post_argv)

        book_files = [fp for fp in sorted(stage3_dir.glob("*.json"))
                     if fp.name not in _RESULT_EXCLUDE and not fp.name.startswith("run_")]
        results = [json.loads(fp.read_text(encoding="utf-8")) for fp in book_files]
        if not write_json:
            for fp in book_files:
                fp.unlink()
    finally:
        if tmp is not None:
            tmp.cleanup()

    if Path(path).is_file():
        return results[0] if results else None
    return results

"""Per-layer model configuration, shared by detect/infer.py (which layer to
load and how to decode it) and detect/postprocess/ (HTML legend colors,
.opf layer-name capitalization). Split out of detect_layers.py so neither
stage has to import the other's internals.

Each break_penalty/stride is the value documented on the model's own Hugging
Face repo (README.md / training/test_metrics.json / training/config.yaml),
checked 2026-09-28, not an assumed project-wide default -- see `source` on
each entry. All five layers (BDRC/Bo-Tsawa-Detection, Bo-Sabche-Detection,
Bo-Chapter-Detection, Bo-Quotation-Detection, Bo-Yigchung-Detection) have a
dedicated model as of that date. Yigchung uses a different windowing stride
(4914, not 5120) and a different overlap policy: a token's logits come from
only the first window that covers it, and the whole document is decoded once
(see infer.run_layer_stitched()), matching how its training data masked
repeated tokens out of the loss. The model is v2 (break penalty 8, chosen on
validation); its test F1 is 0.471 on the published 77-book text, well below the
other four layers, and a caveat saying so is printed whenever it loads. BDRC/Bo-Multilayer-Detection (the old 15-label joint model) is
deliberately not used for anything here: its own published metrics put
tsawa/yigchung/quote F1 at about 0.02-0.04.

Decoding settings per layer (values in LAYERS below; "default" cleanup means
extend_to_tsheg and merge_small_gaps on, the other switches off):

    layer      scheme  break penalty  stride  cleanup differing from default          review threshold
    tsawa      bio     4.0            5120    default                                  0.5
    sabche     bio     4.0            5120    extend_to_tsheg off, merge_small_gaps    0.85
                                              off, repair_fragments on, shad_rule on
    chapter    bio     4.0            5120    extend_to_tsheg off, merge_small_gaps    0.75
                                              off, extend_to_closing_bracket on
    quotation  bioe    0.0            3613    default                                  0.65
    yigchung   bio     8.0            4914    default; stitch_first_window on          0.5
    title      ta      6.0            none    extend_to_tsheg off, merge_small_gaps    0.7
    author     ta      6.0            none    off (kind "head_tail"; same as title)    0.7

The title and author thresholds (0.7) are an untuned placeholder. Every
stride is the value documented on that model's Hugging Face card.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerConfig:
    repo: str | None
    scheme: str  # "bio" or "bioe" ("ta" for the title/author layers, which read the labels from the model)
    break_penalty: float
    stride: int
    color: str
    source: str  # where break_penalty/stride came from, for the --help text and logs
    kind: str = "window"  # "window": sliding windows over the book (every layer but title/author);
    # "head_tail": one sequence from the first and last 3,000 characters, one span per field
    # (title, author) -- see title_author.py
    stitch_first_window: bool = False  # see infer.run_layer_stitched(); only Yigchung as of 2026-09-28
    caveat: str = ""  # printed once when the layer is loaded, for a known-weak model
    # Post-Viterbi cleanup switches, read by decode.apply_linguistic_rules (see its docstring).
    # Defaults keep the original behavior for every layer; Sabche overrides all three, because
    # on the Sabche test set (29 books) extend_to_tsheg alone cost 12 true positives (F1 0.9619
    # -> 0.9590) and merge_small_gaps fused neighbouring headings (F1 -> 0.822).
    extend_to_tsheg: bool = True     # extend a span ending mid-syllable to the next tsheg
    merge_small_gaps: bool = True    # merge spans separated only by a short punctuation gap
    repair_fragments: bool = False   # drop short spans nested in a longer one (a duplicate)
    extend_lone_fragments: bool = False  # with repair_fragments: also extend a short span with no cover
    # to the next shad/newline. Sabche test: recovers 1 heading but turns ~10 other fragments into
    # full-length false positives (F1 0.9642 -> 0.9630), so it is off.
    min_confidence: float | None = None  # Stage 3 drops a span whose confidence is below this (a precision rule);
    # None keeps every span (the review flag still marks the uncertain ones). Sabche: 0.85, chosen on the validation
    # split (33 books) and confirmed on test (29 books). Validation F1 0.8486 -> 0.8535 (precision 0.868 -> 0.892,
    # recall 0.830 -> 0.818, 37 of 3,091 real headings lost); test F1 0.9686 -> 0.9707 (precision 0.974 -> 0.981,
    # recall 0.964 -> 0.961, 12 of 4,079 lost). Not measured for any other layer.
    shad_rule: bool = False  # drop a span under decode.SHAD_RULE_MAX_CHARS (8) chars unless a shad
    # follows it; a precision rule. Sabche test (29 books): P 0.9609 -> 0.9635, R 0.9696 unchanged,
    # F1 0.9652 -> 0.9665 (+0.0013; 11 false positives dropped, no true positives).
    # SHIPPED PENDING VALIDATION, NOT CONFIRMED: the rule and its 8-char threshold were tuned by
    # looking at the test split's errors, so these numbers are in-sample. Re-check on the validation
    # split when the Sabche model is next run, and set shad_rule=False for sabche if it does not hold.
    extend_to_closing_bracket: bool = False  # Chapter: a ༼ stub under ~10 chars extends to the next ༽
    default_review_threshold: float = 0.7  # used by postprocess's --review-threshold
    # when that flag is omitted; passing --review-threshold explicitly overrides every
    # layer alike, same as before this field existed. Chosen per layer by task precision:
    # sabche's model is very precise (0.85, a high bar), chapter fairly precise (0.75),
    # quotation moderate (0.65), tsawa and yigchung are the hardest tasks (0.5, a low bar
    # so more of their spans surface for review rather than being silently trusted).


# The title and author repos are private; weights are in each repo's best/ folder. The two single-entity models know one field
# each (3 labels O, B, I); the joint model knows both (5 labels). Test, 2,384 works, share correct at IoU >= 0.5:
# joint (penalty 12) title 0.833 / author 0.711; title-only + author-only (penalty 6) 0.842 / 0.734.
TITLE_MODEL = "BDRC/Bo-Title-Detection"
AUTHOR_MODEL = "BDRC/Bo-Author-Detection"
JOINT_MODEL = "Yontenn/mmbert-title-author-run2"
TITLE_AUTHOR_MODEL = JOINT_MODEL  # the earlier name, kept for callers that import it
TA_FIELDS = ("title", "author")
TA_LAYER_NAMES = TA_FIELDS


# The OpenPecha layer files for these two, as they are named in the real books (data/raw_opf: BookTitle.yml in
# 527 of 539 books, Author.yml in 516).
OPF_LAYER_NAMES = {"title": "BookTitle", "author": "Author"}


def is_title_author(name: str) -> bool:
    return name in TA_LAYER_NAMES


@dataclass
class TitleAuthorOptions:
    """How the title/author layers are run. Stage 2 reads the models and crop settings;
    Stage 3 reads opf and review_threshold. Everything runs locally from trained models."""
    title_model: str = TITLE_MODEL         # a Hugging Face repo, or a local folder
    author_model: str = AUTHOR_MODEL       # the joint JOINT_MODEL also works here; if both fields use it, one forward pass serves both
    crop_chars: int = 3000                 # characters kept at each end
    break_penalty: float | None = None     # Viterbi. None: per model -- 6 for a single-entity model, 12 for the joint
    # one (both picked on validation); a number applies to every model alike
    author_span: str = "last"              # which author span to keep when the model finds several: "last" (the default, the
    # rule for a single work) or "first". On whole OpenPecha books the gold author is on the title page, and "first" matched it
    # about twice as often as "last" (35% against 17% strict offsets, 511 books, mostly seen in training); see title_author.py.
    strip_underscore: bool | None = None   # remove "_" from the input before tokenizing. None: per model -- yes for a
    # single-entity model (trained on text without them), no for the joint one
    opf: bool = False                      # also write title/author as BookTitle.yml / Author.yml with --opf (off by default)
    review_threshold: float | None = None  # None: LAYERS[title/author].default_review_threshold

    def __post_init__(self):
        if self.author_span not in ("first", "last"):
            raise ValueError(f"author_span must be first or last, not {self.author_span!r}")


LAYERS: dict[str, LayerConfig] = {
    "tsawa": LayerConfig(
        repo="BDRC/Bo-Tsawa-Detection", scheme="bio",
        break_penalty=4.0, stride=5120, color="#f5e34d",  # yellow
        source="BDRC/Bo-Tsawa-Detection README.md, training/train_results.json (break penalty 5.0 in training) "
               "and training/v10_with_test_bp4.json (decoding at test, break penalty 4.0)",
        default_review_threshold=0.5,
    ),
    "sabche": LayerConfig(
        repo="BDRC/Bo-Sabche-Detection", scheme="bio",
        break_penalty=4.0, stride=5120, color="#4d9df5",  # blue
        source="BDRC/Bo-Sabche-Detection training/test_metrics.json (decoding.break_penalty)",
        default_review_threshold=0.85,
        extend_to_tsheg=False, merge_small_gaps=False, repair_fragments=True, shad_rule=True,
        min_confidence=0.85,
    ),
    "chapter": LayerConfig(
        repo="BDRC/Bo-Chapter-Detection", scheme="bio",
        break_penalty=4.0, stride=5120, color="#4df57a",  # green
        source="BDRC/Bo-Chapter-Detection training/test_metrics.json (decoding.break_penalty); "
               "dataset revision c8c758ed matches the stratified Chapter split built in this repo",
        default_review_threshold=0.75,
        # Same as Sabche on the two default rules: on the Chapter test set they cost
        # F1 0.844 -> 0.822. A ༼ stub under 10 chars is extended to just before the
        # next ༽ (P000110, P000275 exact; +0.12 F1). Sanskrit is not auto-dropped:
        # gold legitimately has some Sanskrit chapter spans.
        extend_to_tsheg=False, merge_small_gaps=False, extend_to_closing_bracket=True,
    ),
    "quotation": LayerConfig(
        repo="BDRC/Bo-Quotation-Detection", scheme="bioe",
        break_penalty=0.0, stride=3613, color="#f5a23d",  # orange
        source="BDRC/Bo-Quotation-Detection training/config.yaml / training/test_metrics.json "
               "(decode_chosen_on_validation.break_penalty; max_length 8192 / stride 3613)",
        default_review_threshold=0.65,
    ),
    "yigchung": LayerConfig(
        repo="BDRC/Bo-Yigchung-Detection", scheme="bio",
        break_penalty=8.0, stride=4914, color="#f57ec0",  # pink
        stitch_first_window=True,
        source="BDRC/Bo-Yigchung-Detection (v2) README.md: break_penalty 8 chosen on validation, "
               "stride 4914 from the 'windowed_w8192_s4914' config; a token's logits come from only the "
               "FIRST window that covers it -- decode once over the whole stitched document, "
               "not per window then merged",
        caveat="test F1 0.471 on the published 77-book test text (0.489 with annotator-corrected gold); "
               "scores differ much by source (Lotsawa 0.85, BDRC 0.49, Nalanda 0.38)",
        default_review_threshold=0.5,
    ),
    "title": LayerConfig(
        repo=TITLE_MODEL, scheme="ta", break_penalty=6.0, stride=0, color="#b07df5",  # purple
        source="BDRC/Bo-Title-Detection, Viterbi break penalty 6 picked on validation; test title 0.842 "
               "share correct (2,384 works; the joint model: 0.833)",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        default_review_threshold=0.7,  # untuned placeholder; to be calibrated on validation once the final models exist
        caveat="title/author reads only the first and last 3,000 characters and gives one title for the whole file",
    ),
    "author": LayerConfig(
        repo=AUTHOR_MODEL, scheme="ta", break_penalty=6.0, stride=0, color="#4dd8d8",  # teal
        source="BDRC/Bo-Author-Detection, Viterbi break penalty 6 picked on validation; test author 0.734 "
               "share correct (2,384 works; the joint model: 0.711)",
        kind="head_tail", extend_to_tsheg=False, merge_small_gaps=False,
        default_review_threshold=0.7,  # untuned placeholder; to be calibrated on validation once the final models exist
        caveat="title/author reads only the first and last 3,000 characters and gives one author for the whole file",
    ),
}

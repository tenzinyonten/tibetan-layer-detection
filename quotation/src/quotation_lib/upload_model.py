"""Upload a trained experiment, its metrics, and a complete model card to the Hub.

The command is experiment-agnostic: point it at any experiment ``config.yaml``.
By default it uploads ``<config directory>/<output_dir>/final`` and discovers
the newest Trainer ``trainer_state.json`` below the output directory.

Example::

    python quotation/src/upload_quotation_model.py \
        --config quotation/configs/train_quotation_model.yaml \
        --repo karma689/mmbert-base-layer-detection-v1.2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml

from .training_utils import LABELS


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
}


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and validate an experiment YAML mapping."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {path}")
    return config


def resolve_paths(
    config_path: Path,
    config: dict[str, Any],
    model_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return the output and final-model directories for an experiment."""
    experiment_dir = config_path.resolve().parent
    output_dir = Path(str(config.get("output_dir", "outputs")))
    if not output_dir.is_absolute():
        output_dir = experiment_dir / output_dir
    if model_dir is None:
        model_dir = output_dir / "final"
    elif not model_dir.is_absolute():
        model_dir = Path.cwd() / model_dir
    return output_dir.resolve(), model_dir.resolve()


def find_trainer_state(output_dir: Path) -> Path:
    """Find the most advanced Trainer state beneath an output directory."""
    candidates = list(output_dir.glob("checkpoint-*/trainer_state.json"))
    root_state = output_dir / "trainer_state.json"
    if root_state.is_file():
        candidates.append(root_state)
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {output_dir}")

    def global_step(path: Path) -> int:
        try:
            with path.open(encoding="utf-8") as handle:
                return int(json.load(handle).get("global_step", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return -1

    return max(candidates, key=lambda path: (global_step(path), path.stat().st_mtime))


def _finite_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("eval_") or key in {"epoch", "step"}:
            if isinstance(value, float) and not math.isfinite(value):
                continue
            metrics[key] = value
    return metrics


def extract_metrics(
    state: dict[str, Any], selection: str = "best"
) -> dict[str, Any]:
    """Extract every evaluation record and select the best or last record."""
    evaluations = [
        _finite_metrics(row)
        for row in state.get("log_history", [])
        if isinstance(row, dict) and any(str(key).startswith("eval_") for key in row)
    ]
    if not evaluations:
        raise ValueError("trainer_state.json contains no evaluation metrics")

    selected = evaluations[-1]
    if selection == "best":
        best_step = state.get("best_global_step")
        if best_step is None:
            best_checkpoint = str(state.get("best_model_checkpoint") or "")
            try:
                best_step = int(Path(best_checkpoint).name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                best_step = None
        if best_step is not None:
            selected = next(
                (row for row in evaluations if row.get("step") == best_step),
                selected,
            )
        else:
            metric_value = state.get("best_metric")
            metric_key = str(state.get("metric_for_best_model") or "")
            if metric_key and not metric_key.startswith("eval_"):
                metric_key = f"eval_{metric_key}"
            if (
                metric_key
                and metric_value is not None
                and any(metric_key in row for row in evaluations)
            ):
                selected = min(
                    evaluations,
                    key=lambda row: abs(
                        float(row.get(metric_key, math.inf)) - float(metric_value)
                    ),
                )

    return {
        "selection": selection,
        "selected_evaluation": selected,
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "global_step": state.get("global_step"),
        "evaluation_history": evaluations,
    }


def flatten_config(
    value: Any, prefix: str = ""
) -> Iterable[tuple[str, Any]]:
    """Yield dotted parameter names without dropping nested config values."""
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_config(child, name)
    else:
        yield prefix, value


def _display(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    escaped = text.replace("|", "\\|")
    return f"`{escaped}`"


def build_model_card(
    repo_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Render a Hub README with complete parameters and selected metrics."""
    selected = metrics["selected_evaluation"]
    dataset = config.get("dataset_repo")
    frontmatter: dict[str, Any] = {
        "language": ["bo"],
        "library_name": "transformers",
        "pipeline_tag": "token-classification",
        "tags": ["tibetan", "token-classification", "layer-detection"],
        "base_model": config.get("checkpoint"),
    }
    if dataset:
        frontmatter["datasets"] = [dataset]
    frontmatter = {key: value for key, value in frontmatter.items() if value is not None}

    metric_rows = "\n".join(
        f"| `{key}` | {_display(value)} |"
        for key, value in sorted(selected.items())
    )
    parameter_rows = "\n".join(
        f"| `{name}` | {_display(value)} |"
        for name, value in flatten_config(config)
    )
    label_rows = "\n".join(f"| {index} | `{label}` |" for index, label in enumerate(LABELS))
    experiment_name = config.get("experiment_name", repo_id.rsplit("/", 1)[-1])

    return f"""---
{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()}
---

# {experiment_name}

Tibetan annotation-layer token classifier fine-tuned from
[`{config.get("checkpoint", "unknown")}`](https://huggingface.co/{config.get("checkpoint", "")}).

## Validation metrics

The values below are from the **{metrics["selection"]}** evaluation record.
The complete evaluation history and original Trainer state are included in
`training/metrics.json` and `training/trainer_state.json`.

| Metric | Value |
|---|---:|
{metric_rows}

## Training parameters

Every value from the experiment's `config.yaml` is listed below. The original
file is also included as `training/config.yaml`.

| Parameter | Value |
|---|---|
{parameter_rows}

## Labels

| ID | BIO label |
|---:|---|
{label_rows}

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)
```

Long texts should use the same window settings shown in the training parameters.
"""


def _copy_model_files(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=_link_or_copy)
        else:
            _link_or_copy(item, target)


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    """Hard-link large artifacts when possible, otherwise copy them."""
    try:
        os.link(source, destination)
        return str(destination)
    except OSError:
        return shutil.copy2(source, destination)


def prepare_upload_folder(
    destination: Path,
    model_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    trainer_state_path: Path,
    metrics: dict[str, Any],
    repo_id: str,
    *,
    include_tokenizer: bool = True,
) -> None:
    """Build a self-contained folder ready for ``HfApi.upload_folder``."""
    destination.mkdir(parents=True, exist_ok=True)
    _copy_model_files(model_dir, destination)

    if include_tokenizer and not any((destination / name).is_file() for name in TOKENIZER_FILES):
        from transformers import AutoTokenizer

        tokenizer_source = config.get("tokenizer_checkpoint") or config.get("checkpoint")
        if not tokenizer_source:
            raise ValueError("Config must define checkpoint or tokenizer_checkpoint")
        AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(destination)

    training_dir = destination / "training"
    training_dir.mkdir()
    shutil.copy2(config_path, training_dir / "config.yaml")
    shutil.copy2(trainer_state_path, training_dir / "trainer_state.json")
    with (training_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (destination / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(build_model_card(repo_id, config, metrics))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config.yaml")
    parser.add_argument("--repo", required=True, help="Hub model repo id (owner/name)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Model directory (default: <experiment>/<output_dir>/final)",
    )
    parser.add_argument(
        "--metrics-from",
        choices=("best", "last"),
        default="best",
        help="Evaluation record displayed in the model card (default: best)",
    )
    parser.add_argument("--private", action="store_true", help="Create a private Hub repo")
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Do not fetch/bundle the tokenizer (only safe if model-dir contains it)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage locally without uploading",
    )
    args = parser.parse_args(argv)

    try:
        config_path = args.config.resolve()
        config = load_experiment_config(config_path)
        output_dir, model_dir = resolve_paths(config_path, config, args.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {model_dir / 'config.json'}")
        if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("pytorch_model*.bin")):
            raise FileNotFoundError(f"No model weights found in {model_dir}")

        trainer_state_path = find_trainer_state(output_dir)
        with trainer_state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        metrics = extract_metrics(state, args.metrics_from)

        with tempfile.TemporaryDirectory(prefix="layer-model-upload-") as temporary:
            upload_dir = Path(temporary)
            prepare_upload_folder(
                upload_dir,
                model_dir,
                config_path,
                config,
                trainer_state_path,
                metrics,
                args.repo,
                include_tokenizer=not args.no_tokenizer,
            )
            print(f"Model: {model_dir}")
            print(f"Trainer state: {trainer_state_path}")
            print(f"Selected metrics: {json.dumps(metrics['selected_evaluation'], sort_keys=True)}")
            if args.dry_run:
                print("Dry run complete — validated and staged all upload artifacts.")
                return 0

            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=args.repo,
                repo_type="model",
                private=args.private,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=args.repo,
                repo_type="model",
                folder_path=upload_dir,
                commit_message=f"Upload {config.get('experiment_name', 'trained model')}",
            )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"upload_model: {exc}", file=sys.stderr)
        return 1

    print(f"Pushed model, metrics, parameters, and model card to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

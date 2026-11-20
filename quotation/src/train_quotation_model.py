#!/usr/bin/env python3
"""Fine-tune mmBERT on Quotation with BIOE labels and boundary-weighted loss.

    python train.py --prepare-only     # data + schedule + weights, no GPU work
    python train.py --smoke            # 200 steps + one eval, proves the loop
    python train.py                    # the run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]   # the quotation/ folder

if importlib.util.find_spec("quotation_lib") is None:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quotation_lib.bioe_layer import (  # noqa: E402
    BioeScheme,
    build_scheme,
    load_bioe_splits,
    pinned_class_weights,
    viterbi_decode,
    evaluate_logits,
    compute_metrics_factory,
    bias_vector,
    sweep_decode,
    best_decode,
)
from quotation_lib.window_data import (  # noqa: E402
    require_pinned_revision,
    steps_per_epoch,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def code_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def configure_wandb(cfg: dict) -> None:
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])


def build_schedule(cfg: dict, n_train: int, n_val: int) -> dict:
    batch = int(cfg["train_batch_size"])
    accum = int(cfg["gradient_accumulation_steps"])
    per_epoch = steps_per_epoch(n_train, batch, accum)
    epochs = int(cfg["num_train_epochs"])
    reference = int(cfg.get("warmup_reference_epochs", epochs))
    warmup = int(round(per_epoch * float(cfg.get("warmup_ratio", 0.2)) * reference))
    evals_per_epoch = max(1, int(cfg.get("evals_per_epoch", 4)))
    eval_steps = max(1, per_epoch // evals_per_epoch)
    patience_epochs = int(cfg.get("early_stopping_patience", 4))
    patience_evals = patience_epochs * evals_per_epoch
    print(f"windows: train={n_train} val={n_val}")
    print(
        f"steps/epoch={per_epoch}  total={per_epoch * epochs}  "
        f"(batch={batch} accum={accum} epochs={epochs})"
    )
    print(
        f"warmup_steps={warmup}  "
        f"({cfg.get('warmup_ratio', 0.2)} × {per_epoch} × "
        f"{reference} reference epochs, not × {epochs})"
    )
    print(f"eval_steps={eval_steps}  ({evals_per_epoch}× per epoch)")
    print(
        f"early stopping: {patience_epochs} epochs → {patience_evals} evals "
        f"on {cfg.get('metric_for_best_model', 'iou50_f1')} "
        "(the HF callback counts evals, not epochs)"
    )
    return {
        "steps_per_epoch": per_epoch,
        "total_steps": per_epoch * epochs,
        "warmup_steps": warmup,
        "eval_steps": eval_steps,
        "evals_per_epoch": evals_per_epoch,
        "patience_evals": patience_evals,
    }


def load_backbone(cfg: dict, scheme) -> object:
    from transformers import AutoModelForTokenClassification

    base_kwargs = {
        "num_labels": scheme.n_labels,
        "id2label": scheme.id2label,
        "label2id": scheme.label2id,
    }
    requested = cfg.get("attn_implementation")
    if requested:
        base_kwargs["attn_implementation"] = requested
    try:
        return AutoModelForTokenClassification.from_pretrained(
            cfg["checkpoint"], **base_kwargs
        )
    except Exception as flash_error:  # noqa: BLE001
        if requested != "flash_attention_2":
            raise
        fallback = cfg.get("attn_fallback_implementation", "sdpa")
        print(
            "Flash Attention 2 could not be initialized; "
            f"retrying with {fallback!r}. Original error: {flash_error}",
            file=sys.stderr,
        )
        base_kwargs["attn_implementation"] = fallback
        return AutoModelForTokenClassification.from_pretrained(
            cfg["checkpoint"], **base_kwargs
        )


def write_run_config(cfg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def _headline(name: str, metrics: dict) -> str:
    return (
        f"{name:12s} f1={metrics.get('iou50_f1', 0.0):.4f}  "
        f"P={metrics.get('iou50_precision', 0.0):.4f}  "
        f"R={metrics.get('iou50_recall', 0.0):.4f}  "
        f"token_recall={metrics.get('token_recall_inside_gold', 0.0):.4f}  "
        f"pred/gold={metrics.get('iou50_overproposal', 0.0):.2f}x  "
        f"n_gold={int(metrics.get('iou50_n_gold', 0))}  "
        f"e_recall={metrics.get('e_recall', 0.0):.4f}  "
        f"f1_stitched={metrics.get('iou50_f1_stitched', 0.0):.4f}  "
        f"f1_downgrade={metrics.get('iou50_f1_downgrade', 0.0):.4f}"
    )


def run_final_evaluation(trainer, scheme, cfg: dict, val, test, output_dir: Path) -> dict:
    import numpy as np

    print("\n=== final evaluation ===")
    saved_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    try:
        val_out = trainer.predict(val.dataset, metric_key_prefix="valfinal")
        val_rows = sweep_decode(
            val_out.predictions, val_out.label_ids, scheme,
            book_ids=val.book_ids, window_indices=val.window_indices
        )
        chosen = best_decode(val_rows)
        penalty = float(chosen["break_penalty"])
        bias = float(chosen["bias"])
        print(
            f"decode chosen on VALIDATION: break_penalty={penalty:g} bias={bias:g} "
            f"(val iou50_f1 {chosen['iou50_f1']:.4f})"
        )

        label_bias = bias_vector(scheme, bias) if bias else None
        val_metrics = evaluate_logits(
            val_out.predictions,
            val_out.label_ids,
            scheme,
            break_penalty=penalty,
            label_bias=label_bias,
            book_ids=val.book_ids,
            window_indices=val.window_indices,
        )

        test_out = trainer.predict(test.dataset, metric_key_prefix="testfinal")
        test_metrics = evaluate_logits(
            test_out.predictions,
            test_out.label_ids,
            scheme,
            break_penalty=penalty,
            label_bias=label_bias,
            book_ids=test.book_ids,
            window_indices=test.window_indices,
        )
        test_rows = sweep_decode(
            test_out.predictions, test_out.label_ids, scheme,
            book_ids=None, window_indices=None
        )
    finally:
        trainer.compute_metrics = saved_metrics

    print(_headline("VALIDATION", val_metrics))
    print(_headline("TEST", test_metrics))
    if "iou50_f1_excl_top_book" in val_metrics:
        print(
            f"  validation without its largest book: "
            f"{val_metrics['iou50_f1_excl_top_book']:.4f} "
            f"over {int(val_metrics.get('excl_top_book_n_gold', 0))} gold spans"
        )
    if "iou50_f1_excl_top_book" in test_metrics:
        print(
            f"  test without its largest book:       "
            f"{test_metrics['iou50_f1_excl_top_book']:.4f} "
            f"over {int(test_metrics.get('excl_top_book_n_gold', 0))} gold spans"
        )
    best_on_test = best_decode(test_rows)
    print(
        f"  (diagnostic only) best decode measured ON test would be "
        f"penalty={best_on_test['break_penalty']:g} bias={best_on_test['bias']:g} "
        f"→ {best_on_test['iou50_f1']:.4f}. Not the headline."
    )

    # Read the joint-model rescore results if they exist
    joint_results = {}
    joint_file = output_dir / "joint_model_on_new_test.json"
    if joint_file.exists():
        try:
            joint_results = json.loads(joint_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    payload = {
        "experiment": cfg.get("experiment_name"),
        "code_commit": cfg.get("code_commit"),
        "target_layer": scheme.layer,
        "dataset": {
            "repo": cfg.get("dataset_repo"),
            "config": cfg.get("dataset_config_name"),
            "revision": cfg.get("dataset_revision"),
        },
        "best_val_metric_in_loop": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "decode_chosen_on_validation": {"break_penalty": penalty, "bias": bias},
        "validation": val_metrics,
        "test": test_metrics,
        "validation_sweep": val_rows,
        "test_sweep_diagnostic_only": test_rows,
        "success_criteria": cfg.get("success_criteria", {}),
        "joint_model_on_new_test": joint_results.get("joint_model_on_new_test"),
        "joint_model_on_new_test_masked3": joint_results.get("joint_model_on_new_test_masked3"),
    }

    def _plain(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "test_results.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_plain),
        encoding="utf-8",
    )
    print(f"wrote {target}")
    return payload


def push_final(cfg: dict, final_dir: Path) -> None:
    repo_id = cfg.get("hub_model_id")
    if not repo_id:
        print(
            "hub_model_id is not set — the model stays on this box only."
        )
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=str(repo_id),
            private=bool(cfg.get("hub_private", True)),
            exist_ok=True,
        )
        api.upload_folder(repo_id=str(repo_id), folder_path=str(final_dir))
        print(f"pushed {final_dir} → https://huggingface.co/{repo_id}")
    except Exception as error:  # noqa: BLE001
        print(f"Hub push failed ({error}); the model is still at {final_dir}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train_quotation_model.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="skip the end-of-run test evaluation (it is on by default)",
    )
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)
    cfg["code_commit"] = code_commit()
    require_pinned_revision(cfg.get("dataset_revision"))

    scheme = build_scheme(str(cfg["target_layer"]))
    print(f"code_commit={cfg['code_commit']}")
    print(
        f"dataset {cfg['dataset_repo']} config={cfg['dataset_config_name']} "
        f"revision={cfg['dataset_revision']}"
    )
    print(f"target layer: {scheme.layer} → labels {list(scheme.labels)}")
    criteria = cfg.get("success_criteria") or {}
    if criteria:
        print("success_criteria:")
        for key, value in criteria.items():
            print(f"  {key}: {value}")

    # Load BIOE splits
    splits = load_bioe_splits(cfg)
    train = splits["train"]
    val = splits.get("validation")
    if val is None:
        raise SystemExit("windowed dataset has no validation split")
    test = splits.get("test")
    
    want_test = (
        bool(cfg.get("evaluate_test_at_end", True))
        and not cli_args.no_test
        and not cli_args.smoke
    )
    if want_test and test is None:
        raise SystemExit(
            "evaluate_test_at_end is on but the windowed dataset has no test split"
        )
        
    if cli_args.limit_windows:
        train = train.select(cli_args.limit_windows)
        val = val.select(cli_args.limit_windows)
        if test is not None:
            test = test.select(cli_args.limit_windows)

    if cli_args.smoke:
        val = val.select(64)

    train_ds = train.dataset
    val_ds = val.dataset
    test_ds = test.dataset if test is not None else None

    # Class weights
    weight_values = pinned_class_weights(cfg["class_weights"], scheme)
    print(
        "class weights: "
        + ", ".join(
            f"{scheme.labels[i]}={weight_values[i]:.4f}"
            for i in range(scheme.n_labels)
        )
    )

    schedule = build_schedule(cfg, len(train_ds), len(val_ds))

    # Prepare-only assertions
    if cli_args.prepare_only:
        # Check assertions: 4 labels, revision w2.0, 6195/645/657 windows, weights, warmup 1548, eval_steps 193, 16 evals patience, B==E per split
        assert scheme.n_labels == 4, f"Expected 4 labels, got {scheme.n_labels}"
        assert cfg["dataset_revision"] == "w2.0", f"Expected revision w2.0, got {cfg['dataset_revision']}"
        assert len(train_ds) == 6195, f"Expected 6195 train windows, got {len(train_ds)}"
        assert len(val_ds) == 645, f"Expected 645 val windows, got {len(val_ds)}"
        if test_ds is not None:
            assert len(test_ds) == 657, f"Expected 657 test windows, got {len(test_ds)}"
        assert weight_values == [0.07, 5.0, 1.0, 5.0], f"Expected weights [0.07, 5.0, 1.0, 5.0], got {weight_values}"
        # ceil(6195 / 2) // 4 = 774 steps/epoch, × 0.2 × 10 reference epochs
        assert schedule["warmup_steps"] == 1548, f"Expected 1548 warmup steps, got {schedule['warmup_steps']}"
        assert schedule["eval_steps"] == 193, f"Expected 193 eval steps, got {schedule['eval_steps']}"
        assert schedule["patience_evals"] == 16, f"Expected 16 patience evals, got {schedule['patience_evals']}"
        print("All prepare-only assertions passed successfully!")
        return 0

    import numpy as np  # noqa: F401
    import torch
    from torch import nn
    from transformers import (
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=cfg.get("pad_to_multiple_of", 8)
    )
    model = load_backbone(cfg, scheme)
    weights = torch.tensor(weight_values, dtype=torch.float)
    n_labels = scheme.n_labels

    positive_ids = (scheme.b_id, scheme.i_id, scheme.e_id)

    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reset_signal_counters()

        def _reset_signal_counters(self) -> None:
            self._sig_steps = 0
            self._sig_empty = 0
            self._sig_positive_tokens = 0
            self._sig_windows = 0
            self._sig_windows_with_positive = 0
            self._sig_micro_in_step = getattr(self, "_sig_micro_in_step", 0)
            self._sig_step_positive = getattr(self, "_sig_step_positive", 0)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=weights.to(logits.device),
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, n_labels), labels.view(-1))

            if model.training:
                with torch.no_grad():
                    positive = torch.zeros_like(labels, dtype=torch.bool)
                    for label_id in positive_ids:
                        positive |= labels == label_id
                    per_window = positive.sum(dim=-1)
                    total = int(per_window.sum())
                    self._sig_windows += int(labels.size(0))
                    self._sig_windows_with_positive += int((per_window > 0).sum())
                    self._sig_positive_tokens += total
                    self._sig_step_positive += total
                    self._sig_micro_in_step += 1
                    accum = max(1, int(self.args.gradient_accumulation_steps))
                    if self._sig_micro_in_step >= accum:
                        self._sig_steps += 1
                        self._sig_empty += int(self._sig_step_positive == 0)
                        self._sig_micro_in_step = 0
                        self._sig_step_positive = 0
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if self._sig_steps:
                logs = {
                    **logs,
                    "signal_empty_step_frac": self._sig_empty / self._sig_steps,
                    "signal_positive_tokens_per_step": (
                        self._sig_positive_tokens / max(self._sig_steps, 1)
                    ),
                    "signal_windows_with_positive_frac": (
                        self._sig_windows_with_positive / max(self._sig_windows, 1)
                    ),
                }
                self._reset_signal_counters()
            return super().log(logs, *args, **kwargs)

    configure_wandb(cfg)
    if cfg.get("tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = REPO_ROOT / cfg["output_dir"]
    smoke = bool(cli_args.smoke)
    if smoke:
        print(f"smoke: 200 steps, one eval on {len(val)} val windows")
    val.check_alignment()
    write_run_config(cfg, output_dir)

    training_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=int(schedule["warmup_steps"]),
        max_grad_norm=cfg["max_grad_norm"],
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=cfg.get("dataloader_pin_memory", True),
        eval_strategy="steps",
        eval_steps=200 if smoke else int(schedule["eval_steps"]),
        save_strategy="no" if smoke else cfg.get("save_strategy", "steps"),
        save_steps=int(schedule["eval_steps"]),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=False if smoke else cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "iou50_f1"),
        greater_is_better=cfg.get("greater_is_better", True),
        seed=cfg["seed"],
        logging_steps=1 if smoke else cfg.get("logging_steps", 20),
        remove_unused_columns=False,
        report_to=[] if smoke else cfg.get("report_to", []),
        run_name=cfg["experiment_name"] + ("-smoke" if smoke else ""),
    )
    if smoke:
        training_kwargs["max_steps"] = 200
        training_kwargs["warmup_steps"] = 20
    training_args = TrainingArguments(**training_kwargs)

    callbacks = []
    if not smoke:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(schedule["patience_evals"])
            )
        )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(
            scheme,
            break_penalty=float(cfg.get("in_loop_break_penalty", 2.0)),
            book_ids=val.book_ids,
            window_indices=val.window_indices,
        ),
        callbacks=callbacks,
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    elapsed = time.perf_counter() - started

    if smoke:
        metrics = trainer.evaluate()
        print(json.dumps({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2))
        print(f"smoke: 200 steps in {elapsed:.0f}s ({elapsed / 200:.2f}s/step)")
        return 0

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"training finished in {elapsed / 3600:.2f} h; best = {trainer.state.best_metric}")

    if test is not None:
        run_final_evaluation(trainer, scheme, cfg, val, test, output_dir)

    push_final(cfg, final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

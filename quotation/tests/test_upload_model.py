from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quotation_lib.upload_model import (
    build_model_card,
    extract_metrics,
    find_trainer_state,
    flatten_config,
    prepare_upload_folder,
    resolve_paths,
)


class UploadModelTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "best_global_step": 20,
            "best_metric": 0.8,
            "best_model_checkpoint": "/tmp/outputs/checkpoint-20",
            "global_step": 30,
            "log_history": [
                {"loss": 1.0, "step": 10},
                {"eval_loss": 0.5, "eval_f1": 0.6, "step": 10, "epoch": 1},
                {"eval_loss": 0.3, "eval_f1": 0.8, "step": 20, "epoch": 2},
                {"eval_loss": 0.4, "eval_f1": 0.7, "step": 30, "epoch": 3},
            ],
        }

    def test_extract_metrics_selects_best_global_step_and_keeps_history(self):
        metrics = extract_metrics(self.state)

        self.assertEqual(metrics["selected_evaluation"]["step"], 20)
        self.assertEqual(len(metrics["evaluation_history"]), 3)
        self.assertNotIn("loss", metrics["evaluation_history"][0])

    def test_extract_metrics_can_select_last_evaluation(self):
        metrics = extract_metrics(self.state, "last")
        self.assertEqual(metrics["selected_evaluation"]["step"], 30)

    def test_flatten_config_preserves_every_leaf(self):
        flattened = dict(
            flatten_config(
                {
                    "learning_rate": 1e-5,
                    "wandb": {"project": "layers", "mode": "online"},
                    "masked_layers": ["BookTitle"],
                    "dataset_revision": None,
                }
            )
        )
        self.assertEqual(
            flattened,
            {
                "learning_rate": 1e-5,
                "wandb.project": "layers",
                "wandb.mode": "online",
                "masked_layers": ["BookTitle"],
                "dataset_revision": None,
            },
        )

    def test_model_card_lists_all_parameters_metrics_and_labels(self):
        config = {
            "experiment_name": "test-run",
            "checkpoint": "owner/base",
            "dataset_repo": "owner/data",
            "learning_rate": 1e-5,
            "wandb": {"project": "layers"},
        }
        card = build_model_card("owner/model", config, extract_metrics(self.state))

        self.assertIn("`learning_rate`", card)
        self.assertIn("`wandb.project`", card)
        self.assertIn("`eval_f1`", card)
        self.assertIn("`B-QUOTE`", card)
        self.assertIn('model_id = "owner/model"', card)

    def test_resolve_paths_uses_config_directory(self):
        config_path = Path("/tmp/experiments/run/config.yaml")
        output, model = resolve_paths(config_path, {"output_dir": "artifacts"})
        self.assertEqual(output, Path("/tmp/experiments/run/artifacts"))
        self.assertEqual(model, Path("/tmp/experiments/run/artifacts/final"))

    def test_find_trainer_state_uses_highest_global_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (10, 30, 20):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step}), encoding="utf-8"
                )
            self.assertEqual(
                find_trainer_state(output).parent.name,
                "checkpoint-30",
            )

    def test_prepare_upload_folder_includes_reproducibility_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            config_path = root / "config.yaml"
            config_path.write_text(
                "experiment_name: test\ncheckpoint: owner/base\n", encoding="utf-8"
            )
            state_path = root / "trainer_state.json"
            state_path.write_text(json.dumps(self.state), encoding="utf-8")
            destination = root / "staged"

            prepare_upload_folder(
                destination,
                model_dir,
                config_path,
                {"experiment_name": "test", "checkpoint": "owner/base"},
                state_path,
                extract_metrics(self.state),
                "owner/model",
                include_tokenizer=False,
            )

            self.assertTrue((destination / "model.safetensors").is_file())
            self.assertTrue((destination / "README.md").is_file())
            self.assertTrue((destination / "training/config.yaml").is_file())
            saved_metrics = json.loads(
                (destination / "training/metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(saved_metrics["evaluation_history"]), 3)


if __name__ == "__main__":
    unittest.main()

"""Phase 5 model and evaluator contracts; never open the TrashNet test split."""

import tempfile
import unittest
import json
from pathlib import Path

import torch
import yaml

from src.dataset import ROOT, TrashNetDataset
from src.compare_models import choose_architecture
from src.evaluate import calculate_metrics, check_test_lock
from src.freeze_selection import build_lock
from src.labels import LABELS
from src.model import ARCHITECTURES, build_model, enter_train_mode, set_trainable_stage
from src.select_threshold import sweep_validation
from src.train import file_sha256


class Phase5Tests(unittest.TestCase):
    def test_all_architectures_have_six_outputs_and_freeze(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_model(pretrained=False, architecture=architecture)
                with torch.inference_mode():
                    self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape),
                                     (1, len(LABELS)))
                set_trainable_stage(model, "classifier")
                enter_train_mode(model, "classifier")
                self.assertTrue(any(p.requires_grad for p in model.parameters()))
                self.assertTrue(any(not p.requires_grad for p in model.parameters()))
                set_trainable_stage(model, "finetune")
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            build_model(pretrained=True, architecture="cnn_baseline")

    def test_e3_config_reuses_phase3_without_changing_mapping(self):
        original = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
        phase5 = yaml.safe_load((ROOT / "configs/phase5_e3.yaml").read_text())
        self.assertEqual(original, phase5)
        e4 = yaml.safe_load((ROOT / "configs/phase5_e4.yaml").read_text())
        self.assertEqual({**original, "augmentation": False}, e4)

    def test_no_augmentation_is_deterministic(self):
        data = TrashNetDataset("train", augmentation=False)
        first, _ = data[0]
        second, _ = data[0]
        self.assertTrue(torch.equal(first, second))

    def test_metrics_and_rejection_known_answers(self):
        rows = [{"split": "val", "true_label": "glass", "predicted_label": "glass",
                 "confidence": 0.95},
                {"split": "val", "true_label": "glass", "predicted_label": "metal",
                 "confidence": 0.55},
                {"split": "val", "true_label": "metal", "predicted_label": "metal",
                 "confidence": 0.80}]
        result = calculate_metrics(rows, 0.60)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["accepted_accuracy"], 1.0)
        self.assertEqual(result["confusion_matrix"][LABELS.index("glass")][LABELS.index("metal")], 1)
        sweep = sweep_validation(rows)
        self.assertFalse(sweep["target_met"])
        self.assertEqual(sweep["selected_threshold"], 0.65)
        with self.assertRaises(ValueError):
            sweep_validation([{**rows[0], "split": "test"}])

    def test_test_split_needs_matching_freeze_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            checkpoint = path / "best.pt"
            manifest = path / "manifest.csv"
            checkpoint.write_bytes(b"ckpt")
            manifest.write_bytes(b"manifest")
            with self.assertRaises(ValueError):
                check_test_lock(None, checkpoint, manifest)
            lock = path / "lock.json"
            lock.write_text('{"allow_trashnet_test": true, "checkpoint_sha256": "wrong", "manifest_sha256": "wrong", "threshold": 0.65}')
            with self.assertRaises(ValueError):
                check_test_lock(lock, checkpoint, manifest)

    def test_freeze_record_joins_matching_validation_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = path / "manifest.csv"
            manifest.write_bytes(b"locked manifest")
            checkpoint = path / "best.pt"
            torch.save({"smoke": False, "manifest_sha256": file_sha256(manifest),
                        "architecture": "mobilenet_v3_small", "seed": 42,
                        "preprocessing": {"image_size": 224}}, checkpoint)
            predictions = path / "predictions.csv"
            predictions.write_text("split,true_label,predicted_label,confidence\nval,glass,glass,0.9\n")
            comparison = path / "comparison.json"
            comparison.write_text(json.dumps({"test_split_used": False,
                                              "selected_architecture": "mobilenet_v3_small",
                                              "selected_seed42_checkpoint_sha256": file_sha256(checkpoint)}))
            metrics = path / "metrics.json"
            metrics.write_text(json.dumps({"split": "val", "n": 1, "macro_f1": 1 / 6,
                                           "checkpoint_sha256": file_sha256(checkpoint),
                                           "manifest_sha256": file_sha256(manifest)}))
            sweep = path / "sweep.json"
            sweep.write_text(json.dumps({"selected_threshold": 0.0, "target_met": True,
                                         "predictions_sha256": file_sha256(predictions),
                                         "validation_n": 1}))
            lock = build_lock(checkpoint, manifest, comparison, metrics, predictions, sweep,
                              "Selected from validation, before test")
            self.assertEqual(lock["threshold"], 0.0)
            lock_path = path / "lock.json"
            lock_path.write_text(json.dumps(lock))
            self.assertEqual(check_test_lock(lock_path, checkpoint, manifest), lock)
            secondary = path / "other.pt"
            secondary.write_bytes(b"secondary checkpoint")
            lock["threshold"] = 0.65
            lock["secondary_test_checkpoints"] = {
                "e1": {"checkpoint_sha256": file_sha256(secondary), "threshold": 0.0}
            }
            lock_path.write_text(json.dumps(lock))
            self.assertEqual(check_test_lock(lock_path, checkpoint, manifest)["threshold"], 0.65)
            self.assertEqual(check_test_lock(lock_path, secondary, manifest)["threshold"], 0.0)
            predictions.write_text("modified")
            with self.assertRaises(ValueError):
                build_lock(checkpoint, manifest, comparison, metrics, predictions, sweep, "selection")

    def test_three_seed_selection_uses_f1_tolerance_then_latency(self):
        def runs(score, latency, parameters):
            return [{"seed": seed, "macro_f1": score,
                     "median_latency_ms": latency if seed == 42 else None,
                     "parameter_count": parameters if seed == 42 else None}
                    for seed in (42, 123, 2026)]
        result = choose_architecture({"e1": runs(.70, 1.0, 390000),
                                      "e2": runs(.86, 7.0, 11000000),
                                      "e3": runs(.85, 4.0, 1500000),
                                      "e4": runs(.83, None, None)})
        self.assertEqual(result["selected_experiment"], "e3")
        self.assertEqual(result["eligible"], ["e2", "e3"])
        self.assertAlmostEqual(result["e3_minus_e4_mean_macro_f1"], .02)
        with self.assertRaises(ValueError):
            choose_architecture({"e1": runs(.70, 1.0, 390000)})


if __name__ == "__main__":
    unittest.main()

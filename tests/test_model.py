"""Model, checkpoint and inference checks; never download pretrained weights."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, ROOT
from src.labels import LABELS
from src.model import (ARCHITECTURE, build_model, enter_train_mode,
                       preprocessing_metadata, set_trainable_stage)
from src.predict import load_predictor, predict_image


class ModelTests(unittest.TestCase):
    def test_six_outputs_and_freeze(self):
        model = build_model(pretrained=False)
        self.assertEqual(model.classifier[-1].out_features, len(LABELS))
        set_trainable_stage(model, "classifier")
        enter_train_mode(model, "classifier")
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertFalse(model.features.training)
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        set_trainable_stage(model, "finetune")
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))

    def test_checkpoint_roundtrip_matches_scores(self):
        image = ROOT / "data/raw/dataset-resized/glass/glass105.jpg"
        if not image.exists():
            self.skipTest("TrashNet is not prepared")
        model = build_model(pretrained=False).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roundtrip.pt"
            torch.save({"architecture": ARCHITECTURE, "labels": list(LABELS),
                        "preprocessing": preprocessing_metadata(),
                        "model_state": model.state_dict()}, path)
            loaded_model, transform, _ = load_predictor(path)
            original_scores = predict_image(model, transform, image)["scores"]
            loaded_scores = predict_image(loaded_model, transform, image)["scores"]
            for label in LABELS:
                self.assertAlmostEqual(original_scores[label], loaded_scores[label], places=7)
            self.assertAlmostEqual(sum(loaded_scores.values()), 1.0, places=6)
            self.assertEqual(tuple(IMAGENET_MEAN), tuple(preprocessing_metadata()["mean"]))
            self.assertEqual(tuple(IMAGENET_STD), tuple(preprocessing_metadata()["std"]))

    def test_smoke_resume_preserves_history_if_available(self):
        checkpoint = ROOT / "runs/phase3_smoke_v2/last.pt"
        if not checkpoint.exists():
            self.skipTest("Local resume smoke run not present")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual([row["stage"] for row in payload["history"]],
                         ["classifier", "finetune"])
        self.assertEqual(payload["stage"], "complete")
        self.assertTrue(payload["smoke"])


if __name__ == "__main__":
    unittest.main()

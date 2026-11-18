"""Cover the web specialist's routing rules without downloading weights."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from src.bottle_routing import BottleRefinement
from src.web_app import Classifier, InputError


class BottleRoutingTests(unittest.TestCase):
    def setUp(self):
        self.router = BottleRefinement(Path("unused.pt"), 0.65, "cpu")
        self.image = Image.new("RGB", (16, 16))
        self.base = {"label": "glass", "confidence": 0.925,
                     "scores": {"glass": 0.925, "plastic": 0.07, "cardboard": 0.005}}
        self.timing = {"model_ms": 1.0, "preprocessing_ms": 0.0}

    def configure_detector(self, *scores):
        detector = Mock(return_value=[{"labels": torch.tensor([1] * len(scores)),
                                       "scores": torch.tensor(scores)}])
        self.router.detector = detector, Mock(), 1
        return detector

    def test_skips_detection_when_e2_scores_do_not_indicate_the_material_pair(self):
        detector = self.configure_detector(0.90)
        for scores in ({"glass": 0.94, "plastic": 0.04, "cardboard": 0.02},
                       {"glass": 0.55, "plastic": 0.14, "cardboard": 0.31},
                       {"glass": 0.55, "plastic": 0.30, "cardboard": 0.15}):
            base = {**self.base, "scores": scores}
            self.assertIs(self.router.refine(self.image, base, self.timing), base)
        detector.assert_not_called()

    def test_does_not_consult_e5_for_zero_or_multiple_bottles(self):
        with patch("src.bottle_routing.predict_pil") as predict:
            for scores in ((), (0.25,), (0.82, 0.22)):
                with self.subTest(scores=scores):
                    self.configure_detector(*scores)
                    self.assertIs(self.router.refine(self.image, self.base, self.timing), self.base)
            predict.assert_not_called()

    def test_consulted_e5_can_update_the_pair_but_only_above_its_threshold(self):
        self.configure_detector(0.33, 0.12)
        self.router.specialist = Mock(), Mock()
        with patch("src.bottle_routing.predict_pil") as predict:
            for confidence, applied in ((0.60, False), (0.68, True)):
                with self.subTest(confidence=confidence):
                    predict.return_value = (
                        {"label": "plastic", "confidence": confidence,
                         "scores": {"plastic": confidence, "glass": 1 - confidence}},
                        {"model_ms": 2.0, "preprocessing_ms": 1.0},
                    )
                    result = self.router.refine(self.image, self.base, self.timing)
                    self.assertEqual(result["routing"]["applied"], applied)
                    self.assertEqual(result["routing"]["base_label"], "glass")
                    self.assertEqual(result["label"], "plastic" if applied else "glass")
                    self.assertEqual(result["scores"]["cardboard"], 0.005)
                    self.assertAlmostEqual(sum(result["scores"].values()), 1.0)
                    if not applied:
                        self.assertEqual(result["scores"], self.base["scores"])

    def test_e5_is_an_internal_specialist_even_when_ready(self):
        with patch.object(BottleRefinement, "from_workspace", return_value=self.router):
            classifier = Classifier("cpu")
        self.assertEqual(set(classifier.options), {"e1", "e2", "e3", "e4"})
        with self.assertRaises(InputError):
            classifier.option("e5")

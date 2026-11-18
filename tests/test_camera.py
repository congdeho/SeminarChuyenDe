"""Phase 4 camera logic tests; no physical camera is required."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
from src.camera import (PredictionSmoother, central_roi_bounds,
                        predict_bgr_roi, run_camera)
from src.capture_dev import validate_request
from src.dataset import ROOT
from src.labels import LABELS
from src.predict import load_predictor, predict_image


def fake_prediction(label: str, score: float) -> dict:
    remainder = (1 - score) / (len(LABELS) - 1)
    return {"label": label, "confidence": score,
            "scores": {name: score if name == label else remainder for name in LABELS}}


class CameraTests(unittest.TestCase):
    def test_roi_coordinates(self):
        self.assertEqual(central_roi_bounds((480, 640, 3), 0.5), (200, 120, 240, 240))
        with self.assertRaises(ValueError):
            central_roi_bounds((480, 640, 3), 0)

    def test_smoothing_resets_stale_label(self):
        smoother = PredictionSmoother(window=10, min_votes=7, threshold=0.65)
        for _ in range(9):
            self.assertIsNone(smoother.update(fake_prediction("glass", 0.9)).label)
        self.assertEqual(smoother.update(fake_prediction("glass", 0.9)).label, "glass")
        for _ in range(4):
            state = smoother.update(fake_prediction("plastic", 0.9))
        self.assertEqual(state.status, "uncertain")
        self.assertIsNone(state.label)
        for _ in range(4):
            state = smoother.update(fake_prediction("plastic", 0.9))
        self.assertEqual(state.status, "stable")
        self.assertEqual(state.label, "plastic")
        smoother.reset()
        self.assertEqual(smoother.update(fake_prediction("plastic", 0.9)).status, "warming_up")

    def test_threshold_is_not_object_detection(self):
        smoother = PredictionSmoother(window=10, min_votes=7, threshold=0.65)
        for _ in range(10):
            state = smoother.update(fake_prediction("glass", 0.6))
        self.assertEqual(state.status, "uncertain")

    def test_capture_requires_prior_label_and_object_id(self):
        validate_request("glass", "glass_dev_01", "S1", "wood", "window", 5)
        with self.assertRaises(ValueError):
            validate_request("glass", "plastic_dev_01", "S1", "wood", "window", 5)
        with self.assertRaises(ValueError):
            validate_request("glass", "glass_dev_01", "S1", "wood", "window", 6)

    def test_saved_roi_matches_image_predictor(self):
        checkpoint = ROOT / "runs/local_pretrained_probe/best.pt"
        sample = ROOT / "data/raw/dataset-resized/glass/glass105.jpg"
        if not checkpoint.exists() or not sample.exists():
            self.skipTest("Full checkpoint or TrashNet sample is unavailable")
        model, transform, _ = load_predictor(checkpoint)
        bgr = cv2.imread(str(sample))
        with tempfile.TemporaryDirectory() as folder:
            roi_file = Path(folder) / "roi.png"
            self.assertTrue(cv2.imwrite(str(roi_file), bgr))
            camera_prediction, _ = predict_bgr_roi(model, transform, bgr)
            image_prediction = predict_image(model, transform, roi_file)
            for label in LABELS:
                self.assertAlmostEqual(camera_prediction["scores"][label],
                                       image_prediction["scores"][label], places=7)

    def test_headless_video_pipeline(self):
        checkpoint = ROOT / "runs/local_pretrained_probe/best.pt"
        sample = ROOT / "data/raw/dataset-resized/glass/glass105.jpg"
        if not checkpoint.exists() or not sample.exists():
            self.skipTest("Full checkpoint or TrashNet sample is unavailable")
        frame = cv2.imread(str(sample))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / "input.mp4"
            output = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10.0, (frame.shape[1], frame.shape[0]))
            if not output.isOpened():
                self.skipTest("OpenCV MP4 writer is unavailable")
            for _ in range(8):
                output.write(frame)
            output.release()
            with redirect_stdout(StringIO()):
                summary = run_camera(checkpoint_path=checkpoint, input_video=video,
                                     run_dir=folder / "run", device="cpu", headless=True,
                                     max_frames=5, record_video=True, window=3,
                                     min_votes=2, threshold=0.0)
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(summary["stop_reason"], "frame_limit")
            self.assertTrue((folder / "run/demo.mp4").is_file())
            self.assertTrue((folder / "run/summary.json").is_file())
            self.assertEqual(len((folder / "run/predictions.csv").read_text().splitlines()), 6)
            self.assertGreater(summary["effective_fps"], 0)


if __name__ == "__main__":
    unittest.main()

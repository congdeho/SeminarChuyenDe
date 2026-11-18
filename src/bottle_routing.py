"""Check for a single bottle before consulting the plastic/glass specialist."""

import hashlib
import json
from functools import cached_property
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from src.dataset import ROOT
from src.material_model import BOTTLE_ARCHITECTURE
from src.predict import load_predictor, predict_pil

WEIGHTS = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
DETECTOR_FILE = ROOT / ".cache/torch/hub/checkpoints" / WEIGHTS.url.rsplit("/", 1)[1]
DETECTOR_SHA256 = "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"
BOTTLE_RUN = ROOT / "runs/material_bottles"
MATERIAL_LABELS = {"glass", "plastic"}
MIN_ALTERNATIVE_SCORE = 0.05
MIN_PAIR_SCORE = 0.90
MIN_BOTTLE_SCORE = 0.30
MIN_EXTRA_BOTTLE_SCORE = 0.20


def _verify_checksum(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected checkpoint checksum: {path.name}")


class BottleRefinement:
    def __init__(self, checkpoint: Path, threshold: float, device: str):
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_workspace(cls, device: str):
        checkpoint = BOTTLE_RUN / "bottle_only.pt"
        lock = json.loads((BOTTLE_RUN / "bottle_only_lock.json").read_text(encoding="utf-8"))
        _verify_checksum(checkpoint, lock["checkpoint_sha256"])
        _verify_checksum(DETECTOR_FILE, DETECTOR_SHA256)
        threshold = float(lock["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("Invalid bottle specialist threshold")
        return cls(checkpoint, threshold, device)

    @cached_property
    def detector(self):
        categories = WEIGHTS.meta["categories"]
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=len(categories),
        )
        model.load_state_dict(torch.load(DETECTOR_FILE, map_location="cpu", weights_only=True))
        return model.eval(), WEIGHTS.transforms(), categories.index("bottle")

    @cached_property
    def specialist(self):
        model, transform, checkpoint = load_predictor(self.checkpoint, self.device)
        if checkpoint["architecture"] != BOTTLE_ARCHITECTURE or checkpoint.get("smoke"):
            raise ValueError("Expected the trained bottle specialist checkpoint")
        return model, transform

    def refine(self, image: Image.Image, prediction: dict, timing: dict) -> dict:
        scores = prediction["scores"]
        first, second = sorted(scores, key=scores.__getitem__, reverse=True)[:2]
        if {first, second} != MATERIAL_LABELS or scores[second] < MIN_ALTERNATIVE_SCORE:
            return prediction
        pair_score = scores[first] + scores[second]
        if pair_score < MIN_PAIR_SCORE:
            return prediction

        detector, transform, bottle_id = self.detector
        start = perf_counter()
        # Bound preprocessing memory for full-resolution uploads.
        detector_input = ImageOps.contain(image, (640, 640)) if max(image.size) > 640 else image
        with torch.inference_mode():
            detected = detector([transform(detector_input)])[0]
        bottles = detected["scores"][detected["labels"] == bottle_id]
        candidates = bottles[bottles >= MIN_EXTRA_BOTTLE_SCORE]
        timing["model_ms"] += (perf_counter() - start) * 1000
        if candidates.numel() != 1 or candidates.item() < MIN_BOTTLE_SCORE:
            return prediction

        opinion, extra_timing = predict_pil(*self.specialist, image, self.device)
        for stage, elapsed in extra_timing.items():
            timing[stage] += elapsed
        applied = opinion["confidence"] >= self.threshold
        routing = {"model": "e5", "applied": applied,
                   "base_label": prediction["label"], "base_confidence": prediction["confidence"],
                   "specialist_label": opinion["label"], "specialist_confidence": opinion["confidence"]}
        if not applied:
            return {**prediction, "routing": routing}

        updated = scores.copy()
        for label in MATERIAL_LABELS:
            updated[label] = pair_score * opinion["scores"][label]
        winner = max(updated, key=updated.__getitem__)
        return {"scores": updated, "label": winner, "confidence": updated[winner], "routing": routing}

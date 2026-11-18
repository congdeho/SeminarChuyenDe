"""Predict one image with a trained checkpoint, on CPU by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision import transforms

from src.labels import LABELS
from src.material_model import ARCHITECTURE as MATERIAL_ARCHITECTURE, BOTTLE_ARCHITECTURE, load_material_model
from src.model import ARCHITECTURES, build_model


def load_predictor(checkpoint_path: Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    material_options = (MATERIAL_ARCHITECTURE, BOTTLE_ARCHITECTURE)
    if architecture not in (*ARCHITECTURES, *material_options) or checkpoint["labels"] != list(LABELS):
        raise ValueError("Checkpoint architecture/label order does not match this project")
    prep = checkpoint["preprocessing"]
    transform = transforms.Compose([
        transforms.Resize(prep["resize_short_side"]),
        transforms.CenterCrop(prep["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(prep["mean"], prep["std"]),
    ])
    if architecture in material_options:
        model, transform = load_material_model(checkpoint, transform, device)
    else:
        model = build_model(pretrained=False, architecture=architecture)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
    return model, transform, checkpoint


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def predict_pil(model, transform, image: Image.Image, device: str = "cpu") -> tuple[dict, dict]:
    """Shared image/webcam inference. Timing synchronizes asynchronous GPU work."""
    start = perf_counter()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    _synchronize(device)
    after_preprocessing = perf_counter()
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1).squeeze(0).cpu().tolist()
    _synchronize(device)
    after_model = perf_counter()
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    prediction = {"label": LABELS[winner], "confidence": probabilities[winner],
                  "scores": {label: score for label, score in zip(LABELS, probabilities)}}
    timing = {"preprocessing_ms": (after_preprocessing - start) * 1000,
              "model_ms": (after_model - after_preprocessing) * 1000}
    return prediction, timing


def predict_image(model, transform, image_path: Path, device: str = "cpu") -> dict:
    with Image.open(image_path) as image:
        prediction, _ = predict_pil(model, transform, image, device)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, transform, checkpoint = load_predictor(args.checkpoint, args.device)
    result = predict_image(model, transform, args.image, args.device)
    result["checkpoint_smoke"] = checkpoint["smoke"]
    result["checkpoint_manifest_sha256"] = checkpoint["manifest_sha256"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

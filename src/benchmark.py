"""Batch-1 model latency on a validation image; never reads the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from src.dataset import DEFAULT_MANIFEST, ROOT, read_manifest
from src.predict import _synchronize, load_predictor
from src.train import file_sha256


def benchmark(checkpoint_path: Path, device: str, warmup: int = 20,
              iterations: int = 100, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    model, transform, checkpoint = load_predictor(checkpoint_path, device)
    if checkpoint.get("smoke") or checkpoint["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Requires a full checkpoint matching the manifest")
    val_path = next(row["path"] for row in read_manifest(manifest_path) if row["split"] == "val")
    with Image.open(ROOT / val_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        _synchronize(device)
        times = []
        for _ in range(iterations):
            start = perf_counter()
            model(tensor)
            _synchronize(device)
            times.append((perf_counter() - start) * 1000)
    median_ms = float(np.median(times))
    return {"architecture": checkpoint["architecture"], "seed": checkpoint["seed"],
            "device": device, "batch_size": 1, "image_split": "val",
            "warmup": warmup, "iterations": iterations,
            "median_ms": median_ms, "p95_ms": float(np.percentile(times, 95)),
            "model_only_fps_estimate": 1000 / median_ms,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "scope_note": "Model forward only, one repeated validation image; not webcam FPS."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.checkpoint, args.device, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

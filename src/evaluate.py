"""Evaluate one locked checkpoint on TrashNet validation or final test.

Test evaluation requires a separate freeze record with exact checkpoint and
manifest SHA-256. This prevents accidental use of test during model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import DEFAULT_MANIFEST, ROOT, read_manifest
from src.labels import LABELS
from src.predict import load_predictor, predict_image
from src.train import file_sha256

FIELDS = ("split", "path", "object_id", "true_label", "predicted_label", "confidence",
          "accepted", "scores_json")


def calculate_metrics(rows: list[dict], threshold: float = 0.0) -> dict:
    if not rows:
        raise ValueError("No predictions to evaluate")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1]")
    true = [row["true_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    accepted = [row for row in rows if float(row["confidence"]) >= threshold]
    matrix = confusion_matrix(true, predicted, labels=list(LABELS))
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float),
                           where=support != 0)
    report = classification_report(true, predicted, labels=list(LABELS),
                                   output_dict=True, zero_division=0)
    correct = sum(a == b for a, b in zip(true, predicted))
    accepted_correct = sum(row["true_label"] == row["predicted_label"] for row in accepted)
    return {"n": len(rows), "labels": list(LABELS), "accuracy": correct / len(rows),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_normalized": normalized.tolist(),
            "threshold": threshold, "accepted_n": len(accepted),
            "coverage": len(accepted) / len(rows),
            "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
            "correct_and_accepted_rate": accepted_correct / len(rows)}


def check_test_lock(lock_path: Path | None, checkpoint_path: Path,
                    manifest_path: Path) -> dict:
    if lock_path is None:
        raise ValueError("Test split requires --lock with a frozen Phase 5 decision record")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("allow_trashnet_test") is not True:
        raise ValueError("Freeze record does not authorize TrashNet test")
    checkpoint_sha = file_sha256(checkpoint_path)
    authorized = [lock, *lock.get("secondary_test_checkpoints", {}).values()]
    matching = next((item for item in authorized if item.get("checkpoint_sha256") == checkpoint_sha), None)
    if matching is None:
        raise ValueError("Checkpoint does not match freeze record")
    if lock.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Manifest does not match freeze record")
    return {**lock, "threshold": matching["threshold"]}


def evaluate_split(checkpoint_path: Path, split: str, output_dir: Path,
                   manifest_path: Path = DEFAULT_MANIFEST, device: str = "cpu",
                   threshold: float = 0.0, lock_path: Path | None = None) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Only val and test are valid evaluation splits")
    if split == "test":
        lock = check_test_lock(lock_path, checkpoint_path, manifest_path)
        if threshold != float(lock["threshold"]):
            raise ValueError("Test threshold must exactly match the freeze record")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    checkpoint_sha = file_sha256(checkpoint_path)
    manifest_sha = file_sha256(manifest_path)
    model, transform, checkpoint = load_predictor(checkpoint_path, device)
    if checkpoint.get("smoke"):
        raise ValueError("Refusing to evaluate a smoke checkpoint")
    if checkpoint["manifest_sha256"] != manifest_sha:
        raise ValueError("Checkpoint and manifest do not match")
    data = [row for row in read_manifest(manifest_path) if row["split"] == split]
    if not data:
        raise ValueError(f"No {split} images in manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(data, 1):
            result = predict_image(model, transform, ROOT / row["path"], device)
            predictions.append({"split": split, "path": row["path"], "object_id": row["object_id"],
                                "true_label": row["label"], "predicted_label": result["label"],
                                "confidence": result["confidence"],
                                "accepted": result["confidence"] >= threshold,
                                "scores_json": json.dumps(result["scores"])})
            if index % 100 == 0:
                print(f"Evaluated {index}/{len(data)} {split} images", flush=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    metrics = calculate_metrics(predictions, threshold)
    metrics.update({"split": split, "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha, "manifest_sha256": manifest_sha,
                    "architecture": checkpoint["architecture"], "seed": checkpoint["seed"],
                    "device": device, "test_lock": str(lock_path.resolve()) if lock_path else None})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                             encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    result = evaluate_split(args.checkpoint, args.split, args.output_dir, args.manifest,
                            args.device, args.threshold, args.lock)
    print(json.dumps({key: result[key] for key in ("split", "n", "accuracy", "macro_f1",
                                                  "coverage", "accepted_accuracy")}, indent=2))


if __name__ == "__main__":
    main()

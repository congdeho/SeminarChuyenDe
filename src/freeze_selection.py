"""Freeze the validation-chosen checkpoint and threshold before TrashNet test."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from src.dataset import DEFAULT_MANIFEST
from src.evaluate import calculate_metrics
from src.select_threshold import sweep_validation
from src.train import file_sha256


def build_lock(checkpoint_path: Path, manifest_path: Path, comparison_path: Path,
               validation_metrics_path: Path, predictions_path: Path,
               threshold_sweep_path: Path, reason: str) -> dict:
    if not reason.strip():
        raise ValueError("A validation-based selection reason is required")
    checkpoint_sha = file_sha256(checkpoint_path)
    manifest_sha = file_sha256(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    metrics = json.loads(validation_metrics_path.read_text(encoding="utf-8"))
    sweep = json.loads(threshold_sweep_path.read_text(encoding="utf-8"))
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if checkpoint.get("smoke") or checkpoint["manifest_sha256"] != manifest_sha:
        raise ValueError("Checkpoint is smoke or does not match the manifest")
    if (comparison.get("test_split_used") is not False or
            comparison.get("selected_architecture") != checkpoint["architecture"] or
            comparison.get("selected_seed42_checkpoint_sha256") != checkpoint_sha or
            checkpoint["seed"] != 42):
        raise ValueError("Checkpoint is not the seed-42 winner of the three-seed comparison")
    if (metrics.get("split") != "val" or metrics.get("checkpoint_sha256") != checkpoint_sha or
            metrics.get("manifest_sha256") != manifest_sha):
        raise ValueError("Validation metrics do not match checkpoint/manifest")
    if (sweep.get("predictions_sha256") != file_sha256(predictions_path) or
            sweep.get("validation_n") != metrics.get("n")):
        raise ValueError("Threshold sweep does not match validation predictions")
    recomputed_metrics = calculate_metrics(predictions)
    recomputed_sweep = sweep_validation(predictions)
    if (recomputed_metrics["n"] != metrics["n"] or
            abs(recomputed_metrics["macro_f1"] - metrics["macro_f1"]) > 1e-9 or
            recomputed_sweep["selected_threshold"] != sweep.get("selected_threshold") or
            recomputed_sweep["target_met"] != sweep.get("target_met")):
        raise ValueError("Validation metrics or threshold sweep disagree with predictions")
    threshold = sweep.get("selected_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Invalid selected threshold")
    return {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
            "allow_trashnet_test": True,
            "selection_reason": reason.strip(),
            "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
            "manifest": str(manifest_path.resolve()), "manifest_sha256": manifest_sha,
            "architecture": checkpoint["architecture"], "seed": checkpoint["seed"],
            "preprocessing": checkpoint["preprocessing"], "threshold": float(threshold),
            "threshold_target_met_on_validation": bool(sweep["target_met"]),
            "validation_n": metrics["n"], "validation_macro_f1": metrics["macro_f1"],
            "validation_metrics_sha256": file_sha256(validation_metrics_path),
            "validation_predictions_sha256": file_sha256(predictions_path),
            "threshold_sweep_sha256": file_sha256(threshold_sweep_path),
            "comparison_sha256": file_sha256(comparison_path),
            "secondary_test_checkpoints": {
                key: {"checkpoint_sha256": row["checkpoint_sha256"], "threshold": 0.0}
                for key, row in comparison.get("seed42_checkpoints", {}).items()
                if row["checkpoint_sha256"] != checkpoint_sha
            },
            "offline_video": {"roi_scale": 0.70, "window": 10, "min_votes": 7,
                              "threshold": float(threshold)},
            "scope_note": "Locks only the static-image TrashNet test; webcam and camera-domain test were skipped."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--comparison", type=Path, required=True,
                        help="Output of compare_models after all three seeds for E1–E4")
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold-sweep", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = build_lock(args.checkpoint, args.manifest, args.comparison, args.validation_metrics,
                      args.predictions, args.threshold_sweep, args.reason)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: lock[key] for key in ("checkpoint_sha256", "threshold",
                                                "threshold_target_met_on_validation")}, indent=2))


if __name__ == "__main__":
    main()

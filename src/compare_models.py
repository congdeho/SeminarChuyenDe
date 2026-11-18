"""Compare three-seed validation runs and select an architecture without test data."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

from src.dataset import DEFAULT_MANIFEST, ROOT
from src.model import ARCHITECTURE
from src.train import file_sha256

SEEDS = (42, 123, 2026)
EXPERIMENTS = {"e1": "cnn_baseline", "e2": "resnet18",
               "e3": ARCHITECTURE, "e4": ARCHITECTURE}


def choose_architecture(results: dict) -> dict:
    """Require complete E1–E4 data; choose among E1–E3 by the written rule."""
    if set(results) != set(EXPERIMENTS):
        raise ValueError("Selection requires E1–E4")
    for experiment, runs in results.items():
        if {row["seed"] for row in runs} != set(SEEDS) or len(runs) != len(SEEDS):
            raise ValueError(f"{experiment} requires exactly three distinct planned seeds")
    summary = {}
    for experiment, runs in results.items():
        scores = [float(row["macro_f1"]) for row in runs]
        first = next(row for row in runs if row["seed"] == 42)
        summary[experiment] = {"architecture": EXPERIMENTS[experiment],
                               "seed_scores": {str(row["seed"]): row["macro_f1"] for row in runs},
                               "mean_macro_f1": statistics.mean(scores),
                               "sd_macro_f1": statistics.stdev(scores),
                               "seed42_median_latency_ms": first["median_latency_ms"],
                               "seed42_parameter_count": first["parameter_count"]}
    best_mean = max(summary[key]["mean_macro_f1"] for key in ("e1", "e2", "e3"))
    eligible = [key for key in ("e1", "e2", "e3")
                if best_mean - summary[key]["mean_macro_f1"] <= 0.02]
    selected = min(eligible, key=lambda key:
                   (summary[key]["seed42_median_latency_ms"],
                    summary[key]["seed42_parameter_count"]))
    selected_seed42 = next(row for row in results[selected] if row["seed"] == 42)
    return {"selected_experiment": selected,
            "selected_architecture": EXPERIMENTS[selected],
            "selected_seed42_checkpoint_sha256": selected_seed42.get("checkpoint_sha256"),
            "selection_rule": "Within 0.02 mean validation macro-F1 of best; lowest seed42 batch-1 median MPS latency, then parameters.",
            "eligible": eligible, "summary": summary,
            "runs": results,
            "seed42_checkpoints": {key: next(row for row in runs if row["seed"] == 42)
                                   for key, runs in results.items()},
            "e3_minus_e4_mean_macro_f1": summary["e3"]["mean_macro_f1"] - summary["e4"]["mean_macro_f1"],
            "test_split_used": False}


def collect_results(project_root: Path = ROOT, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest_sha = file_sha256(manifest_path)
    results = {}
    for experiment, architecture in EXPERIMENTS.items():
        config_path = project_root / f"configs/phase5_{experiment}.yaml"
        planned_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        runs = []
        for seed in SEEDS:
            run_dir = (project_root / "runs/local_pretrained_probe" if experiment == "e3" and seed == 42
                       else project_root / f"runs/phase5/{experiment}_seed{seed}")
            checkpoint_path = run_dir / "best.pt"
            metrics_path = run_dir / "best_validation_metrics.json"
            summary_path = run_dir / "run_summary.json"
            if not checkpoint_path.exists() or not metrics_path.exists() or not summary_path.exists():
                raise FileNotFoundError(f"Missing full run for {experiment} seed {seed}: {run_dir}")
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if run_summary.get("complete") is not True or run_summary.get("smoke") is not False:
                raise ValueError(f"Run is incomplete or smoke: {run_dir}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (checkpoint.get("smoke") or checkpoint["architecture"] != architecture or
                    checkpoint["seed"] != seed or checkpoint["manifest_sha256"] != manifest_sha or
                    checkpoint["config"] != {**planned_config, "seed": seed}):
                raise ValueError(f"Checkpoint/config mismatch: {checkpoint_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["validation"]
            if metrics["macro_f1"] != checkpoint["best_val_macro_f1"]:
                raise ValueError(f"Validation metrics/checkpoint mismatch: {checkpoint_path}")
            row = {"seed": seed, "macro_f1": metrics["macro_f1"],
                   "accuracy": metrics["accuracy"], "checkpoint": str(checkpoint_path),
                   "checkpoint_sha256": file_sha256(checkpoint_path),
                   "manifest_sha256": manifest_sha,
                   "config_path": str(config_path.relative_to(project_root)),
                   "config_sha256": file_sha256(config_path),
                   "validation_metrics_path": str(metrics_path.relative_to(project_root)),
                   "validation_metrics_sha256": file_sha256(metrics_path),
                   "device": run_summary["device"], "epochs_run": run_summary["epochs_run"]}
            if seed == 42 and experiment != "e4":
                benchmark_path = project_root / f"runs/phase5/{experiment}_seed42_benchmark_mps.json"
                if not benchmark_path.exists():
                    raise FileNotFoundError(f"Missing MPS benchmark: {benchmark_path}")
                benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
                if benchmark["checkpoint_sha256"] != row["checkpoint_sha256"]:
                    raise ValueError(f"Benchmark/checkpoint mismatch: {benchmark_path}")
                row["median_latency_ms"] = benchmark["median_ms"]
                row["parameter_count"] = benchmark["parameter_count"]
            if seed == 42 and experiment == "e4":
                # E4 is an augmentation ablation, not a candidate architecture.
                row["median_latency_ms"] = None
                row["parameter_count"] = None
            runs.append(row)
        results[experiment] = runs
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = choose_architecture(collect_results())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("selected_experiment", "eligible", "e3_minus_e4_mean_macro_f1")}, indent=2))


if __name__ == "__main__":
    main()

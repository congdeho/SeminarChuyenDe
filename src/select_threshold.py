"""Choose a static-image rejection threshold using validation predictions only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluate import calculate_metrics
from src.train import file_sha256


def sweep_validation(predictions: list[dict]) -> dict:
    if not predictions or any(row.get("split") != "val" for row in predictions):
        raise ValueError("Threshold selection requires validation-only predictions")
    candidates = []
    for step in range(20):
        threshold = round(step * 0.05, 2)
        metrics = calculate_metrics(predictions, threshold)
        candidates.append({key: metrics[key] for key in
                           ("threshold", "coverage", "accepted_accuracy", "accepted_n")})
    valid = [row for row in candidates if row["coverage"] >= 0.70 and
             row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= 0.90]
    selected = sorted(valid, key=lambda row: (-row["coverage"], row["threshold"]))[0] if valid else None
    return {"selected_threshold": selected["threshold"] if selected else 0.65,
            "target_met": selected is not None,
            "selection_rule": "Max coverage subject to accepted accuracy >= .90 and coverage >= .70; tie lowest threshold. Fallback .65.",
            "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = sweep_validation(rows)
    result["predictions_sha256"] = file_sha256(args.predictions)
    result["validation_n"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_threshold", "target_met")}, indent=2))


if __name__ == "__main__":
    main()

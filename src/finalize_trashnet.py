"""Apply the frozen Phase 5 protocol after all planned training runs finish."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.camera import run_camera
from src.compare_models import ROOT, choose_architecture, collect_results
from src.dataset import DEFAULT_MANIFEST
from src.error_gallery import make_gallery
from src.evaluate import evaluate_split
from src.freeze_selection import build_lock
from src.train import file_sha256


def write_json(path: Path, content: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion(metrics: dict, output: Path) -> None:
    counts = np.asarray(metrics["confusion_matrix"])
    normalized = np.asarray(metrics["confusion_matrix_normalized"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained")
    for ax, values, title, fmt in zip(axes, (counts, normalized), ("Số ảnh", "Tỷ lệ theo nhãn thật"), ("d", ".0%")):
        ax.imshow(values, cmap="Blues", vmin=0)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    color="white" if value > 0.55 * values.max() else "#102833")
        ax.set(xticks=range(len(metrics["labels"])), yticks=range(len(metrics["labels"])),
               xticklabels=metrics["labels"], yticklabels=metrics["labels"],
               xlabel="Nhãn dự đoán", ylabel="Nhãn thật", title=title)
        ax.tick_params(axis="x", labelrotation=34)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize_video(video_dir: Path, fixture_path: Path) -> dict:
    metadata = load_json(fixture_path.with_suffix(".json"))
    with (video_dir / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    decoded = 0
    video = cv2.VideoCapture(str(video_dir / "demo.mp4"))
    try:
        while video.read()[0]:
            decoded += 1
    finally:
        video.release()
    if decoded != len(predictions) or decoded != metadata["segments"][-1]["end_frame"]:
        raise ValueError("Video frame count disagrees with the fixture or prediction log")
    segments = []
    for segment in metadata["segments"]:
        frames = predictions[segment["start_frame"] - 1:segment["end_frame"]]
        correct_offsets = [i for i, row in enumerate(frames) if row["stable_label"] == segment["label"]]
        segments.append({**segment, "stable_correct_frames": len(correct_offsets),
                         "first_correct_offset_frames": correct_offsets[0] if correct_offsets else None,
                         "last_stable_label": frames[-1]["stable_label"] or None})
    return {"decoded_frames": decoded, "segments": segments, "source": metadata["source"]}


def finalize(output: Path) -> dict:
    comparison = choose_architecture(collect_results())
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "comparison.json"
    if not comparison_path.exists():
        write_json(comparison_path, comparison)
    elif load_json(comparison_path) != comparison:
        raise ValueError("Comparison changed since this output directory was initialized")
    selected = comparison["selected_experiment"]
    checkpoint = Path(comparison["seed42_checkpoints"][selected]["checkpoint"])
    validation = ROOT / f"runs/phase5/{selected}_seed42_val"
    sweep = validation / ("threshold_sweep_v2.json" if selected == "e3" else "threshold_sweep.json")
    lock_path = output / "selection_lock.json"
    candidate = build_lock(checkpoint, DEFAULT_MANIFEST, comparison_path, validation / "metrics.json",
                           validation / "predictions.csv", sweep, comparison["selection_rule"])
    if not lock_path.exists():
        write_json(lock_path, candidate)
    lock = load_json(lock_path)
    if {k: v for k, v in candidate.items() if k != "created_at"} != {
            k: v for k, v in lock.items() if k != "created_at"}:
        raise ValueError("Saved lock disagrees with the current validation evidence")

    fixture = ROOT / "runs/camera_fixture/validation_sequence.mp4"
    video_dir = output / "offline_video"
    if not (video_dir / "summary.json").exists():
        run_camera(checkpoint, input_video=fixture, run_dir=video_dir, device="cpu",
                   headless=True, record_video=True, **lock["offline_video"])
    video_summary = load_json(video_dir / "summary.json")
    if (video_summary["checkpoint_sha256"] != lock["checkpoint_sha256"] or
            any(video_summary[key] != value for key, value in lock["offline_video"].items())):
        raise ValueError("Offline demo disagrees with the locked settings")
    video_checks = summarize_video(video_dir, fixture)

    test = {}
    for experiment, row in comparison["seed42_checkpoints"].items():
        test_dir = output / f"{experiment}_test"
        threshold = lock["threshold"] if experiment == selected else 0.0
        metrics_path = test_dir / "metrics.json"
        if not metrics_path.exists():
            evaluate_split(Path(row["checkpoint"]), "test", test_dir, device="cpu",
                           threshold=threshold, lock_path=lock_path)
        metrics = load_json(metrics_path)
        if (metrics["checkpoint_sha256"] != row["checkpoint_sha256"] or metrics["split"] != "test" or
                metrics["manifest_sha256"] != lock["manifest_sha256"] or metrics["threshold"] != threshold):
            raise ValueError(f"Saved results disagree with the locked protocol: {metrics_path}")
        test[experiment] = metrics
    selected_dir = output / f"{selected}_test"
    fig_path = output / "confusion.png"
    if not fig_path.exists():
        plot_confusion(test[selected], fig_path)
    gallery = output / "test_errors_top12.png"
    if not gallery.exists():
        make_gallery(selected_dir / "predictions.csv", gallery)
    summary = {"comparison": comparison, "lock_sha256": file_sha256(lock_path),
               "test": test, "video": video_checks}
    if not (output / "summary.json").exists():
        write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/phase5/final")
    args = parser.parse_args()
    summary = finalize(args.output)
    selected = summary["comparison"]["selected_experiment"]
    print(json.dumps({"selected": selected, "test_macro_f1": summary["test"][selected]["macro_f1"],
                      "video_frames": summary["video"]["decoded_frames"]}, indent=2))


if __name__ == "__main__":
    main()

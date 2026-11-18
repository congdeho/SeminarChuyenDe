"""Make a repeatable validation-only video for offline camera-pipeline testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.dataset import ROOT, read_manifest
from src.labels import LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/camera_fixture/validation_sequence.mp4")
    parser.add_argument("--frames-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.frames_per_class < 1:
        raise ValueError("frames-per-class must be positive")
    rows = read_manifest()
    selected = [next(row for row in rows if row["split"] == "val" and row["label"] == label)
                for label in LABELS]
    frames = [cv2.imread(str(ROOT / row["path"])) for row in selected]
    if any(frame is None for frame in frames):
        raise RuntimeError("Cannot read a selected validation image")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError("Validation image dimensions differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             20.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create MP4")
    try:
        for frame in frames:
            for _ in range(args.frames_per_class):
                writer.write(frame)
    finally:
        writer.release()
    metadata = {"source": "TrashNet validation only (not camera capture)",
                "frames_per_class": args.frames_per_class, "fps": 20,
                "segments": [{"label": row["label"], "path": row["path"],
                              "start_frame": index * args.frames_per_class + 1,
                              "end_frame": (index + 1) * args.frames_per_class}
                             for index, row in enumerate(selected)]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Created {args.output} ({len(frames) * args.frames_per_class} validation-only frames)")


if __name__ == "__main__":
    main()

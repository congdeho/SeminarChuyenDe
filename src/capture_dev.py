"""Capture labeled camera_dev photos WITHOUT showing model predictions.

Choose a physical object and its label first. Press s for each viewpoint, q to quit.
The final 300-image camera test is deliberately NOT supported here.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2

from src.camera import central_roi_bounds
from src.dataset import ROOT
from src.labels import LABELS

OUTPUT = ROOT / "data/camera_dev"
FIELDS = ("path", "label", "object_id", "split", "session_id", "captured_at",
          "background", "lighting", "label_reviewed", "review_note", "exclude_reason")


def validate_request(label: str, object_id: str, session_id: str, background: str,
                     lighting: str, max_images: int) -> None:
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", object_id):
        raise ValueError("object_id must contain only letters, digits, _ or -")
    if not object_id.startswith(f"{label}_dev_"):
        raise ValueError(f"Use object_id like {label}_dev_01")
    if not all(value.strip() for value in (session_id, background, lighting)):
        raise ValueError("session_id, background and lighting are required")
    if not 1 <= max_images <= 5:
        raise ValueError("Capture 1–5 views per physical object")


def existing_rows(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_capture(label: str, object_id: str, session_id: str, background: str,
                lighting: str, camera_index: int = 0, max_images: int = 5,
                roi_scale: float = 0.70) -> int:
    validate_request(label, object_id, session_id, background, lighting, max_images)
    metadata = OUTPUT / "metadata.csv"
    rows = existing_rows(metadata)
    prior = [row for row in rows if row["object_id"] == object_id]
    if prior and any(row["label"] != label for row in prior):
        raise ValueError("object_id already belongs to another label")
    if len(prior) + max_images > 5:
        raise ValueError(f"{object_id} already has {len(prior)} images; maximum is 5")
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Cannot open webcam; check macOS Camera permission for Terminal")
    saved = 0
    try:
        while saved < max_images:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames")
            x, y, width, height = central_roi_bounds(frame.shape, roi_scale)
            roi = frame[y:y + height, x:x + width]
            preview = frame.copy()
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 180, 0), 2)
            cv2.rectangle(preview, (0, 0), (min(frame.shape[1], 760), 58), (20, 20, 20), -1)
            cv2.putText(preview, f"{object_id} | {label} | {len(prior)+saved}/5",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "s save ROI | q quit | no predictions shown",
                        (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Camera development capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue
            sequence = len(prior) + saved + 1
            output = OUTPUT / "images" / f"{object_id}_{sequence:02d}.png"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), roi):
                raise OSError(f"Could not save {output}")
            row = {"path": output.relative_to(ROOT).as_posix(), "label": label,
                   "object_id": object_id, "split": "camera_dev", "session_id": session_id,
                   "captured_at": datetime.now().astimezone().isoformat(),
                   "background": background, "lighting": lighting,
                   "label_reviewed": "no", "review_note": "", "exclude_reason": ""}
            with metadata.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            saved += 1
            print(f"Saved {output} ({len(prior)+saved}/5)", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"This session saved {saved}; metadata: {metadata}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--roi-scale", type=float, default=0.70)
    args = parser.parse_args()
    run_capture(args.label, args.object_id, args.session_id, args.background,
                args.lighting, args.camera_index, args.max_images, args.roi_scale)


if __name__ == "__main__":
    main()

"""Phase 4 webcam demo with a central ROI, smoothing and local diagnostics.

Keys: q/Esc quit; s save the current ROI (unlabeled); m mark a physical object
switch for response-time measurement; r clear the smoothing window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import ROOT
from src.labels import LABELS
from src.predict import load_predictor, predict_pil

DEFAULT_CHECKPOINT = ROOT / "runs/local_pretrained_probe/best.pt"


@dataclass(frozen=True)
class DisplayState:
    status: str
    label: str | None
    mean_score: float
    votes: int
    count: int


class PredictionSmoother:
    def __init__(self, window: int = 10, min_votes: int = 7, threshold: float = 0.65):
        if window < 1 or not 1 <= min_votes <= window or not 0 <= threshold <= 1:
            raise ValueError("Invalid smoothing settings")
        self.window = window
        self.min_votes = min_votes
        self.threshold = threshold
        self.entries: deque[dict] = deque(maxlen=window)

    def reset(self) -> None:
        self.entries.clear()

    def update(self, prediction: dict) -> DisplayState:
        self.entries.append(prediction)
        count = len(self.entries)
        if count < self.window:
            return DisplayState("warming_up", None, 0.0, 0, count)
        votes = Counter(entry["label"] for entry in self.entries)
        candidate, count_votes = votes.most_common(1)[0]
        mean_score = sum(entry["scores"][candidate] for entry in self.entries) / count
        if count_votes >= self.min_votes and mean_score >= self.threshold:
            return DisplayState("stable", candidate, mean_score, count_votes, count)
        # Never display a stale label after the window becomes uncertain.
        return DisplayState("uncertain", None, mean_score, count_votes, count)


def central_roi_bounds(frame_shape: tuple[int, ...], scale: float = 0.70) -> tuple[int, int, int, int]:
    if not 0 < scale <= 1:
        raise ValueError("ROI scale must be within (0, 1]")
    height, width = frame_shape[:2]
    side = max(1, round(min(height, width) * scale))
    x = (width - side) // 2
    y = (height - side) // 2
    return x, y, side, side


def predict_bgr_roi(model, transform, roi_bgr: np.ndarray, device: str = "cpu") -> tuple[dict, dict]:
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    start = perf_counter()
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    color_ms = (perf_counter() - start) * 1000
    prediction, timing = predict_pil(model, transform, pil, device)
    timing["bgr_to_rgb_ms"] = color_ms
    return prediction, timing


def select_device(value: str) -> str:
    if value != "auto":
        if value not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {value}")
        if value == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this process; choose cpu or run in macOS Terminal")
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return value
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def _draw_overlay(frame: np.ndarray, bounds: tuple[int, int, int, int],
                  prediction: dict, state: DisplayState, fps: float, model_ms: float) -> np.ndarray:
    x, y, w, h = bounds
    result = frame.copy()
    color = (0, 180, 0) if state.status == "stable" else (0, 180, 255)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    display = f"{state.label}: {state.mean_score:.2f}" if state.label else state.status.replace("_", " ")
    lines = [f"State: {display}",
             f"Raw: {prediction['label']} {prediction['confidence']:.2f}  votes: {state.votes}/{state.count}",
             f"Pipeline FPS: {fps:.1f}  model: {model_ms:.1f} ms",
             "q quit | s save ROI | m mark switch | r reset"]
    cv2.rectangle(result, (0, 0), (min(frame.shape[1], 750), 104), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(result, line, (10, 23 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(result, (0, frame.shape[0] - 25),
                  (min(frame.shape[1], 750), frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(result, "Softmax is NOT unknown-object detection", (10, frame.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _write_capture(path: Path, roi_bgr: np.ndarray, prediction: dict,
                   state: DisplayState, bounds: tuple[int, int, int, int],
                   checkpoint_sha: str, source: str, metadata_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), roi_bgr):
        raise OSError(f"Could not save image: {path}")
    row = {"path": str(path.resolve()), "captured_at": datetime.now().astimezone().isoformat(),
           "source": source, "roi_x": bounds[0], "roi_y": bounds[1], "roi_w": bounds[2],
           "roi_h": bounds[3], "raw_label": prediction["label"],
           "raw_score": prediction["confidence"], "stable_label": state.label or "",
           "status": state.status, "scores_json": json.dumps(prediction["scores"]),
           "checkpoint_sha256": checkpoint_sha, "ground_truth_label": "",
           "object_id": ""}
    fields = list(row)
    with metadata_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def run_camera(checkpoint_path: Path = DEFAULT_CHECKPOINT, camera_index: int = 0,
               input_video: Path | None = None, run_dir: Path | None = None,
               device: str = "auto", roi_scale: float = 0.70, threshold: float = 0.65,
               window: int = 10, min_votes: int = 7, headless: bool = False,
               max_seconds: float | None = None, max_frames: int | None = None,
               record_video: bool = False) -> dict:
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    selected_device = select_device(device)
    model, transform, checkpoint = load_predictor(checkpoint_path, selected_device)
    if checkpoint.get("smoke"):
        raise ValueError("Refusing to demo with a smoke checkpoint")
    smoother = PredictionSmoother(window, min_votes, threshold)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = run_dir or ROOT / "runs/camera" / session_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = _sha256(checkpoint_path)
    source = str(input_video) if input_video is not None else f"camera:{camera_index}"
    capture = cv2.VideoCapture(str(input_video) if input_video is not None else camera_index)
    if not capture.isOpened():
        capture.release()
        message = f"Cannot open {source}. Check camera permission, index and other apps using it."
        (run_dir / "error.json").write_text(json.dumps({"source": source,
                                                        "error": message}, indent=2) + "\n",
                                             encoding="utf-8")
        raise RuntimeError(message)
    writer = None
    prediction_stream = (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8")
    prediction_writer = csv.DictWriter(
        prediction_stream,
        fieldnames=("frame", "captured_at", "raw_label", "raw_score", "stable_label",
                    "stable_score", "status", "votes", "window_count", "fps", "model_ms"),
    )
    prediction_writer.writeheader()
    metrics = {name: [] for name in ("capture_ms", "bgr_to_rgb_ms", "preprocessing_ms",
                                    "model_ms", "render_ms", "loop_ms")}
    switch_delays = []
    marker = None
    previous_stable = None
    frame_count = failures = save_count = 0
    started = perf_counter()
    last_frame_time = started
    fps_ewma = 0.0
    stop_reason = "unknown"
    try:
        while True:
            loop_start = perf_counter()
            if max_seconds is not None and loop_start - started >= max_seconds:
                stop_reason = "duration_limit"
                break
            if max_frames is not None and frame_count >= max_frames:
                stop_reason = "frame_limit"
                break
            read_start = perf_counter()
            ok, frame = capture.read()
            metrics["capture_ms"].append((perf_counter() - read_start) * 1000)
            if not ok or frame is None:
                smoother.reset()
                failures += 1
                if input_video is not None:
                    stop_reason = "video_end"
                    break
                if failures >= 3:
                    stop_reason = "camera_read_failure"
                    break
                continue
            failures = 0
            frame_count += 1
            bounds = central_roi_bounds(frame.shape, roi_scale)
            x, y, w, h = bounds
            roi_bgr = frame[y:y + h, x:x + w]
            prediction, timing = predict_bgr_roi(model, transform, roi_bgr, selected_device)
            state = smoother.update(prediction)
            for key in ("bgr_to_rgb_ms", "preprocessing_ms", "model_ms"):
                metrics[key].append(timing[key])
            now = perf_counter()
            instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-9)
            fps_ewma = instantaneous_fps if frame_count == 1 else 0.8 * fps_ewma + 0.2 * instantaneous_fps
            last_frame_time = now
            if marker is not None and marker[1] is not None and state.label and state.label != marker[1]:
                switch_delays.append((now - marker[0]) * 1000)
                marker = None
            if state.label:
                previous_stable = state.label
            prediction_writer.writerow({"frame": frame_count,
                                        "captured_at": datetime.now().astimezone().isoformat(),
                                        "raw_label": prediction["label"],
                                        "raw_score": prediction["confidence"],
                                        "stable_label": state.label or "",
                                        "stable_score": state.mean_score if state.label else "",
                                        "status": state.status, "votes": state.votes,
                                        "window_count": state.count, "fps": fps_ewma,
                                        "model_ms": timing["model_ms"]})
            render_start = perf_counter()
            annotated = _draw_overlay(frame, bounds, prediction, state, fps_ewma,
                                      timing["model_ms"])
            if record_video:
                if writer is None:
                    output_path = run_dir / "demo.mp4"
                    camera_fps = capture.get(cv2.CAP_PROP_FPS)
                    output_fps = camera_fps if 1 <= camera_fps <= 60 else 20.0
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             output_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video: {output_path}")
                writer.write(annotated)
            if not headless:
                cv2.imshow("Seminar waste classifier", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "user_quit"
                    break
                if key == ord("s"):
                    save_count += 1
                    snapshot = run_dir / "unlabeled_captures" / f"roi_{save_count:04d}.png"
                    _write_capture(snapshot, roi_bgr, prediction, state, bounds,
                                   checkpoint_sha, source, run_dir / "captures.csv")
                    print(f"Saved unlabeled ROI: {snapshot}", flush=True)
                if key == ord("m"):
                    marker = (perf_counter(), previous_stable)
                    print(f"Marked physical switch from {previous_stable}", flush=True)
                if key == ord("r"):
                    smoother.reset()
                    previous_stable = None
                    marker = None
            metrics["render_ms"].append((perf_counter() - render_start) * 1000)
            metrics["loop_ms"].append((perf_counter() - loop_start) * 1000)
    finally:
        prediction_stream.close()
        capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    elapsed = perf_counter() - started
    summary = {"source": source, "device": selected_device,
               "checkpoint": str(checkpoint_path.resolve()),
               "checkpoint_sha256": checkpoint_sha,
               "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
               "roi_scale": roi_scale, "threshold": threshold, "window": window,
               "min_votes": min_votes, "frames": frame_count, "saved_unlabeled_rois": save_count,
               "elapsed_seconds": elapsed, "effective_fps": frame_count / elapsed if elapsed else 0.0,
               "stop_reason": stop_reason, "headless": headless,
               "timings_ms": {key: _summarize(values) for key, values in metrics.items()},
               "manual_switch_delays_ms": switch_delays,
               "manual_switch_delay_note": "Time from pressing m to a different stable label; includes the user's physical swap and smoothing delay.",
               "video_path": str(run_dir / "demo.mp4") if record_video and writer is not None else None}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if stop_reason == "camera_read_failure":
        raise RuntimeError(f"Camera stopped returning frames; see {run_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--input-video", type=Path, default=None, help="Offline smoke-test source")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--roi-scale", type=float, default=0.70)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-votes", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--record", action="store_true", help="Save annotated local MP4")
    args = parser.parse_args()
    try:
        run_camera(args.checkpoint, args.camera_index, args.input_video, args.run_dir,
                   args.device, args.roi_scale, args.threshold, args.window,
                   args.min_votes, args.headless, args.max_seconds, args.max_frames,
                   args.record)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Camera demo error: {exc}\n")


if __name__ == "__main__":
    main()

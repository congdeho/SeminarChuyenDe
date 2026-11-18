"""Local browser interface for the seminar's trained checkpoints."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter, sleep
from urllib.parse import parse_qs, urlsplit

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from src.bottle_routing import BottleRefinement
from src.camera import PredictionSmoother, central_roi_bounds, select_device
from src.dataset import ROOT
from src.labels import LABELS, LABELS_VI
from src.predict import load_predictor, predict_pil

STATIC = Path(__file__).with_name("web_ui")
IMAGE_LIMIT = 20 * 1024 * 1024
VIDEO_LIMIT = 150 * 1024 * 1024
ROI_SCALE = 0.70
WINDOW = 10
MIN_VOTES = 7
DEFAULT_MODEL = "e2"
VIDEO_SUFFIXES = {"mp4", "m4v", "mov", "avi", "webm", "mkv", "mpeg", "mpg"}
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOption:
    name: str
    description: str
    checkpoint: Path
    threshold: float
    labels: tuple[str, ...] = LABELS


MODELS = {
    "e2": ModelOption("ResNet18", "Trọng số ImageNet, có augmentation. Checkpoint demo Phase 5.",
                      ROOT / "runs/phase5/e2_seed42/best.pt", 0.60),
    "e1": ModelOption("CNN cơ sở", "Huấn luyện từ đầu, có augmentation.",
                      ROOT / "runs/phase5/e1_seed42/best.pt", 0.65),
    "e3": ModelOption("MobileNetV3-Small", "Trọng số ImageNet, có augmentation.",
                      ROOT / "runs/local_pretrained_probe/best.pt", 0.65),
    "e4": ModelOption("MobileNetV3-Small · không augmentation", "Trọng số ImageNet, tắt augmentation khi huấn luyện.",
                      ROOT / "runs/phase5/e4_seed42/best.pt", 0.70),
}
SAMPLE_IMAGE = ROOT / "data/raw/dataset-resized/glass/glass105.jpg"
SAMPLE_VIDEO = ROOT / "runs/camera_fixture/validation_sequence.mp4"


class InputError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class Classifier:
    def __init__(self, device: str):
        self.device = device
        self.models: dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.options = MODELS.copy()
        try:
            self.bottle_refinement = BottleRefinement.from_workspace(device)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.bottle_refinement = None
            LOG.warning("Automatic E5 check unavailable: %s", exc)
        else:
            base = self.options["e2"]
            self.options["e2"] = replace(
                base, description=f"{base.description} Tự kiểm tra chai nhựa/thủy tinh bằng E5 khi cần.",
            )

    def option(self, key: str) -> ModelOption:
        option = self.options.get(key)
        if option is None:
            raise InputError("Mã mô hình không hợp lệ.")
        if not option.checkpoint.is_file():
            raise InputError(f"Thiếu checkpoint của {option.name}.", HTTPStatus.NOT_FOUND)
        return option

    def predictor(self, key: str):
        if key not in self.models:
            model, transform, checkpoint = load_predictor(self.option(key).checkpoint, self.device)
            if checkpoint.get("smoke"):
                raise InputError("Checkpoint thử kỹ thuật không dùng cho phân loại.")
            self.models[key] = model, transform
        return self.models[key]

    def _predict(self, key: str, image: Image.Image) -> tuple[dict, dict]:
        with self.lock:
            prediction, timing = predict_pil(*self.predictor(key), image, self.device)
            if key == "e2" and self.bottle_refinement is not None:
                prediction = self.bottle_refinement.refine(image, prediction, timing)
        return prediction, timing

    def image(self, key: str, body: bytes, crop: bool) -> dict:
        try:
            with Image.open(io.BytesIO(body)) as opened:
                if opened.width * opened.height > 30_000_000:
                    raise InputError("Ảnh vượt giới hạn 30 megapixel.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InputError("Không đọc được ảnh. Thử ảnh JPEG, PNG hoặc WebP.") from exc
        bounds = None
        if crop:
            bounds = central_roi_bounds((image.height, image.width), ROI_SCALE)
            x, y, w, h = bounds
            image = image.crop((x, y, x + w, y + h))
        prediction, timing = self._predict(key, image)
        return {**prediction, "model": key, "timing_ms": timing, "roi": bounds}

    def video_frame(self, key: str, frame) -> dict:
        height, width = frame.shape[:2]
        if width > 720:
            frame = cv2.resize(frame, (720, max(1, round(height * 720 / width))), interpolation=cv2.INTER_AREA)
        x, y, w, h = central_roi_bounds(frame.shape, ROI_SCALE)
        start = perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
        color_ms = (perf_counter() - start) * 1000
        prediction, timing = self._predict(key, image)
        timing["bgr_to_rgb_ms"] = color_ms
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise InputError("Không thể hiển thị khung hình video.")
        return {**prediction, "model": key, "timing_ms": timing,
                "frame_width": frame.shape[1], "frame_height": frame.shape[0],
                "roi": [x, y, w, h], "jpeg": base64.b64encode(jpeg).decode("ascii")}


class LocalServer(ThreadingHTTPServer):
    def __init__(self, port: int, classifier: Classifier):
        super().__init__(("127.0.0.1", port), Handler)
        self.classifier = classifier
        self.video_slot = threading.BoundedSemaphore()


class Handler(BaseHTTPRequestHandler):
    server: LocalServer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _headers(self, status: HTTPStatus, content_type: str, size: int | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Permissions-Policy", "camera=(self), microphone=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; "
                         "media-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        if size is not None:
            self.send_header("Content-Length", str(size))
        self.end_headers()

    def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _host(self) -> str:
        host = self.headers.get("Host", "")
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise InputError("Mở ứng dụng bằng địa chỉ localhost hoặc 127.0.0.1.", HTTPStatus.FORBIDDEN)
        return host

    def _length(self, max_size: int) -> int:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise InputError("Kích thước tệp không hợp lệ.") from exc
        if size <= 0:
            raise InputError("Tệp rỗng.")
        if size > max_size:
            raise InputError(f"Tệp vượt giới hạn {max_size // (1024 ** 2)} MiB.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return size

    def _file(self, path: Path, content_type: str):
        if not path.is_file():
            raise InputError("Tệp mẫu không có trên máy này.", HTTPStatus.NOT_FOUND)
        with path.open("rb") as stream:
            self._headers(HTTPStatus.OK, content_type, path.stat().st_size)
            shutil.copyfileobj(stream, self.wfile)

    def do_GET(self):
        try:
            self._host()
            path = urlsplit(self.path).path
            if path == "/api/models":
                self._json({"default": DEFAULT_MODEL, "device": self.server.classifier.device,
                            "window": WINDOW, "min_votes": MIN_VOTES, "roi_scale": ROI_SCALE,
                            "models": [{"id": key, "name": opt.name, "description": opt.description,
                                        "threshold": opt.threshold, "available": opt.checkpoint.is_file(),
                                        "labels": list(opt.labels)}
                                       for key, opt in self.server.classifier.options.items()],
                            "labels": dict(zip(LABELS, LABELS_VI))})
                return
            files = {"/": (STATIC / "index.html", "text/html; charset=utf-8"),
                     "/app.css": (STATIC / "app.css", "text/css; charset=utf-8"),
                     "/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
                     "/stats.js": (STATIC / "stats.js", "text/javascript; charset=utf-8"),
                     "/sample.jpg": (SAMPLE_IMAGE, "image/jpeg"),
                     "/sample.mp4": (SAMPLE_VIDEO, "video/mp4")}
            if path not in files:
                raise InputError("Không tìm thấy trang.", HTTPStatus.NOT_FOUND)
            self._file(*files[path])
        except InputError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (ConnectionError, TimeoutError):
            pass

    def do_POST(self):
        try:
            host = self._host()
            if self.headers.get("Origin") not in (None, f"http://{host}"):
                raise InputError("Yêu cầu phải đến từ ứng dụng cục bộ.", HTTPStatus.FORBIDDEN)
            url = urlsplit(self.path)
            options = parse_qs(url.query)
            key = options.get("model", [DEFAULT_MODEL])[0]
            model = self.server.classifier.option(key)
            if url.path == "/api/predict":
                length = self._length(IMAGE_LIMIT)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise InputError("Ảnh tải lên chưa đầy đủ.")
                crop = options.get("source", ["image"])[0] == "webcam"
                self._json(self.server.classifier.image(key, body, crop))
            elif url.path == "/api/video":
                self._video(key, model, options.get("format", ["mp4"])[0])
            else:
                raise InputError("Không tìm thấy chức năng.", HTTPStatus.NOT_FOUND)
        except (ConnectionError, TimeoutError):
            pass
        except InputError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (OSError, RuntimeError, ValueError):
            LOG.exception("Local inference failed")
            self._json({"error": "Suy luận gặp lỗi. Kiểm tra checkpoint và nhật ký Terminal."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _video(self, key: str, model: ModelOption, extension: str):
        length = self._length(VIDEO_LIMIT)
        slot = self.server.video_slot
        if not slot.acquire(timeout=1.2):
            raise InputError("Đang xử lý video khác. Thử lại sau vài giây.", HTTPStatus.CONFLICT)
        try:
            suffix = extension if extension in VIDEO_SUFFIXES else "video"
            with TemporaryDirectory(prefix="waste-preview-") as folder:
                path = Path(folder) / f"input.{suffix}"
                with path.open("wb") as target:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise InputError("Video tải lên chưa đầy đủ.")
                        target.write(chunk)
                        remaining -= len(chunk)
                self._stream_video(key, model, path)
        finally:
            slot.release()

    def _stream_video(self, key: str, model: ModelOption, path: Path):
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise InputError("Không mở được video. Thử MP4, MOV, AVI hoặc WebM.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not 1 <= fps <= 240:
                fps = 25.0
            step = max(1, round(fps / 10))
            reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            count = max(0, int(reported_count)) if isfinite(reported_count) else 0
            smoother = PredictionSmoother(WINDOW, MIN_VOTES, model.threshold)
            self._headers(HTTPStatus.OK, "application/x-ndjson; charset=utf-8")
            frame_index = 0
            start = monotonic()
            self._line({"type": "video", "duration_s": count / fps})
            try:
                while ok:
                    result = self.server.classifier.video_frame(key, frame)
                    delay = start + frame_index / fps - monotonic()
                    if delay > 0:
                        sleep(delay)
                    self._line({"type": "frame", "time_s": frame_index / fps,
                                "stable": asdict(smoother.update(result)), **result})
                    for _ in range(step - 1):
                        if not capture.grab():
                            ok = False
                            break
                    if ok:
                        ok, frame = capture.read()
                    frame_index += step
                self._line({"type": "done"})
            except (RuntimeError, ValueError, cv2.error, InputError):
                LOG.exception("Video stream failed")
                self._line({"type": "error", "error": "Không thể tiếp tục phân loại video."})
        finally:
            capture.release()

    def _line(self, row: dict):
        self.wfile.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.wfile.flush()

    def log_request(self, code="-", size="-"):
        if str(code).startswith(("4", "5")):
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "auto", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        classifier = Classifier(select_device(args.device))
        with LocalServer(args.port, classifier) as server:
            print(f"Mở http://127.0.0.1:{server.server_port} · {classifier.device.upper()} · Ctrl+C để dừng", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Không thể chạy giao diện: {exc}\n")


if __name__ == "__main__":
    main()

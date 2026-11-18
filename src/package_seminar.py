"""Package a reproducible CPU demo and write the matching seminar results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from src.compare_models import EXPERIMENTS, ROOT, SEEDS
from src.finalize_trashnet import load_json, summarize_video
from src.train import file_sha256


NAMES = {"e1": "CNN từ đầu", "e2": "ResNet18 pretrained",
         "e3": "MobileNetV3-Small có augmentation", "e4": "MobileNetV3-Small không augmentation"}
FIXTURE = ROOT / "runs/camera_fixture/validation_sequence.mp4"


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%".replace(".", ",")


def build_results(data: dict, section_level: str = "##") -> str:
    lines = [data["status"], "", f"{section_level} Validation TrashNet · 378 ảnh", ""]
    if data["test"]:
        selected = data["selected"]
        lines += ["| Phương án | Seed 42 | Seed 123 | Seed 2026 | Macro-F1 trung bình ± SD |",
                  "|---|---:|---:|---:|---:|"]
        for key, row in data["validation"].items():
            seed_values = " | ".join(percent(row["seed_scores"][str(seed)]) for seed in SEEDS)
            lines.append(f'| {key.upper()} {NAMES[key]} | {seed_values} | '
                         f'{percent(row["mean_macro_f1"])} ± {percent(row["sd_macro_f1"])} |')
        delta = (data['validation']['e3']['mean_macro_f1'] - data['validation']['e4']['mean_macro_f1']) * 100
        eligible = ", ".join(key.upper() for key in data["eligible"])
        lines += ["", "SD là độ lệch chuẩn mẫu của ba lượt chạy trên cùng tập validation.",
                  f'E3 trừ E4 bằng **{delta:+.2f} điểm phần trăm macro-F1** tính theo trung bình ba seed. '
                  f'{eligible} nằm trong khoảng cách 0,02 tới macro-F1 trung bình cao nhất. '
                  f'Chọn **{selected.upper()} {NAMES[selected]} seed 42** theo latency trung vị trong nhóm này.', ""]
    else:
        lines += ["| Phương án | Accuracy seed 42 | Macro-F1 seed 42 |",
                  "|---|---:|---:|"]
        for key, row in data["validation"].items():
            lines.append(f'| {key.upper()} {NAMES[key]} | {percent(row["accuracy"])} | {percent(row["macro_f1"])} |')

    lines += ["", f"{section_level} Tốc độ suy luận trên Mac MPS", "",
              "| Phương án seed 42 | Median (ms) | p95 (ms) | Forward ước tính (ảnh/s) | Tham số | Checkpoint (MiB) |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, row in data["validation"].items():
        lines.append(f'| {key.upper()} | {row["median_ms"]:.2f} | {row["p95_ms"]:.2f} | '
                     f'{row["model_only_fps_estimate"]:.1f} | {row["parameter_count"]:,} | '
                     f'{row["checkpoint_size_bytes"] / (1024 ** 2):.2f} |')
    lines += ["", "Mỗi phép đo chạy forward batch 1 trên một ảnh validation, warm-up 20 lượt, đo 100 lượt và đồng bộ MPS. Kích thước là checkpoint huấn luyện gồm metadata và trạng thái optimizer nếu có.", ""]
    if data["test"]:
        lines += [f"{section_level} Test TrashNet · 378 ảnh", "",
                  "| Phương án seed 42 | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|"]
        for key, row in data["test"].items():
            lines.append(f'| {key.upper()} | {percent(row["accuracy"])} | {percent(row["macro_f1"])} | {percent(row["weighted_f1"])} |')
        metrics = data["test"][selected]
        threshold_status = (
            "Ngưỡng này đạt các tiêu chí coverage và accuracy trên validation."
            if data["threshold_target_met"] else
            "Lưới ngưỡng trên validation không có điểm đạt đồng thời hai tiêu chí. Áp dụng ngưỡng dự phòng 0,65 theo kế hoạch."
        )
        val_gate = data["validation_gate"]
        lines += ["", "Ba chỉ số trong bảng được tính bằng nhãn argmax trên toàn bộ tập test.", "",
                  f'Checkpoint demo {selected.upper()} giữ ngưỡng validation {data["threshold"]:.2f}. '
                  f'{threshold_status} '
                  f'Trên validation, coverage đạt {percent(val_gate["coverage"])} và accuracy trên số ảnh được nhận đạt {percent(val_gate["accepted_accuracy"])}. '
                  f'Coverage test {percent(metrics["coverage"])} ({metrics["accepted_n"]}/{metrics["n"]}). '
                  f'Accuracy trong số ảnh được chấp nhận {percent(metrics["accepted_accuracy"])}. '
                  f'Tỷ lệ trả nhãn đúng trên toàn test {percent(metrics["correct_and_accepted_rate"])}.', "",
                  f"{section_level} Chỉ số từng lớp của checkpoint demo", "",
                  "| Nhãn thật | Precision | Recall | F1 | Số ảnh |", "|---|---:|---:|---:|---:|"]
        for label in metrics["labels"]:
            row = metrics["classification_report"][label]
            lines.append(f'| {label} | {percent(row["precision"])} | {percent(row["recall"])} | '
                         f'{percent(row["f1-score"])} | {int(row["support"])} |')
        labels = metrics["labels"]
        confusion = [(count, labels[i], labels[j])
                     for i, row in enumerate(metrics["confusion_matrix"])
                     for j, count in enumerate(row) if i != j]
        max_count, actual, predicted = max(confusion, key=lambda item: item[0])
        min_recall_label = min(labels, key=lambda label: metrics["classification_report"][label]["recall"])
        min_recall = metrics["classification_report"][min_recall_label]["recall"]
        errors = data["errors"]
        top_pairs = Counter((row["true_label"], row["predicted_label"]) for row in errors).most_common(3)
        pairs = ", ".join(f"{actual} bị gán {predicted} ({count})" for (actual, predicted), count in top_pairs)
        lines += ["", f'Lớp `{min_recall_label}` có recall thấp nhất ({percent(min_recall)}). '
                  f'Ô ngoài đường chéo lớn nhất ghi nhận {max_count} ảnh `{actual}` bị gán `{predicted}`.', "",
                  "![Ma trận nhầm lẫn theo số ảnh và tỷ lệ](evidence/confusion.png)", "",
                  f"{section_level} Các lỗi có điểm dự đoán cao nhất", "",
                  f"Các cặp nhãn xuất hiện nhiều nhất trong nhóm {len(errors)} lỗi được xếp theo điểm softmax là {pairs}.", "",
                  "| Ảnh | Nhãn thật | Nhãn đoán | Điểm softmax | Qua ngưỡng |",
                  "|---|---|---|---:|---|"]
        for row in errors:
            lines.append(f'| {Path(row["path"]).name} | {row["true_label"]} | {row["predicted_label"]} | '
                         f'{float(row["confidence"]):.4f} | {"Có" if row["accepted"] == "True" else "Không"} |')
        lines += ["", "![Ảnh sai có điểm dự đoán cao nhất](evidence/test_errors_top12.png)", "",
                  "Các ảnh trong hình được chọn theo thứ tự điểm dự đoán sai. Chúng minh họa rủi ro của ngưỡng softmax với lỗi tự tin cao.", ""]
        if data["error_notes"]:
            lines += [data["error_notes"].strip(), ""]
    lines += [f"{section_level} Demo offline", "", f'Gói dùng {data["model"]} seed 42 và ngưỡng {data["threshold"]:.2f}. '
              f'Video đầu ra giải mã đủ {data["video"]["decoded_frames"]} frame từ sáu ảnh validation. '
              "Ảnh webcam thật và FPS webcam chưa có số đo.", ""]
    lines += ["| Nhãn của đoạn | Frame có nhãn ổn định đúng | Nhãn ở cuối đoạn |",
              "|---|---:|---|"]
    for segment in data["video"]["segments"]:
        frame_count = segment["end_frame"] - segment["start_frame"] + 1
        last_label = segment["last_stable_label"] or "Chưa chắc chắn"
        lines.append(f'| {segment["label"]} | {segment["stable_correct_frames"]}/{frame_count} | {last_label} |')
    lines += ["", "Các frame trong mỗi đoạn lặp lại cùng một ảnh. Khi đổi đoạn, cửa sổ smoothing vẫn chứa dự đoán của đoạn trước. Đọc bảng để kiểm tra hành vi chuyển nhãn trên chuỗi này.", ""]
    return "\n".join(lines)


def package(output: Path, use_preview: bool, error_notes: Path | None = None) -> dict:
    final = ROOT / "runs/phase5/final"
    if use_preview:
        selected = "e3"
        video = ROOT / "outputs/phase6/video_seed42_preview"
        test = {}
        aggregates = {}
        eligible = []
        errors = []
        threshold_target_met = None
        validation_gate = None
    else:
        final_summary = load_json(final / "summary.json")
        selected = final_summary["comparison"]["selected_experiment"]
        video = final / "offline_video"
        test = final_summary["test"]
        aggregates = final_summary["comparison"]["summary"]
        eligible = final_summary["comparison"]["eligible"]
        lock = load_json(final / "selection_lock.json")
        threshold_target_met = lock["threshold_target_met_on_validation"]
        sweep_name = "threshold_sweep_v2.json" if selected == "e3" else "threshold_sweep.json"
        sweep = load_json(ROOT / f"runs/phase5/{selected}_seed42_val" / sweep_name)
        validation_gate = next(row for row in sweep["candidates"] if row["threshold"] == lock["threshold"])
        with (final / f"{selected}_test/predictions.csv").open(newline="", encoding="utf-8") as stream:
            mistakes = [row for row in csv.DictReader(stream) if row["true_label"] != row["predicted_label"]]
        errors = sorted(mistakes, key=lambda row: float(row["confidence"]), reverse=True)[:12]

    validation = {}
    for key in EXPERIMENTS:
        metrics = load_json(ROOT / f"runs/phase5/{key}_seed42_val/metrics.json")
        benchmark = load_json(ROOT / f"runs/phase5/{key}_seed42_benchmark_mps.json")
        validation[key] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                           **{k: benchmark[k] for k in ("median_ms", "p95_ms", "model_only_fps_estimate",
                                                         "parameter_count", "checkpoint_size_bytes")},
                           **{k: aggregates[key][k] for k in ("mean_macro_f1", "sd_macro_f1", "seed_scores")
                              if key in aggregates}}
    settings = load_json(video / "summary.json")
    checkpoint = Path(settings["checkpoint"])
    if file_sha256(checkpoint) != settings["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from those used to record the demo")
    data = {"status": ("Bản luyện bảo vệ · Kết quả validation seed 42. Đánh giá test đang chờ các seed còn lại."
                       if use_preview else "Bản kết quả · Đủ ba seed và đã khóa đánh giá test TrashNet."),
            "model": NAMES[selected], "selected": selected, "threshold": settings["threshold"],
            "threshold_target_met": threshold_target_met, "validation_gate": validation_gate,
            "eligible": eligible, "validation": validation, "test": test, "errors": errors,
            "error_notes": error_notes.read_text(encoding="utf-8") if error_notes else "",
            "video": summarize_video(video, FIXTURE)}
    results = "# Kết quả seminar\n\n" + build_results(data)
    output.mkdir(parents=True, exist_ok=True)
    (output / "KET_QUA.md").write_text(results, encoding="utf-8")
    (output / "slide_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_sample = load_json(FIXTURE.with_suffix(".json"))["segments"][0]
    sample = ROOT / validation_sample["path"]
    runtime_files = {f"src/{p.name}": p for p in (ROOT / "src").glob("*.py")}
    files = {**runtime_files, **{name: ROOT / name for name in ("pyproject.toml", "uv.lock", ".python-version")},
             "BAO_VE_SEMINAR.md": ROOT / "docs/BAO_VE_SEMINAR.md",
             "sample.jpg": sample, "validation_sequence.mp4": FIXTURE,
             "validation_sequence.json": FIXTURE.with_suffix(".json"),
             "best.pt": checkpoint, "demo.mp4": video / "demo.mp4",
             "demo_summary.json": video / "summary.json"}
    if test:
        for name in ("comparison.json", "selection_lock.json", "summary.json", "confusion.png", "test_errors_top12.png"):
            files[f"evidence/{name}"] = final / name
        for key in EXPERIMENTS:
            config = f"phase5_{key}.yaml"
            files[f"configs/{config}"] = ROOT / "configs" / config
            benchmark = f"{key}_seed42_benchmark_mps.json"
            files[f"evidence/benchmarks/{benchmark}"] = ROOT / "runs/phase5" / benchmark
            for name in ("metrics.json", "predictions.csv"):
                files[f"evidence/{key}_test/{name}"] = final / f"{key}_test" / name
            for row in final_summary["comparison"]["runs"][key]:
                source = ROOT / row["validation_metrics_path"]
                if file_sha256(source) != row["validation_metrics_sha256"]:
                    raise ValueError(f"Validation metrics changed after freezing: {source}")
                files[f'evidence/validation/{key}_seed{row["seed"]}.json'] = source
        (output / "evidence").mkdir(exist_ok=True)
        for name in ("confusion.png", "test_errors_top12.png"):
            copyfile(final / name, output / "evidence" / name)
    report = (ROOT / "docs/BAO_CAO_SEMINAR.md").read_text(encoding="utf-8")
    if test:
        before, rest = report.split("## 5. Kết quả thực nghiệm\n\n", 1)
        _, after = rest.split("## 6. Triển khai và giới hạn", 1)
        metrics = test[selected]
        abstract_result = (f'Trong 12 lượt huấn luyện, quy tắc chọn dựa trên validation và latency '
                           f'chốt {NAMES[selected]} seed 42 cho demo. Trên 378 ảnh test TrashNet, '
                           f'checkpoint này đạt accuracy {percent(metrics["accuracy"])} và macro-F1 '
                           f'{percent(metrics["macro_f1"])} trên toàn bộ ảnh.\n\n')
        before = before.replace("\n## 1. Bài toán", "\n" + abstract_result + "## 1. Bài toán", 1)
        report = (before.replace("ba seed dự kiến", "ba seed") + "## 5. Kết quả thực nghiệm\n\n"
                  + build_results(data, "###") + "\n## 6. Triển khai và giới hạn" + after)
    (output / "BAO_CAO_SEMINAR.md").write_text(report, encoding="utf-8")
    readme = f"""# Demo seminar TrashNet

{data['status']}

`demo.mp4` có thể phát trực tiếp. Để chạy lại bằng CPU, giải nén và mở Terminal tại thư mục này. Cài `uv` và dùng `uv sync --locked` để chuẩn bị Python và dependency. Nếu máy đã có cache phù hợp, dùng `uv sync --locked --offline`. Sau khi cài, các lệnh demo không cần tải trọng số hay ảnh.

```bash
uv run --locked --offline python -m src.predict --checkpoint best.pt --image sample.jpg --device cpu
uv run --locked --offline python -m src.camera --checkpoint best.pt --input-video validation_sequence.mp4 --headless --record --device cpu --threshold {data['threshold']:.2f} --run-dir repro_demo
```

Muốn chạy lại lần nữa, dùng thư mục `--run-dir` mới. Webcam dùng cùng lệnh với `--checkpoint best.pt --device cpu --threshold {data['threshold']:.2f}` và bỏ `--input-video`, `--headless`, `--record`. Chế độ webcam cần quyền Camera và chưa được nghiệm thu trong gói này. Các ảnh và video đầu vào lấy từ TrashNet validation, nguồn https://github.com/garythung/trashnet.
"""
    package_path = output / "Demo_Seminar_TrashNet.zip"
    with TemporaryDirectory(dir=output, prefix=".package-") as temporary:
        staged_zip = Path(temporary) / package_path.name
        with ZipFile(staged_zip, "x", compression=ZIP_DEFLATED) as archive:
            checksums = []
            for name, source in sorted(files.items()):
                archive.write(source, name)
                checksums.append(f"{file_sha256(source)}  {name}")
            for name, contents in {"README.md": readme, "KET_QUA.md": results,
                                   "BAO_CAO_SEMINAR.md": report}.items():
                archive.writestr(name, contents)
                checksums.append(f"{hashlib.sha256(contents.encode()).hexdigest()}  {name}")
            archive.writestr("SHA256SUMS", "\n".join(checksums) + "\n")
        package_path.hardlink_to(staged_zip)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Package the existing seed 42 validation rehearsal")
    parser.add_argument("--error-notes", type=Path, help="Visual observations about the reviewed test errors")
    args = parser.parse_args()
    result = package(args.output, args.preview, args.error_notes)
    print(result["status"])


if __name__ == "__main__":
    main()

"""Create a review sheet of the most confident mistakes from one evaluated split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.dataset import ROOT


def make_gallery(predictions_path: Path, output_path: Path, limit: int = 12) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {row["split"] for row in rows} not in ({"val"}, {"test"}):
        raise ValueError("Predictions must belong to exactly one validation or test split")
    errors = sorted((row for row in rows if row["true_label"] != row["predicted_label"]),
                    key=lambda row: float(row["confidence"]), reverse=True)[:limit]
    if not errors:
        raise ValueError("No wrong predictions to display")
    cell_width, cell_height = 300, 270
    columns = 3
    rows_count = (len(errors) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(errors):
        left = (index % columns) * cell_width
        top = (index // columns) * cell_height
        with Image.open(ROOT / row["path"]) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (cell_width - 16, 215))
        sheet.paste(thumbnail, (left + (cell_width - thumbnail.width) // 2,
                                top + (215 - thumbnail.height) // 2))
        draw.text((left + 8, top + 218), f"{index + 1}. {Path(row['path']).name}", fill="black")
        draw.text((left + 8, top + 237),
                  f"true {row['true_label']} -> {row['predicted_label']} {float(row['confidence']):.3f}",
                  fill=(170, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    sheet.save(output_path)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    errors = make_gallery(args.predictions, args.output, args.limit)
    for row in errors:
        print(f"{row['path']}: {row['true_label']} -> {row['predicted_label']} "
              f"{float(row['confidence']):.3f}")


if __name__ == "__main__":
    main()

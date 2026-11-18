"""Prepare clear-plastic and glass bottle crops from the official TACO annotations."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import ROOT

SOURCE = ROOT / "data/raw/taco/annotations.json"
DEST = ROOT / "data/processed/material_bottles"
TYPES = {"Clear plastic bottle": "plastic", "Glass bottle": "glass"}
HOSTS = {"img.tacodataset.org", "olm-s3.s3.eu-west-1.amazonaws.com"}
MAX_BYTES = 20 * 1024 * 1024


def source_bytes(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in HOSTS:
        raise ValueError("Unrecognized TACO image URL")
    with urlopen(Request(url, headers={"User-Agent": "SeminarTrashNet/1.0"}), timeout=12) as response:
        resolved = urlsplit(response.url)
        if resolved.scheme != "https" or resolved.hostname not in HOSTS:
            raise ValueError("Unrecognized TACO image redirect")
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("TACO image is larger than 20 MiB")
    return payload


def pull(job):
    photo, items, output = job
    urls = dict.fromkeys(url for url in (photo.get("flickr_640_url"), photo.get("flickr_url")) if url)
    for url in urls:
        try:
            with Image.open(io.BytesIO(source_bytes(url))) as raw:
                picture = ImageOps.exif_transpose(raw).convert("RGB")
            break
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    else:
        return []
    sx, sy = picture.width / photo["width"], picture.height / photo["height"]
    if abs(sx / sy - 1) > 0.025:
        return []
    saved = []
    for row in items:
        x, y, w, h = row["bbox"]
        if min(w * sx, h * sy) < 24:
            continue
        left, top = max(0, int((x - .12 * w) * sx)), max(0, int((y - .12 * h) * sy))
        right = min(picture.width, round((x + 1.12 * w) * sx))
        bottom = min(picture.height, round((y + 1.12 * h) * sy))
        if right <= left or bottom <= top:
            continue
        crop = picture.crop((left, top, right, bottom))
        crop.thumbnail((512, 512), Image.Resampling.LANCZOS)
        target = output / f'{row["annotation_id"]}.jpg'
        crop.save(target, quality=93)
        saved.append({k: row[k] for k in ("split", "label", "image_id", "annotation_id")}
                     | {"path": target.relative_to(ROOT).as_posix(), "source_url": url, "source_page": photo.get("flickr_url")})
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    categories = {cat["id"]: TYPES[cat["name"]] for cat in data["categories"] if cat["name"] in TYPES}
    rows = [{"image_id": entry["image_id"], "annotation_id": entry["id"], "bbox": entry["bbox"],
             "label": categories[entry["category_id"]]}
            for entry in data["annotations"] if entry["category_id"] in categories and not entry.get("iscrowd")]
    rows.sort(key=lambda row: (row["image_id"], row["annotation_id"]))
    # All views from one source photo go to one partition before any model is used.
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(rows)), [row["label"] for row in rows],
                                                   [row["image_id"] for row in rows])):
        for index in indices:
            rows[index]["split"] = ("test", "val", "train", "train", "train")[fold]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    images = {image["id"]: image for image in data["images"]}
    args.output.mkdir(parents=True, exist_ok=False)
    crops = args.output / "images"
    crops.mkdir()
    jobs = [(images[key], items, crops) for key, items in grouped.items()]
    saved = []
    with ThreadPoolExecutor(max_workers=12) as workers:
        for index, result in enumerate(workers.map(pull, jobs), 1):
            saved.extend(result)
            if index % 50 == 0:
                print(f"Checked {index}/{len(jobs)} original images; kept {len(saved)} crops", flush=True)
    if not saved:
        raise RuntimeError("No usable TACO images downloaded")
    with (args.output / "manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    print(sorted(Counter((row["split"], row["label"]) for row in saved).items()))


if __name__ == "__main__":
    main()

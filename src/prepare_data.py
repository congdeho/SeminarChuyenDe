"""Audit TrashNet and create a reproducible, group-aware data manifest.

Run from the repository root: python -m src.prepare_data
Raw images are never modified. Reports and manifests live in ignored data/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.labels import LABELS

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/dataset-resized"
ARCHIVE = ROOT / "data/raw/trashnet-dataset-resized.zip"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "data/reports"
SOURCE_URL = "https://github.com/garythung/trashnet/blob/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip"
SOURCE_COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
EXPECTED_ARCHIVE_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
EXPECTED_COUNTS = {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594, "plastic": 482, "trash": 137}
MANIFEST_FIELDS = ("path", "label", "split", "object_id", "source", "group_id", "sha256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hashes(image: Image.Image) -> tuple[int, int]:
    gray = image.convert("L")
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash_bits = (horizontal[:, 1:] > horizontal[:, :-1]).flatten()
    dhash = sum(int(bit) << offset for offset, bit in enumerate(dhash_bits))
    square = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(square)[:8, :8]
    phash_bits = (dct > np.median(dct[1:, :])).flatten()
    phash = sum(int(bit) << offset for offset, bit in enumerate(phash_bits))
    return dhash, phash


def inspect_images(raw_root: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    issues: list[dict] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(ROOT).as_posix()
        label = path.parent.name
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            issues.append({"path": relative, "issue": "unexpected_extension"})
            continue
        if label not in LABELS:
            issues.append({"path": relative, "issue": "unexpected_label"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                dhash, phash = perceptual_hashes(image)
                width, height = image.size
                mode = image.mode
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues.append({"path": relative, "issue": f"unreadable: {type(exc).__name__}"})
            continue
        records.append({"path": relative, "label": label, "sha256": sha256_file(path),
                        "width": width, "height": height, "mode": mode,
                        "dhash": dhash, "phash": phash})
    return records, issues


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        self.parent[self.find(first)] = self.find(second)


def find_duplicate_candidates(records: list[dict]) -> tuple[list[dict], UnionFind]:
    """Group exact copies and conservative same-label near copies.

    Candidate hashes are screening signals, not proof that two images show one object.
    Cross-label candidates are reported but never merged automatically.
    """
    groups = UnionFind(len(records))
    candidates = []
    exact = {}
    for index, record in enumerate(records):
        previous = exact.get(record["sha256"])
        if previous is not None:
            if record["label"] == records[previous]["label"]:
                groups.union(index, previous)
            candidates.append({"path_a": records[previous]["path"], "path_b": record["path"],
                               "label_a": records[previous]["label"], "label_b": record["label"],
                               "dhash_distance": 0, "phash_distance": 0, "kind": "exact"})
        else:
            exact[record["sha256"]] = index
    for first in range(len(records)):
        a = records[first]
        for second in range(first + 1, len(records)):
            b = records[second]
            pd = (a["phash"] ^ b["phash"]).bit_count()
            if pd > 5:
                continue
            dd = (a["dhash"] ^ b["dhash"]).bit_count()
            if dd > 6 or a["sha256"] == b["sha256"]:
                continue
            kind = "near_same_label" if a["label"] == b["label"] else "near_cross_label_review"
            candidates.append({"path_a": a["path"], "path_b": b["path"],
                               "label_a": a["label"], "label_b": b["label"],
                               "dhash_distance": dd, "phash_distance": pd, "kind": kind})
            if kind == "near_same_label":
                groups.union(first, second)
    candidates.sort(key=lambda row: (row["kind"], row["phash_distance"], row["dhash_distance"],
                                     row["path_a"], row["path_b"]))
    return candidates, groups


def assign_splits(records: list[dict], groups: UnionFind, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[groups.find(index)].append(index)
    by_label = defaultdict(list)
    for indices in grouped.values():
        labels = {records[index]["label"] for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Duplicate group crosses labels: {indices}")
        by_label[next(iter(labels))].append(indices)
    manifest = []
    for class_index, label in enumerate(LABELS):
        rng = random.Random(seed + class_index)
        class_groups = sorted(by_label[label], key=lambda g: records[min(g)]["path"])
        rng.shuffle(class_groups)
        class_groups.sort(key=len, reverse=True)  # stable tie order remains seeded
        total = sum(map(len, class_groups))
        targets = {"train": round(total * .70), "val": round(total * .15)}
        targets["test"] = total - targets["train"] - targets["val"]
        counts = Counter()
        assigned = {}
        for indices in class_groups:
            # Choose the greatest proportional deficit; group sizes stay intact.
            split = max(("train", "val", "test"),
                        key=lambda name: ((targets[name] - counts[name]) / max(targets[name], 1),
                                          {"train": 2, "val": 1, "test": 0}[name]))
            counts[split] += len(indices)
            for index in indices:
                assigned[index] = split
        for indices in class_groups:
            group_key = hashlib.sha256("|".join(sorted(records[i]["path"] for i in indices)).encode()).hexdigest()[:16]
            # A hash-similar group is not a verified physical object identity.
            object_id = ""
            for index in indices:
                record = records[index]
                manifest.append({"path": record["path"], "label": label,
                                 "split": assigned[index], "object_id": object_id,
                                 "source": "trashnet", "group_id": group_key,
                                 "sha256": record["sha256"]})
    return sorted(manifest, key=lambda row: row["path"])


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_charts(records: list[dict], candidates: list[dict], report_dir: Path) -> None:
    counts = Counter(row["label"] for row in records)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(LABELS, [counts[label] for label in LABELS], color="#4285a4")
    ax.bar_label(bars)
    ax.set_ylabel("Số ảnh")
    ax.set_title("TrashNet — phân bố sáu lớp")
    fig.tight_layout()
    fig.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Fixed deterministic examples, plus first candidate pairs for visual review.
    rng = random.Random(42)
    chosen = []
    for label in LABELS:
        paths = sorted(row["path"] for row in records if row["label"] == label)
        chosen.extend((label, path) for path in rng.sample(paths, min(6, len(paths))))
    sheet = Image.new("RGB", (6 * 180, 6 * 170), "white")
    draw = ImageDraw.Draw(sheet)
    for pos, (label, path) in enumerate(chosen):
        with Image.open(ROOT / path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (166, 140))
        x, y = (pos % 6) * 180, (pos // 6) * 170
        sheet.paste(thumb, (x + (180 - thumb.width) // 2, y))
        draw.text((x + 5, y + 144), f"{label}: {Path(path).name}", fill="black")
    sheet.save(report_dir / "contact_sheet.jpg", quality=90)

    review = candidates[: min(40, len(candidates))]
    if review:
        pairs = Image.new("RGB", (2 * 250, len(review) * 190), "white")
        draw = ImageDraw.Draw(pairs)
        for row_number, row in enumerate(review):
            for col, key in enumerate(("path_a", "path_b")):
                with Image.open(ROOT / row[key]) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (240, 160))
                pairs.paste(thumb, (col * 250 + (250 - thumb.width) // 2, row_number * 190))
                draw.text((col * 250 + 4, row_number * 190 + 163), Path(row[key]).name, fill="black")
        pairs.save(report_dir / "duplicate_candidates.jpg", quality=85)


def validate_manifest(manifest: list[dict]) -> None:
    if len({row["path"] for row in manifest}) != len(manifest):
        raise ValueError("Manifest has duplicate paths")
    if set(row["label"] for row in manifest) != set(LABELS):
        raise ValueError("Manifest does not contain all six classes")
    for split in ("train", "val", "test"):
        if set(row["label"] for row in manifest if row["split"] == split) != set(LABELS):
            raise ValueError(f"Split {split} does not contain all six classes")
    group_splits = defaultdict(set)
    sha_splits = defaultdict(set)
    for row in manifest:
        group_splits[row["group_id"]].add(row["split"])
        sha_splits[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("Group leakage across splits")
    if any(len(splits) > 1 for splits in sha_splits.values()):
        raise ValueError("Exact duplicate leakage across splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not ARCHIVE.is_file() or not RAW_ROOT.is_dir():
        raise SystemExit("Missing TrashNet ZIP/extracted images; see README Phase 2")
    archive_sha = sha256_file(ARCHIVE)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit(f"Archive checksum mismatch: {archive_sha}")
    records, issues = inspect_images(RAW_ROOT)
    counts = Counter(row["label"] for row in records)
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected dataset counts: {dict(counts)}; expected {EXPECTED_COUNTS}")
    candidates, groups = find_duplicate_candidates(records)
    by_sha = defaultdict(list)
    for row in records:
        by_sha[row["sha256"]].append(row)
    conflicting_sha = {sha for sha, rows in by_sha.items() if len({row["label"] for row in rows}) > 1}
    for row in records:
        if row["sha256"] in conflicting_sha:
            issues.append({"path": row["path"], "issue": "exact_duplicate_conflicting_labels_excluded"})
    usable = [row for row in records if row["sha256"] not in conflicting_sha]
    usable_candidates, usable_groups = find_duplicate_candidates(usable)
    manifest = assign_splits(usable, usable_groups, args.seed)
    validate_manifest(manifest)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(PROCESSED / "trashnet_manifest.csv", MANIFEST_FIELDS, manifest)
    write_csv(REPORTS / "image_audit.csv", ("path", "label", "sha256", "width", "height", "mode", "dhash", "phash"), records)
    write_csv(REPORTS / "issues.csv", ("path", "issue"), issues)
    write_csv(REPORTS / "duplicate_candidates.csv",
              ("path_a", "path_b", "label_a", "label_b", "dhash_distance", "phash_distance", "kind"), candidates)
    make_charts(records, candidates, REPORTS)
    split_counts = {split: {label: sum(row["split"] == split and row["label"] == label for row in manifest)
                            for label in LABELS} for split in ("train", "val", "test")}
    group_sizes = Counter(usable_groups.find(i) for i in range(len(usable)))
    report = {"date": date.today().isoformat(), "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
              "archive_sha256": archive_sha, "manifest_sha256": sha256_file(PROCESSED / "trashnet_manifest.csv"),
              "seed": args.seed, "source_images": len(records), "usable_images": len(usable),
              "excluded_conflicting_label_images": len(records) - len(usable),
              "class_counts_source": dict(counts),
              "split_counts": split_counts, "image_sizes": dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
              "image_modes": dict(Counter(r["mode"] for r in records)),
              "issues": len(issues), "duplicate_candidates": dict(Counter(c["kind"] for c in candidates)),
              "candidate_multi_image_groups": sum(size > 1 for size in group_sizes.values()),
              "usable_duplicate_candidates": dict(Counter(c["kind"] for c in usable_candidates)),
              "limitations": ["Perceptual hashes are only candidate screening; no complete object IDs in source.",
                              "Same object can still occur across splits if not detected by hashes.",
                              "TrashNet white-background photographs do not represent webcam conditions."]}
    (REPORTS / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

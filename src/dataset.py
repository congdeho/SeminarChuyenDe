"""TrashNet Dataset/DataLoader using the locked Phase 2 manifest."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.labels import LABELS, LABEL_TO_INDEX

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/processed/trashnet_manifest.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10, fill=(255, 255, 255)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "label", "split", "object_id", "source", "group_id", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing required fields: {path}")
    for row in rows:
        if row["label"] not in LABELS or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid manifest label/split: {row}")
    return rows


class TrashNetDataset(Dataset):
    def __init__(self, split: str, manifest_path: Path = DEFAULT_MANIFEST,
                 image_size: int = 224, project_root: Path = ROOT,
                 augmentation: bool = True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.project_root = project_root
        self.rows = [row for row in read_manifest(manifest_path) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split {split}")
        self.transform = (train_transform(image_size) if split == "train" and augmentation
                          else eval_transform(image_size))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.project_root / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABEL_TO_INDEX[row["label"]]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_train_val_loaders(manifest_path: Path = DEFAULT_MANIFEST, batch_size: int = 32,
                            num_workers: int = 0, seed: int = 42,
                            image_size: int = 224,
                            augmentation: bool = True) -> tuple[DataLoader, DataLoader]:
    """Only train and validation are exposed to model selection in Phase 3."""
    generator = torch.Generator().manual_seed(seed)
    train = TrashNetDataset("train", manifest_path, image_size, augmentation=augmentation)
    val = TrashNetDataset("val", manifest_path, image_size)
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "worker_init_fn": _seed_worker, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train, shuffle=True, generator=generator, **common),
            DataLoader(val, shuffle=False, **common))


def build_test_loader(manifest_path: Path = DEFAULT_MANIFEST, batch_size: int = 32,
                      num_workers: int = 0, image_size: int = 224) -> DataLoader:
    """Use only for the final, one-time locked evaluation in Phase 5."""
    return DataLoader(TrashNetDataset("test", manifest_path, image_size),
                      batch_size=batch_size, shuffle=False, num_workers=num_workers)


def train_class_weights(manifest_path: Path = DEFAULT_MANIFEST) -> torch.Tensor:
    """Inverse-frequency weights from train split only; optional for Phase 3."""
    counts = {label: 0 for label in LABELS}
    for row in read_manifest(manifest_path):
        if row["split"] == "train":
            counts[row["label"]] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every class needs train examples")
    total = sum(counts.values())
    return torch.tensor([total / (len(LABELS) * counts[label]) for label in LABELS], dtype=torch.float32)

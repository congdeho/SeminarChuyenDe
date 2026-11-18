"""Phase 2 smoke test; only loads train and validation data."""

from __future__ import annotations

from collections import Counter
import random

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from src.dataset import (IMAGENET_MEAN, IMAGENET_STD, ROOT, build_train_val_loaders,
                         train_class_weights, train_transform)
from src.labels import LABELS


def main() -> None:
    train, val = build_train_val_loaders(batch_size=8, num_workers=0)
    images, targets = next(iter(train))
    val_images, val_targets = next(iter(val))
    assert images.shape == (8, 3, 224, 224)
    assert val_images.shape == (8, 3, 224, 224)
    assert torch.isfinite(images).all() and torch.isfinite(val_images).all()
    assert all(0 <= index < len(LABELS) for index in targets.tolist() + val_targets.tolist())
    print(f"train={len(train.dataset)} val={len(val.dataset)}")
    print(f"train batch={tuple(images.shape)} val batch={tuple(val_images.shape)}")
    print(f"train labels={dict(Counter(LABELS[i] for i in targets.tolist()))}")
    print(f"class weights (train only)={train_class_weights().tolist()}")
    undo_normalize = transforms.Normalize(
        mean=tuple(-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)),
        std=tuple(1 / std for std in IMAGENET_STD),
    )
    sheet = Image.new("RGB", (5 * 224, 6 * 254), "white")
    draw = ImageDraw.Draw(sheet)
    rng = random.Random(42)
    for row_index, class_name in enumerate(LABELS):
        sample = rng.choice([row for row in train.dataset.rows if row["label"] == class_name])
        with Image.open(ROOT / sample["path"]) as original:
            image = original.convert("RGB")
            sheet.paste(transforms.CenterCrop(224)(transforms.Resize(256)(image)), (0, row_index * 254))
            draw.text((4, row_index * 254 + 230), f"{class_name}: eval", fill="black")
            for index in range(1, 5):
                augmented = train_transform()(image)
                preview = transforms.ToPILImage()(undo_normalize(augmented).clamp(0, 1))
                sheet.paste(preview, (index * 224, row_index * 254))
                draw.text((index * 224 + 4, row_index * 254 + 230), f"train: random {index}", fill="black")
    report_path = ROOT / "data/reports/augmented_examples.jpg"
    sheet.save(report_path, quality=90)
    print(f"augmentation preview={report_path.relative_to(ROOT)}")
    print("Phase 2 DataLoader smoke test: PASS (test split not loaded)")


if __name__ == "__main__":
    main()

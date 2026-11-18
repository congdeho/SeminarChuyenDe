"""The three Phase 5 classifiers and their checkpoint metadata."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (MobileNet_V3_Small_Weights, ResNet18_Weights,
                                mobilenet_v3_small, resnet18)

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.labels import LABELS

ARCHITECTURE = "mobilenet_v3_small"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
ARCHITECTURES = ("cnn_baseline", "resnet18", ARCHITECTURE)


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks = []
        input_channels = 3
        for output_channels in (32, 64, 128, 256):
            blocks.extend((nn.Conv2d(input_channels, output_channels, 3, padding=1),
                           nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)))
            input_channels = output_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(256, len(LABELS)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(pretrained: bool = True, architecture: str = ARCHITECTURE) -> nn.Module:
    if architecture == ARCHITECTURE:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(LABELS))
    elif architecture == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    elif architecture == "cnn_baseline":
        if pretrained:
            raise ValueError("The baseline CNN must start from random weights")
        model = BaselineCNN()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model._seminar_architecture = architecture
    return model


def set_trainable_stage(model: nn.Module, stage: str) -> None:
    if stage not in {"classifier", "finetune"}:
        raise ValueError(f"Unknown training stage: {stage}")
    architecture = getattr(model, "_seminar_architecture", ARCHITECTURE)
    if architecture == "resnet18":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = stage == "finetune" or name.startswith("fc.")
    else:
        for parameter in model.features.parameters():
            parameter.requires_grad = stage == "finetune"
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def enter_train_mode(model: nn.Module, stage: str) -> None:
    model.train()
    if stage == "classifier":
        # BatchNorm running statistics in the frozen backbone must not change.
        if getattr(model, "_seminar_architecture", ARCHITECTURE) == "resnet18":
            model.eval()
            model.fc.train()
        else:
            model.features.eval()


def preprocessing_metadata(image_size: int = 224) -> dict:
    return {"image_size": image_size, "resize_short_side": 256,
            "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
            "color_mode": "RGB", "crop": "center", "interpolation": "bilinear"}

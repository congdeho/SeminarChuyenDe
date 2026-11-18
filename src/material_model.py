"""A glass/plastic specialist that preserves the four unrelated class scores."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import resnet18

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.labels import LABELS, LABEL_TO_INDEX
from src.model import build_model

ARCHITECTURE = "resnet18_material"
BOTTLE_ARCHITECTURE = "bottle_material"
GLASS = LABEL_TO_INDEX["glass"]
PLASTIC = LABEL_TO_INDEX["plastic"]
PEERS = {"e1": "cnn_baseline", "e3": "mobilenet_v3_small", "e4": "mobilenet_v3_small"}
DETAIL_NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def full_view(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image)
    edges = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0)
    pad_color = tuple(int(value) for value in np.median(edges, axis=0))
    padded = ImageOps.pad(image, (224, 224), Image.Resampling.BILINEAR, color=pad_color)
    return DETAIL_NORMALIZE(padded)


def build_encoder() -> nn.Module:
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    return encoder


def embeddings(encoder: nn.Module, detail: torch.Tensor) -> torch.Tensor:
    views = torch.cat((detail, detail.flip(-1)))
    original, mirrored = encoder(views).chunk(2)
    return (original + mirrored) * 0.5


def pair_gate(logits: torch.Tensor) -> torch.Tensor:
    top = logits.topk(2, dim=-1).indices
    return (top == GLASS).any(dim=-1) & (top == PLASTIC).any(dim=-1)


def update_pair(logits: torch.Tensor, plastic_log_odds: torch.Tensor) -> torch.Tensor:
    pair_total = logits[:, (GLASS, PLASTIC)].logsumexp(dim=-1)
    result = logits.clone()
    result[:, GLASS] = pair_total + F.logsigmoid(-plastic_log_odds)
    result[:, PLASTIC] = pair_total + F.logsigmoid(plastic_log_odds)
    return result


def agree_to_switch(base_odds: torch.Tensor, specialist_odds: torch.Tensor,
                    peer_odds: torch.Tensor, margin: float) -> torch.Tensor:
    plastic = (base_odds <= 0) & (specialist_odds > margin) & (peer_odds > margin)
    glass = (base_odds > 0) & (specialist_odds < -margin) & (peer_odds < -margin)
    return plastic | glass


class MaterialTransform:
    def __init__(self, base, use_specialist: bool):
        self.base = base
        self.use_specialist = use_specialist

    def __call__(self, image: Image.Image) -> torch.Tensor:
        center = self.base(image)
        if not self.use_specialist:
            return center.unsqueeze(0)
        return torch.stack((center, full_view(image)))


class BottleModel(nn.Module):
    """Restricted to two materials for an image already known to show a bottle."""

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.head = nn.Linear(512, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        odds = self.head(embeddings(self.encoder, inputs)).flatten()
        logits = odds.new_full((inputs.shape[0], len(LABELS)), -torch.inf)
        logits[:, GLASS] = 0
        logits[:, PLASTIC] = odds
        return logits


class MaterialModel(nn.Module):
    def __init__(self, specialist_weight: float, peer_id: str | None, peer_weight: float,
                 agreement_margin: float | None = None):
        super().__init__()
        if not (0 <= specialist_weight <= 1 and 0 <= peer_weight <= 1 - specialist_weight):
            raise ValueError("Invalid material-model weights")
        if peer_weight and peer_id not in PEERS:
            raise ValueError("Material peer checkpoint is required")
        if agreement_margin is not None and (agreement_margin < 0 or not (specialist_weight and peer_weight)):
            raise ValueError("Agreement requires two opinions and a nonnegative margin")
        self.base = build_model(pretrained=False, architecture="resnet18")
        self.specialist_weight = specialist_weight
        self.peer_weight = peer_weight
        self.agreement_margin = agreement_margin
        self.encoder = build_encoder() if specialist_weight else None
        self.head = nn.Linear(512, 1) if specialist_weight else None
        self.peer = build_model(pretrained=False, architecture=PEERS[peer_id]) if peer_weight and peer_id else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs[:, 0])
        mask = pair_gate(base)
        if not mask.any():
            return base
        selected = base[mask]
        base_odds = selected[:, PLASTIC] - selected[:, GLASS]
        odds = base_odds * (1 - self.specialist_weight - self.peer_weight)
        if self.encoder is not None:
            specialist_odds = self.head(embeddings(self.encoder, inputs[mask, 1])).flatten()
            odds = odds + self.specialist_weight * specialist_odds
        if self.peer is not None:
            peer = self.peer(inputs[mask, 0])
            peer_odds = peer[:, PLASTIC] - peer[:, GLASS]
            odds = odds + self.peer_weight * peer_odds
        updated = update_pair(selected, odds)
        if self.agreement_margin is not None:
            switch = agree_to_switch(base_odds, specialist_odds, peer_odds, self.agreement_margin)
            updated = torch.where(switch[:, None], updated, selected)
        base[mask] = updated
        return base


def load_material_model(checkpoint: dict, base_transform, device: str):
    if checkpoint["architecture"] == BOTTLE_ARCHITECTURE:
        model = BottleModel()
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device).eval(), full_view
    config = checkpoint["material_config"]
    model = MaterialModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    transform = MaterialTransform(base_transform, model.encoder is not None)
    return model.to(device).eval(), transform

"""Backbone registry + light classification head for the cpm pipeline.

Default (disqualification-proof): ImageNet ``convnext_tiny`` (timm), used FROZEN as a feature
extractor over the 480x160x3 composite. The head is a light MLP over
``concat(feature, onehot(masked_region))`` so it knows which panel is dead.

OpenPhenom-S/16 (CA-MAE) and ChannelViT are optional, license-gated drop-ins (see
``LICENSE_OK``). They consume the ``channels`` input (dead panel dropped) — wired only after the
Shipd rules clear their license. Fine-tuning 22M params on 288 images overfits; keep them frozen.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from . import config

# gate for non-commercial / channel-agnostic backbones — flip only after reading the rules
LICENSE_OK = os.environ.get("CPM_LICENSE_OK", "0") == "1"

# name -> (feat_dim, input_mode)
BACKBONES = {
    "imagenet_convnext_tiny": (768, "composite"),
    "imagenet_convnext_nano": (640, "composite"),
    "openphenom": (384, "channels"),      # CA-MAE ViT-S/16, ~22M params
    "channelvit": (384, "channels"),
}


def input_mode(name: str) -> str:
    return BACKBONES[name][1]


def feat_dim(name: str) -> int:
    return BACKBONES[name][0]


def backbone_weight_path(name: str):
    return config.WEIGHTS_DIR / name / "backbone.pt"


def build_backbone(name: str, device="cpu") -> nn.Module:
    """Return a FROZEN backbone whose forward(image) -> (B, feat_dim) pooled features.

    Offline-safe: if weights/<name>/backbone.pt exists it is loaded with pretrained=False (no
    network), so an internet-off Kaggle kernel works. Otherwise the pretrained weights are
    downloaded (online) and can be exported via :func:`export_backbone`.
    """
    if name.startswith("imagenet_"):
        import timm
        arch = name.replace("imagenet_", "")
        local = backbone_weight_path(name)
        if local.exists():
            m = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
            m.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            m = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="avg")
    elif name in ("openphenom", "channelvit"):
        if not LICENSE_OK:
            raise RuntimeError(
                f"backbone '{name}' is license-gated. Set CPM_LICENSE_OK=1 only after confirming "
                "the Shipd rules permit its weight license for a prize submission."
            )
        m = _load_channel_agnostic(name)
    else:
        raise KeyError(f"unknown backbone {name}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def export_backbone(name: str, backbone: nn.Module) -> None:
    """Persist the backbone weights for offline bundling (Kaggle internet-off kernel)."""
    if not name.startswith("imagenet_"):
        return
    dst = backbone_weight_path(name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        torch.save(backbone.state_dict(), dst)


def _load_channel_agnostic(name: str) -> nn.Module:  # pragma: no cover (gated)
    if name == "openphenom":
        # from huggingface_hub import ... ; returns a CA-MAE encoder taking (B, C, H, W) with
        # variable C. Apply PCA-CenterScale / TVN post-processing on the embeddings downstream.
        raise NotImplementedError("wire recursionpharma/OpenPhenom loader once license clears")
    raise NotImplementedError(f"{name} loader not wired")


class Head(nn.Module):
    """Light head over concat(feature, mask_onehot). kind in {'linear','mlp'}."""

    def __init__(self, in_dim: int, n_mask=config.N_PANELS, num_classes=config.NUM_CLASSES,
                 kind="mlp", hidden=256, p_drop=0.3, proj_dim=128):
        super().__init__()
        d = in_dim + n_mask
        self.kind = kind
        if kind == "linear":
            self.classifier = nn.Linear(d, num_classes)
            self.proj = nn.Linear(d, proj_dim)
        else:
            self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                                       nn.ReLU(inplace=True), nn.Dropout(p_drop))
            self.classifier = nn.Linear(hidden, num_classes)
            self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, feat, mask):
        z = torch.cat([feat, mask], dim=1)
        h = z if self.kind == "linear" else self.trunk(z)
        # normalized projection for the optional SupCon auxiliary loss
        proj = nn.functional.normalize(self.proj(h), dim=1)
        return self.classifier(h), proj

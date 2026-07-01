"""Dataset + slot-preserving augmentation for the cpm pipeline.

Two input-construction modes behind one contract:
  * ``composite`` (default / ImageNet stem): the 480x160x3 RGB composite as-is (dead panel
    included), ImageNet-normalized -> tensor (3, 160, 480).
  * ``channels`` (channel-agnostic / OpenPhenom stem): each panel reduced to one grayscale
    marker (panels are single-marker under a fixed LUT — verified in preprocess), stacked ->
    tensor (3, 160, 160) with the dead channel = 0. The backbone adapter decides whether to
    drop the dead channel; the dataset stays backbone-agnostic.

Augmentation is **slot-preserving**: panels are column slots indexed by ``masked_region``, so a
global horizontal flip would swap panels and desync the mask. Allowed: global vertical flip,
per-panel horizontal flip, per-panel mild brightness/gamma. **No panel dropout** — train and
test already share the exactly-one-black-panel distribution; a second mask is off-distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config

_TEST_LABEL = -1


def _load_norm(data_dir: Path) -> dict:
    p = data_dir / "norm_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"live_mean": [0.16, 0.16, 0.16], "live_std": [0.17, 0.17, 0.17],
            "imagenet_mean": list(config.IMAGENET_MEAN), "imagenet_std": list(config.IMAGENET_STD)}


def _augment(img: np.ndarray, dead: int, rng: np.random.Generator) -> np.ndarray:
    """Slot-preserving augmentation on a uint8 (H,480,3) composite. Keeps dead panel at 0."""
    img = img.astype(np.float32)
    # global vertical flip (rows) — does NOT move column slots
    if rng.random() < 0.5:
        img = img[::-1, :, :]
    for p in range(config.N_PANELS):
        if p == dead:
            continue
        sl = config.panel_slice(p)
        panel = img[:, sl, :]
        if rng.random() < 0.5:                       # per-panel horizontal flip (in slot)
            panel = panel[:, ::-1, :]
        gain = rng.uniform(0.85, 1.15)               # mild brightness
        gamma = rng.uniform(0.9, 1.1)                # mild gamma
        panel = 255.0 * gain * np.power(np.clip(panel / 255.0, 0, 1), gamma)
        img[:, sl, :] = np.clip(panel, 0, 255)
    return np.ascontiguousarray(img)


class CPMDataset(Dataset):
    def __init__(self, rows, data_dir=config.DATA_DIR, mode="composite", train=False,
                 norm="imagenet", seed=0):
        assert mode in ("composite", "channels")
        assert norm in ("imagenet", "live")
        self.rows = list(rows)
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.train = train
        self.seed = seed
        st = _load_norm(self.data_dir)
        if mode == "composite":
            key = "imagenet" if norm == "imagenet" else "live"
            self.mean = np.array(st[f"{key}_mean"], np.float32).reshape(3, 1, 1)
            self.std = np.array(st[f"{key}_std"], np.float32).reshape(3, 1, 1)
        else:  # channels: standardize grayscale by scalar live stats
            self.g_mean = float(np.mean(st["live_mean"]))
            self.g_std = float(np.mean(st["live_std"]))

    def __len__(self):
        return len(self.rows)

    def _label(self, r) -> int:
        lab = r.get("moa_label")
        return config.LABEL2IDX[lab] if lab else _TEST_LABEL

    def __getitem__(self, i):
        r = self.rows[i]
        dead = config.REGION2PANEL[r["masked_region"]]
        img = np.asarray(Image.open(self.data_dir / r["panel_path"]).convert("RGB"))  # (160,480,3)
        if self.train:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            img = _augment(img, dead, rng)
        img = img.astype(np.float32) / 255.0

        if self.mode == "composite":
            x = np.transpose(img, (2, 0, 1))                         # (3,160,480)
            x = (x - self.mean) / self.std
        else:  # channels: (3,160,160) per-panel grayscale, dead channel = 0
            chans = []
            for p in range(config.N_PANELS):
                g = img[:, config.panel_slice(p), :].mean(axis=2)    # single-marker -> grayscale
                if p == dead:
                    g = np.zeros_like(g)
                else:
                    g = (g - self.g_mean) / self.g_std
                chans.append(g)
            x = np.stack(chans, axis=0)

        mask = np.zeros(config.N_PANELS, np.float32)
        mask[dead] = 1.0                                             # one-hot: which panel is DEAD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "mask": torch.from_numpy(mask),                         # (3,)
            "label": torch.tensor(self._label(r), dtype=torch.long),
            "sample_id": r["sample_id"],
        }

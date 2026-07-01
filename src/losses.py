"""Losses for the cpm pipeline.

Primary: label-smoothed cross-entropy. Note (arXiv:2402.06855) label smoothing alone can
entrench low-variance / spurious features, so it is paired with — not a substitute for — the
compound-invariant backbone and the supervised-contrastive auxiliary.

Auxiliary (optional): supervised contrastive loss keyed on MoA. Positives are same-MoA pairs;
because compound is the nuisance, same-MoA/different-compound pairs pull the representation
toward compound invariance. This only helps the multi-compound classes — a thin class has one
compound, so its only positives are same-compound and teach nothing about invariance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_smoothed_ce(smoothing: float = 0.1) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=smoothing)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalized projections."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, proj: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = proj.device
        b = proj.shape[0]
        if b < 2:
            return proj.new_tensor(0.0)
        sim = proj @ proj.t() / self.t                       # (B,B)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(b, dtype=torch.bool, device=device)
        pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
        exp = torch.exp(sim).masked_fill(eye, 0.0)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        pos_cnt = pos.sum(1)
        valid = pos_cnt > 0
        if not valid.any():
            return proj.new_tensor(0.0)
        mean_log_prob_pos = (pos * log_prob).sum(1)[valid] / pos_cnt[valid]
        return -mean_log_prob_pos.mean()

"""Adaptive contact-map thresholding utilities for SobolevRNA.

The RNA-FM HWS path starts from a dense pairwise pseudo-probability matrix
``P_ij = (cos(e_i, e_j) + 1) / 2``.  A single global threshold can over-admit
long-range contacts in repetitive megascale RNAs, so these helpers make the
threshold a monotone function of sequence separation ``s = |i - j|`` while
retaining an exact static-threshold compatibility mode.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

DEFAULT_THETA0 = 0.85
DEFAULT_THETA_MAX = 0.94
DEFAULT_ALPHA = 0.02
DEFAULT_MIN_SEQ_SEP = 6
DEFAULT_CHUNK_SIZE = 500
DEFAULT_SEPARATION_BINS: dict[str, tuple[int, int | None]] = {
    "short": (6, 24),
    "medium": (25, 100),
    "long": (101, None),
}


def threshold_for_sequence_separation(
    separation,
    theta0: float = DEFAULT_THETA0,
    theta_max: float = DEFAULT_THETA_MAX,
    alpha: float = DEFAULT_ALPHA,
    beta: float | None = None,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    mode: str = "power",
) -> np.ndarray:
    """Return the contact threshold for one or many sequence separations.

    ``mode='power'`` implements ``theta(s) = min(theta_max,
    theta0 * (s / max(min_seq_sep, 1))**alpha)``.  ``mode='log'`` implements the
    logarithmic alternative from the research note.  ``mode='static'`` returns
    ``theta0`` exactly and is useful for A/B reproducibility.
    """

    sep = np.asarray(separation, dtype=np.float32)
    sep_anchor = max(float(min_seq_sep), 1.0)
    safe_sep = np.maximum(sep, sep_anchor)

    if mode in {"static", "fixed", "none", "off"}:
        theta = np.full_like(safe_sep, float(theta0), dtype=np.float32)
    elif mode == "power":
        theta = float(theta0) * np.power(safe_sep / sep_anchor, float(alpha))
    elif mode == "log":
        if beta is None:
            span = max(math.log(1000.0 / sep_anchor), 1e-8)
            beta = (float(theta_max) - float(theta0)) / span
        theta = float(theta0) + float(beta) * np.log(safe_sep / sep_anchor)
    else:
        raise ValueError(f"unknown adaptive threshold mode: {mode}")

    return np.clip(theta, float(theta0), float(theta_max)).astype(np.float32)


def binarize_contact_probability(
    contact_prob,
    theta0: float = DEFAULT_THETA0,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    adaptive: bool = True,
    theta_max: float = DEFAULT_THETA_MAX,
    alpha: float = DEFAULT_ALPHA,
    beta: float | None = None,
    mode: str = "power",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Binarize a square contact probability matrix with separation-aware cuts."""

    prob = np.asarray(contact_prob, dtype=np.float32)
    if prob.ndim != 2 or prob.shape[0] != prob.shape[1]:
        raise ValueError("contact_prob must be a square matrix")

    n = prob.shape[0]
    contact_map = np.zeros((n, n), dtype=np.float32)
    cols = np.arange(n, dtype=np.int32)[None, :]
    threshold_mode = mode if adaptive else "static"
    step = max(1, int(chunk_size))

    for start in range(0, n, step):
        end = min(start + step, n)
        rows = np.arange(start, end, dtype=np.int32)[:, None]
        sep = np.abs(rows - cols)
        theta = threshold_for_sequence_separation(
            sep,
            theta0=theta0,
            theta_max=theta_max,
            alpha=alpha,
            beta=beta,
            min_seq_sep=min_seq_sep,
            mode=threshold_mode,
        )
        contact_map[start:end] = prob[start:end] > theta

    for offset in range(-int(min_seq_sep), int(min_seq_sep) + 1):
        np.fill_diagonal(contact_map[max(0, -offset):, max(0, offset):], 0.0)

    return np.maximum(contact_map, contact_map.T).astype(np.float32)


def contact_density_by_separation_bins(
    contact_map,
    bins: Mapping[str, tuple[int, int | None]] | None = None,
) -> dict[str, float]:
    """Measure predicted contact density in short/medium/long separation bins."""

    cmap = np.asarray(contact_map)
    if cmap.ndim != 2 or cmap.shape[0] != cmap.shape[1]:
        raise ValueError("contact_map must be a square matrix")
    bins = bins or DEFAULT_SEPARATION_BINS
    n = cmap.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    sep = cols - rows
    values = cmap[rows, cols] > 0
    out: dict[str, float] = {}
    for name, (lo, hi) in bins.items():
        mask = sep >= int(lo)
        if hi is not None:
            mask &= sep <= int(hi)
        out[name] = float(values[mask].mean()) if mask.any() else math.nan
    return out


def fit_contact_decay_gamma(
    contact_map,
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
    max_seq_sep: int | None = None,
    n_bins: int = 24,
) -> float:
    """Fit ``P(contact | s) ~ s^-gamma`` from a binary contact map."""

    cmap = np.asarray(contact_map)
    if cmap.ndim != 2 or cmap.shape[0] != cmap.shape[1]:
        raise ValueError("contact_map must be a square matrix")
    n = cmap.shape[0]
    if n <= min_seq_sep + 1:
        return math.nan
    max_sep = int(max_seq_sep or (n - 1))
    max_sep = min(max_sep, n - 1)
    if max_sep <= min_seq_sep:
        return math.nan

    rows, cols = np.triu_indices(n, k=min_seq_sep)
    sep = cols - rows
    values = cmap[rows, cols] > 0
    edges = np.unique(
        np.round(np.geomspace(min_seq_sep, max_sep + 1, num=max(3, n_bins + 1))).astype(int)
    )
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sep >= lo) & (sep < hi)
        if not mask.any():
            continue
        density = float(values[mask].mean())
        if density <= 0.0:
            continue
        xs.append(math.log((lo + hi - 1) / 2.0))
        ys.append(math.log(density))
    if len(xs) < 2:
        return math.nan
    slope, _intercept = np.polyfit(np.asarray(xs), np.asarray(ys), deg=1)
    return float(-slope)


def matthews_corrcoef_by_separation_bins(
    predicted,
    truth,
    bins: Mapping[str, tuple[int, int | None]] | None = None,
) -> dict[str, float]:
    """Compute MCC between predicted and true contacts by sequence-separation bin."""

    pred = np.asarray(predicted) > 0
    true = np.asarray(truth) > 0
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[0] != pred.shape[1]:
        raise ValueError("predicted and truth must be same-shape square matrices")
    bins = bins or DEFAULT_SEPARATION_BINS
    n = pred.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    sep = cols - rows
    pred_vals = pred[rows, cols]
    true_vals = true[rows, cols]
    out: dict[str, float] = {}
    for name, (lo, hi) in bins.items():
        mask = sep >= int(lo)
        if hi is not None:
            mask &= sep <= int(hi)
        if not mask.any():
            out[name] = math.nan
            continue
        p = pred_vals[mask]
        t = true_vals[mask]
        tp = float(np.sum(p & t))
        tn = float(np.sum(~p & ~t))
        fp = float(np.sum(p & ~t))
        fn = float(np.sum(~p & t))
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out[name] = (tp * tn - fp * fn) / denom if denom else 0.0
    return out

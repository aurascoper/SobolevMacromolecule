"""Calibration for the cpm pipeline — the corrected, sign-independent design.

Two things are fit offline on the dense-class OOF logits (legitimate: dense classes have
held-out-compound OOF predictions):
  * global temperature T (NLL-minimizing) on all dense OOF;
  * thin_shrink: fit by shrinking the 2-compound LOCO_CLASSES — which, by our compound-grouped
    CV, are already leave-one-compound-out predictions — to the factor that minimizes dense OOF
    log-loss. That factor is the empirically-optimal damping for "a class seen through a single
    (other) compound," i.e. the closest measurable analog to a thin class. Transferred to the
    true thin classes as a LOWER bound (they have strictly less diversity).

At inference (K fold models available), :func:`calibrate_probs` applies the corrected transform:
  1. temperature-scale each fold's logits;
  2. shrink thin-class probability mass by thin_shrink and RENORMALIZE — sign-independent, unlike
     the buggy `logits[:,cls] /= T` which raises thin prob on the majority of non-thin images;
  3. blend mildly toward the 1/12 prior weighted by ensemble BALD disagreement
     (H(mean) - mean_k H(p_k)), NOT the entropy of the mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import config


# ---------------------------------------------------------------------------
# core numerics
# ---------------------------------------------------------------------------
def _softmax(logits, T=1.0, axis=-1):
    z = logits / T
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=axis)


def _shrink(probs, classes, s):
    p = probs.copy()
    p[:, classes] *= s
    return p / p.sum(1, keepdims=True)


def logloss(probs, labels):
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.log(p[np.arange(len(labels)), labels]).mean())


# ---------------------------------------------------------------------------
# offline fitting on dense OOF
# ---------------------------------------------------------------------------
def fit_temperature(logits, labels):
    def nll(logT):
        return logloss(_softmax(logits, np.exp(logT)), labels)
    r = minimize_scalar(nll, bounds=(np.log(0.25), np.log(10.0)), method="bounded")
    return float(np.exp(r.x))


def fit_shrink(logits, labels, T, classes):
    """Shrink factor s in (0,1] on `classes` minimizing dense-OOF log-loss."""
    base = _softmax(logits, T)

    def ll(s):
        return logloss(_shrink(base, classes, s), labels)
    r = minimize_scalar(ll, bounds=(0.05, 1.0), method="bounded")
    return float(r.x)


# ---------------------------------------------------------------------------
# inference-time transform (corrected)
# ---------------------------------------------------------------------------
def calibrate_probs(logits_stack, T, thin_shrink, thin_classes=None, bald_pow=2.0):
    """logits_stack: (K, B, C) from K fold models -> calibrated (B, C) probabilities."""
    thin_classes = config.THIN_CLASSES if thin_classes is None else thin_classes
    logits_stack = np.asarray(logits_stack, dtype=np.float64)
    p = _softmax(logits_stack, T, axis=-1)                 # (K,B,C)
    mean = p.mean(0)                                       # (B,C)

    total_H = _entropy(mean)                               # (B,)
    mean_H = _entropy(p).mean(0)                           # (B,)
    bald = np.clip(total_H - mean_H, 0, None)              # ensemble disagreement (MI/BALD)
    denom = bald.max() if bald.max() > 1e-9 else 1.0
    bald = bald / denom                                    # ~[0,1]

    ps = _shrink(mean, thin_classes, thin_shrink)          # sign-independent thin damping
    w = (bald ** bald_pow)[:, None]                        # mild blend toward prior
    uniform = np.full_like(ps, config.UNIFORM_PRIOR)
    return (1.0 - w) * ps + w * uniform


# ---------------------------------------------------------------------------
# CLI: fit + report
# ---------------------------------------------------------------------------
def fit_and_report(oof_path, thin_extra=1.0):
    d = np.load(oof_path, allow_pickle=True)
    logits, labels = d["logits"], d["label"]
    C = config.NUM_CLASSES

    T = fit_temperature(logits, labels)
    loco_shrink = fit_shrink(logits, labels, T, config.LOCO_CLASSES)
    thin_shrink = float(np.clip(loco_shrink * thin_extra, 0.02, 1.0))  # lower bound, optional extra

    uni = np.full((len(labels), C), config.UNIFORM_PRIOR)
    report = {
        "backbone_oof": str(oof_path),
        "temperature": T,
        "loco_shrink": loco_shrink,
        "thin_shrink": thin_shrink,
        "loco_classes": config.LOCO_CLASSES,
        "thin_classes": config.THIN_CLASSES,
        "logloss_uniform": logloss(uni, labels),
        "logloss_raw": logloss(_softmax(logits, 1.0), labels),
        "logloss_temponly": logloss(_softmax(logits, T), labels),
        "logloss_temp_loco_shrink": logloss(
            _shrink(_softmax(logits, T), config.LOCO_CLASSES, loco_shrink), labels),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--thin-extra", type=float, default=1.0,
                    help="<1 damps thin classes harder than the LOCO lower bound")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = fit_and_report(args.oof, args.thin_extra)
    tag = Path(args.oof).name.replace("oof_", "").replace(".npz", "")
    out = args.out or str(config.REPO_ROOT / f"calib_{tag}.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))
    # guardrails: calibrated must beat uniform and not be worse than temp-only
    assert rep["logloss_temponly"] <= rep["logloss_raw"] + 1e-6, "temperature made it worse"
    assert rep["logloss_temp_loco_shrink"] <= rep["logloss_temponly"] + 1e-6, "LOCO shrink hurt dense OOF"
    assert rep["logloss_temponly"] < rep["logloss_uniform"], "model no better than uniform"
    print(f"\n[calibrate] wrote {out}")


if __name__ == "__main__":
    main()

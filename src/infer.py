"""Offline inference -> submission.csv.

Load the frozen backbone + K trained fold heads, predict on the test panels, ensemble across
folds, apply the corrected calibration (temperature -> sign-independent thin shrink -> BALD
blend), and write probabilities in sample_submission column order.

Run:  python -m src.infer --backbone imagenet_convnext_tiny --calib calib_imagenet_convnext_tiny.json --out submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch

from . import calibrate, config, cv, models
from .train import _forward_features, get_device


def _load_heads(backbone, device):
    wdir = config.WEIGHTS_DIR / backbone
    heads = []
    for pt in sorted(wdir.glob("fold*.pt")):
        ck = torch.load(pt, map_location=device)
        h = models.Head(ck["in_dim"], kind=ck["kind"]).to(device)
        h.load_state_dict(ck["state"]); h.eval()
        heads.append(h)
    if not heads:
        raise FileNotFoundError(f"no fold*.pt in {wdir} — run src.train first")
    return heads


@torch.no_grad()
def predict(backbone, rows, calib, device, tta=1):
    mode = models.input_mode(backbone)
    bb = models.build_backbone(backbone, device=device)
    # deterministic features (+ optional TTA augmented views, averaged in feature space)
    feats = [_forward_features(bb, rows, mode, device, train_aug=False, seed=0)]
    for v in range(max(0, tta - 1)):
        feats.append(_forward_features(bb, rows, mode, device, train_aug=True, seed=100 + v))
    feat = np.mean([f[0] for f in feats], axis=0)
    mask = feats[0][1]
    ids = feats[0][3]

    heads = _load_heads(backbone, device)
    ft = torch.tensor(feat, dtype=torch.float32, device=device)
    mt = torch.tensor(mask, dtype=torch.float32, device=device)
    stack = []
    for h in heads:
        lg, _ = h(ft, mt)
        stack.append(lg.cpu().numpy())
    stack = np.stack(stack, 0)                                   # (K, N, C)

    probs = calibrate.calibrate_probs(stack, calib["temperature"], calib["thin_shrink"])
    return ids, probs


def write_submission(ids, probs, out_path, sample_sub=config.SAMPLE_SUB_CSV):
    by_id = {sid: probs[i] for i, sid in enumerate(ids)}
    with open(sample_sub) as fh:
        order = [r["sample_id"] for r in csv.DictReader(fh)]
    assert set(order) == set(by_id), "test ids differ from sample_submission ids"
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id"] + config.PROB_COLS)
        for sid in order:
            p = by_id[sid]
            w.writerow([sid] + [f"{x:.6f}" for x in p])
    return len(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_convnext_tiny")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test-csv", default=str(config.TEST_CSV))
    ap.add_argument("--out", default=str(config.REPO_ROOT / "submission.csv"))
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    calib = json.loads(open(args.calib).read())
    rows = cv.load_rows(args.test_csv)
    ids, probs = predict(args.backbone, rows, calib, device, tta=args.tta)

    # validity guards before writing
    assert probs.shape == (len(rows), config.NUM_CLASSES)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5), "rows must sum to 1"
    assert (probs >= 0).all(), "negative probability"
    n = write_submission(ids, probs, args.out)
    print(f"[infer] wrote {args.out}: {n} rows, {config.NUM_CLASSES} prob cols, "
          f"row-sum≈{probs.sum(1).mean():.4f}, thin-mass mean={probs[:, config.THIN_CLASSES].sum(1).mean():.3f}")


if __name__ == "__main__":
    main()

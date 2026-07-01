"""Extract public.zip, verify the data invariants at full scale, and cache dataset stats.

Run:  python -m src.preprocess --zip public.zip --out data/

Full-dataset checks (not just the 4 spot-checked images):
  * config.verify_against_data — label/compound structure matches constants.
  * verify_masking — EVERY image (train+test) has exactly one all-zero panel, and that panel
    equals REGION2PANEL[masked_region]. This is the load-bearing assumption of the whole
    design; if it ever fails we want to know before training, not after grading.
  * norm_stats — per-channel mean/std over LIVE (non-masked) pixels only, so the dead panel
    does not skew normalization.
  * panel_report — per-panel RGB structure, to decide single- vs multi-marker for the
    channel-agnostic (OpenPhenom) input path.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import config


def extract_zip(zip_path: Path, out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "train.csv"
    if marker.exists() and not force:
        print(f"[extract] {marker} already present — skip (use --force to re-extract)")
        return
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"[extract] done: {sum(1 for _ in out_dir.rglob('*.png'))} panels")


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def _zero_panels(img: np.ndarray) -> list[int]:
    """Indices of panels that are exactly all-zero across all channels."""
    return [p for p in range(config.N_PANELS)
            if not img[:, config.panel_slice(p), :].any()]


def verify_masking(data_dir: Path) -> dict:
    """Assert every image has exactly one all-zero panel == masked_region. Full dataset."""
    n_checked = 0
    for split, csv_name in (("train", "train.csv"), ("test", "test.csv")):
        rows = _read_rows(data_dir / csv_name)
        for r in rows:
            img = np.asarray(Image.open(data_dir / r["panel_path"]).convert("RGB"))
            zeros = _zero_panels(img)
            assert len(zeros) == 1, (
                f"{r['sample_id']}: expected exactly 1 masked panel, found {zeros} "
                f"(shape={img.shape}). Core 'exactly-one-black-panel' assumption violated."
            )
            expect = config.REGION2PANEL[r["masked_region"]]
            assert zeros[0] == expect, (
                f"{r['sample_id']}: black panel is {zeros[0]} but masked_region="
                f"{r['masked_region']} => panel {expect}. masked_region is unreliable."
            )
            n_checked += 1
    print(f"[verify_masking] OK — {n_checked} images: exactly one black panel, matches masked_region")
    return {"n_checked": n_checked}


def compute_norm_stats(data_dir: Path) -> dict:
    """Per-channel mean/std over LIVE (non-masked) pixels of the train split."""
    rows = _read_rows(data_dir / "train.csv")
    # streaming mean/var over live pixels, per RGB channel, in [0,1]
    n = 0
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    for r in rows:
        img = np.asarray(Image.open(data_dir / r["panel_path"]).convert("RGB")).astype(np.float64) / 255.0
        dead = config.REGION2PANEL[r["masked_region"]]
        live = [p for p in range(config.N_PANELS) if p != dead]
        for p in live:
            px = img[:, config.panel_slice(p), :].reshape(-1, 3)
            n += px.shape[0]
            s += px.sum(0)
            ss += (px * px).sum(0)
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    stats = {"live_mean": mean.tolist(), "live_std": std.tolist(),
             "imagenet_mean": list(config.IMAGENET_MEAN), "imagenet_std": list(config.IMAGENET_STD)}
    print(f"[norm_stats] live mean={np.round(mean,3).tolist()} std={np.round(std,3).tolist()}")
    return stats


def panel_report(data_dir: Path, n_sample: int = 40) -> dict:
    """Per-panel RGB structure. If, within a live panel, the three RGB channels are highly
    correlated, that panel is effectively single-marker (grayscale under a fixed LUT) and the
    channel-agnostic path can reduce it to one intensity map; low correlation => multi-marker.
    """
    rows = _read_rows(data_dir / "train.csv")[:n_sample]
    per_panel_corr = {p: [] for p in range(config.N_PANELS)}
    for r in rows:
        img = np.asarray(Image.open(data_dir / r["panel_path"]).convert("RGB")).astype(np.float64)
        dead = config.REGION2PANEL[r["masked_region"]]
        for p in range(config.N_PANELS):
            if p == dead:
                continue
            px = img[:, config.panel_slice(p), :].reshape(-1, 3)
            if px.std() < 1e-6:
                continue
            c = np.corrcoef(px.T)  # 3x3
            per_panel_corr[p].append(float(np.nanmin([c[0, 1], c[0, 2], c[1, 2]])))
    summary = {}
    for p, vals in per_panel_corr.items():
        if vals:
            m = float(np.mean(vals))
            summary[f"panel_{p}"] = {
                "min_rgb_corr_mean": round(m, 3),
                "interpretation": "single-marker (grayscale LUT)" if m > 0.9 else "multi-marker (true RGB)",
            }
    print(f"[panel_report] {json.dumps(summary)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=config.REPO_ROOT / "public.zip")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-masking", action="store_true", help="skip the full-dataset mask check")
    args = ap.parse_args()

    if args.zip.exists():
        extract_zip(args.zip, args.out, force=args.force)
    else:
        print(f"[extract] {args.zip} not found — assuming data already at {args.out}")

    summary = config.verify_against_data(args.out / "train.csv")
    print(f"[verify_against_data] OK — {json.dumps(summary)}")

    if not args.skip_masking:
        verify_masking(args.out)

    stats = compute_norm_stats(args.out)
    (args.out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    report = panel_report(args.out)
    (args.out / "panel_report.json").write_text(json.dumps(report, indent=2))
    print("[preprocess] complete.")


if __name__ == "__main__":
    main()

"""Assemble the offline-inference bundle and (optionally) push it as a Kaggle dataset.

The bundle contains everything the internet-off inference kernel needs:
  src/                     — pipeline code
  weights/<backbone>/      — backbone.pt (ImageNet weights, so no HF download) + fold*.pt heads
  calib_<backbone>.json    — temperature + thin_shrink

Run:  python upload_kaggle_dataset.py --backbone imagenet_convnext_tiny [--push]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from src import config

BUNDLE = config.REPO_ROOT / "kaggle_bundle"
SLUG = "sobolevcellular-cpm-bundle"


def build_bundle(backbone: str) -> Path:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "weights").mkdir(parents=True)
    # code
    shutil.copytree(config.REPO_ROOT / "src", BUNDLE / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # weights (backbone + heads)
    src_w = config.WEIGHTS_DIR / backbone
    assert (src_w / "backbone.pt").exists(), f"missing {src_w/'backbone.pt'} (run src.train)"
    assert list(src_w.glob("fold*.pt")), f"no fold heads in {src_w}"
    shutil.copytree(src_w, BUNDLE / "weights" / backbone)
    # calibration
    calib = config.REPO_ROOT / f"calib_{backbone}.json"
    assert calib.exists(), f"missing {calib} (run src.calibrate)"
    shutil.copy(calib, BUNDLE / calib.name)
    # dataset metadata for the kaggle CLI
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps({
        "title": "SobolevCellular CPM bundle",
        "id": f"REPLACE_USERNAME/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    print(f"[bundle] {BUNDLE}  ({size/1e6:.1f} MB)")
    return BUNDLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_convnext_tiny")
    ap.add_argument("--push", action="store_true", help="create/version via the kaggle CLI")
    args = ap.parse_args()
    build_bundle(args.backbone)
    if args.push:
        meta = BUNDLE / "dataset-metadata.json"
        print("[bundle] set the 'id' username in", meta, "then this creates/versions it")
        exists = subprocess.run(["kaggle", "datasets", "status", "-p", str(BUNDLE)],
                                capture_output=True).returncode == 0
        cmd = ["kaggle", "datasets", "version", "-p", str(BUNDLE), "-m", "update"] if exists \
            else ["kaggle", "datasets", "create", "-p", str(BUNDLE)]
        subprocess.run(cmd, check=True)
    else:
        print("[bundle] built. Edit dataset-metadata.json id, then re-run with --push (needs kaggle CLI).")


if __name__ == "__main__":
    main()

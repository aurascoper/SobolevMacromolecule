"""Data-contract tests for the cpm pipeline. Requires `python -m src.preprocess` to have run
(data/ populated). Runnable via pytest or directly: python tests/test_cpm_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, cv, dataset  # noqa: E402


def _zero_panels(img):
    return [p for p in range(config.N_PANELS) if not img[:, config.panel_slice(p), :].any()]


def _train_rows():
    return cv.load_rows(config.TRAIN_CSV)


def test_config_matches_data():
    s = config.verify_against_data()
    assert config.THIN_CLASSES == [2, 5, 9]
    assert s["thin_labels"] == ["moa_02", "moa_05", "moa_09"]
    assert config.LOCO_CLASSES == [0, 1, 6, 7, 11]


def test_dataset_mask_matches_black_panel():
    rows = _train_rows()[:20]
    ds = dataset.CPMDataset(rows, mode="composite", train=False)
    for i, r in enumerate(rows):
        item = ds[i]
        dead = int(item["mask"].argmax())
        assert item["mask"].sum() == 1.0
        raw = np.asarray(Image.open(config.DATA_DIR / r["panel_path"]).convert("RGB"))
        assert _zero_panels(raw) == [dead], f"{r['sample_id']}: mask {dead} != black {_zero_panels(raw)}"
        assert dead == config.REGION2PANEL[r["masked_region"]]


def test_channels_mode_dead_channel_zero():
    rows = _train_rows()[:10]
    ds = dataset.CPMDataset(rows, mode="channels", train=False)
    for i in range(len(rows)):
        item = ds[i]
        x = item["image"].numpy()            # (3,160,160)
        dead = int(item["mask"].argmax())
        assert np.all(x[dead] == 0.0)
        live = [c for c in range(3) if c != dead]
        assert all(np.any(x[c] != 0.0) for c in live)


def test_augment_slot_preserving():
    rows = _train_rows()[:15]
    for seed in (0, 1, 7):
        ds = dataset.CPMDataset(rows, mode="channels", train=True, seed=seed)
        for i, r in enumerate(rows):
            item = ds[i]
            dead = int(item["mask"].argmax())
            # dead panel/channel must still be dead after augmentation (slots preserved)
            assert np.all(item["image"].numpy()[dead] == 0.0)
            assert dead == config.REGION2PANEL[r["masked_region"]]


def test_cv_folds_pin_thin_and_cover_dense():
    rows = _train_rows()
    folds = cv.make_folds(rows, n_splits=5, seed=42)
    cov = cv.oof_coverage(rows, folds)
    assert cov["thin_covered"] == 0, "thin classes must never be in OOF"
    assert cov["dense_covered"] == cov["dense_total"], "every dense sample needs one OOF pred"
    # every fold trains on all thin compounds
    thin_comp = {r["compound_id"] for r in rows if config.LABEL2IDX[r["moa_label"]] in config.THIN_CLASSES}
    for train_idx, _ in folds:
        tr_comp = {rows[i]["compound_id"] for i in train_idx}
        assert thin_comp <= tr_comp, "a fold is missing a thin compound in train"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")

"""Central constants + data-verified invariants for the cpm (cell-phenotype MoA) pipeline.

The single most dangerous class of bug in this competition is a silent index desync: the
earlier blueprint used thin-class indices ``[1, 4, 8]`` (off-by-one) which would have damaged
three healthy classes while leaving the real single-compound classes exposed. Every constant
here that is derived from the data is *re-checked against the data* by :func:`verify_against_data`
so it fails loudly rather than silently.

Measured ground truth (public.zip, 2026-06-27):
  - 288 train / 144 test panels, 480x160 uint8 RGB = three 160x160 side-by-side panels.
  - exactly one panel pre-masked to (0,0,0) per image; masked_region in {region_0,1,2}.
  - 12 classes moa_00..moa_11, 24 images each, 25 unique compounds.
  - single-compound (thin) classes: moa_02, moa_05, moa_09  ->  indices 2, 5, 9.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (env-overridable so the offline Kaggle kernel can point DATA/WEIGHTS at
# the mounted competition data + bundled-dataset weights without editing code).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CPM_DATA_DIR", REPO_ROOT / "data"))
WEIGHTS_DIR = Path(os.environ.get("CPM_WEIGHTS_DIR", REPO_ROOT / "weights"))
FEAT_CACHE = Path(os.environ.get("CPM_FEAT_CACHE", REPO_ROOT / ".feat_cache"))
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
NUM_CLASSES = 12
LABELS = [f"moa_{i:02d}" for i in range(NUM_CLASSES)]      # sorted == natural order
LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}
IDX2LABEL = {i: lab for i, lab in enumerate(LABELS)}
PROB_COLS = [f"prob_{lab}" for lab in LABELS]              # submission column order
UNIFORM_PRIOR = 1.0 / NUM_CLASSES

# Encoding sanity: moa_0X -> integer X. (self-consistent; the *data* check is below.)
assert all(LABEL2IDX[f"moa_{i:02d}"] == i for i in range(NUM_CLASSES)), "label->idx desync"

# ---------------------------------------------------------------------------
# Compound structure (measured; verified against data at runtime)
# ---------------------------------------------------------------------------
# n_compounds per class index:
N_COMPOUNDS_PER_CLASS = {
    0: 2, 1: 2, 2: 1, 3: 3, 4: 3, 5: 1, 6: 2, 7: 2, 8: 3, 9: 1, 10: 3, 11: 2,
}
# Single-compound "thin" classes — the confounded, hardest-to-generalize labels.
THIN_CLASSES = [2, 5, 9]
# Two-compound classes — the closest structural analog to a thin class under leave-one-
# compound-out (one held-out compound => zero in-distribution compounds), used to calibrate
# `thin_shrink` as a LOWER bound on what the true thin classes need.
LOCO_CLASSES = [0, 1, 6, 7, 11]
DENSE_CLASSES = [i for i in range(NUM_CLASSES) if i not in THIN_CLASSES]

assert THIN_CLASSES == sorted(i for i, n in N_COMPOUNDS_PER_CLASS.items() if n == 1)
assert LOCO_CLASSES == sorted(i for i, n in N_COMPOUNDS_PER_CLASS.items() if n == 2)

# ---------------------------------------------------------------------------
# Panel / mask geometry
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 160, 480
PANEL = 160
N_PANELS = 3
REGIONS = ["region_0", "region_1", "region_2"]
REGION2PANEL = {"region_0": 0, "region_1": 1, "region_2": 2}
assert IMG_W == PANEL * N_PANELS

# ImageNet stats (for the default RGB composite stem). Applied to the WHOLE composite;
# the dead panel stays a consistent constant train==test because it is (0,0,0) in both.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def panel_slice(panel_idx: int) -> slice:
    """Column slice for panel ``panel_idx`` in a (H, 480, C) composite."""
    return slice(panel_idx * PANEL, (panel_idx + 1) * PANEL)


# ---------------------------------------------------------------------------
# Runtime verification against the actual CSV — the load-bearing assert.
# ---------------------------------------------------------------------------
def verify_against_data(train_csv: Path | str = TRAIN_CSV) -> dict:
    """Recompute label/compound structure from train.csv and assert it matches constants.

    Raises AssertionError with an explicit message if the data ever disagrees with the
    hard-coded THIN_CLASSES / LOCO_CLASSES / label encoding. Returns a small summary dict.
    """
    train_csv = Path(train_csv)
    with train_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{train_csv} is empty"

    labels = sorted({r["moa_label"] for r in rows})
    assert labels == LABELS, f"label set mismatch: data={labels} const={LABELS}"

    comp_per_class = defaultdict(set)
    for r in rows:
        comp_per_class[LABEL2IDX[r["moa_label"]]].add(r["compound_id"])
    n_comp = {k: len(v) for k, v in comp_per_class.items()}
    assert n_comp == N_COMPOUNDS_PER_CLASS, (
        f"compound-per-class mismatch:\n  data ={dict(sorted(n_comp.items()))}"
        f"\n  const={N_COMPOUNDS_PER_CLASS}"
    )

    data_thin = sorted(i for i, n in n_comp.items() if n == 1)
    assert data_thin == THIN_CLASSES, (
        f"THIN_CLASSES desync: data single-compound classes={data_thin} "
        f"({[IDX2LABEL[i] for i in data_thin]}) but const THIN_CLASSES={THIN_CLASSES}. "
        "This is exactly the off-by-one bug the pipeline exists to prevent — STOP."
    )

    class_counts = Counter(LABEL2IDX[r["moa_label"]] for r in rows)
    regions = sorted({r["masked_region"] for r in rows})
    assert regions == REGIONS, f"masked_region values mismatch: {regions}"

    return {
        "n_rows": len(rows),
        "n_classes": len(labels),
        "class_counts": dict(sorted(class_counts.items())),
        "thin_classes": data_thin,
        "thin_labels": [IDX2LABEL[i] for i in data_thin],
        "n_unique_compounds": len({r["compound_id"] for r in rows}),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify_against_data(), indent=2))

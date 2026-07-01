"""Cross-validation splits for the cpm pipeline.

Compound-grouped K-fold with a hard rule: **the single compound of each thin class
(moa_02/05/09) is pinned into every training fold and never held out.** If a thin compound
landed in a validation fold, that fold would have zero training examples of the class and could
never learn it — the structural contradiction the earlier plan missed.

Consequences (by design):
  * Thin-class samples have NO out-of-fold (OOF) prediction — they are always in train.
  * Every dense (multi-compound) sample gets exactly one OOF prediction.
  * For the 2-compound LOCO_CLASSES, grouping by compound means that whenever one compound is
    in validation the model trained only on the *other* compound — i.e. the OOF prediction is
    already a leave-one-compound-out prediction. calibrate.py uses this to estimate how
    overconfident the model is on a single unseen compound (the thin-class analog).
"""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from . import config


def load_rows(csv_path) -> list[dict]:
    with Path(csv_path).open() as fh:
        return list(csv.DictReader(fh))


def _label_idx(r) -> int:
    return config.LABEL2IDX[r["moa_label"]]


def make_folds(rows, n_splits: int = 5, seed: int = 42):
    """Return list of (train_idx, val_idx) into ``rows``.

    Thin-class rows are appended to every train_idx and excluded from every val_idx.
    Dense rows are split by StratifiedGroupKFold(group=compound_id, y=label).
    """
    thin_idx = [i for i, r in enumerate(rows) if _label_idx(r) in config.THIN_CLASSES]
    dense_idx = [i for i, r in enumerate(rows) if _label_idx(r) not in config.THIN_CLASSES]

    y = [_label_idx(rows[i]) for i in dense_idx]
    groups = [rows[i]["compound_id"] for i in dense_idx]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for tr, va in sgkf.split(dense_idx, y, groups):
        train_idx = [dense_idx[j] for j in tr] + thin_idx
        val_idx = [dense_idx[j] for j in va]
        folds.append((sorted(train_idx), sorted(val_idx)))

    # sanity: no compound spans train and val within a fold; thin never in val
    for train_idx, val_idx in folds:
        tr_comp = {rows[i]["compound_id"] for i in train_idx}
        va_comp = {rows[i]["compound_id"] for i in val_idx}
        assert tr_comp.isdisjoint(va_comp), "compound leakage across a fold"
        assert all(_label_idx(rows[i]) not in config.THIN_CLASSES for i in val_idx), \
            "thin-class sample leaked into validation"
    return folds


def oof_coverage(rows, folds) -> dict:
    """Diagnostic: which sample indices ever appear in a val fold (get an OOF prediction)."""
    covered = set()
    for _, val_idx in folds:
        covered.update(val_idx)
    dense = [i for i, r in enumerate(rows) if _label_idx(r) not in config.THIN_CLASSES]
    thin = [i for i, r in enumerate(rows) if _label_idx(r) in config.THIN_CLASSES]
    return {
        "n_folds": len(folds),
        "dense_covered": sum(i in covered for i in dense),
        "dense_total": len(dense),
        "thin_covered": sum(i in covered for i in thin),   # expected 0
        "thin_total": len(thin),
    }

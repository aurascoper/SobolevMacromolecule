"""K-fold training on cached frozen-backbone features.

Pattern: extract features ONCE with the frozen backbone (optionally with K slot-preserving
augmented views cached alongside), then train a light head per fold on the cached features.
Produces per-fold head weights + an out-of-fold (OOF) logit file consumed by calibrate.py.

Run:  python -m src.train --backbone imagenet_convnext_tiny --folds 5 --epochs 150
Smoke (CPU): add --limit 96 --epochs 30 --aug-views 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config, cv, losses, models
from .dataset import CPMDataset


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _forward_features(backbone, rows, mode, device, train_aug, seed, batch=32):
    ds = CPMDataset(rows, mode=mode, train=train_aug, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    feats, masks, labels, ids = [], [], [], []
    for b in dl:
        f = backbone(b["image"].to(device)).float().cpu()
        feats.append(f)
        masks.append(b["mask"])
        labels.append(b["label"])
        ids.extend(b["sample_id"])
    return (torch.cat(feats).numpy(), torch.cat(masks).numpy(),
            torch.cat(labels).numpy(), np.array(ids))


def extract_features(backbone_name, rows, device, aug_views=1, batch=32):
    """Deterministic features for all rows + optional K augmented views (train aug)."""
    mode = models.input_mode(backbone_name)
    cache = config.FEAT_CACHE / f"{backbone_name}_train_a{aug_views}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    config.FEAT_CACHE.mkdir(exist_ok=True)
    backbone = models.build_backbone(backbone_name, device=device)
    models.export_backbone(backbone_name, backbone)          # save for offline bundling
    det = _forward_features(backbone, rows, mode, device, train_aug=False, seed=0, batch=batch)
    out = {"feat_det": det[0], "mask": det[1], "label": det[2], "id": det[3]}
    aug = []
    for v in range(max(0, aug_views - 1)):
        fa = _forward_features(backbone, rows, mode, device, train_aug=True, seed=1 + v, batch=batch)
        aug.append(fa[0])
    out["feat_aug"] = np.stack(aug, 0) if aug else np.zeros((0,) + det[0].shape, np.float32)
    np.savez(cache, **out)
    return out


def _train_head(Xtr, Mtr, Ytr, Xva, Mva, in_dim, device, epochs, kind, smoothing,
                supcon_w, lr, seed):
    torch.manual_seed(seed)
    head = models.Head(in_dim, kind=kind).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = losses.label_smoothed_ce(smoothing)
    sc = losses.SupConLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Mtr_t = torch.tensor(Mtr, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(Ytr, dtype=torch.long, device=device)
    n = Xtr_t.shape[0]
    bs = min(64, n)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            logits, proj = head(Xtr_t[idx], Mtr_t[idx])
            loss = ce(logits, Ytr_t[idx])
            if supcon_w > 0:
                loss = loss + supcon_w * sc(proj, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    head.eval()
    with torch.no_grad():
        va_logits, _ = head(torch.tensor(Xva, dtype=torch.float32, device=device),
                            torch.tensor(Mva, dtype=torch.float32, device=device))
    return head, va_logits.cpu().numpy()


def _macro_f1(logits, labels):
    from sklearn.metrics import f1_score
    return f1_score(labels, logits.argmax(1), average="macro", labels=list(range(config.NUM_CLASSES)),
                    zero_division=0)


def _logloss(logits, labels):
    p = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p = np.clip(p, 1e-9, 1)
    return float(-np.log(p[np.arange(len(labels)), labels]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_convnext_tiny")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--supcon", type=float, default=0.0, help="SupCon aux weight (0=off)")
    ap.add_argument("--aug-views", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="subsample train rows for smoke test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[train] backbone={args.backbone} device={device} folds={args.folds}")
    config.verify_against_data()
    rows = cv.load_rows(config.TRAIN_CSV)
    if args.limit:
        rows = rows[:args.limit]

    fe = extract_features(args.backbone, rows, device, aug_views=args.aug_views)
    feat_det, mask, label, ids = fe["feat_det"], fe["mask"], fe["label"], fe["id"]
    feat_aug = fe["feat_aug"]                                      # (V, N, D) or (0,...)
    in_dim = models.feat_dim(args.backbone)

    folds = cv.make_folds(rows, n_splits=args.folds, seed=args.seed)
    print(f"[train] OOF coverage: {cv.oof_coverage(rows, folds)}")

    wdir = config.WEIGHTS_DIR / args.backbone
    wdir.mkdir(parents=True, exist_ok=True)
    oof_logits = np.full((len(rows), config.NUM_CLASSES), np.nan, np.float32)

    for k, (tr, va) in enumerate(folds):
        Xtr, Mtr, Ytr = feat_det[tr], mask[tr], label[tr]
        if feat_aug.shape[0]:                                     # append augmented views of train
            Xtr = np.concatenate([Xtr] + [feat_aug[v][tr] for v in range(feat_aug.shape[0])], 0)
            Mtr = np.concatenate([Mtr] * (1 + feat_aug.shape[0]), 0)
            Ytr = np.concatenate([Ytr] * (1 + feat_aug.shape[0]), 0)
        head, va_logits = _train_head(Xtr, Mtr, Ytr, feat_det[va], mask[va], in_dim, device,
                                      args.epochs, args.head, args.smoothing, args.supcon,
                                      args.lr, args.seed + k)
        oof_logits[va] = va_logits
        torch.save({"state": head.state_dict(), "kind": args.head, "in_dim": in_dim,
                    "backbone": args.backbone}, wdir / f"fold{k}.pt")
        f1 = _macro_f1(va_logits, label[va]); ll = _logloss(va_logits, label[va])
        print(f"[fold {k}] n_val={len(va)} macroF1={f1:.3f} logloss={ll:.3f}")

    # OOF metrics on dense classes (thin classes have no OOF by design)
    dense = ~np.isnan(oof_logits[:, 0])
    dl_ids, dl_lab = ids[dense], label[dense]
    dl_logits = oof_logits[dense]
    f1 = _macro_f1(dl_logits, dl_lab); ll = _logloss(dl_logits, dl_lab)
    print(f"[OOF dense] n={dense.sum()} macroF1={f1:.3f} logloss={ll:.3f} (uniform LL={np.log(12):.3f})")

    np.savez(config.REPO_ROOT / f"oof_{args.backbone}.npz",
             logits=dl_logits, label=dl_lab, id=dl_ids)
    (config.REPO_ROOT / f"metrics_{args.backbone}.json").write_text(
        json.dumps({"oof_dense_macroF1": f1, "oof_dense_logloss": ll,
                    "backbone": args.backbone, "folds": args.folds}, indent=2))
    print(f"[train] saved heads -> {wdir}, OOF -> oof_{args.backbone}.npz")


if __name__ == "__main__":
    main()

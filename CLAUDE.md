# SobolevCellular — Channel-Dropout Multiplex Cell-Phenotype (MoA) Classification

Active project: a 12-class mechanism-of-action (MoA) image classifier for the **Eris / Shipd**
"Channel-Dropout Multiplex Cell-Phenotype" competition. Input is a 480×160 RGB Cell-Painting
composite = three 160×160 panels with **exactly one panel blacked out** (`masked_region`).
Submission is a probability per class (`prob_moa_00..11`) scored on calibrated log-loss over a
**compound-blind** private test.

(Repo renamed from `SobolevMacromolecule`; the RNA-structure files at the root — `SobolevRNA.ipynb`,
`sobolev_*.py`, `bdbv_*.py`, `docs/whitepaper.md` — are legacy and unrelated to this pipeline.)

## Stack
- Python 3.10+, numpy, pandas, scikit-learn, scipy, PyTorch, **torchvision**, Pillow.
- The **submission notebook (`solution.ipynb`) uses only Kaggle-Docker libraries** (the list
  above) — NO `timm`. `timm` is used only by the local `src/` dev pipeline.

## Run / test
```sh
# local dev pipeline (timm backbones, feature caching)
python -m src.preprocess --zip ~/Downloads/public.zip --out data/
python -m src.train      --backbone imagenet_convnext_small --folds 5
python -m src.calibrate  --oof oof_imagenet_convnext_small.npz
python -m src.infer      --backbone imagenet_convnext_small --calib calib_imagenet_convnext_small.json
python3 tests/test_cpm_pipeline.py         # 5 data-contract tests

# Eris submission dry-run (mirrors the grading layout)
#   stage ./dataset/public -> data/, run solution.ipynb via nbconvert, check ./working/submission.csv
```

## Submission (Eris) — the deliverable is `solution.ipynb`
Self-contained; **reads `./dataset/public/`, writes `./working/submission.csv`**; Kaggle-Docker
libs only; **< 30 min on an A10G** (24GB VRAM). Defensive weight load (torchvision pretrained →
graceful fallback, logged). Eris rewards **iterative submissions** (baseline→best; 6 credits,
+1/4h) and rubric-scores *how* you solve — the notebook narrates its reasoning in markdown.

## Conventions (the corrected-blueprint invariants — do not regress)
- **Thin (single-compound) classes = `moa_02/05/09` → indices `[2,5,9]`**, re-derived from the
  data and asserted (`src/config.verify_against_data`). Never hard-code `[1,4,8]`.
- **No panel dropout.** Train ≡ test already have exactly one black panel; a second mask is
  off-distribution. Augmentation, if any, is within-surviving-panel only (slot-preserving).
- **Backbone frozen, head light.** N=288 — do not fine-tune the backbone. Default composite RGB
  stem = `convnext_small`; feed `concat(feature, onehot(masked_region))`.
- **Compound-grouped CV**, thin compounds pinned into every train fold (thin classes get no OOF).
- **Calibration:** temperature on dense OOF; **sign-correct probability-space thin-shrink**
  (`p[:,[2,5,9]] *= s; renormalize` — never divide logits); `s` from LOCO on the 2-compound
  classes `[0,1,6,7,11]` as a lower bound; **BALD** (`H(mean)−mean H(p_k)`) blend toward the 1/12
  prior at inference.
- **Normalization excludes the black panel** (compute stats on live pixels / use fixed ImageNet
  stats on the composite, where the dead panel is a consistent constant train==test).

## Gotchas
- Data lives in `~/Downloads/public.zip` (288 train / 144 test panels); `data/`, `weights/`,
  `.feat_cache/`, `submission*.csv`, `.eris_dryrun/` are gitignored.
- Every panel is a **single marker under a fixed color LUT** (RGB corr ≥0.998) — so the
  channel-agnostic path can reduce each surviving panel to one grayscale channel.
- OpenPhenom-S/16 (CA-MAE, channel-agnostic — arXiv:2404.10242) is a possible stronger backbone
  but is **Non-Commercial licensed**; benchmarking is fine, submission use is a rules gate. It is
  also not a Kaggle-Docker library, so it can't go in `solution.ipynb` without an allowed attach.

# SobolevCellular

**Channel-Dropout Multiplex Cell-Phenotype (MoA) Classification — Eris / Shipd**

12-class mechanism-of-action (MoA) prediction from 3-panel Cell-Painting images in which **one
panel is blacked out**. The submission is a probability per class scored on calibrated log-loss,
with a compound-blind private test — so the whole design targets *unseen-compound generalization*,
not in-distribution accuracy.

## The two structural difficulties (and the design that answers them)

1. **Compound confounding.** Three classes (`moa_02/05/09`) are backed by a *single compound*
   each — their label is collinear with that compound's batch/staining signature. We (a) validate
   with **compound-grouped CV** so the score reflects unseen-compound generalization, pinning the
   single-compound classes into every training fold (they get no OOF — correct, not a bug), and
   (b) **damp confidence** on those classes at calibration via a sign-correct probability-space
   shrink calibrated on the 2-compound classes (the LOCO analog).
2. **One missing panel.** Exactly one of three 160×160 panels is `(0,0,0)` in *both* train and
   test (`masked_region` names which). Train and test already share this distribution → there is
   **no channel-dropout augmentation to add**; a second mask would be off-distribution. We feed
   `masked_region` as a one-hot so the head knows which panel is absent.

## Method

Frozen ImageNet `convnext_small` features (chosen over tiny/base by compound-blind OOF) + a light
MLP head over `concat(feature, onehot(masked_region))`, K-fold ensemble, then temperature +
sign-correct thin-class shrinkage + ensemble-disagreement (BALD) blending toward the uniform prior.
No LLM is used anywhere in the pipeline.

**Compound-blind OOF floor:** dense-OOF log-loss **1.366** vs uniform 2.485 (torchvision
convnext_small; macro-F1 0.495). Runs end-to-end in **~80 s on CPU** (seconds on the A10G).

## Submission — Eris

The submitted artifact is **`solution.ipynb`**: self-contained, reads `./dataset/public/`, writes
`./working/submission.csv`, uses only Kaggle-Docker libraries (numpy/pandas/scikit-learn/scipy/
pytorch/torchvision), and completes well under the 30-min A10G budget. Weight loading is
defensive (torchvision pretrained → graceful fallback, logged).

## Layout

- **`solution.ipynb`** — the self-contained Eris submission notebook (the deliverable).
- **`src/`** — the local reference/dev pipeline mirroring the notebook's design, split into
  `config, preprocess, dataset, cv, models, losses, train, calibrate, infer`; run via `python -m
  src.<module>`. `tests/test_cpm_pipeline.py` pins the data contracts.
- **`approach.md`** — the corrected design writeup + measured data facts.
- **`setup_eris.sh`** — A10G environment bootstrap.

## Legacy

This repo was renamed from `SobolevMacromolecule`; the prior RNA-structure project's files
(`SobolevRNA.ipynb`, `sobolev_*.py`, `bdbv_*.py`, `docs/whitepaper.md`, etc.) remain at the repo
root as legacy and are unrelated to the CPM pipeline.

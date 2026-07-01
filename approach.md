# Approach — Channel-Dropout Multiplex Cell-Phenotype (MoA) Classification

12-class mechanism-of-action (MoA) classification from Cell-Painting panels. Data ships as
`public.zip`. This document is the corrected design; it supersedes an earlier blueprint that
contained load-bearing bugs (documented under "Corrections" below).

## Data ground truth (measured from public.zip — do not re-derive)
- **288 train PNG, 144 test PNG.** Native format **480×160 `uint8` RGB** = three side-by-side
  **160×160 panels** (false-color: panel0 blue-ish, panel1 magenta, panel2 green; fixed
  per-panel LUT, all three RGB channels populated in a live panel).
- **Exactly one panel is pre-masked to `(0,0,0)` per image, in BOTH train and test.**
  `masked_region ∈ {region_0,region_1,region_2}` names the dead panel. The masked panel's mean
  is *exactly* 0.0 (non-negative pixels ⇒ every pixel is 0). Never zero-masked, never two.
  Region frequencies ≈ uniform (train 93/91/104, test 53/43/48).
- **12 classes `moa_00..moa_11`, 24 images each.** 25 unique compounds.
  **Single-compound classes = `moa_02`, `moa_05`, `moa_09` (indices 2, 5, 9).** Others 2–3
  compounds. Compound is the confounder: a 1-compound class's label is collinear with that
  compound's batch/staining signature.
- **Submission:** `sample_id, prob_moa_00 … prob_moa_11` (probabilities; uniform 1/12 baseline).
- **Leak checks:** mask ⟂ compound; mask varies within each thin class; `P(mask|moa)` ~flat
  (mild skew on moa_00 region_2=0.58, moa_07 region_1=0.50 — monitor, don't redesign).

## Consequences that shape the design
1. **Train and test already share the "exactly one black panel" distribution.** There is no
   masking mismatch to correct → **panel/channel dropout is OFF.** Re-masking is impossible
   (dead pixels are destroyed) and a second mask fabricates a 2-black state test never contains.
   Regularization is **within the two surviving panels only** (crops/flips/mild brightness).
2. **At N=288 the dominant lever is a strong *frozen* backbone**, not augmentation. Lever order:
   (1) backbone, (2) thin-class log-loss protection, (3) calibration, (4) SupCon aux, (5) aug.

## Method
- **Backbone (default, disqualification-proof):** ImageNet `convnext_tiny` (timm), **frozen**
  features + light MLP head over `concat(feature, onehot(masked_region))`. Feeds the 480×160×3
  composite as-is. **OpenPhenom-S/16 (CA-MAE)** is a license-gated drop-in (see Rules gate); its
  `channels` input drops the dead panel and feeds the surviving markers.
- **CV:** compound-grouped K-fold. **Thin-class single compounds are pinned into every training
  fold** (never held out — else a fold can't learn that class). No valid OOF for thin classes.
- **Calibration:** global temperature fit on *dense-class* OOF; thin-class confidence set by a
  **sign-correct probability-space shrink** (`p[:,[2,5,9]] *= thin_shrink; renormalize`), with
  `thin_shrink` estimated by **LOCO on the 2-compound classes** and treated as a lower bound;
  ensemble **BALD** disagreement (`H(mean) − mean H(p_k)`) blends mildly toward the 1/12 prior.
- **Loss:** label-smoothed CE (primary) + optional SupCon (MoA positive, compound nuisance) —
  SupCon helps only multi-compound classes.

## Rules gate (human — before committing OpenPhenom)
Read two Shipd clauses: (a) is scoring **internet-off**? (b) does prize eligibility require a
**commercially usable** weight license? OpenPhenom's Non-Commercial EULA only clears if both
pass. Fallback: ChannelViT + JUMP-CP weights *if* permissive + bundle-able, else ImageNet only.

## Corrections vs. the earlier blueprint
1. Thin-class indices were `[1,4,8]` (off-by-one) → **`[2,5,9]`** (asserted in `src/config.py`).
2. Damping divided logits (sign-dependent, raised thin-class prob on non-thin images) →
   **probability-space shrink + renormalize.**
3. Panel dropout ("drop 1–2 else keep all three") is undefined here (data is pre-masked) →
   **removed.**
4. "Epistemic uncertainty = entropy of mean" → **BALD mutual information.**
5. Class-dependent dropout would inject a mask-count confounder → **dropout is class-uniform /
   off.**

## Layout
`src/{config,preprocess,dataset,cv,models,losses,train,calibrate,infer}.py`,
`notebooks/kaggle_{train,infer}.ipynb`, `upload_kaggle_dataset.py`, `setup_eris.sh`.
See `/Users/aurascoper/.claude/plans/the-plan-at-users-aurascoper-claude-plan-swift-puppy.md`.

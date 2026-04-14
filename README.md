# SobolevRNA

**Megascale RNA 3D Structure Prediction · 0.509 private TM-score · Stanford RNA 3D Folding Part 2 (Kaggle)**

Multi-model RNA 3D structure prediction pipeline with a custom physics relaxation engine, designed for megascale targets (up to 4,640 nt). Achieves a private/public score inversion (0.387 public → 0.509 private) through physics-guided generalization rather than benchmark overfitting.

📄 **[Full technical whitepaper with derivations](docs/whitepaper.md)** · [Reviewer notes](docs/reviewer-notes-opus.md) · [Kaggle competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)

> **Reproducibility note:** `SobolevRNA.ipynb` is Kaggle-native — model weights are mounted as Kaggle datasets under `/kaggle/input/`. Running locally requires substituting paths to locally downloaded weights for RNA-FM, Boltz-1, Protenix, and RibonanzaNet-2. See [ATTRIBUTION.md](ATTRIBUTION.md) for source links to each model.

---

## Architecture

```
Input Sequence (L nucleotides)
        │
        ├─── L ≤ 1022 nt ──► RNA-FM (single pass, 640-dim embeddings)
        │
        └─── L > 1022 nt ──► HWS (Hierarchical Windowed Sensor)
                                   sliding window + taper blend
                                          │
                              ┌───────────▼───────────┐
                              │  Global Contact Map    │
                              │  C_ij ∈ {0,1}^{N×N}  │
                              └───────────┬───────────┘
                                          │
             ┌─────────────── Routing by L ──────────────────┐
             │                                               │
        L ≤ 512 nt                                     L > 512 nt
             │                                               │
   ┌─────────▼──────────┐                    ┌──────────────▼────────────┐
   │  Ensemble: Boltz-1 │                    │  SHR Megascale Path       │
   │  + Protenix (N=5)  │                    │  (Stochastic Hamiltonian  │
   └─────────┬──────────┘                    │   Relaxation, JAX x64)    │
             │                               └──────────────┬────────────┘
             └──────────────────┬────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  SHR Physics Polish   │
                    │  (E_bond + E_rep +    │
                    │   E_DL + E_Rg)        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Hungarian Chain Map  │
                    │  + Kabsch Alignment   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  submission.csv       │
                    │  (C1' coordinates)    │
                    └───────────────────────┘
```

---

## HWS — Hierarchical Windowed Sensor

RNA-FM has a hard architectural truncation at 1,022 nt. For megascale targets (e.g. 9MME = 4,640 nt), a single-pass embedding is impossible. HWS extracts embeddings via overlapping windows and blends them with a linear taper to eliminate hard boundary artifacts.

**Parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $W$ | 1022 nt | RNA-FM max window |
| $s$ | 768 nt | Stride between windows |
| $\tau$ | 128 nt | Taper length at boundaries |
| $d$ | 640 | Embedding dimension |

**Weighted accumulation:**

For each window $w$ covering positions $[a_w, b_w)$, a taper weight $\omega_{w}(i)$ is computed for each position $i$:

$$\omega_{w}(i) = \begin{cases} \frac{i - a_w}{\tau} & i \in [a_w,\; a_w + \tau) \\ 1 & i \in [a_w + \tau,\; b_w - \tau) \\ \frac{b_w - i}{\tau} & i \in [b_w - \tau,\; b_w) \end{cases}$$

The blended embedding at position $i$ is:

$$\tilde{e}(i) = \frac{\sum_w \omega_{w}(i)\; e_w(i)}{\sum_w \omega_{w}(i)}$$

The blended embeddings are converted to a global pairwise contact map via cosine similarity:

$$C_{ij} = \mathbf{1}\!\left[\frac{\tilde{e}(i) \cdot \tilde{e}(j)}{\|\tilde{e}(i)\|\|\tilde{e}(j)\|} > \theta\right] \cdot \mathbf{1}[|i - j| \geq 6]$$

with $\theta = 0.85$ and minimum sequence separation 6.

---

## SHR — Stochastic Hamiltonian Relaxation

The physics engine operates on C1′ backbone coordinates $\mathbf{x} \in \mathbb{R}^{N \times 3}$. The Hamiltonian has four terms:

$$H(\mathbf{x}) = E_{\text{bond}} + E_{\text{rep}} + E_{\text{DL}} + E_{R_g}$$

### Bond Term

$$E_{\text{bond}} = k_{\text{bond}} \sum_{i=1}^{N-1} \left(\|\mathbf{x}_{i+1} - \mathbf{x}_i\| - d_0\right)^2$$

$d_0 = 5.95$ Å (ideal C1′–C1′ backbone distance), $k_{\text{bond}} = 100.0$ (fixed across all tiers — prevents chain breakage under compaction).

### Steric Repulsion Term

$$E_{\text{rep}} = \sum_{\substack{i < j \\ |i-j| > 1}} \left(\max\!\left(0,\; \sigma_{\text{clash}} - \|\mathbf{x}_i - \mathbf{x}_j\|\right)\right)^2$$

$\sigma_{\text{clash}} \in [2.6, 3.0]$ Å, length-dependent (see dynamic schedule below). The gradient is regularized with $\epsilon = 10^{-2}$ to bound the maximum force at ~60 Å$^{-1}$ per pair (vs. ~60,000 without regularization).

### Deep Learning Contact Restraint Term

$$E_{\text{DL}} = w_{\text{DL}} \sum_{(i,j) \in \mathcal{C}} \left(\max\!\left(0,\; \|\mathbf{x}_i - \mathbf{x}_j\| - d_{\text{contact}}\right)\right)^2$$

$\mathcal{C}$ is the contact set from a 50/50 consensus of the HWS pipeline (RNA-FM embeddings) and RibonanzaNet-2 predictions, thresholded to binary. $d_{\text{contact}} = 8.0$ Å, and $w_{\text{DL}} \in [10.0, 25.0]$ scales with sequence length.

### Radius of Gyration Term (Flory Scaling)

$$E_{R_g} = k_{R_g} \left(\max\!\left(0,\; R_g^{\text{target}} - R_g(\mathbf{x})\right)\right)^2$$

The Flory-scaling target enforces physical compaction:

$$R_g^{\text{target}} = 3.5 \cdot N^{0.45} \quad \text{(Å)}$$

$$R_g(\mathbf{x}) = \sqrt{\frac{1}{N}\sum_i \|\mathbf{x}_i - \bar{\mathbf{x}}\|^2}$$

with $k_{R_g} = 1.0$.

---

## Sobolev H¹ Gradient Preconditioning

Gradient descent on $H$ with raw gradients causes high-frequency noise to dominate updates — a known pathology in chain relaxation. The gradient is preconditioned in spectral space via a Sobolev H¹ seminorm filter:

$$\hat{\nabla}_k = \frac{(\text{DCT-II}\;\nabla H)_k}{1 + \alpha k^2}$$

$$\widetilde{\nabla H} = \text{IDCT-II}\!\left(\hat{\nabla}_k\right)$$

with $\alpha = 10.0$. This damps wavenumber $k$ by factor $(1 + \alpha k^2)^{-1}$, suppressing oscillatory modes while preserving the low-frequency compaction signal. The learning rate follows a linear decay schedule:

$$\eta(t) = \eta_0 \left(1 - \frac{t}{T}\right)$$

---

## Dynamic Parameter Schedule

Parameters adapt linearly with sequence length across three tiers:

$$t = \text{clamp}\!\left(\frac{N - 1000}{4640 - 1000},\; 0,\; 1\right)$$

| Parameter | $N \leq 1000$ | $N \in (1000, 3000]$ | $N > 3000$ |
|-----------|--------------|----------------------|------------|
| $w_{\text{DL}}$ | 10.0 | $10.0 + 15.0\,t$ | $10.0 + 15.0\,t$ |
| $\sigma_{\text{clash}}$ | 3.0 Å | $3.0 - 0.4\,t$ Å | $3.0 - 0.4\,t$ Å |
| $T$ (steps) | 1,000 | 1,500 | 8,000 |
| $\eta_0$ | 0.02 | 0.01 | 0.005 |
| $K$ (seeds) | 5 | 3 | 1 |

9MME at $N = 4640$ targets $R_g^{\text{target}} = 3.5 \times 4640^{0.45} \approx 156$ Å, requiring $w_{\text{DL}} \approx 25.0$ to achieve compaction.

---

## Kabsch Alignment (Chain Stitching)

Consecutive 3D chunks are aligned via the Kabsch algorithm (Kabsch 1976). Given mobile anchor $\mathbf{M} \in \mathbb{R}^{m \times 3}$ and target anchor $\mathbf{T} \in \mathbb{R}^{m \times 3}$:

$$\mathbf{H} = (\mathbf{M} - \boldsymbol{\mu}_M)^\top (\mathbf{T} - \boldsymbol{\mu}_T)$$

$$\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top \quad \text{(SVD)}$$

$$\mathbf{D} = \text{diag}(1,\; 1,\; \text{sgn}(\det(\mathbf{V}^\top \mathbf{U}^\top))) \quad \text{(chirality guard)}$$

$$\mathbf{R} = \mathbf{V}\, \mathbf{D}\, \mathbf{U}^\top, \qquad \mathbf{T}_{\text{translate}} = \boldsymbol{\mu}_T - \mathbf{R}\,\boldsymbol{\mu}_M$$

The seam between aligned chunks is blended with a linear taper over the overlap region to prevent harmonic-force spikes in the subsequent SHR polish.

---

## Results

| Set | TM-score |
|-----|---------|
| Public leaderboard | 0.38650 |
| **Private leaderboard** | **0.50934** |

The private > public inversion reflects that SHR physics generalizes to novel RNA families (the private set composition) better than pure deep learning baselines.

---

## Dependencies

| Model | Source | Role |
|-------|--------|------|
| RNA-FM | Chen et al. 2022 | Sequence embeddings (HWS input) |
| RibonanzaNet-2 | Shujun717/Kaggle | Contact map (E_DL fallback) |
| Boltz-1 | odat1248/Kaggle | 3D seed generation (L ≤ 800 nt) |
| Protenix v1 | qiweiyin/Kaggle | 3D generation (L ≤ 512 nt) |
| USalign | Zhang Lab | TM-score evaluation |

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits and licenses.

---

## Citation

If you build on HWS or SHR, please cite:

```
@misc{kinder2026rna,
  author = {Kinder, Hunter},
  title  = {SobolevRNA: Megascale RNA 3D Structure Prediction with Hierarchical
            Windowed Sensing and Stochastic Hamiltonian Relaxation},
  year   = {2026},
  url    = {https://github.com/aurascoper/SobolevRNA}
}
```

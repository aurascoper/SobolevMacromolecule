# SobolevMacromolecule

**Megascale RNA 3D Structure Prediction · 0.509 private TM-score · Stanford RNA 3D Folding Part 2 (Kaggle)**

> **Safety and scope:** This repository is for non-infectious computational
> structure modeling, relaxation, and visualization. It does not contain
> infectious material, reverse-genetics workflows, synthesis-ready constructs,
> or laboratory protocols. See [SAFETY.md](SAFETY.md).

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

$$
\omega_{w}(i) =
\left\{
\begin{array}{ll}
\frac{i - a_w}{\tau} & i \in [a_w,\; a_w + \tau) \\
1 & i \in [a_w + \tau,\; b_w - \tau) \\
\frac{b_w - i}{\tau} & i \in [b_w - \tau,\; b_w)
\end{array}
\right.
$$

The blended embedding at position $i$ is:

$$\tilde{e}(i) = \frac{\sum_w \omega_{w}(i)\; e_w(i)}{\sum_w \omega_{w}(i)}$$

The blended embeddings are converted to a global pairwise contact map via cosine similarity. The cosine is rescaled to $P_{ij} = (S_{ij}+1)/2$, then binarized with a sequence-separation threshold:

$$
\theta(s) = \min\left(\theta_{\max},\; \theta_0 \left(\frac{\max(s, 6)}{6}\right)^\alpha\right),\qquad s = |i-j|
$$

$$
C_{ij} = \mathbf{1}[P_{ij} > \theta(|i-j|)] \cdot \mathbf{1}[|i-j| \geq 6]
$$

Defaults are $\theta_0 = 0.85$, $\theta_{\max} = 0.94$, and $\alpha = 0.02$. This preserves the original local threshold while requiring stronger RNA-FM evidence for long-range contacts that would otherwise over-saturate megascale contact maps.

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

$\mathcal{C}$ is the contact set from a 50/50 consensus of the HWS pipeline (RNA-FM embeddings) and RibonanzaNet-2 predictions, thresholded to binary. The RNA-FM side uses the adaptive $\theta(|i-j|)$ rule above, while RibonanzaNet-2 contributes its learned pairwise channel. $d_{\text{contact}} = 8.0$ Å, and $w_{\text{DL}} \in [10.0, 25.0]$ scales with sequence length.

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

## Guarded Hybrid Polish for External Candidate Pipelines

SobolevRNA now includes `sobolev_polish_gate.py`, a production-safe bridge for
using SHR as a post-processing layer on strong external candidates such as the
Stanford RNA 3D Folding Part 2 1st-place ensemble. The intended integration is
serial:

```
external predictors (Boltz2 / Protenix / DRFold2 / RNAPro / TBM)
        → C1' candidate slots
        → SobolevRNA contact map C
        → guarded shr_polish
        → accept/reject gate
        → submission.csv
```

The bridge deliberately calls `shr_polish`, not `shr_refine_single`. External
DL/TBM candidates already have plausible global folds, so the polish stage uses
the input coordinates as the initial condition and never adds stochastic
Gaussian jitter. The polish Hamiltonian is the local C1′ objective

$$
H_{\mathrm{polish}}(X; C)
= E_{\mathrm{bond}}(X)
+ E_{\mathrm{steric}}(X)
+ E_{\mathrm{DL}}(X; C),
$$

with $E_{R_g}$ disabled during polish by setting $R_g^{\mathrm{target}} = 0$.
This prevents the post-processor from imposing a new global compaction basin on
a model that may already have the right topology. The terms are

$$
E_{\mathrm{bond}}(X)
= k_{\mathrm{bond}}\sum_{i=1}^{N-1}
\left(\lVert x_{i+1}-x_i\rVert_2 - 5.95\right)^2,
$$

$$
E_{\mathrm{steric}}(X)
= \sum_{j>i+1}
\left[
\max\left(
0,\;
\sigma_{\mathrm{clash}}
- \sqrt{\lVert x_i-x_j\rVert_2^2 + 10^{-2}}
\right)
\right]^2,
$$

$$
E_{\mathrm{DL}}(X; C)
= w_{\mathrm{DL}}\sum_{i,j} C_{ij}
\left[
\max\left(0,\lVert x_i-x_j\rVert_2 - 8.0\right)
\right]^2.
$$

The gradient update uses the same Sobolev $H^1$ resolvent as SHR, implemented
with the correct JAX DCT path:

$$
\widetilde{\nabla H}
= \mathrm{IDCT}_{II}\left(
\frac{\mathrm{DCT}_{II}(\nabla H)_k}{1+\alpha k^2}
\right),
$$

where `sobolev_polish_gate.py` imports `jax.scipy.fft as jfft` and enables
`jax_enable_x64=True`. The production defaults are
$\alpha = 5.0$, `clip = 2.0`, $w_{\mathrm{DL}} = 2.0$,
$k_{\mathrm{bond}} = 100.0$, $\sigma_{\mathrm{clash}} = 3.0$, 2000 steps, and
learning rate 0.01.

### Accept/Reject Gate

In a fixed best-of-5 submission, post-processing is not automatically additive:
replacing a raw candidate can reduce the best slot if the refined structure
drifts away from the native fold. The bridge therefore accepts a polished
candidate only if all checks pass:

1. Shape matches the raw candidate and all coordinates are finite, non-sentinel,
   and within the sanitizer bound.
2. Bond violations do not increase, where a violation is an adjacent C1′
   distance with $\lvert d - 5.95\rvert > 2.0$ Å.
3. Steric clashes do not increase, using KDTree pairs below 3.0 Å and excluding
   adjacent residues.
4. $H_{\mathrm{polish}}(X_{\mathrm{refined}}; C) <
   H_{\mathrm{polish}}(X_{\mathrm{raw}}; C)$.
5. $R_g(X_{\mathrm{refined}})$ lies in
   $[0.7, 1.5]\cdot 3.5N^{0.45}$.
6. $\max_i \lVert x_{i+1}-x_i\rVert_2 < 12.0$ Å.
7. `tm_self >= 0.85`, computed by Kabsch-aligning refined coordinates to raw
   coordinates and applying a TM-style C1′ similarity over valid residues.

Rejected candidates leave the original external prediction unchanged. Accepted
candidates are recorded in `sobolev_polished_slots.csv`; all candidates receive
metrics and reject reasons in `sobolev_polish_report.csv`.

### Runtime Controls

| Variable | Default | Effect |
|---|---:|---|
| `SOBOLEVRNA_POLISH` | `1` | Set to `0` to disable the gate and preserve raw slots |
| `SOBOLEVRNA_POLISH_SLOTS` | all | Optional comma-separated slot allowlist, e.g. `1,2,3` |
| `SOBOLEVRNA_POLISH_STEPS` | `2000` | Number of polish steps |
| `SOBOLEVRNA_POLISH_LR` | `0.01` | Polish learning rate |

The module also accepts the earlier prototype prefix `SOBOLERNA_*` as a
backward-compatible alias.

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

## Adaptive Contact Calibration

`adaptive_contacts.py` provides the pure-NumPy contact-map utilities used to tune this layer outside the Kaggle notebook. The intended A/B protocol is:

1. Match the predicted contact-density decay curve to polymer scaling, $P(s) \propto s^{-\gamma}$, targeting $\gamma \approx 1.0$ to $1.4$ for folded RNA.
2. Report MCC or F1 separately for short ($6 \leq s \leq 24$), medium ($25 \leq s \leq 100$), and long-range ($s > 100$) contacts when labels are available.
3. Track downstream SHR energy and polish-gate `tm_self`; good thresholds reduce energetic frustration without moving accepted structures out of their raw-model basin.

---

## Sobolev Macromolecule Extension

The Sobolev $H^1$ preconditioner is polymer-agnostic: it smooths high-frequency gradient modes along an indexed chain. `sobolev_macromolecule.py` turns that into a small object-oriented factory. The JAX optimization loop is shared; the factory swaps only the bead identity, physical constants, optional bending term, and expected restraint frontends:

| Domain | Bead | $d_0$ | Compaction | Frontend restraints |
|---|---|---:|---|---|
| RNA | C1′ | 5.95 Å | $R_g = 3.5N^{0.45}$ | RNA-FM, RibonanzaNet-2, templates |
| Proteins | Cα | 3.80 Å | $R_g = R_0N^{0.33}$ | ESM-3, Chai-1, Boltz/Protenix confidence maps |
| dsDNA | C1′ or P | ~4.8 Å | worm-like-chain bending term, not Flory collapse | Boltz/Protenix/AF3 nucleic-acid complex restraints |

For dsDNA, disable the RNA-style collapse basin and add bending plus inter-strand Watson-Crick restraints so the optimizer preserves helix stiffness rather than crushing the duplex.

```python
from sobolev_macromolecule import create_macromolecule, watson_crick_contact_map

protein_engine = create_macromolecule("protein")
protein_terms = protein_engine.energy_terms(ca_coords, contact_map=protein_contacts)

dna_engine = create_macromolecule("dsdna", bend_stiffness=8.0)
watson_crick_contacts = watson_crick_contact_map(n_base_pairs=1000)
polished_dna = dna_engine.polish(p_coords, contact_map=watson_crick_contacts)
```

The presets are deliberately lightweight. A new frontend only needs to emit a square contact/restraint matrix in the same residue order as the coordinates; `SobolevMacromolecule.polish()` handles the shared bond, steric, contact, radius-of-gyration, optional bending, and Sobolev $H^1$ gradient-preconditioned update. For dsDNA, `watson_crick_contact_map()` builds the explicit inter-strand pairing restraints while the bending term preserves helix stiffness.

### Mixed Complexes

`SobolevComplex` extends the same engine to CRISPR-Cas9-like protein/RNA/DNA assemblies. The key change is tensor masking: bonds are evaluated only inside chains, steric radii and bond lengths are looked up from each bead's molecular type, radius-of-gyration basins are computed per chain, and the Sobolev DCT filter is applied independently to each chain gradient slice so unrelated chain endpoints are never smoothed together.

```python
from sobolev_macromolecule import SobolevComplex

sequence, complex_engine = SobolevComplex.from_fasta("""
>Cas9|protein
MKK...
>guide|rna
GGA...
>target|dsdna
ATGC...
""")

terms = complex_engine.energy_terms(coords, contact_map=af3_or_boltz_contacts)
polished = complex_engine.polish(coords, contact_map=af3_or_boltz_contacts)
```

The complex contact Hamiltonian reports `contacts_intra` and `contacts_inter` separately. `ComplexSpec(w_intra=..., w_inter=...)` lets interface restraints carry a different weight from intra-chain folding restraints, which is useful when preserving docking geometry matters more than relaxing internal monomer noise.

### Graph Sobolev Macro Backend

The Kaggle notebook's original SHR cells are too linear and monolithic to swap the 1D DCT for a graph filter in place. The extracted `sobolev_macromolecule.py` layer is the modular boundary: `SobolevMacromolecule` keeps single-chain DCT smoothing, `SobolevComplex` keeps per-chain DCT smoothing for central-dogma complexes, and `SobolevMacro` replaces the DCT with a graph Laplacian spectral filter for branched or non-polymer systems.

```python
from sobolev_macromolecule import SlabPotential, create_macro_graph

glycan = create_macro_graph(
    node_types=["glycan", "glycan", "glycan", "glycan"],
    bonds=[(0, 1, 1.4), (1, 2, 1.4), (1, 3, 1.6)],
)

membrane_patch = create_macro_graph(
    node_types=["lipid_tail", "lipid_head"],
    bonds=[],
    slab=SlabPotential(half_thickness=15.0),
)

filtered_gradient = glycan.smooth_gradient(raw_gradient)
polished = glycan.polish(coords, contact_map=boltz_or_af3_restraints)
```

`SobolevMacro` uses `GraphSpec` tensors for node types, bead radii, arbitrary covalent edges, contact weights, and optional implicit membrane slab potentials. The Sobolev filter is
`U @ ((U.T @ gradient) / (1 + alpha * lambda))`, where `U` and `lambda` come from the graph Laplacian `L = D - A`. This acts like the old DCT on a line, but also handles glycan trees, ligand bond graphs, disconnected systems, lipid patches, and coarse MARTINI-style beads without smoothing across non-edges.

### Whole-Cell Visualization Bridge

Whole-cell scenes should be rendered by an instanced WebGPU/OpenUSD/Unreal frontend, not by loading every bead into a desktop molecular viewer. `sobolev_visualization.py` adds the thin adapter layer: binary coordinate frames for SHM/gRPC/WebSocket transport, renderer-ready `(N, 4, 4)` instance matrices, abundance-to-asset-id expansion, JSON scene manifests, minimal OpenUSD PointInstancer export, and far-field coarse graining around an active-site sphere.

See [`docs/whole-cell-visualization.md`](docs/whole-cell-visualization.md) for the multi-scale data integration and cloud rendering architecture, including the recommended split between A100/H100-class SobolevMacro compute and L4/G5-class Pixel Streaming visualization.

---

## BDBV Visualization — Safety, Scope, and Provenance

This repository includes a **non-infectious, integrative, multi-scale
structural visualization of a Bundibugyo ebolavirus–associated virion
architecture**, assembled from public structural components, homologous
templates, and computationally relaxed placement/packing coordinates.
This is a **hypothesis-generating model**, not an experimentally
validated complete virion reconstruction and not a genome or
infectious-virus reconstruction.

### PDB Accession Audit

The initially assigned PDB IDs have been corrected after review against
RCSB annotations:

- **6N7J**: BDBV223 Fab bound to a 16-residue synthetic GP stalk peptide
  (X-ray, 3.68 Å). This is NOT a full GP trimer — it is a small stalk
  epitope + antibody complex. Set to `None` in the ChimeraX script until
  a full GP trimer structure is properly vetted.
- **4LDB**: Zaire ebolavirus (EBOV) VP40 dimer (X-ray, 2.6 Å). This is
  NOT BDBV VP40 — it is a homologous template (~60% identity). Labeled
  as a homolog in the provenance table.
- **7ZPE**: Human branched-chain keto acid dehydrogenase kinase —
  completely unrelated to ebolavirus. This was a wrong PDB ID. Removed.

### What This Repository Does Not Contain

- No nucleotide or amino acid sequences of any pathogen
- No synthesis-ready construct designs or cloning vectors
- No reverse-genetics systems, minigenomes, replicons, or rescue protocols
- No wet-lab protocols of any kind
- No "one-click" pipeline that turns a sequence into a buildable pathogen

### Appropriate Use

- **Education**: visualizing filovirus architecture for students and trainees
- **Structural interpretation**: communicating antibody-epitope geography
- **Defensive research communication**: illustrating virion-scale organization
- **Method demonstration**: showing the Sobolev/graph-relaxation engine
- **Scene-building toolkit**: the visualization adapters are domain-agnostic

### Documentation

- [SAFETY.md](SAFETY.md) — full safety and scope statement
- [PROVENANCE.md](PROVENANCE.md) — component-level provenance table
  (source, organism, method, resolution, chain IDs, copy number,
  placement rule, uncertainty) with the PDB accession audit
- [docs/pathogen-modeling-provenance-template.md](docs/pathogen-modeling-provenance-template.md) —
  blank provenance template for future pathogen-associated models

### Video

> **⚠️ Caveat:** This is an **integrative non-infectious BDBV-associated
> virion visualization**, not a complete coordinate-exact reconstruction.
> It is a hypothesis-generating model assembled from public structural
> components, homologous templates, and computationally relaxed
> placement/packing coordinates. See [PROVENANCE.md](PROVENANCE.md) for
> the full provenance table and PDB accession audit.

**[📷 BDBV Virion Visualization (35s, 1720×1080)](docs/media/bdbv_virion_visualization.mp4)**

The video shows the 1,081 nm Bundibugyo ebolavirus–associated virion
architecture visualization in UCSF ChimeraX, with instance transforms
from the Sobolev graph relaxation engine placing PDB structures at
JAX-computed coordinates along the virion envelope.

*If the video does not render inline, download it from
[docs/media/bdbv_virion_visualization.mp4](docs/media/bdbv_virion_visualization.mp4)
and open locally.*

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

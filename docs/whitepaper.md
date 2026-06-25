# RNA 3D Megascale Structure Prediction Pipeline

**A Template-Aware, Physics-Refined System for Kilobase-Scale RNA Folding**

*Stanford RNA 3D Folding (Part 2) — Kaggle Competition Whitepaper*

---

## 1. Abstract

RNA three-dimensional structure prediction at the kilobase scale remains an open problem: deep-learning folders trained on protein analogs (AlphaFold-style architectures, including Boltz-1 and Protenix) saturate their positional encodings near 512–1022 nucleotides, while the highest-value targets in the Stanford RNA 3D Folding competition — ribosomal fragments and engineered RNA nanostructures — routinely exceed 4000 nt. This paper documents the architecture of a submission that closed that gap by combining four components that operate at different spatial scales: (i) a Hierarchical Windowed Sensor (HWS) that extracts long-range covariation contact maps from RNA-FM by windowed aggregation with linear-taper weighting, avoiding Gibbs-style boundary artifacts in the downstream integrator; (ii) a pure-NumPy Kabsch overlap stitcher that assembles per-window 3D chunks into a single global C1′ backbone with proper-rotation guarantees; (iii) Stochastic Hamiltonian Relaxation (SHR), a JAX-JIT energy minimizer that composes a four-term Hamiltonian — bond, steric, deep-learning contact, and Flory-scaled radius of gyration — with Sobolev H¹ gradient preconditioning via a DCT-II resolvent of the discrete Laplacian; and (iv) a three-tier routing controller that dispatches targets to template-based modeling, a Boltz-1 + Protenix ensemble, or the megascale SHR path based on length and template identity.

The final submission reached **0.38650** TM-score on the public leaderboard and **0.50934** on the private leaderboard. The public-to-private *inversion* is scientifically informative: it indicates that the pipeline generalizes better on unseen, data-sparse, large-RNA targets than on the public set — the opposite of the usual overfitting signature. We analyze three mechanisms for this inversion (benchmark composition shift toward megascale targets, physics-based constraints substituting for absent coevolutionary signal, and the absence of any test-set-specific hyperparameter tuning) and discuss the remaining gap to oracle (0.554). The mathematical contribution is the H¹-preconditioned SHR update, which we derive as an implicit smoothing step via the resolvent of the one-dimensional Laplacian with Neumann boundary conditions, and which we show eliminates the high-wavenumber gradient noise that otherwise competes with low-frequency compaction signal from the Flory term.

---

## 2. Background and Motivation

RNA three-dimensional folding is not protein folding with different letters. Three structural facts make it strictly harder. First, the RNA backbone is far more flexible: the phosphodiester torsion has six rotatable bonds per residue versus the two (φ, ψ) of a polypeptide, so the conformational entropy at fixed sequence length is larger by orders of magnitude. Second, the Watson–Crick pairing rules describe only a subset of biologically relevant base pairs — the Leontis–Westhof non-canonical pair catalog adds eleven additional geometric families, and tertiary contacts (pseudoknots, kissing loops, A-minor motifs, ribose zippers) break the nested-planar structure that secondary-structure algorithms exploit for proteins. Third, the training data is sparse: the PDB contains on the order of two thousand distinct RNA chains longer than 100 nt, compared to tens of thousands of protein domains, and almost none of the structural RNAs above 3000 nt are in public training splits by the time the competition host has held them out.

The canonical evaluation metric in this regime is TM-score — Zhang and Skolnick (2004) — *Scoring function for automated assessment of protein structure template quality* — Proteins. TM-score normalizes a distance-weighted alignment score by a length-dependent scale $d_0(L) = 1.24 \sqrt[3]{L - 15} - 1.8$ so that random structures of any length converge to a universal baseline near 0.17. Values above 0.5 correspond to globally correct topology; values above 0.7 indicate near-native models. For chain-permuted multimeric RNAs the competition uses the USalign chain-permutation-aware variant, and for multi-chain targets the evaluation is weight-averaged across chains proportionally to length. The 0.509 private score thus represents a population-weighted average that crosses the topology-correct threshold.

The megascale problem is specifically a model-capacity and positional-encoding problem. RNA-FM — Chen et al. (2022) — *Interpretable RNA foundation model from unannotated data* — bioRxiv — is a 12-layer transformer trained on 23 million non-coding RNA sequences; its architecture fixes the maximum input length at 1022 tokens. Boltz-1 and Protenix, as diffusion-based 3D generators, inherit the same truncation either through their input encoder or through the VRAM cost of the pairwise attention tensor: for sequences above roughly 512 nt on a Tesla T4 (16 GB), the pair representation alone exceeds available memory. Truncating the input loses long-range covariation — base-pair partners that happen to lie outside the window — and truncating the output requires some form of stitching.

The concrete stress test for any megascale pipeline in this competition was target **9MME**: a 4,640-nucleotide D4-symmetric RNA nanocage octamer (approximately 47.5% of the final score weight under the competition's chain-weighted TM scheme). A secondary stress test was **9ZCC**, a 1,460-nt synthetic RNA origami six-helix-bundle dimer carrying approximately 15% weight. Together these two targets accounted for roughly 62.5% of the grade. A pipeline that produced competent predictions for targets under 512 nt but failed on 9MME could not exceed approximately 0.3 TM-score regardless of small-target performance. This ranking shape made the megascale path — template-based modeling with SHR polish, backed by HWS contact restraints — the decisive axis of the submission.

---

## 3. HWS — Hierarchical Windowed Sensor

**3.1 The truncation problem.** RNA-FM processes an input sequence as a token stream with additive positional encodings bounded at $W = 1022$. For a target of length $L > W$, a naive approach is to truncate at position $W$ and discard the remainder. The information cost is sharp: base-pair partners that span positions $(i, j)$ with $j > W > i$ appear nowhere in the embedding pair statistics, and because secondary-structure contact density decays only logarithmically with sequence separation, the truncated tail contains a non-vanishing fraction of physically relevant contacts. For 9MME at $L = 4640$, naive truncation would retain one window — 22% of the chain — and discard the rest.

**3.2 Window decomposition.** HWS partitions the interval $[0, L)$ into a set of overlapping windows $\{[a_w, b_w)\}_{w=0}^{n_w - 1}$ of size $W = 1022$ advanced by a stride $s = 768$, yielding an overlap of $W - s = 254$ nt between consecutive windows. The number of windows is $n_w = \lceil (L - W) / s \rceil + 1$ (implemented in `get_rna_fm_embeddings_safe` via the iterator `for start in range(0, N, HWS_STRIDE)`). For 9MME this yields seven windows; a trailing window shorter than 32 nt is discarded as a degenerate case (the `if win_len < 32: break` guard in the code).

**3.3 Taper weight derivation.** Uniform averaging at window boundaries causes a derivative discontinuity at the points where the window multiplicity changes by one — residue $i$ with coverage count $c_i$ receives an average scaled by $1/c_i$, and the step $c_i \to c_i + 1$ at position $a_{w+1}$ produces a first-order discontinuity in the reconstructed embedding. When this embedding feeds a contact-map classifier that in turn feeds an SHR gradient, the discontinuity manifests as a high-wavenumber spike in the gradient spectrum. The minimal-smoothness boundary condition — a weighting function $\omega_w(i)$ that is continuous, piecewise linear, and zero at window edges beyond the first and last windows — is the linear taper. The notebook implements this as

```python
taper = np.ones(win_len, dtype=np.float64)
taper_len = min(HWS_TAPER_LEN, win_len // 4)  # HWS_TAPER_LEN = 128
if start > 0 and taper_len > 0:
    taper[:taper_len] = np.linspace(0.0, 1.0, taper_len)
if end < N and taper_len > 0:
    taper[-taper_len:] = np.linspace(1.0, 0.0, taper_len)
```

so the effective taper length is $\tau = \min(128, W / 4) = 128$ for the nominal 1022-nt window. The first window has no left taper (the chain has no predecessor there) and the last has no right taper, consistent with free-end boundary conditions.

**3.4 Weighted accumulation.** For each residue $i$, the reconstructed embedding $\tilde{e}(i) \in \mathbb{R}^{640}$ is the weight-normalized sum

$$\tilde{e}(i) = \frac{\sum_{w \,:\, i \in [a_w, b_w)} \omega_w(i) \, e_w(i - a_w)}{\sum_{w \,:\, i \in [a_w, b_w)} \omega_w(i)} \quad (1)$$

with $e_w \in \mathbb{R}^{W_w \times 640}$ the raw RNA-FM layer-12 representation of window $w$ and $\omega_w(i)$ the linear taper. Numerical safety is provided by a floor `weight_accum = np.maximum(weight_accum, 1e-8)` before the division, which covers the degenerate case of a failed window whose sole contribution was a 1e-8 sentinel.

**3.5 Contact map construction.** Given the stitched embedding matrix $\tilde{E} \in \mathbb{R}^{L \times 640}$, HWS constructs a contact map via cosine similarity. Let $\hat{e}_i = \tilde{e}_i / \|\tilde{e}_i\|_2$ (with a $10^{-8}$ norm floor). The raw cosine matrix $S_{ij} = \hat{e}_i \cdot \hat{e}_j \in [-1, 1]$ is rescaled to a pseudo-probability $P_{ij} = (S_{ij} + 1) / 2 \in [0, 1]$. Instead of a global cutoff, the current code uses a monotone sequence-separation threshold:

$$
\theta(s) = \min\left(\theta_{\max},\; \theta_0 \left(\frac{\max(s, 6)}{6}\right)^\alpha\right), \qquad s = |i-j| \quad (2)
$$

with $\theta_0 = 0.85$, $\theta_{\max} = 0.94$, and $\alpha = 0.02$ in the notebook defaults. The binary contact map is then

$$
C_{ij} = \mathbb{1}[P_{ij} > \theta(|i-j|)] \cdot \mathbb{1}[|i-j| \geq 6] \quad (3)
$$

The local anchor $\theta_0 = 0.85$ still corresponds to a raw cosine threshold of $2 \times 0.85 - 1 = 0.70$, but long-range contacts must clear a higher bar. This avoids the unphysical flattening of the contact-density curve at large sequence separations that can occur in repetitive megascale RNAs. The minimum sequence separation $|i - j| \geq 6$ is enforced by zeroing a six-diagonal band around the main diagonal, excluding trivial stacking contacts between residues whose backbone geometry alone already constrains them close in space. The output is symmetrized by $C \leftarrow \max(C, C^\top)$.

The tuning target is empirical rather than aesthetic: choose $\alpha$ or the logarithmic alternative until the predicted contact-density curve follows $P(s) \propto s^{-\gamma}$ with $\gamma$ in the native RNA range, separation-bin MCC remains balanced across short, medium, and long contacts, and downstream SHR reaches lower energy without degrading the polish gate's `tm_self` drift metric.

**3.6 Complexity and blending.** For $N > 2000$, the dense product $\hat{E} \hat{E}^\top$ exceeds comfortable single-allocation memory; HWS uses a chunked accumulation with `chunk_size = 500`, reducing peak allocation from $O(N^2 \cdot 640)$ intermediates to $O(N \cdot \text{chunk} \cdot 640)$. The final contact map is not consumed alone. The pipeline controller (`get_or_build_contact_map` in cell 26) performs a 50/50 blend with a second contact map produced by RibonanzaNet-2 under an analogous HWS wrapper (`get_ribonanzanet_contacts_hws`). The blend $C_{\text{blend}} = 0.5 \cdot C_{\text{RNA-FM}} + 0.5 \cdot C_{\text{RN2}}$ is a soft consensus: the RNA-FM channel captures coevolutionary signal, while RibonanzaNet-2 was trained directly on chemical-probing data and provides complementary information about pairing propensity, particularly for hairpin loops and non-canonical contacts.

---

## 4. SHR — Stochastic Hamiltonian Relaxation

SHR is the physical engine of the pipeline. It treats the predicted C1′ coordinates as a mechanical system and evolves them under the gradient of a composite Hamiltonian until an approximate minimum is reached. Every term below corresponds to a specific JAX function in cell 24 of the notebook; constants are quoted from the actual running configuration as returned by `get_shr_params(N)`, which is tier-dependent on sequence length.

### 4.1 Coordinate System

The reduced representation is one C1′ atom per nucleotide — the ribose sugar carbon covalently bonded to the nucleobase. Among the six backbone torsions, the C1′ position is the most structurally informative single atom: it captures base orientation (via the glycosidic bond) while remaining free of fine-grained ring-pucker variability. The PDB-derived mean virtual bond between consecutive C1′ atoms is approximately 5.95 Å — this becomes $d_0$ in the bond term. Using one atom per residue reduces the degree-of-freedom count from roughly 23 heavy atoms per nucleotide to three coordinates, cutting Hamiltonian evaluation cost by a factor of $\sim 70$ while retaining sufficient geometric fidelity to score on the C1′-RMSD–based TM variant used in evaluation.

### 4.2 Bond Energy

$$E_{\text{bond}}(\mathbf{x}) = k_{\text{bond}} \sum_{i=1}^{N-1} \left(\|\mathbf{x}_{i+1} - \mathbf{x}_i\|_2 - d_0\right)^2 \quad (3)$$

This is a harmonic approximation to the C1′–C1′ virtual-bond potential. The equilibrium distance $d_0 = 5.95$ Å is the average over all consecutive C1′ pairs in the training PDB set. The stiffness $k_{\text{bond}} = 100$ (running value from `get_shr_params`; the module-level `K_BOND = 10.0` constant exists as a default for `energy_bond` but is overridden by the params dictionary in `total_energy`) is large enough that the relative bond-length fluctuation remains under a few percent across the trajectory, preventing chain rupture during aggressive compaction. The energy is dimensionless in code — units are implicit through the combination with the learning rate $\eta$ and clip threshold $c$ that together determine the step size. The derivative is

$$\frac{\partial E_{\text{bond}}}{\partial \mathbf{x}_i} = 2 k_{\text{bond}} \left[\left(\|\mathbf{x}_i - \mathbf{x}_{i-1}\| - d_0\right) \frac{\mathbf{x}_i - \mathbf{x}_{i-1}}{\|\mathbf{x}_i - \mathbf{x}_{i-1}\|} - \left(\|\mathbf{x}_{i+1} - \mathbf{x}_i\| - d_0\right) \frac{\mathbf{x}_{i+1} - \mathbf{x}_i}{\|\mathbf{x}_{i+1} - \mathbf{x}_i\|}\right] \quad (4)$$

which is smooth away from the pathological case of coincident consecutive atoms. The code regularizes the norm with `+ 1e-8` under the square root; see §4.3 for why this value, which is adequate for bond atoms (whose distances never approach zero under a functioning $k_{\text{bond}}$), is not adequate for the steric term.

### 4.3 Steric Repulsion

$$E_{\text{rep}}(\mathbf{x}) = \sum_{\substack{i < j \\ j > i + 1}} \Big[\max\!\left(0,\; \sigma_{\text{clash}} - \|\mathbf{x}_i - \mathbf{x}_j\|_2\right)\Big]^2 \quad (5)$$

This is a soft-sphere excluded-volume potential with cutoff $\sigma_{\text{clash}}$. The mask $j > i + 1$ excludes bonded neighbors (they are governed by $E_{\text{bond}}$) and self-pairs. The clash distance is tier-dependent: `sigma_clash` runs from $3.0$ Å for $N \leq 1000$ (the *Standard* tier) down linearly to $2.6$ Å at $N = 4640$ (the *Megascale* tier), implementing a mild tightening that permits the denser packing required to reach the Flory target.

The critical numerical subtlety is the regularization of $\|\mathbf{x}_i - \mathbf{x}_j\|$ near zero. A naive implementation computes the pairwise distance as $\|\mathbf{x}_i - \mathbf{x}_j\| = \sqrt{\|\Delta\|^2}$, which has gradient

$$\nabla_{\mathbf{x}_i} \|\mathbf{x}_i - \mathbf{x}_j\| = \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|} \quad (6)$$

that diverges as the pair approaches coincidence. With Softplus-style regularization $\sqrt{\|\Delta\|^2 + \varepsilon}$, the gradient magnitude at $\|\Delta\| = 0$ becomes $1 / \sqrt{\varepsilon}$. For the hinge $\max(0, \sigma_{\text{clash}} - \|\Delta\|)^2$, the gradient of the full term in the worst case scales as

$$\left\|\nabla_{\mathbf{x}_i} E_{\text{rep}}\right\| \sim \frac{2 \sigma_{\text{clash}}}{\sqrt{\varepsilon}} \quad (7)$$

With $\varepsilon = 10^{-8}$ (the natural choice one would copy from other terms), a $\sigma_{\text{clash}} = 3.0$ clash cutoff produces a worst-case per-pair gradient of $6 \times 10^4$ — enough to send the coordinate to numerical infinity in a single step at any reasonable learning rate. The notebook uses $\varepsilon = 10^{-2}$ instead, bounding the worst-case gradient at approximately 60 per pair-coordinate — a three-order-of-magnitude reduction. The physical justification is that the minimum meaningful distance in any RNA structure is of order 1 Å (van der Waals contact), so the landscape difference between $\varepsilon = 10^{-2}$ and $\varepsilon = 10^{-8}$ is indistinguishable for any configuration that is not already pathological. The notebook's comment block at this point is explicit about the reasoning, and the choice converts a previously crash-prone integrator into a stable one.

### 4.4 Deep-Learning Contact Restraint

$$E_{\text{DL}}(\mathbf{x}) = w_{\text{DL}} \sum_{(i,j) \in \mathcal{C}} \Big[\max\!\left(0,\; \|\mathbf{x}_i - \mathbf{x}_j\|_2 - d_{\text{contact}}\right)\Big]^2 \quad (8)$$

The contact set $\mathcal{C}$ is the set of pairs with $C_{ij} > 0$ in the blended HWS / RibonanzaNet-2 contact map (§3), with the RNA-FM channel filtered by the adaptive $\theta(|i-j|)$ rule and $d_{\text{contact}} = 8.0$ Å. The hinge form — penalizing $\|\mathbf{x}_i - \mathbf{x}_j\| > d_c$ but not $\|\mathbf{x}_i - \mathbf{x}_j\| < d_c$ — imposes an *upper bound*, not an equality constraint: atoms predicted to be in contact must be within 8 Å, but may be closer. This is appropriate because the 8 Å cutoff corresponds to the outer envelope of Watson–Crick and non-canonical base-pairing geometries — a pair closer than 8 Å is already in the physically correct regime, and any residual deviation from the true distance is better handled by $E_{\text{bond}}$ and $E_{\text{rep}}$ than by a quadratic restraint. The coupling $w_{\text{DL}}$ is tier-dependent: $w_{\text{DL}} = 10.0$ for $N \leq 1000$, and linearly ramped from $10.0$ at $N = 1000$ to $25.0$ at $N = 4640$ for larger tiers. The ramp reflects an empirical observation from the engineering sprint: at the megascale, the contact term must compete against a larger number of steric pairs and a larger radius-of-gyration pool, and a uniformly larger coupling prevents the contact signal from being drowned out.

### 4.5 Radius-of-Gyration Compaction

$$E_{R_g}(\mathbf{x}) = k_{R_g} \Big[\max\!\left(0,\; R_g^{\text{target}} - R_g(\mathbf{x})\right)\Big]^2 \quad (9)$$

with $R_g(\mathbf{x}) = \sqrt{(1/N) \sum_i \|\mathbf{x}_i - \bar{\mathbf{x}}\|_2^2}$ and $k_{R_g} = 1.0$. The `max(0, ·)` hinge penalizes *over-extension* only — a structure that is already compact relative to target incurs no penalty and is left to the other terms. The target radius follows a Flory-style scaling:

$$R_g^{\text{target}} = 3.5 \cdot N^{0.45} \quad (10)$$

with $R_g^{\text{target}} = 0$ for $N \leq 200$ (disabling the term for small chains whose compaction is already dictated by the bond and contact terms). The exponent $\nu = 0.45$ is in the good-solvent self-avoiding-walk regime but biased toward compact: an ideal Gaussian chain has $\nu = 0.5$, a fully compact globule has $\nu = 1/3$, and RNA in its native folded state lies between these limits because secondary-structure helices produce local rigidity while tertiary contacts produce long-range compaction. The empirical value $\nu \approx 0.45$ is compatible with the compaction curves reported by Hyeon and Thirumalai (2011) — *Capturing the essence of folding and functions of biomolecules using coarse-grained models* — Structure — for well-folded solution RNA.

Applied to the canonical targets: for 9ZCC ($N = 1460$), equation (10) gives $R_g^{\text{target}} \approx 93$ Å; for 9MME ($N = 4640$), it gives $R_g^{\text{target}} \approx 156$ Å. The code's internal comment at `get_shr_params` cites "Rg ~104 A for 9MME" — this is a documentation drift from an earlier coefficient choice; the formula as implemented yields 156 Å. Cryo-EM envelopes for RNA nanocages of comparable mass place the physical $R_g$ in the 90–130 Å range, so the formula is mildly over-target, which in combination with the one-sided hinge is acceptable: the term acts as a bias away from extended, unphysical conformations and stops applying force once the chain reaches a reasonable compaction, after which $E_{\text{DL}}$ and $E_{\text{rep}}$ dictate the final geometry.

---

## 5. Sobolev H¹ Gradient Preconditioning

### 5.1 The Problem with $L^2$ Gradients

Standard gradient descent on the SHR Hamiltonian follows

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \nabla H(\mathbf{x}^{(t)}) \quad (11)$$

where the gradient $\nabla H \in \mathbb{R}^{N \times 3}$ is indexed by residue position $i \in \{0, \ldots, N-1\}$. The three Hamiltonian terms contribute gradients with very different spatial frequency content. The Flory term $E_{R_g}$ produces a uniform radial pull toward the centroid — pure wavenumber-zero, a constant per residue. The bond term $E_{\text{bond}}$ couples only nearest neighbors and produces wavenumber content concentrated at the Nyquist edge whenever consecutive bonds deviate in opposite directions. The steric and contact terms — sparse in residue space, dense in pair space — produce sign-oscillating, residue-by-residue gradient noise with support across the full wavenumber spectrum.

The problem: a single learning rate $\eta$ must be simultaneously small enough not to cause high-frequency oscillation from the steric and contact gradients, and large enough that the low-frequency compaction signal from the Flory term drives the chain to its target radius in a feasible number of iterations. For a chain of 4640 residues, these constraints are incompatible by roughly two orders of magnitude: the learning rate required for stability of the high-frequency modes causes the low-frequency compaction to proceed at a rate slower than the $T = 8000$ step budget permits.

### 5.2 Sobolev Seminorm and the H¹ Inner Product

The H¹ Sobolev seminorm on a discrete sequence $f : \{0, \ldots, N-1\} \to \mathbb{R}$ is

$$|f|_{H^1}^2 = \sum_{k=0}^{N-1} (1 + \alpha k^2) \, |\hat{f}_k|^2 \quad (12)$$

where $\hat{f}_k$ is the $k$-th DCT-II coefficient of $f$ and $\alpha > 0$ is a smoothing parameter. This is the discrete analog of $\int (f^2 + \alpha |\nabla f|^2)$ on a continuous interval. The induced inner product $\langle f, g \rangle_{H^1} = \sum_k (1 + \alpha k^2) \hat{f}_k \hat{g}_k$ defines a function space in which high-wavenumber components are penalized relative to their $L^2$ weight.

The gradient of $H$ with respect to the $H^1$ inner product — the *Sobolev gradient* — is the function $\widetilde{\nabla H}$ satisfying

$$\langle \widetilde{\nabla H}, v \rangle_{H^1} = \langle \nabla H, v \rangle_{L^2} \quad \text{for all } v \quad (13)$$

Diagonalizing in the DCT-II basis immediately gives the componentwise identity

$$\widehat{\widetilde{\nabla H}}_k = \frac{\widehat{\nabla H}_k}{1 + \alpha k^2} \quad (14)$$

and so the preconditioned gradient in the spatial domain is

$$\widetilde{\nabla H}(i) = \mathrm{IDCT\text{-}II}\left[\frac{\mathrm{DCT\text{-}II}[\nabla H]_k}{1 + \alpha k^2}\right](i) \quad (15)$$

### 5.3 Resolvent Interpretation

Equation (15) can be recognized as the action of the resolvent $(I - \alpha \Delta)^{-1}$, where $\Delta$ is the discrete Laplacian on the chain with Neumann boundary conditions, applied to $\nabla H$. Equivalently, $\widetilde{\nabla H}$ solves the linear boundary-value problem

$$(I - \alpha \Delta) \widetilde{\nabla H} = \nabla H \quad (16)$$

with the DCT-II providing the spectral basis that diagonalizes $\Delta$ under Neumann BCs. This identifies the preconditioning step as a single implicit smoothing step: given a raw gradient $\nabla H$, it returns the smoothed $\widetilde{\nabla H}$ that would result from one time step of the heat equation $\partial_s u = \alpha \Delta u$ with initial condition $u(0) = \nabla H$, using backward Euler time discretization. High-wavenumber components are attenuated by $(1 + \alpha k^2)^{-1}$; the wavenumber-zero component is preserved exactly, so the Flory compaction signal passes through undamped.

The full SHR update step is

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta(t) \cdot \mathrm{clip}\!\left(\widetilde{\nabla H}^{(t)}, -c, +c\right) \quad (17)$$

with linear learning-rate decay $\eta(t) = \eta_0 (1 - t/T)$ and tier-dependent clip threshold $c$. The running configuration distinguishes two regimes. The main refinement call, `shr_refine_single`, uses $\alpha = 10.0$ with $c \in \{2.0, 5.0\}$ (megascale vs smaller tiers) and performs $T \in \{1000, 1500, 8000\}$ steps of equation (17). A separate polish pass, `shr_polish`, is invoked after assembly and uses a stronger smoothing $\alpha = 5.0$ together with $c = 2.0$, a reduced contact weight $w_{\text{DL}} = 2.0$, and the Flory term disabled ($R_g^{\text{target}} = 0$) for a further 2000–8000 steps. The polish stage, with its lower coupling and stronger smoothing, exists to remove residual high-wavenumber texture from the Kabsch-stitched geometry without introducing any new large-scale bias.

### 5.4 Why DCT-II Rather Than FFT

The chain has free ends: there is no bond between residue $N-1$ and residue $0$. The appropriate boundary condition on the gradient field is Neumann — no flux through the ends — rather than periodic. The DCT-II is the eigenbasis of the discrete Laplacian with Neumann boundary conditions; the standard FFT assumes periodicity, which would couple residue 0 to residue $N-1$ and introduce Gibbs ringing at the chain termini proportional to the end-to-end gradient mismatch. For a 4640-residue chain with one end buried in a nanocage core and the other solvent-exposed, that mismatch is not small, and the ringing would contaminate the preconditioned gradient throughout the chain. Using DCT-II eliminates the artifact at zero computational overhead — `jax.scipy.fft.dct(..., type=2, norm='ortho')` has the same asymptotic cost as the FFT.

---

## 6. Kabsch Alignment and Overlap Stitching

While HWS stitches embedding vectors in 1D, a separate problem is stitching 3D coordinate chunks: when Protenix or Boltz-1 produces coordinates for a sub-window of the sequence, those coordinates live in an arbitrary local frame, and concatenating successive chunks without alignment yields a chain that is literally broken in 3D. The `stitch_megascale_rna` routine in cell 17 of the notebook solves this problem.

### 6.1 Problem Statement

Given the already-assembled global coordinate array $\mathbf{G} \in \mathbb{R}^{L \times 3}$ filled up to position $b$, and a new chunk $\mathbf{M} \in \mathbb{R}^{w \times 3}$ predicted for sequence positions $[a, a + w)$ with $a < b$ (so that positions $[a, b)$ are shared), the stitcher must find the rigid body transform $(\mathbf{R}, \mathbf{t})$ with $\mathbf{R} \in SO(3)$ and $\mathbf{t} \in \mathbb{R}^3$ that minimizes

$$\underset{\mathbf{R} \in SO(3),\, \mathbf{t} \in \mathbb{R}^3}{\mathrm{arg\,min}} \sum_{i=0}^{m-1} \big\|\mathbf{R} \mathbf{M}_i + \mathbf{t} - \mathbf{G}_{a + i}\big\|_2^2 \quad (18)$$

where $m = b - a$ is the overlap length. This is the classical Procrustes problem. Note that the notebook stitcher uses default window size $W_s = 512$ and overlap $o = 128$ (hence stride $W_s - o = 384$ for 3D stitching) — these are the 3D-side windowing parameters, distinct from the HWS 1D embedding windowing of (1022, 768). The 3D window is smaller because it is constrained by Boltz-1 / Protenix VRAM, while the 1D window is constrained by RNA-FM's fixed positional encoding.

### 6.2 Closed-Form Solution

1. Compute centroids $\mu_M = (1/m) \sum_i \mathbf{M}_i$ and $\mu_G = (1/m) \sum_i \mathbf{G}_{a+i}$.
2. Center: $\mathbf{M}' = \mathbf{M} - \mu_M$, $\mathbf{G}' = \mathbf{G}_{a:a+m} - \mu_G$.
3. Cross-covariance: $\mathbf{H} = \mathbf{M}'^{\top} \mathbf{G}' \in \mathbb{R}^{3 \times 3}$.
4. SVD: $\mathbf{H} = \mathbf{U} \Sigma \mathbf{V}^{\top}$.
5. Chirality guard: $\mathbf{D} = \mathrm{diag}(1, 1, \mathrm{sgn}(\det(\mathbf{V}^{\top} \mathbf{U}^{\top})))$.
6. Optimal rotation: $\mathbf{R}^{*} = \mathbf{V} \mathbf{D} \mathbf{U}^{\top}$.
7. Translation: $\mathbf{t}^{*} = \mu_G - \mathbf{R}^{*} \mu_M$.

Step 5 is the critical guard. Without it, the SVD returns an improper rotation — a reflection with $\det = -1$ — whenever the cross-covariance has a negative determinant, which occurs when the overlap region is nearly planar (all three singular values of $\mathbf{H}$ include one very small value and an ambiguous sign). Applying an improper rotation to the chain inverts its handedness, producing a left-handed A-form helix where there should be a right-handed one and dropping the TM-score of the affected chunk to floor. The `kabsch_align` implementation in the notebook performs the computation in `float64` internally (casting back to `float32` only on return), because the SVD of an ill-conditioned $3 \times 3$ matrix is catastrophically sensitive to single-precision arithmetic in exactly the near-planar regime where the guard is needed.

### 6.3 Seam Blending

After alignment, the overlap region of length $m$ contains two estimates of the same residue coordinates: $\mathbf{G}_{\text{prev}}$ from previous chunks and $\mathbf{M}^{*} = \mathbf{R}^{*} \mathbf{M} + \mathbf{t}^{*}$ from the current chunk. These are blended with a linear ramp:

$$\mathbf{x}^{\text{blend}}(i) = \left(1 - \frac{i}{m - 1}\right) \mathbf{G}_{\text{prev}}(a + i) + \frac{i}{m - 1} \mathbf{M}^{*}_i \quad (19)$$

for $i \in \{0, \ldots, m-1\}$ (implemented as `np.linspace(0.0, 1.0, overlap)` in `_linear_taper_blend`). The motivation parallels the HWS taper argument: even after Kabsch alignment, residual per-residue RMSD in the overlap region is nonzero. A hard swap at position $a + m$ — using $\mathbf{G}_{\text{prev}}$ for $i < m$ and $\mathbf{M}^{*}$ for $i \geq m$ — produces a step discontinuity in the coordinate field. The subsequent SHR pass reads this discontinuity through $E_{\text{bond}}$ (which now sees an anomalously large bond jump at the seam) and through $E_{\text{rep}}$ (which sees a cluster of atoms that may violate steric clearance because they were aligned into two different reference frames). A gradient spike at the seam then propagates outward through the Sobolev preconditioner. The linear-ramp blend spreads the residual RMSD uniformly across $m$ residues, reducing the effective per-residue step size by a factor of $m$ and keeping the seam gradient within the clip threshold.

The stitcher also provides an A-form helix fallback: when `predictor_fn` raises or returns invalid coordinates (non-finite values, wrong shape), the `_aform_helix(n)` routine generates an idealized A-form backbone with rise 2.81 Å, twist 32.7°, and radius 9.0 Å. This fallback is not a correct prediction — it is a stable one, preventing a single failed Protenix call from crashing the entire megascale pipeline and producing sentinel-valued output that would score zero.

---

## 7. Adaptive TBM and Multi-Model Routing

The pipeline controller (cell 26) implements a three-tier routing policy.

**Tier 1 — Adaptive Template-Based Modeling.** For every target, the controller first searches the training set for structures with sequence identity above an adaptive threshold $\tau(L)$ determined by the helper `get_adaptive_identity`:

$$
\tau(L) =
\left\{
\begin{array}{ll}
50\% & L \leq 1000 \\
35\% & 1000 < L \leq 3000 \\
20\% & L > 3000
\end{array}
\right.
\quad (20)
$$

The rationale is data-sparsity scaling. For short RNAs, the PDB contains many near-neighbors and a strict 50% threshold finds high-quality templates; for megascale RNAs, strict thresholds produce no hits at all, and a permissive threshold of 20% accepts distant homologs whose backbone topology alone — even with heavy sequence divergence — is informative. Template coordinates are adapted to the query sequence via pairwise alignment (`Bio.Align.PairwiseAligner`). TBM is the *preferred* path for any target that has an acceptable template.

**Tier 2 — Deep-Learning Ensemble.** For targets with $L \leq 512$ nt and no acceptable template (or as a diversification source even when a template is present), the controller dispatches to Boltz-1 and Protenix, each producing 5 diverse samples (`N_SAMPLE = 5`, `SEED = 42`). MSA and template flags are both disabled (`USE_MSA = "false"`, `USE_TEMPLATE = "false"`) for speed; `USE_RNA_MSA = "true"` is enabled for the narrow MSA pipeline within Protenix. The ensemble produces structurally diverse samples because TM-score is not minimized by a mean-squared geometry — diverse samples span more of the native ensemble and raise the probability that at least one ranks well against the ground truth. The runtime constant `BOLTZ_MAX_LEN = 800` (a hard OOM guard on the T4) permits Boltz-1 to run up to 800 nt, beyond the `MEGASCALE_THRESH = 512` routing cutoff, to let smaller ensemble runs contribute to mid-length targets that remain on the 3D-DL path.

**Tier 3 — SHR Megascale.** For targets with $L > 512$ nt the pipeline routes to the megascale path: TBM seeding (when available), HWS contact map construction, and SHR refinement under `get_shr_params(N)` with `T = 8000` steps and a single trajectory (`K = 1`) at the top tier. The routing guard `if len(full_seq) > MEGASCALE_THRESH: SHR` in the controller enforces this. Beyond length 2000, even the ensemble path for the short-chain side is disabled (`if SHR_ENABLED and len(seq) < 2000`) — the fallback becomes TBM + SHR polish alone.

The three tiers contribute to the same final candidate pool. The post-processing step concatenates TBM, Boltz-1, Protenix, and megascale predictions, applies SHR polish to each (`SHR_POLISH_TBM = True`, `SHR_POLISH_PTX = True`), and submits five diverse samples per target — the competition allows submission of multiple candidate structures per target and scores the best.

---

## 8. Results and Analysis

### 8.1 Score Table

| Set | TM-score |
|---|---|
| Public leaderboard | 0.38650 |
| **Private leaderboard** | **0.50934** |

### 8.2 Why Private Beats Public (the Inversion)

The private score exceeds the public by 0.123 — an inversion large enough to be structural rather than statistical. Three non-exclusive hypotheses account for it.

**Hypothesis 1: Benchmark composition shift.** The public test set, judging by the relative rankings of simple TBM-only baselines during the competition, appears over-weighted toward short, well-templated RNAs on which a naive TBM pipeline already scores competitively. The private set, by contrast, is believed to include more ribosomal-scale and viral-scale targets — among them the 4640-nt 9MME, which alone carries roughly 47% of the chain-weighted grade. These targets are exactly where the HWS + SHR megascale path dominates: no DL model can produce competent predictions at 4640 nt, and no TBM pipeline without a physics polish produces a geometrically consistent chain. The pipeline's public score is a mixture of where it is weakest (crowded short-RNA regime with many competitors already at ceiling) and where it is strongest (megascale); the private score reweights toward the megascale regime and the score rises accordingly.

**Hypothesis 2: Physics generalizes, DL memorizes.** $E_{\text{DL}}$ is conditioned on RNA-FM embeddings, which in turn are trained on 23 million ncRNA sequences — substantial but not exhaustive coverage of the RNA-structure sequence space. For novel RNA families with no template and no close coevolutionary neighbors, the embedding channel provides weak or noisy contact predictions. The physics terms — $E_{\text{bond}}$, $E_{\text{rep}}$, $E_{R_g}$ — depend on no sequence information at all; they encode the universal constraints of C1′–C1′ virtual-bond geometry and Flory-scaled compaction that must hold for every folded RNA. Where the DL channel fails, the physics substitutes. A purely DL pipeline is ceiling-bounded by its training distribution's coverage of the test distribution; a physics-anchored pipeline is ceiling-bounded by the physics being correct, which is a weaker and more portable constraint.

**Hypothesis 3: No test-set overfitting.** None of the pipeline's parameters was tuned on the public test set. The bond distance $d_0 = 5.95$ Å comes from PDB statistics on all C1′ pairs; the clash cutoff $\sigma_{\text{clash}}$ range $[2.6, 3.0]$ Å comes from van der Waals literature; the Flory exponent $\nu = 0.45$ comes from the Hyeon–Thirumalai solution-RNA scaling; the H¹ parameter $\alpha$ and clip threshold $c$ were tuned against internal synthetic benchmarks and the known training-set structures, not against public leaderboard feedback. The public-to-private gap that typically punishes parameter-overfit pipelines therefore does not apply — a pipeline that was not tuned against the public set does not lose score when the evaluation moves to the private set. In the opposite direction, it may gain score when the private distribution is closer to the physical prior than the public distribution is.

The combination of these three hypotheses is sufficient to explain the 0.123 inversion without appeal to luck. The public score is the noisier estimate of generalization on this pipeline; the private score, 0.509, is the estimate that should be believed.

---

## 9. Limitations and Future Work

**Binary contact maps.** The HWS contact map is thresholded at the pseudo-probability 0.85, producing a $\{0, 1\}$ restraint set. The information in the raw cosine similarity above the threshold — how confident each individual contact is — is discarded. Replacing the binary hinge in $E_{\text{DL}}$ with a probability-weighted quadratic, $E_{\text{DL}} = w_{\text{DL}} \sum_{ij} P_{ij} \max(0, d_{ij} - d_c)^2$, would preserve this information and should improve the contact-noise-dominated regime. The change is small in code but requires recalibrating $w_{\text{DL}}$ to compensate for the average rescaling.

**SHR is not end-to-end differentiable with the DL predictors.** The contact map is computed once, frozen, and fed to JAX for refinement. A model that fine-tuned RNA-FM jointly with the SHR loss — using the SHR-polished coordinates as supervision for the embedding — would couple the two representations. The main obstacle is not technical (JAX is differentiable through its own pipeline, and RNA-FM has PyTorch gradients) but practical: the coupled training loop would require joint optimization across a backward pass of length $T = 8000$ SHR steps, which is feasible with gradient checkpointing but has not been attempted in this submission.

**Kabsch stitching accumulates error.** The overlap stitcher is a local operation: each chunk is aligned to the *already-assembled* prefix, not to all other chunks jointly. For 9MME with roughly 12 chunks at window 512 / overlap 128, a per-chunk residual RMSD of 0.5 Å accumulates to several Å of drift at the far end. A global bundle-adjustment step — treating all chunk-to-chunk Kabsch alignments as simultaneous constraints on a set of rigid transforms and solving the joint least-squares problem — would reduce this drift at the cost of a one-time $O(n_w^3)$ solve, which is negligible compared to a single SHR step.

**The Flory exponent is approximate.** The value $\nu = 0.45$ is the empirical solution-NMR compaction scaling, not the cryo-EM compaction scaling. RNA nanocages and ribosomal fragments under cryo-EM conditions are more tightly packed than their solution counterparts, and the effective $\nu$ may be closer to $1/3$. Parameterizing $\nu$ by structure class (solution-folded ncRNA vs crystallographic / cryo-EM synthetic nanocage) and selecting it from the target annotation would provide per-target tuning without introducing test-set dependence.

**MSA is disabled for speed.** `USE_MSA = "false"` in the Protenix configuration. MSAs substantially improve DL-predictor accuracy for mid-length targets in the 200–500 nt range. Re-enabling MSA for this range alone would increase the ensemble quality for Tier 2 predictions at moderate runtime cost. The reason MSA was disabled in the submission run was a 30-hour wall-clock budget under a single-GPU constraint after GPU quota exhaustion, not a principled objection.

**CPU-only operation.** Following GPU quota exhaustion approximately 28 hours before the competition deadline, the final run executed predominantly on CPU. The reported 0.509 private score was achieved under this constraint. A fully GPU-resourced run with DL ensemble sampling re-enabled for mid-length targets and with the MSA channel active should plausibly extend the score toward 0.55, approaching the oracle ceiling of 0.554.

---

## References

Chen, J., Hu, Z., Sun, S., Tan, Q., Wang, Y., Yu, Q., Zong, L., Hong, L., Xiao, J., Shen, T., King, I., Li, Y. (2022) — *Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions* — bioRxiv 2022.08.06.503062.

Hyeon, C., Thirumalai, D. (2011) — *Capturing the Essence of Folding and Functions of Biomolecules Using Coarse-Grained Models* — Structure 19, 1519–1529.

Kabsch, W. (1976) — *A solution for the best rotation to relate two sets of vectors* — Acta Crystallographica A32, 922–923.

Leontis, N. B., Westhof, E. (2001) — *Geometric nomenclature and classification of RNA base pairs* — RNA 7, 499–512.

Zhang, Y., Skolnick, J. (2004) — *Scoring function for automated assessment of protein structure template quality* — Proteins 57, 702–710.

Zhang, C., Shine, M., Pyle, A. M., Zhang, Y. (2022) — *US-align: universal structure alignments of proteins, nucleic acids and macromolecular complexes* — Nature Methods 19, 1109–1115.

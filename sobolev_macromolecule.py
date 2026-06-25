"""Universal Sobolev H1 macromolecule polish engine.

The original SobolevRNA notebook is RNA-first: C1' beads, RNA-FM/RibonanzaNet
restraints, and RNA-specific physical constants.  This module factors out the
parts that are polymer-generic so the same Sobolev preconditioned optimizer can
be configured for RNA, proteins, and double-stranded DNA.

JAX is imported lazily by ``SobolevMacromolecule.polish``.  The factory and
NumPy energy diagnostics remain usable in lightweight environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

RgMode = Literal["none", "upper_bound", "lower_bound"]


class JaxUnavailableError(RuntimeError):
    """Raised when polish is requested without a working JAX installation."""


@dataclass(frozen=True)
class RestraintFrontend:
    """Describes the external model family expected to supply restraints."""

    name: str
    role: str
    output: str


@dataclass(frozen=True)
class MacromoleculeConfig:
    """Physical constants for a one-bead-per-residue polymer Hamiltonian."""

    name: str
    bead_name: str
    ideal_bond_distance: float
    bond_stiffness: float
    sigma_clash: float
    contact_distance: float
    contact_weight: float
    rg_prefactor: float
    rg_exponent: float
    rg_mode: RgMode
    rg_min_residues: int
    bend_stiffness: float
    bend_target_cos: float
    sobolev_alpha: float
    gradient_clip: float
    default_lr: float
    default_steps: int
    frontends: tuple[RestraintFrontend, ...]

    def with_overrides(self, **kwargs) -> "MacromoleculeConfig":
        """Return a copy with selected constants replaced."""

        return replace(self, **kwargs)

    def expected_rg(self, n_residues: int) -> float:
        if self.rg_mode == "none" or n_residues < self.rg_min_residues:
            return 0.0
        return float(self.rg_prefactor * (int(n_residues) ** self.rg_exponent))


@dataclass(frozen=True)
class ChainSpec:
    """Residue slice and molecular type for one chain in a complex."""

    chain_id: str
    kind: str
    start: int
    end: int
    compaction_enabled: bool | None = None
    sobolev_alpha: float | None = None

    @property
    def length(self) -> int:
        return int(self.end) - int(self.start)

    @property
    def config(self) -> MacromoleculeConfig:
        return get_macromolecule_config(self.kind)

    def uses_compaction(self) -> bool:
        if self.compaction_enabled is not None:
            return bool(self.compaction_enabled)
        return self.config.rg_mode != "none"


@dataclass(frozen=True)
class ComplexSpec:
    """Chain-aware topology and contact weights for a heterogeneous complex."""

    chains: tuple[ChainSpec, ...]
    w_intra: float = 2.0
    w_inter: float = 3.0

    @property
    def n_residues(self) -> int:
        return max((chain.end for chain in self.chains), default=0)

    def validate(self, n_residues: int | None = None) -> None:
        if not self.chains:
            raise ValueError("ComplexSpec requires at least one chain")
        expected_start = 0
        for chain in self.chains:
            if chain.start != expected_start:
                raise ValueError(
                    f"chain {chain.chain_id!r} starts at {chain.start}, "
                    f"expected {expected_start}"
                )
            if chain.end <= chain.start:
                raise ValueError(f"chain {chain.chain_id!r} has non-positive length")
            get_macromolecule_config(chain.kind)
            expected_start = chain.end
        if n_residues is not None and expected_start != int(n_residues):
            raise ValueError(
                f"complex covers {expected_start} residues, expected {n_residues}"
            )

    def chain_index_vector(self) -> np.ndarray:
        self.validate()
        out = np.empty(self.n_residues, dtype=np.int32)
        for idx, chain in enumerate(self.chains):
            out[chain.start : chain.end] = idx
        return out


RNA_FRONTENDS = (
    RestraintFrontend("RNA-FM", "sequence embedding contacts", "contact_map"),
    RestraintFrontend("RibonanzaNet-2", "pairwise RNA structural signal", "contact_map"),
    RestraintFrontend("templates", "template-derived C1' contacts", "contact_map"),
)

PROTEIN_FRONTENDS = (
    RestraintFrontend("ESM-3", "single-sequence protein embeddings", "contact_map"),
    RestraintFrontend("Chai-1", "protein confidence and pair restraints", "contact_map"),
    RestraintFrontend("Boltz/Protenix", "candidate coordinates and confidence", "contact_map"),
)

DSDNA_FRONTENDS = (
    RestraintFrontend("Boltz/Protenix/AF3", "nucleic-acid complex coordinates", "contact_map"),
    RestraintFrontend("Watson-Crick pairing", "inter-strand base-pair restraints", "contact_map"),
)


PRESETS: dict[str, MacromoleculeConfig] = {
    "rna": MacromoleculeConfig(
        name="rna",
        bead_name="C1'",
        ideal_bond_distance=5.95,
        bond_stiffness=100.0,
        sigma_clash=3.0,
        contact_distance=8.0,
        contact_weight=2.0,
        rg_prefactor=3.5,
        rg_exponent=0.45,
        rg_mode="upper_bound",
        rg_min_residues=200,
        bend_stiffness=0.0,
        bend_target_cos=1.0,
        sobolev_alpha=5.0,
        gradient_clip=2.0,
        default_lr=0.01,
        default_steps=2000,
        frontends=RNA_FRONTENDS,
    ),
    "protein": MacromoleculeConfig(
        name="protein",
        bead_name="C_alpha",
        ideal_bond_distance=3.80,
        bond_stiffness=100.0,
        sigma_clash=4.2,
        contact_distance=8.0,
        contact_weight=2.0,
        rg_prefactor=3.25,
        rg_exponent=0.33,
        rg_mode="upper_bound",
        rg_min_residues=30,
        bend_stiffness=0.0,
        bend_target_cos=1.0,
        sobolev_alpha=5.0,
        gradient_clip=2.0,
        default_lr=0.01,
        default_steps=2000,
        frontends=PROTEIN_FRONTENDS,
    ),
    "dsdna": MacromoleculeConfig(
        name="dsdna",
        bead_name="P",
        ideal_bond_distance=4.8,
        bond_stiffness=100.0,
        sigma_clash=4.0,
        contact_distance=10.5,
        contact_weight=2.0,
        rg_prefactor=0.0,
        rg_exponent=0.0,
        rg_mode="none",
        rg_min_residues=0,
        bend_stiffness=5.0,
        bend_target_cos=1.0,
        sobolev_alpha=5.0,
        gradient_clip=1.5,
        default_lr=0.005,
        default_steps=2000,
        frontends=DSDNA_FRONTENDS,
    ),
}

ALIASES = {
    "rna": "rna",
    "nucleic_acid": "rna",
    "protein": "protein",
    "peptide": "protein",
    "ca": "protein",
    "c_alpha": "protein",
    "dna": "dsdna",
    "dsdna": "dsdna",
    "double_stranded_dna": "dsdna",
}


def get_macromolecule_config(kind: str, **overrides) -> MacromoleculeConfig:
    """Return a preset config, optionally overriding selected constants."""

    key = ALIASES.get(kind.strip().lower())
    if key is None:
        known = ", ".join(sorted(ALIASES))
        raise ValueError(f"unknown macromolecule kind {kind!r}; expected one of: {known}")
    config = PRESETS[key]
    return config.with_overrides(**overrides) if overrides else config


def complex_from_chain_lengths(
    chain_lengths: Sequence[tuple[str, str, int]],
    w_intra: float = 2.0,
    w_inter: float = 3.0,
) -> ComplexSpec:
    """Build a contiguous ComplexSpec from ``(chain_id, kind, length)`` rows."""

    chains: list[ChainSpec] = []
    start = 0
    for chain_id, kind, length in chain_lengths:
        n = int(length)
        if n <= 0:
            raise ValueError(f"chain {chain_id!r} length must be positive")
        chains.append(ChainSpec(str(chain_id), str(kind), start, start + n))
        start += n
    spec = ComplexSpec(tuple(chains), w_intra=float(w_intra), w_inter=float(w_inter))
    spec.validate(start)
    return spec


def parse_complex_fasta(text: str) -> tuple[str, ComplexSpec]:
    """Parse a multi-entry FASTA into a concatenated sequence and ComplexSpec.

    Headers may use ``>chain_id|kind`` or include ``kind=protein`` style tokens.
    Example: ``>Cas9|protein`` followed by a protein sequence, then
    ``>guide|rna`` and ``>target|dsdna``.
    """

    records: list[tuple[str, str, str]] = []
    current_id: str | None = None
    current_kind: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal chunks, current_id, current_kind
        if current_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"chain {current_id!r} has no sequence")
        records.append((current_id, current_kind or "rna", sequence))
        chunks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            pieces = header.replace("|", " ").replace(";", " ").split()
            if not pieces:
                raise ValueError("empty FASTA header")
            current_id = pieces[0]
            current_kind = None
            for piece in pieces[1:]:
                if piece.startswith("kind=") or piece.startswith("type="):
                    current_kind = piece.split("=", 1)[1]
                elif ALIASES.get(piece.lower()) is not None:
                    current_kind = piece
            chunks = []
        else:
            if current_id is None:
                raise ValueError("FASTA sequence found before first header")
            chunks.append(line)
    flush()

    if not records:
        raise ValueError("no FASTA records found")

    chain_lengths = [(chain_id, kind, len(sequence)) for chain_id, kind, sequence in records]
    sequence = "".join(sequence for _chain_id, _kind, sequence in records)
    return sequence, complex_from_chain_lengths(chain_lengths)


def _as_contact_map(contact_map, n_residues: int) -> np.ndarray:
    if contact_map is None:
        return np.zeros((n_residues, n_residues), dtype=np.float64)
    cmap = np.asarray(contact_map, dtype=np.float64)
    if cmap.shape != (n_residues, n_residues):
        raise ValueError(
            f"contact_map has shape {cmap.shape}, expected {(n_residues, n_residues)}"
        )
    return cmap


def _complex_vectors(spec: ComplexSpec) -> dict[str, np.ndarray]:
    spec.validate()
    n = spec.n_residues
    chain_idx = spec.chain_index_vector()
    ideal = np.empty(n, dtype=np.float64)
    bond_k = np.empty(n, dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)
    contact_distance = np.empty(n, dtype=np.float64)
    for chain in spec.chains:
        cfg = chain.config
        ideal[chain.start : chain.end] = cfg.ideal_bond_distance
        bond_k[chain.start : chain.end] = cfg.bond_stiffness
        sigma[chain.start : chain.end] = cfg.sigma_clash
        contact_distance[chain.start : chain.end] = cfg.contact_distance

    same_chain = chain_idx[:, None] == chain_idx[None, :]
    pair_sigma = 0.5 * (sigma[:, None] + sigma[None, :])
    pair_contact_distance = 0.5 * (
        contact_distance[:, None] + contact_distance[None, :]
    )
    contact_weight = np.where(same_chain, spec.w_intra, spec.w_inter).astype(np.float64)
    return {
        "chain_idx": chain_idx,
        "ideal": ideal,
        "bond_k": bond_k,
        "sigma": sigma,
        "contact_distance": contact_distance,
        "same_chain": same_chain,
        "pair_sigma": pair_sigma,
        "pair_contact_distance": pair_contact_distance,
        "contact_weight": contact_weight,
    }


def _complex_neighbor_mask(spec: ComplexSpec) -> np.ndarray:
    n = spec.n_residues
    idx = np.arange(n)
    upper = idx[None, :] > idx[:, None]
    same_chain = spec.chain_index_vector()[:, None] == spec.chain_index_vector()[None, :]
    adjacent_same_chain = same_chain & (np.abs(idx[:, None] - idx[None, :]) <= 1)
    return upper & ~adjacent_same_chain


def radius_of_gyration(coords: np.ndarray) -> float:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        raise ValueError("coords must have shape (N, 3)")
    centered = arr - arr.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def energy_terms_numpy(
    coords: np.ndarray,
    config: MacromoleculeConfig,
    contact_map: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate Hamiltonian terms with NumPy for diagnostics and tests."""

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    n = len(arr)
    cmap = _as_contact_map(contact_map, n)

    if n > 1:
        diffs = arr[1:] - arr[:-1]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1) + 1e-8)
        bond = config.bond_stiffness * float(
            np.sum((dists - config.ideal_bond_distance) ** 2)
        )
    else:
        bond = 0.0

    if n > 2:
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-2)
        idx = np.arange(n)
        mask = idx[None, :] > idx[:, None] + 1
        violations = np.maximum(config.sigma_clash - dist, 0.0)
        steric = float(np.sum((violations * mask) ** 2))
    else:
        steric = 0.0

    if cmap.any():
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
        violations = np.maximum(dist - config.contact_distance, 0.0)
        contacts = config.contact_weight * float(np.sum(cmap * (violations**2)))
    else:
        contacts = 0.0

    rg_energy = 0.0
    if config.rg_mode != "none":
        target = config.expected_rg(n)
        if target > 0.0:
            rg = radius_of_gyration(arr)
            if config.rg_mode == "upper_bound":
                rg_energy = max(rg - target, 0.0) ** 2
            elif config.rg_mode == "lower_bound":
                rg_energy = max(target - rg, 0.0) ** 2
            else:
                raise ValueError(f"unknown rg_mode {config.rg_mode!r}")

    bend = 0.0
    if config.bend_stiffness > 0.0 and n > 2:
        diffs = arr[1:] - arr[:-1]
        lengths = np.sqrt(np.sum(diffs * diffs, axis=1) + 1e-8)
        unit = diffs / lengths[:, None]
        cosines = np.sum(unit[1:] * unit[:-1], axis=1)
        bend = config.bend_stiffness * float(
            np.sum((cosines - config.bend_target_cos) ** 2)
        )

    total = bond + steric + contacts + rg_energy + bend
    return {
        "bond": float(bond),
        "steric": float(steric),
        "contacts": float(contacts),
        "rg": float(rg_energy),
        "bend": float(bend),
        "total": float(total),
    }


def total_energy_numpy(
    coords: np.ndarray,
    config: MacromoleculeConfig,
    contact_map: np.ndarray | None = None,
) -> float:
    return energy_terms_numpy(coords, config, contact_map)["total"]


def sobolev_h1_smooth_numpy(gradient: np.ndarray, alpha: float = 5.0) -> np.ndarray:
    """Apply the DCT-II Sobolev H1 filter to one chain's gradient."""

    grad = np.asarray(gradient, dtype=np.float64)
    if grad.ndim != 2 or grad.shape[1] != 3:
        raise ValueError("gradient must have shape (N, 3)")
    if len(grad) == 0:
        return grad.copy()
    from scipy.fft import dct, idct

    k = np.arange(len(grad), dtype=np.float64)
    inv_sigma = 1.0 / (1.0 + float(alpha) * k**2)
    g_hat = dct(grad, type=2, norm="ortho", axis=0)
    return idct(g_hat * inv_sigma[:, None], type=2, norm="ortho", axis=0)


def smooth_gradient_by_chain_numpy(
    gradient: np.ndarray,
    complex_spec: ComplexSpec,
) -> np.ndarray:
    """Apply Sobolev smoothing independently to each chain gradient slice."""

    grad = np.asarray(gradient, dtype=np.float64)
    complex_spec.validate(len(grad))
    smoothed = np.zeros_like(grad)
    for chain in complex_spec.chains:
        alpha = chain.sobolev_alpha
        if alpha is None:
            alpha = chain.config.sobolev_alpha
        smoothed[chain.start : chain.end] = sobolev_h1_smooth_numpy(
            grad[chain.start : chain.end],
            alpha=alpha,
        )
    return smoothed


def energy_terms_complex_numpy(
    coords: np.ndarray,
    complex_spec: ComplexSpec,
    contact_map: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a heterogeneous multi-chain complex Hamiltonian with NumPy."""

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    complex_spec.validate(len(arr))
    vectors = _complex_vectors(complex_spec)
    cmap = _as_contact_map(contact_map, len(arr))

    bond = 0.0
    rg_energy = 0.0
    bend = 0.0
    for chain in complex_spec.chains:
        cfg = chain.config
        chain_coords = arr[chain.start : chain.end]
        if len(chain_coords) > 1:
            diffs = chain_coords[1:] - chain_coords[:-1]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1) + 1e-8)
            bond += cfg.bond_stiffness * float(
                np.sum((dists - cfg.ideal_bond_distance) ** 2)
            )

        if chain.uses_compaction() and cfg.rg_mode != "none":
            target = cfg.expected_rg(len(chain_coords))
            if target > 0.0:
                rg = radius_of_gyration(chain_coords)
                if cfg.rg_mode == "upper_bound":
                    rg_energy += max(rg - target, 0.0) ** 2
                elif cfg.rg_mode == "lower_bound":
                    rg_energy += max(target - rg, 0.0) ** 2
                else:
                    raise ValueError(f"unknown rg_mode {cfg.rg_mode!r}")

        if cfg.bend_stiffness > 0.0 and len(chain_coords) > 2:
            diffs = chain_coords[1:] - chain_coords[:-1]
            lengths = np.sqrt(np.sum(diffs * diffs, axis=1) + 1e-8)
            unit = diffs / lengths[:, None]
            cosines = np.sum(unit[1:] * unit[:-1], axis=1)
            bend += cfg.bend_stiffness * float(
                np.sum((cosines - cfg.bend_target_cos) ** 2)
            )

    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-2)
    mask = _complex_neighbor_mask(complex_spec)
    violations = np.maximum(vectors["pair_sigma"] - dist, 0.0)
    steric = float(np.sum((violations * mask) ** 2))

    if cmap.any():
        contact_dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
        contact_violations = np.maximum(
            contact_dist - vectors["pair_contact_distance"],
            0.0,
        )
        weighted = cmap * vectors["contact_weight"] * (contact_violations**2)
        same = vectors["same_chain"]
        contacts_intra = float(np.sum(weighted * same))
        contacts_inter = float(np.sum(weighted * ~same))
    else:
        contacts_intra = 0.0
        contacts_inter = 0.0
    contacts = contacts_intra + contacts_inter

    total = bond + steric + contacts + rg_energy + bend
    return {
        "bond": float(bond),
        "steric": float(steric),
        "contacts": float(contacts),
        "contacts_intra": float(contacts_intra),
        "contacts_inter": float(contacts_inter),
        "rg": float(rg_energy),
        "bend": float(bend),
        "total": float(total),
    }


def total_energy_complex_numpy(
    coords: np.ndarray,
    complex_spec: ComplexSpec,
    contact_map: np.ndarray | None = None,
) -> float:
    return energy_terms_complex_numpy(coords, complex_spec, contact_map)["total"]


def watson_crick_contact_map(
    n_base_pairs: int,
    weight: float = 1.0,
    antiparallel: bool = True,
) -> np.ndarray:
    """Build an inter-strand dsDNA base-pair contact map.

    The coordinate order is assumed to be strand A followed by strand B.  For
    canonical antiparallel dsDNA, residue ``i`` on strand A pairs with
    ``n_base_pairs - 1 - i`` on strand B.
    """

    n = int(n_base_pairs)
    if n <= 0:
        raise ValueError("n_base_pairs must be positive")
    contact_map = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for i in range(n):
        partner_on_b = n - 1 - i if antiparallel else i
        j = n + partner_on_b
        contact_map[i, j] = contact_map[j, i] = float(weight)
    return contact_map


class SobolevMacromolecule:
    """Config-bound facade for universal Sobolev H1 polishing."""

    def __init__(self, config: MacromoleculeConfig):
        self.config = config

    @classmethod
    def from_kind(cls, kind: str, **overrides) -> "SobolevMacromolecule":
        return cls(get_macromolecule_config(kind, **overrides))

    def expected_rg(self, n_residues: int) -> float:
        return self.config.expected_rg(n_residues)

    def energy_terms(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
    ) -> dict[str, float]:
        return energy_terms_numpy(coords, self.config, contact_map)

    def total_energy(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
    ) -> float:
        return total_energy_numpy(coords, self.config, contact_map)

    def polish(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
        n_steps: int | None = None,
        lr: float | None = None,
    ) -> np.ndarray:
        """Run Sobolev H1 gradient-flow polish with the preset constants."""

        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import jax.scipy.fft as jfft
        except Exception as exc:  # pragma: no cover - depends on local install
            raise JaxUnavailableError(
                "JAX is required for SobolevMacromolecule.polish(); "
                "install the repo requirements to enable optimization."
            ) from exc

        config = self.config
        raw = np.asarray(coords, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        cmap_np = _as_contact_map(contact_map, len(raw))
        steps = int(config.default_steps if n_steps is None else n_steps)
        step_lr = float(config.default_lr if lr is None else lr)

        def energy_bond(x):
            diffs = x[1:] - x[:-1]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)
            return config.bond_stiffness * jnp.sum(
                (dists - config.ideal_bond_distance) ** 2
            )

        def energy_steric(x):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-2)
            idx = jnp.arange(x.shape[0])
            mask = idx[None, :] > idx[:, None] + 1
            violations = jnp.maximum(config.sigma_clash - dist, 0.0)
            return jnp.sum((violations * mask) ** 2)

        def energy_contacts(x, cmap):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            violations = jnp.maximum(dist - config.contact_distance, 0.0)
            return config.contact_weight * jnp.sum(cmap * (violations**2))

        def energy_rg(x):
            if config.rg_mode == "none":
                return jnp.array(0.0, dtype=jnp.float64)
            target = config.expected_rg(x.shape[0])
            if target <= 0.0:
                return jnp.array(0.0, dtype=jnp.float64)
            center = jnp.mean(x, axis=0)
            rg = jnp.sqrt(jnp.mean(jnp.sum((x - center) ** 2, axis=-1)) + 1e-8)
            if config.rg_mode == "upper_bound":
                violation = jnp.maximum(rg - target, 0.0)
            else:
                violation = jnp.maximum(target - rg, 0.0)
            return violation**2

        def energy_bend(x):
            if config.bend_stiffness <= 0.0 or x.shape[0] < 3:
                return jnp.array(0.0, dtype=jnp.float64)
            diffs = x[1:] - x[:-1]
            lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1) + 1e-8)
            unit = diffs / lengths[:, None]
            cosines = jnp.sum(unit[1:] * unit[:-1], axis=1)
            return config.bend_stiffness * jnp.sum(
                (cosines - config.bend_target_cos) ** 2
            )

        def total_energy(x, cmap):
            return (
                energy_bond(x)
                + energy_steric(x)
                + energy_contacts(x, cmap)
                + energy_rg(x)
                + energy_bend(x)
            )

        def sobolev_h1_smooth(gradient):
            n = gradient.shape[0]
            k = jnp.arange(n)
            inv_sigma = 1.0 / (1.0 + config.sobolev_alpha * k**2)
            g_hat = jfft.dct(gradient, type=2, norm="ortho", axis=0)
            return jfft.idct(g_hat * inv_sigma[:, None], type=2, norm="ortho", axis=0)

        @jax.jit
        def step_fn(x, cmap):
            grads = jax.grad(total_energy)(x, cmap)
            smooth_grads = sobolev_h1_smooth(grads)
            clipped = jnp.clip(smooth_grads, -config.gradient_clip, config.gradient_clip)
            return x - step_lr * clipped

        def scan_body(x, _):
            return step_fn(x, contact_map_jax), None

        coords_jax = jnp.array(raw, dtype=jnp.float64)
        contact_map_jax = jnp.array(cmap_np, dtype=jnp.float64)
        final, _ = jax.lax.scan(scan_body, coords_jax, None, length=steps)
        return np.asarray(final)


class SobolevComplex:
    """Chain-aware Sobolev H1 polisher for mixed macromolecular complexes."""

    def __init__(self, complex_spec: ComplexSpec):
        complex_spec.validate()
        self.spec = complex_spec

    @classmethod
    def from_fasta(cls, text: str) -> tuple[str, "SobolevComplex"]:
        sequence, spec = parse_complex_fasta(text)
        return sequence, cls(spec)

    def energy_terms(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
    ) -> dict[str, float]:
        return energy_terms_complex_numpy(coords, self.spec, contact_map)

    def total_energy(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
    ) -> float:
        return total_energy_complex_numpy(coords, self.spec, contact_map)

    def smooth_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return smooth_gradient_by_chain_numpy(gradient, self.spec)

    def polish(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
        n_steps: int = 2000,
        lr: float = 0.005,
    ) -> np.ndarray:
        """Run chain-masked Sobolev H1 polish on a heterogeneous complex."""

        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import jax.scipy.fft as jfft
        except Exception as exc:  # pragma: no cover - depends on local install
            raise JaxUnavailableError(
                "JAX is required for SobolevComplex.polish(); "
                "install the repo requirements to enable optimization."
            ) from exc

        raw = np.asarray(coords, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.spec.validate(len(raw))
        cmap_np = _as_contact_map(contact_map, len(raw))
        vectors = _complex_vectors(self.spec)
        neighbor_mask_np = _complex_neighbor_mask(self.spec)

        pair_sigma = jnp.array(vectors["pair_sigma"], dtype=jnp.float64)
        pair_contact_distance = jnp.array(
            vectors["pair_contact_distance"],
            dtype=jnp.float64,
        )
        contact_weight = jnp.array(vectors["contact_weight"], dtype=jnp.float64)
        neighbor_mask = jnp.array(neighbor_mask_np, dtype=jnp.float64)
        contact_map_jax = jnp.array(cmap_np, dtype=jnp.float64)

        def energy_bond(x):
            total = jnp.array(0.0, dtype=jnp.float64)
            for chain in self.spec.chains:
                cfg = chain.config
                segment = x[chain.start : chain.end]
                if chain.length > 1:
                    diffs = segment[1:] - segment[:-1]
                    dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)
                    total = total + cfg.bond_stiffness * jnp.sum(
                        (dists - cfg.ideal_bond_distance) ** 2
                    )
            return total

        def energy_steric(x):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-2)
            violations = jnp.maximum(pair_sigma - dist, 0.0)
            return jnp.sum((violations * neighbor_mask) ** 2)

        def energy_contacts(x, cmap):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            violations = jnp.maximum(dist - pair_contact_distance, 0.0)
            return jnp.sum(cmap * contact_weight * (violations**2))

        def energy_rg(x):
            total = jnp.array(0.0, dtype=jnp.float64)
            for chain in self.spec.chains:
                cfg = chain.config
                if not chain.uses_compaction() or cfg.rg_mode == "none":
                    continue
                target = cfg.expected_rg(chain.length)
                if target <= 0.0:
                    continue
                segment = x[chain.start : chain.end]
                center = jnp.mean(segment, axis=0)
                rg = jnp.sqrt(
                    jnp.mean(jnp.sum((segment - center) ** 2, axis=-1)) + 1e-8
                )
                if cfg.rg_mode == "upper_bound":
                    violation = jnp.maximum(rg - target, 0.0)
                else:
                    violation = jnp.maximum(target - rg, 0.0)
                total = total + violation**2
            return total

        def energy_bend(x):
            total = jnp.array(0.0, dtype=jnp.float64)
            for chain in self.spec.chains:
                cfg = chain.config
                if cfg.bend_stiffness <= 0.0 or chain.length < 3:
                    continue
                segment = x[chain.start : chain.end]
                diffs = segment[1:] - segment[:-1]
                lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1) + 1e-8)
                unit = diffs / lengths[:, None]
                cosines = jnp.sum(unit[1:] * unit[:-1], axis=1)
                total = total + cfg.bend_stiffness * jnp.sum(
                    (cosines - cfg.bend_target_cos) ** 2
                )
            return total

        def total_energy(x, cmap):
            return (
                energy_bond(x)
                + energy_steric(x)
                + energy_contacts(x, cmap)
                + energy_rg(x)
                + energy_bend(x)
            )

        def sobolev_h1_smooth_complex(gradient):
            out = jnp.zeros_like(gradient)
            for chain in self.spec.chains:
                alpha = chain.sobolev_alpha
                if alpha is None:
                    alpha = chain.config.sobolev_alpha
                segment = gradient[chain.start : chain.end]
                k = jnp.arange(chain.length)
                inv_sigma = 1.0 / (1.0 + float(alpha) * k**2)
                g_hat = jfft.dct(segment, type=2, norm="ortho", axis=0)
                filtered = jfft.idct(
                    g_hat * inv_sigma[:, None],
                    type=2,
                    norm="ortho",
                    axis=0,
                )
                out = out.at[chain.start : chain.end].set(filtered)
            return out

        max_clip = max(chain.config.gradient_clip for chain in self.spec.chains)

        @jax.jit
        def step_fn(x):
            grads = jax.grad(total_energy)(x, contact_map_jax)
            smooth_grads = sobolev_h1_smooth_complex(grads)
            clipped = jnp.clip(smooth_grads, -max_clip, max_clip)
            return x - float(lr) * clipped

        def scan_body(x, _):
            return step_fn(x), None

        coords_jax = jnp.array(raw, dtype=jnp.float64)
        final, _ = jax.lax.scan(scan_body, coords_jax, None, length=int(n_steps))
        return np.asarray(final)


def create_macromolecule(kind: str, **overrides) -> SobolevMacromolecule:
    """Factory entrypoint for RNA/protein/dsDNA Sobolev polishing."""

    return SobolevMacromolecule.from_kind(kind, **overrides)


def create_complex(
    chain_lengths: Sequence[tuple[str, str, int]],
    w_intra: float = 2.0,
    w_inter: float = 3.0,
) -> SobolevComplex:
    """Factory entrypoint for heterogeneous multi-chain complexes."""

    return SobolevComplex(
        complex_from_chain_lengths(
            chain_lengths,
            w_intra=w_intra,
            w_inter=w_inter,
        )
    )


def available_presets() -> tuple[str, ...]:
    return tuple(PRESETS)

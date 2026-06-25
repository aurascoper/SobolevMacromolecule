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


@dataclass(frozen=True)
class GraphBond:
    """Covalent/topological edge in a heterogeneous molecular graph."""

    i: int
    j: int
    ideal_distance: float
    stiffness: float = 100.0


@dataclass(frozen=True)
class SlabPotential:
    """Implicit membrane slab potential centered on the z axis."""

    half_thickness: float = 15.0
    hydrophobic_strength: float = 1.0
    hydrophilic_strength: float = 1.0
    hydrophobic_types: tuple[str, ...] = ("lipid_tail", "hydrophobic")
    hydrophilic_types: tuple[str, ...] = ("lipid_head", "hydrophilic", "protein_loop")


@dataclass(frozen=True)
class GraphNodeFeatures:
    """Per-bead physical properties for SobolevMacro graph simulations."""

    node_types: tuple[str, ...]
    radii: tuple[float, ...]
    masses: tuple[float, ...] | None = None
    charges: tuple[float, ...] | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_types)

    def validate(self) -> None:
        n = self.n_nodes
        if len(self.radii) != n:
            raise ValueError("radii length must match node_types length")
        if self.masses is not None and len(self.masses) != n:
            raise ValueError("masses length must match node_types length")
        if self.charges is not None and len(self.charges) != n:
            raise ValueError("charges length must match node_types length")


@dataclass(frozen=True)
class GraphSpec:
    """Unified heterogeneous graph topology for SobolevMacro."""

    nodes: GraphNodeFeatures
    bonds: tuple[GraphBond, ...]
    contact_distance: float = 8.0
    contact_weight: float = 2.0
    sobolev_alpha: float = 5.0
    gradient_clip: float = 2.0
    slab: SlabPotential | None = None

    @property
    def n_nodes(self) -> int:
        return self.nodes.n_nodes

    def validate(self, n_nodes: int | None = None) -> None:
        self.nodes.validate()
        if n_nodes is not None and int(n_nodes) != self.n_nodes:
            raise ValueError(f"graph has {self.n_nodes} nodes, expected {n_nodes}")
        for bond in self.bonds:
            if bond.i == bond.j:
                raise ValueError("self bonds are not allowed")
            if not (0 <= bond.i < self.n_nodes and 0 <= bond.j < self.n_nodes):
                raise ValueError(f"bond {(bond.i, bond.j)} is outside node range")

    def adjacency_matrix(self) -> np.ndarray:
        self.validate()
        adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for bond in self.bonds:
            adjacency[bond.i, bond.j] = 1.0
            adjacency[bond.j, bond.i] = 1.0
        return adjacency


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


def make_graph_spec(
    node_types: Sequence[str],
    bonds: Sequence[tuple[int, int, float] | GraphBond],
    radii: Sequence[float] | None = None,
    contact_distance: float = 8.0,
    contact_weight: float = 2.0,
    sobolev_alpha: float = 5.0,
    gradient_clip: float = 2.0,
    slab: SlabPotential | None = None,
) -> GraphSpec:
    """Build a heterogeneous graph spec from node labels and bond tuples."""

    node_type_tuple = tuple(str(node_type) for node_type in node_types)
    if radii is None:
        radii = tuple(default_radius_for_type(node_type) for node_type in node_type_tuple)
    graph_bonds: list[GraphBond] = []
    for bond in bonds:
        if isinstance(bond, GraphBond):
            graph_bonds.append(bond)
        else:
            i, j, ideal = bond
            graph_bonds.append(GraphBond(int(i), int(j), float(ideal)))
    spec = GraphSpec(
        nodes=GraphNodeFeatures(node_type_tuple, tuple(float(r) for r in radii)),
        bonds=tuple(graph_bonds),
        contact_distance=float(contact_distance),
        contact_weight=float(contact_weight),
        sobolev_alpha=float(sobolev_alpha),
        gradient_clip=float(gradient_clip),
        slab=slab,
    )
    spec.validate()
    return spec


def default_radius_for_type(node_type: str) -> float:
    """Coarse physical radius lookup for graph sterics."""

    key = node_type.strip().lower()
    if key in {"coarse", "capsid_bead", "martini"}:
        return 15.0
    if key in {"protein", "c_alpha", "ca"}:
        return 4.2
    if key in {"rna", "dna", "c1", "p"}:
        return 3.5
    if key in {"lipid_tail", "lipid_head", "lipid"}:
        return 4.5
    if key in {"ligand", "small_molecule", "atom"}:
        return 1.5
    if key in {"glycan", "sugar", "carbohydrate"}:
        return 2.5
    return 3.0


def graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the unnormalized graph Laplacian L = D - A."""

    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    sym = np.maximum(adj, adj.T)
    degree = np.sum(sym, axis=1)
    return np.diag(degree) - sym


def graph_sobolev_smooth_numpy(
    gradient: np.ndarray,
    adjacency: np.ndarray,
    alpha: float = 5.0,
) -> np.ndarray:
    """Apply graph Laplacian Sobolev smoothing to an arbitrary topology."""

    grad = np.asarray(gradient, dtype=np.float64)
    if grad.ndim != 2 or grad.shape[1] != 3:
        raise ValueError("gradient must have shape (N, 3)")
    laplacian = graph_laplacian(adjacency)
    if laplacian.shape[0] != len(grad):
        raise ValueError("gradient and adjacency sizes differ")
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    spectral = eigvecs.T @ grad
    filtered = spectral / (1.0 + float(alpha) * eigvals[:, None])
    return eigvecs @ filtered


def _bond_adjacency_from_graph(graph_spec: GraphSpec) -> np.ndarray:
    return graph_spec.adjacency_matrix() > 0


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


def energy_terms_graph_numpy(
    coords: np.ndarray,
    graph_spec: GraphSpec,
    contact_map: np.ndarray | None = None,
    contact_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate the unified SobolevMacro graph Hamiltonian with NumPy."""

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    graph_spec.validate(len(arr))
    n = len(arr)

    bond = 0.0
    for edge in graph_spec.bonds:
        dist = float(np.sqrt(np.sum((arr[edge.i] - arr[edge.j]) ** 2) + 1e-8))
        bond += edge.stiffness * (dist - edge.ideal_distance) ** 2

    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-2)
    radii = np.asarray(graph_spec.nodes.radii, dtype=np.float64)
    sigma = radii[:, None] + radii[None, :]
    idx = np.arange(n)
    upper = idx[None, :] > idx[:, None]
    bonded = _bond_adjacency_from_graph(graph_spec)
    steric_mask = upper & ~bonded
    steric = float(np.sum((np.maximum(sigma - dist, 0.0) * steric_mask) ** 2))

    cmap = _as_contact_map(contact_map, n)
    if contact_weights is None:
        weights = np.full((n, n), graph_spec.contact_weight, dtype=np.float64)
    else:
        weights = np.asarray(contact_weights, dtype=np.float64)
        if weights.shape != (n, n):
            raise ValueError(f"contact_weights shape {weights.shape}, expected {(n, n)}")
    if cmap.any():
        contact_dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
        violations = np.maximum(contact_dist - graph_spec.contact_distance, 0.0)
        contacts = float(np.sum(cmap * weights * (violations**2)))
    else:
        contacts = 0.0

    environment = 0.0
    if graph_spec.slab is not None:
        slab = graph_spec.slab
        z_abs = np.abs(arr[:, 2])
        hydrophobic = np.array(
            [node_type in slab.hydrophobic_types for node_type in graph_spec.nodes.node_types],
            dtype=bool,
        )
        hydrophilic = np.array(
            [node_type in slab.hydrophilic_types for node_type in graph_spec.nodes.node_types],
            dtype=bool,
        )
        tail_violation = np.maximum(z_abs - slab.half_thickness, 0.0)
        head_violation = np.maximum(slab.half_thickness - z_abs, 0.0)
        environment = float(
            slab.hydrophobic_strength * np.sum((tail_violation * hydrophobic) ** 2)
            + slab.hydrophilic_strength * np.sum((head_violation * hydrophilic) ** 2)
        )

    total = bond + steric + contacts + environment
    return {
        "bond": float(bond),
        "steric": float(steric),
        "contacts": float(contacts),
        "environment": float(environment),
        "total": float(total),
    }


def total_energy_graph_numpy(
    coords: np.ndarray,
    graph_spec: GraphSpec,
    contact_map: np.ndarray | None = None,
    contact_weights: np.ndarray | None = None,
) -> float:
    return energy_terms_graph_numpy(
        coords,
        graph_spec,
        contact_map=contact_map,
        contact_weights=contact_weights,
    )["total"]


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


class SobolevMacro:
    """Graph-based Sobolev engine for branched and non-polymer systems."""

    def __init__(self, graph_spec: GraphSpec):
        graph_spec.validate()
        self.spec = graph_spec

    def energy_terms(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
        contact_weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        return energy_terms_graph_numpy(
            coords,
            self.spec,
            contact_map=contact_map,
            contact_weights=contact_weights,
        )

    def total_energy(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
        contact_weights: np.ndarray | None = None,
    ) -> float:
        return total_energy_graph_numpy(
            coords,
            self.spec,
            contact_map=contact_map,
            contact_weights=contact_weights,
        )

    def smooth_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return graph_sobolev_smooth_numpy(
            gradient,
            self.spec.adjacency_matrix(),
            alpha=self.spec.sobolev_alpha,
        )

    def polish(
        self,
        coords: np.ndarray,
        contact_map: np.ndarray | None = None,
        contact_weights: np.ndarray | None = None,
        n_steps: int = 2000,
        lr: float = 0.005,
    ) -> np.ndarray:
        """Run graph Laplacian Sobolev polish on a heterogeneous graph."""

        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - depends on local install
            raise JaxUnavailableError(
                "JAX is required for SobolevMacro.polish(); "
                "install the repo requirements to enable optimization."
            ) from exc

        raw = np.asarray(coords, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")
        self.spec.validate(len(raw))
        n = len(raw)
        cmap_np = _as_contact_map(contact_map, n)
        if contact_weights is None:
            contact_weights_np = np.full((n, n), self.spec.contact_weight, dtype=np.float64)
        else:
            contact_weights_np = np.asarray(contact_weights, dtype=np.float64)
            if contact_weights_np.shape != (n, n):
                raise ValueError(
                    f"contact_weights shape {contact_weights_np.shape}, expected {(n, n)}"
                )

        adjacency = jnp.array(self.spec.adjacency_matrix(), dtype=jnp.float64)
        laplacian = jnp.diag(jnp.sum(adjacency, axis=1)) - adjacency
        eigvals, eigvecs = jnp.linalg.eigh(laplacian)
        radii = jnp.array(self.spec.nodes.radii, dtype=jnp.float64)
        bonded = jnp.array(_bond_adjacency_from_graph(self.spec), dtype=bool)
        contact_map_jax = jnp.array(cmap_np, dtype=jnp.float64)
        contact_weights_jax = jnp.array(contact_weights_np, dtype=jnp.float64)
        idx = jnp.arange(n)
        upper = idx[None, :] > idx[:, None]
        steric_mask = upper & ~bonded

        bond_i = jnp.array([edge.i for edge in self.spec.bonds], dtype=jnp.int32)
        bond_j = jnp.array([edge.j for edge in self.spec.bonds], dtype=jnp.int32)
        bond_d0 = jnp.array(
            [edge.ideal_distance for edge in self.spec.bonds],
            dtype=jnp.float64,
        )
        bond_k = jnp.array([edge.stiffness for edge in self.spec.bonds], dtype=jnp.float64)

        if self.spec.slab is None:
            hydrophobic = jnp.zeros(n, dtype=bool)
            hydrophilic = jnp.zeros(n, dtype=bool)
            slab_half = 0.0
            slab_k_hydrophobic = 0.0
            slab_k_hydrophilic = 0.0
        else:
            slab = self.spec.slab
            hydrophobic = jnp.array(
                [node_type in slab.hydrophobic_types for node_type in self.spec.nodes.node_types],
                dtype=bool,
            )
            hydrophilic = jnp.array(
                [node_type in slab.hydrophilic_types for node_type in self.spec.nodes.node_types],
                dtype=bool,
            )
            slab_half = float(slab.half_thickness)
            slab_k_hydrophobic = float(slab.hydrophobic_strength)
            slab_k_hydrophilic = float(slab.hydrophilic_strength)

        def energy_bond(x):
            if len(self.spec.bonds) == 0:
                return jnp.array(0.0, dtype=jnp.float64)
            diffs = x[bond_i] - x[bond_j]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)
            return jnp.sum(bond_k * (dists - bond_d0) ** 2)

        def energy_steric(x):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-2)
            sigma = radii[:, None] + radii[None, :]
            violations = jnp.maximum(sigma - dist, 0.0)
            return jnp.sum((violations * steric_mask) ** 2)

        def energy_contacts(x):
            diff = x[:, None, :] - x[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            violations = jnp.maximum(dist - self.spec.contact_distance, 0.0)
            return jnp.sum(contact_map_jax * contact_weights_jax * (violations**2))

        def energy_environment(x):
            z_abs = jnp.abs(x[:, 2])
            tail_violation = jnp.maximum(z_abs - slab_half, 0.0)
            head_violation = jnp.maximum(slab_half - z_abs, 0.0)
            return (
                slab_k_hydrophobic * jnp.sum((tail_violation * hydrophobic) ** 2)
                + slab_k_hydrophilic * jnp.sum((head_violation * hydrophilic) ** 2)
            )

        def total_energy(x):
            return energy_bond(x) + energy_steric(x) + energy_contacts(x) + energy_environment(x)

        def graph_sobolev_smooth(gradient):
            spectral = eigvecs.T @ gradient
            filtered = spectral / (1.0 + self.spec.sobolev_alpha * eigvals[:, None])
            return eigvecs @ filtered

        @jax.jit
        def step_fn(x):
            grads = jax.grad(total_energy)(x)
            smooth_grads = graph_sobolev_smooth(grads)
            clipped = jnp.clip(
                smooth_grads,
                -self.spec.gradient_clip,
                self.spec.gradient_clip,
            )
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


def create_macro_graph(
    node_types: Sequence[str],
    bonds: Sequence[tuple[int, int, float] | GraphBond],
    radii: Sequence[float] | None = None,
    contact_distance: float = 8.0,
    contact_weight: float = 2.0,
    sobolev_alpha: float = 5.0,
    gradient_clip: float = 2.0,
    slab: SlabPotential | None = None,
) -> SobolevMacro:
    """Factory entrypoint for heterogeneous graph SobolevMacro simulations."""

    return SobolevMacro(
        make_graph_spec(
            node_types,
            bonds,
            radii=radii,
            contact_distance=contact_distance,
            contact_weight=contact_weight,
            sobolev_alpha=sobolev_alpha,
            gradient_clip=gradient_clip,
            slab=slab,
        )
    )


def available_presets() -> tuple[str, ...]:
    return tuple(PRESETS)

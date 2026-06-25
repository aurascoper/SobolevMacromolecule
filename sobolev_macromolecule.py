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


def _as_contact_map(contact_map, n_residues: int) -> np.ndarray:
    if contact_map is None:
        return np.zeros((n_residues, n_residues), dtype=np.float64)
    cmap = np.asarray(contact_map, dtype=np.float64)
    if cmap.shape != (n_residues, n_residues):
        raise ValueError(
            f"contact_map has shape {cmap.shape}, expected {(n_residues, n_residues)}"
        )
    return cmap


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


def create_macromolecule(kind: str, **overrides) -> SobolevMacromolecule:
    """Factory entrypoint for RNA/protein/dsDNA Sobolev polishing."""

    return SobolevMacromolecule.from_kind(kind, **overrides)


def available_presets() -> tuple[str, ...]:
    return tuple(PRESETS)

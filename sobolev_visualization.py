"""Streaming and instancing helpers for SobolevMacro visualization.

The physics engine should stay headless: it produces coordinate tensors, while
WebGPU, OpenUSD, or Unreal Engine consume compact instance updates.  This module
defines a tiny binary frame ABI plus NumPy helpers for renderer-facing instance
matrices and coarse-grained far-field views.  It intentionally has no Unreal,
USD, WebSocket, or gRPC dependency.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MAGIC = b"SMACRO1\0"
VERSION = 1

_FLAG_ASSET_IDS = 1
_HEADER = struct.Struct("<8sHHIqd")


@dataclass(frozen=True)
class CoordinateFrame:
    """Decoded coordinate frame for renderer or network transport adapters."""

    coords: np.ndarray
    frame_index: int = 0
    time_seconds: float = 0.0
    asset_ids: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class CoarseGrainResult:
    """Far-field coarse-grained coordinate view with provenance mapping."""

    coords: np.ndarray
    source_indices: tuple[tuple[int, ...], ...]
    is_coarse: np.ndarray

    @property
    def n_original(self) -> int:
        return sum(len(group) for group in self.source_indices)


@dataclass(frozen=True)
class AssetPrototype:
    """Renderer prototype for one molecular asset species."""

    asset_id: int
    name: str
    path: str = ""
    source: str = ""
    copy_number: int | None = None

    def validate(self) -> None:
        if self.asset_id < 0:
            raise ValueError("asset_id must be non-negative")
        if not self.name:
            raise ValueError("asset prototype name must be non-empty")
        if self.copy_number is not None and self.copy_number < 0:
            raise ValueError("copy_number must be non-negative")

    def to_dict(self) -> dict[str, int | str | None]:
        self.validate()
        return {
            "asset_id": int(self.asset_id),
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "copy_number": self.copy_number,
        }


def validate_coords(coords: np.ndarray | Sequence[Sequence[float]], name: str = "coords") -> np.ndarray:
    """Return coordinates as a finite ``(N, 3)`` float array."""

    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def asset_id_vector_from_counts(
    copy_numbers: Mapping[int, int] | Sequence[tuple[int, int]]
) -> np.ndarray:
    """Expand abundance/copy-number records into an instance asset-id vector."""

    items = sorted(copy_numbers.items()) if isinstance(copy_numbers, Mapping) else list(copy_numbers)
    ids: list[int] = []
    for asset_id, count in items:
        asset_id_int = int(asset_id)
        count_int = int(count)
        if asset_id_int < 0:
            raise ValueError("asset ids must be non-negative")
        if count_int < 0:
            raise ValueError("copy numbers must be non-negative")
        ids.extend([asset_id_int] * count_int)
    return np.asarray(ids, dtype=np.uint32)


def encode_coordinate_frame(
    coords: np.ndarray | Sequence[Sequence[float]],
    frame_index: int = 0,
    time_seconds: float = 0.0,
    asset_ids: Sequence[int] | np.ndarray | None = None,
) -> bytes:
    """Encode one coordinate tensor as a compact little-endian binary frame.

    Layout:
        header: ``<8sHHIqd`` = magic, version, flags, N, frame_index, time
        body:   ``N * 3`` little-endian float32 coordinates
        tail:   optional ``N`` little-endian uint32 asset identifiers

    The payload is suitable for WebSocket messages, gRPC byte fields, or shared
    memory slots.  Transport-level framing and compression are intentionally left
    to the caller.
    """

    arr = validate_coords(coords).astype(np.dtype("<f4"), copy=False)
    n_nodes = int(arr.shape[0])
    flags = 0
    asset_payload = b""

    if asset_ids is not None:
        ids = np.asarray(asset_ids)
        if ids.shape != (n_nodes,):
            raise ValueError(f"asset_ids shape {ids.shape}, expected {(n_nodes,)}")
        if np.any(ids < 0):
            raise ValueError("asset_ids must be non-negative")
        ids = ids.astype(np.dtype("<u4"), copy=False)
        flags |= _FLAG_ASSET_IDS
        asset_payload = np.ascontiguousarray(ids).tobytes(order="C")

    header = _HEADER.pack(
        MAGIC,
        VERSION,
        flags,
        n_nodes,
        int(frame_index),
        float(time_seconds),
    )
    return header + np.ascontiguousarray(arr).tobytes(order="C") + asset_payload


def decode_coordinate_frame(payload: bytes | bytearray | memoryview) -> CoordinateFrame:
    """Decode a binary coordinate frame produced by ``encode_coordinate_frame``."""

    view = memoryview(payload)
    if len(view) < _HEADER.size:
        raise ValueError("payload is shorter than the SobolevMacro frame header")

    magic, version, flags, n_nodes, frame_index, time_seconds = _HEADER.unpack_from(view, 0)
    if magic != MAGIC:
        raise ValueError("payload does not start with SobolevMacro frame magic")
    if version != VERSION:
        raise ValueError(f"unsupported SobolevMacro frame version {version}")

    coords_bytes = int(n_nodes) * 3 * np.dtype("<f4").itemsize
    asset_bytes = int(n_nodes) * np.dtype("<u4").itemsize if flags & _FLAG_ASSET_IDS else 0
    expected = _HEADER.size + coords_bytes + asset_bytes
    if len(view) != expected:
        raise ValueError(f"payload has {len(view)} bytes, expected {expected}")

    coords = np.frombuffer(view, dtype=np.dtype("<f4"), count=int(n_nodes) * 3, offset=_HEADER.size)
    coords = coords.reshape((int(n_nodes), 3)).astype(np.float32, copy=True)
    validate_coords(coords)

    asset_ids = None
    if flags & _FLAG_ASSET_IDS:
        offset = _HEADER.size + coords_bytes
        asset_ids = np.frombuffer(view, dtype=np.dtype("<u4"), count=int(n_nodes), offset=offset)
        asset_ids = asset_ids.astype(np.uint32, copy=True)

    return CoordinateFrame(
        coords=coords,
        frame_index=int(frame_index),
        time_seconds=float(time_seconds),
        asset_ids=asset_ids,
    )


def write_scene_manifest(
    path: str | Path,
    *,
    tomography_map: str | None = None,
    segmentation_mesh: str | None = None,
    coordinate_frame: str | None = None,
    assets: Sequence[AssetPrototype] = (),
    sources: Mapping[str, str] | None = None,
) -> None:
    """Write a JSON manifest tying imaging, assets, abundance, and frames together."""

    for asset in assets:
        asset.validate()
    payload = {
        "schema": "sobolev-whole-cell-scene-v1",
        "tomography_map": tomography_map,
        "segmentation_mesh": segmentation_mesh,
        "coordinate_frame": coordinate_frame,
        "assets": [asset.to_dict() for asset in assets],
        "sources": dict(sources or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_usda_point_instancer(
    path: str | Path,
    coords: np.ndarray | Sequence[Sequence[float]],
    asset_ids: Sequence[int] | np.ndarray,
    prototypes: Sequence[AssetPrototype],
    scene_name: str = "SobolevMacroScene",
) -> None:
    """Write a minimal OpenUSD ASCII PointInstancer scene.

    This is a dependency-free handoff file for render clients.  The generated
    USDA references one prototype per asset species and stores each SobolevMacro
    coordinate as an instanced position.
    """

    arr = validate_coords(coords).astype(np.float32, copy=False)
    ids = np.asarray(asset_ids, dtype=np.int64)
    if ids.shape != (len(arr),):
        raise ValueError(f"asset_ids shape {ids.shape}, expected {(len(arr),)}")
    if np.any(ids < 0):
        raise ValueError("asset_ids must be non-negative")
    if not prototypes:
        raise ValueError("at least one AssetPrototype is required")

    prototype_by_id = {}
    for prototype in prototypes:
        prototype.validate()
        if prototype.asset_id in prototype_by_id:
            raise ValueError(f"duplicate asset prototype id {prototype.asset_id}")
        prototype_by_id[int(prototype.asset_id)] = prototype

    missing = sorted(set(int(asset_id) for asset_id in ids) - set(prototype_by_id))
    if missing:
        raise ValueError(f"missing prototypes for asset ids {missing}")

    ordered_ids = sorted(prototype_by_id)
    proto_index = {asset_id: idx for idx, asset_id in enumerate(ordered_ids)}
    proto_indices = [proto_index[int(asset_id)] for asset_id in ids]
    scene_ident = _usd_identifier(scene_name)

    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{scene_ident}"',
        ")",
        "",
        f'def Xform "{scene_ident}"',
        "{",
        '    def Scope "Prototypes"',
        "    {",
    ]
    for asset_id in ordered_ids:
        prototype = prototype_by_id[asset_id]
        proto_name = _usd_identifier(prototype.name)
        if prototype.path:
            lines.extend(
                [
                    f'        def Xform "{proto_name}" (',
                    f"            references = @{_usd_asset_path(prototype.path)}@",
                    "        )",
                    "        {",
                    "        }",
                ]
            )
        else:
            lines.extend([f'        def Xform "{proto_name}"', "        {", "        }"])
    lines.extend(
        [
            "    }",
            "",
            '    def PointInstancer "Instances"',
            "    {",
            "        rel prototypes = [",
        ]
    )
    for asset_id in ordered_ids:
        proto_name = _usd_identifier(prototype_by_id[asset_id].name)
        lines.append(f"            </{scene_ident}/Prototypes/{proto_name}>,")
    lines.extend(
        [
            "        ]",
            "        int[] protoIndices = [" + ", ".join(str(idx) for idx in proto_indices) + "]",
            "        point3f[] positions = [",
        ]
    )
    for x, y, z in arr:
        lines.append(f"            ({_fmt_float(x)}, {_fmt_float(y)}, {_fmt_float(z)}),")
    lines.extend(["        ]", "    }", "}"])

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def instance_matrices(
    coords: np.ndarray | Sequence[Sequence[float]],
    scales: float | Sequence[float] | np.ndarray = 1.0,
    rotations: np.ndarray | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Build ``(N, 4, 4)`` transform matrices for renderer instancing.

    Coordinates become translations.  ``scales`` may be a scalar or one uniform
    scale per node.  ``rotations`` may be omitted for identity orientation or
    supplied as ``(N, 3, 3)`` rotation matrices.
    """

    arr = validate_coords(coords).astype(dtype, copy=False)
    n_nodes = int(arr.shape[0])

    if rotations is None:
        rot = np.broadcast_to(np.eye(3, dtype=dtype), (n_nodes, 3, 3)).copy()
    else:
        rot = np.asarray(rotations, dtype=dtype)
        if rot.shape != (n_nodes, 3, 3):
            raise ValueError(f"rotations shape {rot.shape}, expected {(n_nodes, 3, 3)}")

    scale_arr = np.asarray(scales, dtype=dtype)
    if scale_arr.ndim == 0:
        scale_arr = np.full(n_nodes, float(scale_arr), dtype=dtype)
    if scale_arr.shape != (n_nodes,):
        raise ValueError(f"scales shape {scale_arr.shape}, expected scalar or {(n_nodes,)}")

    matrices = np.broadcast_to(np.eye(4, dtype=dtype), (n_nodes, 4, 4)).copy()
    matrices[:, :3, :3] = rot * scale_arr[:, None, None]
    matrices[:, :3, 3] = arr
    return matrices


def coarse_grain_far_field(
    coords: np.ndarray | Sequence[Sequence[float]],
    active_center: Sequence[float] | np.ndarray,
    active_radius: float,
    group_size: int = 10,
) -> CoarseGrainResult:
    """Collapse nodes outside an active sphere into centroid beads.

    Nodes inside ``active_radius`` are preserved one-to-one.  Far-field nodes are
    chunked in original index order into groups of ``group_size`` and replaced by
    their centroids.  The returned ``source_indices`` field preserves the mapping
    from each rendered bead back to the original node indices.
    """

    arr = validate_coords(coords).astype(np.float64, copy=False)
    center = np.asarray(active_center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("active_center must have shape (3,)")
    if active_radius < 0:
        raise ValueError("active_radius must be non-negative")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    dist = np.linalg.norm(arr - center[None, :], axis=1)
    active_indices = np.flatnonzero(dist <= float(active_radius))
    far_indices = np.flatnonzero(dist > float(active_radius))

    groups: list[tuple[int, ...]] = [(int(idx),) for idx in active_indices]
    for start in range(0, len(far_indices), int(group_size)):
        chunk = tuple(int(idx) for idx in far_indices[start : start + int(group_size)])
        groups.append(chunk)

    if not groups:
        out = np.empty((0, 3), dtype=np.float64)
        coarse = np.empty((0,), dtype=bool)
    else:
        out = np.vstack([arr[list(group)].mean(axis=0) for group in groups])
        coarse = np.asarray([len(group) > 1 for group in groups], dtype=bool)

    return CoarseGrainResult(
        coords=out.astype(np.float32),
        source_indices=tuple(groups),
        is_coarse=coarse,
    )


def _usd_identifier(value: str) -> str:
    out = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    out = out.strip("_") or "Asset"
    if out[0].isdigit():
        out = "A_" + out
    return out


def _usd_asset_path(value: str) -> str:
    return value.replace("@", "").replace("\n", "")


def _fmt_float(value: float) -> str:
    return format(float(value), ".7g")


__all__ = [
    "AssetPrototype",
    "CoarseGrainResult",
    "CoordinateFrame",
    "MAGIC",
    "VERSION",
    "asset_id_vector_from_counts",
    "coarse_grain_far_field",
    "decode_coordinate_frame",
    "encode_coordinate_frame",
    "instance_matrices",
    "write_scene_manifest",
    "write_usda_point_instancer",
    "validate_coords",
]

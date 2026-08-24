# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Operations for positioning groups of meshes in a scene."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np


class EOverlapDirection(Enum):
    """Direction in which to place an overlapping mesh group."""
    NEGATIVE = "negative"
    CURRENT = "current"
    POSITIVE = "positive"


# Compatibility alias; new code should follow the enum naming convention.
OverlapDirection = EOverlapDirection


class MeshCoords(Protocol):
    """Structural type required by scene positioning operations."""
    coords: np.ndarray


@dataclass(frozen=True, slots=True)
class Bounds3D:
    """Axis-aligned three-dimensional bounds."""
    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    extent: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshGroup:
    """Contiguous range of meshes in a scene."""
    mesh_start: int
    mesh_len: int


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Spacing and capacity of a three-dimensional group grid."""
    gap: tuple[float, float, float]
    max_divs: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BoundsOverlapSpec:
    """Per-axis settings for overlapping two mesh group bounds."""
    overlap_frac: tuple[float, float, float]
    enabled_axes: tuple[bool, bool, bool] = (True, True, True)
    direction: tuple[
        EOverlapDirection, EOverlapDirection, EOverlapDirection
    ] = (
        EOverlapDirection.CURRENT,
        EOverlapDirection.CURRENT,
        EOverlapDirection.CURRENT,
    )
    extra_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def mesh_group_span(mesh_start: int, mesh_len: int) -> MeshGroup:
    """Create a validated contiguous mesh group."""
    if mesh_start < 0:
        raise ValueError("mesh_start must be non-negative.")
    if mesh_len <= 0:
        raise ValueError("mesh_len must be positive.")
    return MeshGroup(mesh_start=mesh_start, mesh_len=mesh_len)


def mesh_group_single(mesh_idx: int) -> MeshGroup:
    """Create a group containing one mesh."""
    return mesh_group_span(mesh_idx, 1)


def _group_indices(meshes: Sequence[MeshCoords], group: MeshGroup) -> range:
    """Return validated indices for a mesh group."""
    if group.mesh_start < 0 or group.mesh_len <= 0:
        raise ValueError(
            "Mesh groups require a non-negative start and positive length.",
        )
    group_end = group.mesh_start + group.mesh_len
    if group_end > len(meshes):
        raise IndexError("Mesh group extends beyond the mesh sequence.")
    return range(group.mesh_start, group_end)


def _validated_coords(coords: np.ndarray) -> np.ndarray:
    """Return coordinates after validating their shape and values."""
    coords_in = np.asarray(coords)
    if coords_in.ndim != 2 or coords_in.shape[1] != 3 or not coords_in.shape[0]:
        raise ValueError(
            "Mesh coordinates must have shape (nodes, 3) and not be empty.",
        )
    if not np.all(np.isfinite(coords_in)):
        raise ValueError("Mesh coordinates must contain only finite values.")
    return coords_in


def bounds_for_coords(coords: np.ndarray) -> Bounds3D:
    """Calculate axis-aligned bounds for coordinates."""
    coords_in = _validated_coords(coords)
    minimum = np.min(coords_in, axis=0)
    maximum = np.max(coords_in, axis=0)
    return Bounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def _bounds_for_indices(
    meshes: Sequence[MeshCoords],
    indices: range,
) -> Bounds3D:
    """Reduce bounds over meshes without concatenating their coordinates."""
    mesh_bounds = [bounds_for_coords(meshes[index].coords) for index in indices]
    minimum = np.min([bounds.minimum for bounds in mesh_bounds], axis=0)
    maximum = np.max([bounds.maximum for bounds in mesh_bounds], axis=0)
    return Bounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def bounds_for_meshes(meshes: Sequence[MeshCoords]) -> Bounds3D:
    """Calculate combined bounds without copying all mesh coordinates."""
    if not meshes:
        raise ValueError("At least one mesh is required.")
    return _bounds_for_indices(meshes, range(len(meshes)))


def bounds_for_mesh_group(
    meshes: Sequence[MeshCoords],
    group: MeshGroup,
) -> Bounds3D:
    """Calculate combined bounds for a contiguous mesh group."""
    return _bounds_for_indices(meshes, _group_indices(meshes, group))


def _vector3(
    values: tuple[float, float, float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate and convert a finite three-component vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    return vector


def translate_mesh_group(
    meshes: Sequence[MeshCoords],
    group: MeshGroup,
    translation: tuple[float, float, float] | np.ndarray,
) -> None:
    """Translate a mesh group in place."""
    translation_array = _vector3(translation, "translation")
    for index in _group_indices(meshes, group):
        coords = _validated_coords(meshes[index].coords)
        if not np.issubdtype(coords.dtype, np.floating):
            raise TypeError("Mesh coordinates must use a floating-point dtype.")
        coords += translation_array


def center_mesh_group_at(
    meshes: Sequence[MeshCoords],
    group: MeshGroup,
    target_center: tuple[float, float, float] | np.ndarray,
) -> None:
    """Translate a mesh group so its bounds have the requested center."""
    bounds = bounds_for_mesh_group(meshes, group)
    target = _vector3(target_center, "target_center")
    translate_mesh_group(meshes, group, target - bounds.center)


def _overlap_sign(current_sep: float, direction: EOverlapDirection) -> float:
    """Resolve an overlap direction to a signed separation."""
    if direction is EOverlapDirection.NEGATIVE:
        return -1.0
    if direction is EOverlapDirection.POSITIVE:
        return 1.0
    if direction is EOverlapDirection.CURRENT:
        return -1.0 if current_sep < 0.0 else 1.0
    raise ValueError(f"Unsupported overlap direction: {direction}.")


def overlap_mesh_group_bounds(
    meshes: Sequence[MeshCoords],
    fixed_group: MeshGroup,
    moving_group: MeshGroup,
    spec: BoundsOverlapSpec,
) -> None:
    """Translate one group to achieve the requested bounds overlap."""
    overlap = _vector3(spec.overlap_frac, "overlap_frac")
    if np.any((overlap < 0.0) | (overlap > 1.0)):
        raise ValueError("overlap_frac values must lie in [0, 1].")
    enabled = np.asarray(spec.enabled_axes)
    if enabled.shape != (3,) or enabled.dtype != np.bool_:
        raise ValueError("enabled_axes must contain three boolean values.")
    if len(spec.direction) != 3:
        raise ValueError("direction must contain three values.")
    extra_offset = _vector3(spec.extra_offset, "extra_offset")
    fixed_bounds = bounds_for_mesh_group(meshes, fixed_group)
    moving_bounds = bounds_for_mesh_group(meshes, moving_group)
    translation = extra_offset.copy()

    for axis in range(3):
        if not enabled[axis]:
            continue
        desired_overlap = overlap[axis] * min(
            fixed_bounds.extent[axis], moving_bounds.extent[axis],
        )
        separation = (
            0.5 * (fixed_bounds.extent[axis] + moving_bounds.extent[axis])
            - desired_overlap
        )
        current_sep = moving_bounds.center[axis] - fixed_bounds.center[axis]
        target = (
            fixed_bounds.center[axis]
            + _overlap_sign(current_sep, spec.direction[axis]) * separation
            + extra_offset[axis]
        )
        translation[axis] = target - moving_bounds.center[axis]
    translate_mesh_group(meshes, moving_group, translation)


def arrange_mesh_groups_grid(
    meshes: Sequence[MeshCoords],
    groups: Sequence[MeshGroup],
    spec: GridSpec,
) -> None:
    """Center mesh groups on a bounded three-dimensional grid."""
    if not groups:
        return
    gap = _vector3(spec.gap, "gap")
    divisions = np.asarray(spec.max_divs)
    if (
        divisions.shape != (3,)
        or not np.issubdtype(divisions.dtype, np.integer)
    ):
        raise ValueError("max_divs must contain three integers.")
    if np.any(divisions <= 0):
        raise ValueError("max_divs values must be positive.")
    if len(groups) > int(np.prod(divisions)):
        raise ValueError("Mesh groups exceed the grid capacity.")
    group_bounds = [bounds_for_mesh_group(meshes, group) for group in groups]
    max_extent = np.max([bounds.extent for bounds in group_bounds], axis=0)
    stride = max_extent + gap
    x_divs, y_divs, _ = (int(value) for value in divisions)
    for index, group in enumerate(groups):
        grid_index = (
            index % x_divs,
            (index // x_divs) % y_divs,
            index // (x_divs * y_divs),
        )
        center_mesh_group_at(meshes, group, np.asarray(grid_index) * stride)


__all__ = [
    "Bounds3D", "BoundsOverlapSpec", "EOverlapDirection", "GridSpec",
    "MeshGroup", "OverlapDirection", "arrange_mesh_groups_grid",
    "bounds_for_coords", "bounds_for_mesh_group", "bounds_for_meshes",
    "center_mesh_group_at", "mesh_group_single", "mesh_group_span",
    "overlap_mesh_group_bounds", "translate_mesh_group",
]

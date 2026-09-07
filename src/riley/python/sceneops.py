# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from riley.python._verifio import _validate_coords, _validate_vec3


class EOverlapDirect(Enum):
    NEGATIVE = "negative"
    CURRENT = "current"
    POSITIVE = "positive"


@dataclass(frozen=True, slots=True)
class Bounds3D:
    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    extent: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshGroup:
    mesh_start: int
    mesh_len: int


@dataclass(frozen=True, slots=True)
class GridSpec:
    gap: tuple[float, float, float]
    max_divs: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BoundsOverlapSpec:
    overlap_frac: tuple[float, float, float]
    enabled_axes: tuple[bool, bool, bool] = (True, True, True)
    direct: tuple[
        EOverlapDirect, EOverlapDirect, EOverlapDirect
    ] = (
        EOverlapDirect.CURRENT,
        EOverlapDirect.CURRENT,
        EOverlapDirect.CURRENT,
    )
    extra_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def create_mesh_group_span(mesh_start: int, mesh_len: int) -> MeshGroup:
    if mesh_start < 0:
        raise ValueError("mesh_start must be non-negative.")

    if mesh_len <= 0:
        raise ValueError("mesh_len must be positive.")

    return MeshGroup(mesh_start=mesh_start, mesh_len=mesh_len)


def create_mesh_group_single(mesh_idx: int) -> MeshGroup:
    return create_mesh_group_span(mesh_idx, 1)


def _get_group_indices(
    coords_list: Sequence[np.ndarray],
    group: MeshGroup,
) -> range:
    if group.mesh_start < 0 or group.mesh_len <= 0:
        raise ValueError(
            "Mesh groups require a non-negative start and positive length.",
        )

    group_end = group.mesh_start + group.mesh_len

    if group_end > len(coords_list):
        raise IndexError("Mesh group extends beyond the mesh sequence.")

    return range(group.mesh_start, group_end)


def calc_bounds_for_coords(coords: np.ndarray) -> Bounds3D:
    coords_in = _validate_coords(coords, "Mesh coordinates")
    minimum = np.min(coords_in, axis=0)
    maximum = np.max(coords_in, axis=0)

    return Bounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def _calc_bounds_for_indices(
    coords_list: Sequence[np.ndarray],
    indices: range,
) -> Bounds3D:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for index in indices:
        bounds = calc_bounds_for_coords(coords_list[index])
        mins.append(bounds.minimum)
        maxs.append(bounds.maximum)

    minimum = np.min(mins, axis=0)
    maximum = np.max(maxs, axis=0)

    return Bounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def calc_bounds_for_coords_list(
    coords_list: Sequence[np.ndarray],
) -> Bounds3D:
    if not coords_list:
        raise ValueError("At least one mesh is required.")

    return _calc_bounds_for_indices(coords_list, range(len(coords_list)))


def calc_bounds_for_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: MeshGroup,
) -> Bounds3D:
    return _calc_bounds_for_indices(
        coords_list,
        _get_group_indices(coords_list, group),
    )


def translate_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: MeshGroup,
    translation: tuple[float, float, float] | np.ndarray,
) -> None:
    translation_array = _validate_vec3(translation, "translation")
    for index in _get_group_indices(coords_list, group):
        coords = coords_list[index]
        _ = _validate_coords(coords, "Mesh coordinates")
        if not np.issubdtype(coords.dtype, np.floating):
            raise TypeError("Mesh coordinates must use a floating-point dtype.")
        coords += translation_array


def center_mesh_group_at(
    coords_list: Sequence[np.ndarray],
    group: MeshGroup,
    target_center: tuple[float, float, float] | np.ndarray,
) -> None:
    bounds = calc_bounds_for_mesh_group(coords_list, group)
    target = _validate_vec3(target_center, "target_center")
    translate_mesh_group(coords_list, group, target - bounds.center)


def _calc_overlap_sign(
    current_sep: float,
    direct: EOverlapDirect,
) -> float:
    if direct is EOverlapDirect.NEGATIVE:
        return -1.0

    if direct is EOverlapDirect.POSITIVE:
        return 1.0

    if direct is EOverlapDirect.CURRENT:
        return -1.0 if current_sep < 0.0 else 1.0

    raise ValueError(f"Unsupported overlap direction: {direct}.")


def overlap_mesh_group_bounds(
    coords_list: Sequence[np.ndarray],
    fixed_group: MeshGroup,
    moving_group: MeshGroup,
    spec: BoundsOverlapSpec,
) -> None:
    overlap = _validate_vec3(spec.overlap_frac, "overlap_frac")
    if np.any((overlap < 0.0) | (overlap > 1.0)):
        raise ValueError("overlap_frac values must lie in [0, 1].")

    enabled = np.asarray(spec.enabled_axes)
    if enabled.shape != (3,) or enabled.dtype != np.bool_:
        raise ValueError("enabled_axes must contain three boolean values.")

    if len(spec.direct) != 3:
        raise ValueError("direct must contain three values.")

    extra_offset = _validate_vec3(spec.extra_offset, "extra_offset")
    fixed_bounds = calc_bounds_for_mesh_group(coords_list, fixed_group)
    moving_bounds = calc_bounds_for_mesh_group(coords_list, moving_group)
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
            + _calc_overlap_sign(current_sep, spec.direct[axis]) * separation
            + extra_offset[axis]
        )

        translation[axis] = target - moving_bounds.center[axis]

    translate_mesh_group(coords_list, moving_group, translation)


def arrange_mesh_groups_grid(
    coords_list: Sequence[np.ndarray],
    groups: Sequence[MeshGroup],
    spec: GridSpec,
) -> None:
    if not groups:
        return

    gap = _validate_vec3(spec.gap, "gap")
    divisions = np.asarray(spec.max_divs)
    if (
        divisions.shape != (3,)
        or not np.issubdtype(divisions.dtype, np.integer)
    ):
        raise ValueError("max_divs must contain three integers.")

    if np.any(divisions <= 0):
        raise ValueError("max_divs values must be positive.")

    grid_capacity = int(np.prod(divisions))
    if len(groups) > grid_capacity:
        raise ValueError("Mesh groups exceed the grid capacity.")

    group_extents: list[np.ndarray] = []
    for group in groups:
        bounds = calc_bounds_for_mesh_group(coords_list, group)
        group_extents.append(bounds.extent)

    max_extent = np.max(group_extents, axis=0)
    stride = max_extent + gap
    x_divs = int(divisions[0])
    y_divs = int(divisions[1])

    for index, group in enumerate(groups):
        grid_index = (
            index % x_divs,
            (index // x_divs) % y_divs,
            index // (x_divs * y_divs),
        )
        center_mesh_group_at(
            coords_list,
            group,
            np.asarray(grid_index) * stride,
        )


mesh_group_span = create_mesh_group_span
mesh_group_single = create_mesh_group_single
bounds_for_coords = calc_bounds_for_coords
bounds_for_meshes = calc_bounds_for_coords_list
bounds_for_mesh_group = calc_bounds_for_mesh_group


__all__ = [
    "Bounds3D",
    "BoundsOverlapSpec",
    "EOverlapDirect",
    "GridSpec",
    "MeshGroup",
    "arrange_mesh_groups_grid",
    "bounds_for_coords",
    "bounds_for_mesh_group",
    "bounds_for_meshes",
    "calc_bounds_for_coords",
    "calc_bounds_for_coords_list",
    "calc_bounds_for_mesh_group",
    "center_mesh_group_at",
    "create_mesh_group_single",
    "create_mesh_group_span",
    "mesh_group_single",
    "mesh_group_span",
    "overlap_mesh_group_bounds",
    "translate_mesh_group",
]

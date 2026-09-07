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

def _verify_finite_f64(
    values: object,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    values_out = np.ascontiguousarray(values, dtype=np.float64)

    if shape is not None and values_out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    finite = np.isfinite(values_out)
    if not np.all(finite):
        raise ValueError(f"{name} must not contain non-finite values.")

    return values_out


def _verify_coords(coords: np.ndarray) -> np.ndarray:
    coords_out = np.ascontiguousarray(coords, dtype=np.float64)

    coords_are_2d = coords_out.ndim == 2
    coords_have_nodes = coords_are_2d and coords_out.shape[0] > 0
    coords_have_xyz = coords_are_2d and coords_out.shape[1] == 3
    if not coords_have_nodes or not coords_have_xyz:
        raise ValueError(
            "coords must have shape (nodes, 3) and not be empty.",
        )

    finite = np.isfinite(coords_out)
    if not np.all(finite):
        raise ValueError("coords must not contain non-finite values.")

    return coords_out


class ESceneOverlapDirect(Enum):
    NEGATIVE = "negative"
    CURRENT = "current"
    POSITIVE = "positive"


@dataclass(slots=True)
class SceneBounds3D:
    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    extent: np.ndarray


@dataclass(frozen=True, slots=True)
class SceneMeshGroup:
    mesh_start: int
    mesh_len: int


@dataclass(frozen=True, slots=True)
class SceneGridSpec:
    gap: tuple[float, float, float]
    max_divs: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SceneBoundsOverlapSpec:
    overlap_frac: tuple[float, float, float]
    enabled_axes: tuple[bool, bool, bool] = (True, True, True)
    direct: tuple[
        ESceneOverlapDirect, ESceneOverlapDirect, ESceneOverlapDirect
    ] = (
        ESceneOverlapDirect.CURRENT,
        ESceneOverlapDirect.CURRENT,
        ESceneOverlapDirect.CURRENT,
    )
    extra_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def scene_create_mesh_group_span(
    mesh_start: int,
    mesh_len: int,
) -> SceneMeshGroup:
    if mesh_start < 0:
        raise ValueError("mesh_start must be non-negative.")

    if mesh_len <= 0:
        raise ValueError("mesh_len must be positive.")

    return SceneMeshGroup(mesh_start=mesh_start, mesh_len=mesh_len)


def scene_create_mesh_group_single(mesh_idx: int) -> SceneMeshGroup:
    return scene_create_mesh_group_span(mesh_idx, 1)


def _get_group_indices(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
) -> range:
    if group.mesh_start < 0 or group.mesh_len <= 0:
        raise ValueError(
            "Mesh groups require a non-negative start and positive length.",
        )

    group_end = group.mesh_start + group.mesh_len

    if group_end > len(coords_list):
        raise IndexError("Mesh group extends beyond the mesh sequence.")

    return range(group.mesh_start, group_end)


def scene_calc_bounds_for_coords(coords: np.ndarray) -> SceneBounds3D:
    coords_in = _verify_coords(coords)
    minimum = np.min(coords_in, axis=0)
    maximum = np.max(coords_in, axis=0)

    return SceneBounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def _calc_bounds_for_indices(
    coords_list: Sequence[np.ndarray],
    indices: range,
) -> SceneBounds3D:

    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for index in indices:
        bounds = scene_calc_bounds_for_coords(coords_list[index])
        mins.append(bounds.minimum)
        maxs.append(bounds.maximum)

    minimum = np.min(mins, axis=0)
    maximum = np.max(maxs, axis=0)

    return SceneBounds3D(
        minimum=minimum,
        maximum=maximum,
        center=0.5 * (minimum + maximum),
        extent=maximum - minimum,
    )


def scene_calc_bounds_for_coords_list(
    coords_list: Sequence[np.ndarray],
) -> SceneBounds3D:
    if not coords_list:
        raise ValueError("At least one mesh is required.")

    return _calc_bounds_for_indices(coords_list, range(len(coords_list)))


def scene_calc_bounds_for_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
) -> SceneBounds3D:
    return _calc_bounds_for_indices(
        coords_list,
        _get_group_indices(coords_list, group),
    )


def scene_translate_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
    translation: tuple[float, float, float] | np.ndarray,
) -> None:
    translation_array = _verify_finite_f64(translation, "translation", (3,))
    for index in _get_group_indices(coords_list, group):
        coords = coords_list[index]
        _ = _verify_coords(coords)
        if not np.issubdtype(coords.dtype, np.floating):
            raise TypeError("Mesh coordinates must use a floating-point dtype.")
        coords += translation_array


def scene_center_mesh_group_at(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
    target_center: tuple[float, float, float] | np.ndarray,
) -> None:
    bounds = scene_calc_bounds_for_mesh_group(coords_list, group)
    target = _verify_finite_f64(target_center, "target_center", (3,))
    scene_translate_mesh_group(coords_list, group, target - bounds.center)


def _calc_overlap_sign(
    current_sep: float,
    direct: ESceneOverlapDirect,
) -> float:
    if direct is ESceneOverlapDirect.NEGATIVE:
        return -1.0

    if direct is ESceneOverlapDirect.POSITIVE:
        return 1.0

    if direct is ESceneOverlapDirect.CURRENT:
        return -1.0 if current_sep < 0.0 else 1.0

    raise ValueError(f"Unsupported overlap direction: {direct}.")


def scene_overlap_mesh_group_bounds(
    coords_list: Sequence[np.ndarray],
    fixed_group: SceneMeshGroup,
    moving_group: SceneMeshGroup,
    spec: SceneBoundsOverlapSpec,
) -> None:
    overlap = _verify_finite_f64(spec.overlap_frac, "overlap_frac", (3,))
    if np.any((overlap < 0.0) | (overlap > 1.0)):
        raise ValueError("overlap_frac values must lie in [0, 1].")

    enabled = np.asarray(spec.enabled_axes)
    if enabled.shape != (3,) or enabled.dtype != np.bool_:
        raise ValueError("enabled_axes must contain three boolean values.")

    if len(spec.direct) != 3:
        raise ValueError("direct must contain three values.")

    extra_offset = _verify_finite_f64(spec.extra_offset, "extra_offset", (3,))
    fixed_bounds = scene_calc_bounds_for_mesh_group(coords_list, fixed_group)
    moving_bounds = scene_calc_bounds_for_mesh_group(coords_list, moving_group)
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

    scene_translate_mesh_group(coords_list, moving_group, translation)


def scene_arrange_mesh_groups_grid(
    coords_list: Sequence[np.ndarray],
    groups: Sequence[SceneMeshGroup],
    spec: SceneGridSpec,
) -> None:
    if not groups:
        return

    gap = _verify_finite_f64(spec.gap, "gap", (3,))
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
        bounds = scene_calc_bounds_for_mesh_group(coords_list, group)
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
        scene_center_mesh_group_at(
            coords_list,
            group,
            np.asarray(grid_index) * stride,
        )


__all__ = [
    "ESceneOverlapDirect",
    "SceneBounds3D",
    "SceneBoundsOverlapSpec",
    "SceneGridSpec",
    "SceneMeshGroup",
    "scene_arrange_mesh_groups_grid",
    "scene_calc_bounds_for_coords",
    "scene_calc_bounds_for_coords_list",
    "scene_calc_bounds_for_mesh_group",
    "scene_center_mesh_group_at",
    "scene_create_mesh_group_single",
    "scene_create_mesh_group_span",
    "scene_overlap_mesh_group_bounds",
    "scene_translate_mesh_group",
]

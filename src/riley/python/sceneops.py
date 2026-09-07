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
    """Verify that an object can be cast to finite contiguous float64 array.

    Parameters
    ----------
    values : object
        Input array-like or sequence of numerical values.
    name : str
        Human-readable identifier for error messages.
    shape : tuple of int, optional
        Expected array shape. If None, any shape is permitted.

    Returns
    -------
    numpy.ndarray
        Contiguous array of dtype `np.float64`.

    Raises
    ------
    ValueError
        If array shape does not match `shape` or contains non-finite values.
    """
    values_out = np.ascontiguousarray(values, dtype=np.float64)

    if shape is not None and values_out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    finite = np.isfinite(values_out)
    if not np.all(finite):
        raise ValueError(f"{name} must not contain non-finite values.")

    return values_out


def _verify_coords(coords: np.ndarray) -> np.ndarray:
    """Verify that node coordinate array has shape (N, 3) and finite values.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape `(N, 3)` and dtype `np.float64`, where `N` is the
        number of nodes and columns represent `(x, y, z)` spatial coordinates.

    Returns
    -------
    numpy.ndarray
        Contiguous array of shape `(N, 3)` and dtype `np.float64`.

    Raises
    ------
    ValueError
        If `coords` is not 2-dimensional with shape `(N, 3)`, is empty,
        or contains non-finite values.
    """
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
    """Direction for bounding box separation during overlap operations.

    Members
    -------
    NEGATIVE
        Move in the negative coordinate direction along the axis.
    CURRENT
        Preserve the current relative sign of separation.
    POSITIVE
        Move in the positive coordinate direction along the axis.
    """

    NEGATIVE = "negative"
    CURRENT = "current"
    POSITIVE = "positive"


@dataclass(slots=True)
class SceneBounds3D:
    """Axis-aligned 3D bounding box extents for geometry.

    Attributes
    ----------
    minimum : numpy.ndarray
        Lower spatial bounds `(x_min, y_min, z_min)` of shape `(3,)` and
        dtype `np.float64`.
    maximum : numpy.ndarray
        Upper spatial bounds `(x_max, y_max, z_max)` of shape `(3,)` and
        dtype `np.float64`.
    center : numpy.ndarray
        Midpoint center coordinates `(xc, yc, zc)` of shape `(3,)` and
        dtype `np.float64`.
    extent : numpy.ndarray
        Total span dimensions `(dx, dy, dz)` of shape `(3,)` and
        dtype `np.float64`.
    """

    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    extent: np.ndarray


@dataclass(frozen=True, slots=True)
class SceneMeshGroup:
    """Contiguous slice of meshes in a scene coordinate list.

    Attributes
    ----------
    mesh_start : int
        Zero-based index of the first mesh in the group.
    mesh_len : int
        Number of meshes in the group.
    """

    mesh_start: int
    mesh_len: int


@dataclass(frozen=True, slots=True)
class SceneGridSpec:
    """Grid layout configuration for arranging mesh groups.

    Attributes
    ----------
    gap : tuple of float
        Spacing gap `(gx, gy, gz)` added between adjacent cell extents.
    max_divs : tuple of int
        Maximum number of cell divisions `(nx, ny, nz)` along each axis.
    """

    gap: tuple[float, float, float]
    max_divs: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SceneBoundsOverlapSpec:
    """Configuration for positioning a moving mesh group relative to fixed.

    Attributes
    ----------
    overlap_frac : tuple of float
        Fraction of overlap `(fx, fy, fz)` in `[0.0, 1.0]` along each axis.
    enabled_axes : tuple of bool, default=(True, True, True)
        Whether overlap adjustment is active on `(x, y, z)` axes.
    direct : tuple of ESceneOverlapDirect, default=(CURRENT, CURRENT, CURRENT)
        Direction policy for placing the moving group relative to fixed.
    extra_offset : tuple of float, default=(0.0, 0.0, 0.0)
        Additional spatial translation offset `(ox, oy, oz)` applied.
    """

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
    """Create a mesh group specifying a contiguous range of meshes.

    Parameters
    ----------
    mesh_start : int
        Zero-based index of the starting mesh. Must be non-negative.
    mesh_len : int
        Number of meshes in the span. Must be positive.

    Returns
    -------
    SceneMeshGroup
        Validated mesh group instance.

    Raises
    ------
    ValueError
        If `mesh_start < 0` or `mesh_len <= 0`.
    """
    if mesh_start < 0:
        raise ValueError("mesh_start must be non-negative.")

    if mesh_len <= 0:
        raise ValueError("mesh_len must be positive.")

    return SceneMeshGroup(mesh_start=mesh_start, mesh_len=mesh_len)


def scene_create_mesh_group_single(mesh_idx: int) -> SceneMeshGroup:
    """Create a mesh group containing a single mesh.

    Parameters
    ----------
    mesh_idx : int
        Zero-based index of the mesh. Must be non-negative.

    Returns
    -------
    SceneMeshGroup
        Mesh group spanning exactly one mesh.
    """
    return scene_create_mesh_group_span(mesh_idx, 1)


def _get_group_indices(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
) -> range:
    """Compute the index range for a mesh group and verify list bounds.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays.
    group : SceneMeshGroup
        Target mesh group.

    Returns
    -------
    range
        Integer index range for the mesh group.

    Raises
    ------
    ValueError
        If `group` parameters are invalid.
    IndexError
        If `group` extends beyond `coords_list`.
    """
    if group.mesh_start < 0 or group.mesh_len <= 0:
        raise ValueError(
            "Mesh groups require a non-negative start and positive length.",
        )

    group_end = group.mesh_start + group.mesh_len

    if group_end > len(coords_list):
        raise IndexError("Mesh group extends beyond the mesh sequence.")

    return range(group.mesh_start, group_end)


def scene_calc_bounds_for_coords(coords: np.ndarray) -> SceneBounds3D:
    """Compute the 3D bounding box for a single node coordinate array.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape `(N, 3)` and dtype `np.float64`, where `N` is the
        number of nodes and columns represent `(x, y, z)` spatial coordinates.

    Returns
    -------
    SceneBounds3D
        Bounding box containing minimum, maximum, center, and extent.

    Raises
    ------
    ValueError
        If `coords` is empty, non-2D, or contains non-finite values.
    """
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
    """Compute combined 3D bounds across a subset of mesh coordinate arrays.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays, each of shape `(N_i, 3)` and dtype
        `np.float64`.
    indices : range
        Integer indices specifying which meshes in `coords_list` to include.

    Returns
    -------
    SceneBounds3D
        Combined bounding box covering all specified meshes.
    """
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
    """Compute combined 3D bounding box for all meshes in a coordinate list.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays, each of shape `(N_i, 3)` and dtype
        `np.float64`.

    Returns
    -------
    SceneBounds3D
        Overall bounding box covering all meshes.

    Raises
    ------
    ValueError
        If `coords_list` is empty.
    """
    if not coords_list:
        raise ValueError("At least one mesh is required.")

    return _calc_bounds_for_indices(coords_list, range(len(coords_list)))


def scene_calc_bounds_for_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
) -> SceneBounds3D:
    """Compute combined 3D bounding box for meshes in a group.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays, each of shape `(N_i, 3)` and dtype
        `np.float64`.
    group : SceneMeshGroup
        Target mesh group span.

    Returns
    -------
    SceneBounds3D
        Combined bounding box for the meshes in `group`.

    Raises
    ------
    ValueError or IndexError
        If `group` parameters are invalid or out of range.
    """
    return _calc_bounds_for_indices(
        coords_list,
        _get_group_indices(coords_list, group),
    )


def scene_translate_mesh_group(
    coords_list: Sequence[np.ndarray],
    group: SceneMeshGroup,
    translation: tuple[float, float, float] | np.ndarray,
) -> None:
    """Translate all mesh coordinates in a group in-place by an offset vector.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays, each of shape `(N_i, 3)` and dtype
        floating-point, modified in-place.
    group : SceneMeshGroup
        Target mesh group.
    translation : tuple of float or numpy.ndarray
        Offset vector `(dx, dy, dz)` of shape `(3,)` and dtype `np.float64`.

    Raises
    ------
    TypeError
        If `coords` does not use a floating-point dtype.
    ValueError or IndexError
        If `group` or `translation` is invalid.
    """
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
    """Translate meshes in a group in-place so their combined center aligns.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays, each of shape `(N_i, 3)` and dtype
        floating-point, modified in-place.
    group : SceneMeshGroup
        Target mesh group.
    target_center : tuple of float or numpy.ndarray
        Desired center coordinates `(xc, yc, zc)` of shape `(3,)` and dtype
        `np.float64`.
    """
    bounds = scene_calc_bounds_for_mesh_group(coords_list, group)
    target = _verify_finite_f64(target_center, "target_center", (3,))
    scene_translate_mesh_group(coords_list, group, target - bounds.center)


def _calc_overlap_sign(
    current_sep: float,
    direct: ESceneOverlapDirect,
) -> float:
    """Calculate the direction sign for overlap positioning.

    Parameters
    ----------
    current_sep : float
        Current spatial separation between group centers along the axis.
    direct : ESceneOverlapDirect
        Direction placement policy.

    Returns
    -------
    float
        Sign multiplier (-1.0 or 1.0).

    Raises
    ------
    ValueError
        If `direct` is not a valid `ESceneOverlapDirect` member.
    """
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
    """Position a moving mesh group relative to a fixed group by overlap.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays modified in-place.
    fixed_group : SceneMeshGroup
        Stationary mesh group reference.
    moving_group : SceneMeshGroup
        Mesh group translated in-place.
    spec : SceneBoundsOverlapSpec
        Overlap fractions, active axes, direction policy, and extra offsets.

    Raises
    ------
    ValueError
        If `spec` parameters are invalid or out of range.
    """
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
    """Arrange multiple mesh groups on a regular 3D grid layout.

    Parameters
    ----------
    coords_list : collections.abc.Sequence of numpy.ndarray
        Sequence of coordinate arrays modified in-place.
    groups : collections.abc.Sequence of SceneMeshGroup
        Mesh groups to position.
    spec : SceneGridSpec
        Grid spacing gaps and maximum cell divisions along each axis.

    Raises
    ------
    ValueError
        If `spec` divisions are invalid or `groups` exceed grid capacity.
    """
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

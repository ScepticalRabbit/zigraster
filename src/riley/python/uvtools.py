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

_Z_PARALLEL_TOL = 0.999


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


class EUVProjPlane(Enum):
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EUVPlanarProjMode(Enum):
    BEST = "best"
    FIT_X = "fit_x"
    FIT_Y = "fit_y"


@dataclass(slots=True)
class UVProjPlane:
    normal: np.ndarray
    origin: np.ndarray


@dataclass(slots=True)
class UVProjAxes:
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray


@dataclass(frozen=True, slots=True)
class UVProjBounds2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True, slots=True)
class UVPixelBBox:
    x_lower: float
    y_lower: float
    x_upper: float
    y_upper: float


UVProjPlaneLike = EUVProjPlane | UVProjPlane | tuple[
    np.ndarray, np.ndarray
]

UVPixelBBoxLike = (
    UVPixelBBox
    | tuple[float, float, float, float]
    | Sequence[float]
    | np.ndarray
)


def _verify_texture_size(
    texture_size: tuple[int, int] | tuple[float, float],
) -> tuple[float, float]:

    texture = _verify_finite_f64(texture_size, "texture_size", (2,))

    if np.any(texture < 2.0):
        raise ValueError("Texture width and height must both be at least 2.")

    return float(texture[0]), float(texture[1])


def _project_coords(
    coords: np.ndarray,
    proj_plane: UVProjPlaneLike,
) -> np.ndarray:

    if proj_plane is EUVProjPlane.XY:
        return coords[:, :2]

    if proj_plane is EUVProjPlane.YZ:
        return coords[:, 1:3]

    if proj_plane is EUVProjPlane.XZ:
        return coords[:, (0, 2)]

    if isinstance(proj_plane, EUVProjPlane):
        raise ValueError(f"Unsupported projection plane: {proj_plane}.")

    if isinstance(proj_plane, UVProjPlane):
        normal_in = proj_plane.normal
        origin_in = proj_plane.origin
    else:
        try:
            normal_in, origin_in = proj_plane
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A custom projection plane must be (normal, origin).",
            ) from error

    normal = _verify_finite_f64(normal_in, "Projection normal", (3,))
    origin = _verify_finite_f64(origin_in, "Projection origin", (3,))
    normal_norm = np.linalg.norm(normal)

    if normal_norm == 0.0:
        raise ValueError("Projection normal must be nonzero.")

    normal = normal / normal_norm

    if abs(normal[2]) < _Z_PARALLEL_TOL:
        u_axis = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    else:
        u_axis = np.cross(normal, np.array((0.0, 1.0, 0.0)))

    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    difference = coords - origin
    return np.column_stack(
        (difference @ u_axis, difference @ v_axis),
    )


def _calc_proj_bounds(
    projected: np.ndarray,
) -> UVProjBounds2D:

    minimum = np.min(projected, axis=0)
    maximum = np.max(projected, axis=0)
    extent = maximum - minimum

    if np.any(extent <= 0.0):
        raise ValueError("Projected mesh has zero area in the chosen plane.")

    return UVProjBounds2D(
        x_min=float(minimum[0]),
        x_max=float(maximum[0]),
        y_min=float(minimum[1]),
        y_max=float(maximum[1]),
    )


def _calc_uvs_from_proj(
    projected: np.ndarray,
    bounds: UVProjBounds2D,
    texture_size: tuple[float, float],
    px_bbox: UVPixelBBox,
    mode: EUVPlanarProjMode,
) -> np.ndarray:

    scale_x = (
        (px_bbox.x_upper - px_bbox.x_lower) / (bounds.x_max - bounds.x_min)
    )
    scale_y = (
        (px_bbox.y_upper - px_bbox.y_lower) / (bounds.y_max - bounds.y_min)
    )

    if mode is EUVPlanarProjMode.FIT_X:
        scale = scale_x
    elif mode is EUVPlanarProjMode.FIT_Y:
        scale = scale_y
    elif mode is EUVPlanarProjMode.BEST:
        scale = min(scale_x, scale_y)
    else:
        raise ValueError(f"Unsupported planar projection mode: {mode}.")

    pixel_center = 0.5 * np.array(
        (px_bbox.x_lower + px_bbox.x_upper, px_bbox.y_lower + px_bbox.y_upper),
    )

    mesh_center = 0.5 * np.array(
        (bounds.x_min + bounds.x_max, bounds.y_min + bounds.y_max),
    )

    pixels = pixel_center + (projected - mesh_center) * scale
    texture_width, texture_height = texture_size

    uvs = np.empty((projected.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = pixels[:, 0] / (texture_width - 1.0)
    uvs[:, 1] = 1.0 - pixels[:, 1] / (texture_height - 1.0)

    return uvs


def project_uvs_planar_bbox(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    px_bbox: UVPixelBBoxLike,
    proj_plane: UVProjPlaneLike,
    mode: EUVPlanarProjMode = EUVPlanarProjMode.BEST,
) -> np.ndarray:

    coords_in = _verify_coords(coords)
    texture_size_in = _verify_texture_size(texture_size)

    if not isinstance(mode, EUVPlanarProjMode):
        raise ValueError(f"Unsupported planar projection mode: {mode}.")

    if isinstance(px_bbox, UVPixelBBox):
        px_x_lower = float(px_bbox.x_lower)
        px_y_lower = float(px_bbox.y_lower)
        px_x_upper = float(px_bbox.x_upper)
        px_y_upper = float(px_bbox.y_upper)
        bounds = (px_x_lower, px_y_lower, px_x_upper, px_y_upper)
        _ = _verify_finite_f64(bounds, "px_bbox", (4,))
    else:
        px_bounds = _verify_finite_f64(px_bbox, "px_bbox", (4,))
        px_x_lower = float(px_bounds[0])
        px_y_lower = float(px_bounds[1])
        px_x_upper = float(px_bounds[2])
        px_y_upper = float(px_bounds[3])

    if px_x_upper <= px_x_lower or px_y_upper <= px_y_lower:
        raise ValueError("px_bbox upper bounds must exceed lower bounds.")

    bbox = UVPixelBBox(
        x_lower=px_x_lower,
        y_lower=px_y_lower,
        x_upper=px_x_upper,
        y_upper=px_y_upper,
    )

    projected = _project_coords(coords_in, proj_plane)
    bounds_2d = _calc_proj_bounds(projected)

    return _calc_uvs_from_proj(
        projected, bounds_2d, texture_size_in, bbox, mode,
    )


def project_uvs_planar_centered(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    uv_span_max: float = 1.0,
    proj_plane: UVProjPlaneLike = EUVProjPlane.XY,
) -> np.ndarray:

    coords_in = _verify_coords(coords)
    texture_width, texture_height = _verify_texture_size(texture_size)

    if not np.isfinite(uv_span_max) or not 0.0 < uv_span_max <= 1.0:
        raise ValueError(
            "uv_span_max must be finite and in the interval (0, 1].",
        )

    projected = _project_coords(coords_in, proj_plane)
    bounds = _calc_proj_bounds(projected)
    aspect_ratio_ratio = (
        (bounds.x_max - bounds.x_min) / (bounds.y_max - bounds.y_min)
        / (texture_width / texture_height)
    )

    if aspect_ratio_ratio > 1.0:
        u_span = uv_span_max
        v_span = u_span / aspect_ratio_ratio
        mode = EUVPlanarProjMode.FIT_X
    else:
        v_span = uv_span_max
        u_span = v_span * aspect_ratio_ratio
        mode = EUVPlanarProjMode.FIT_Y

    u_min = 0.5 * (1.0 - u_span)
    v_min = 0.5 * (1.0 - v_span)

    px_bbox = UVPixelBBox(
        x_lower=u_min * (texture_width - 1.0),
        y_lower=v_min * (texture_height - 1.0),
        x_upper=(1.0 - u_min) * (texture_width - 1.0),
        y_upper=(1.0 - v_min) * (texture_height - 1.0),
    )

    return _calc_uvs_from_proj(
        projected, bounds, (texture_width, texture_height), px_bbox, mode,
    )


__all__ = [
    "EUVPlanarProjMode",
    "EUVProjPlane",
    "UVPixelBBox",
    "UVProjAxes",
    "UVProjBounds2D",
    "UVProjPlane",
    "project_uvs_planar_bbox",
    "project_uvs_planar_centered",
]

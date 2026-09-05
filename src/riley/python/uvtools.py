# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from riley.python._verifio import _validate_coords, _validate_finite_f64


class EProjPlane(Enum):
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EPlanarProjMode(Enum):
    BEST = "best"
    FIT_X = "fit_x"
    FIT_Y = "fit_y"


@dataclass(frozen=True, slots=True)
class ProjPlane:
    normal: np.ndarray
    origin: np.ndarray


ProjPlaneLike = EProjPlane | ProjPlane | tuple[
    np.ndarray, np.ndarray
]


def _validate_texture_size(
    texture_size: tuple[int, int] | tuple[float, float],
) -> tuple[float, float]:

    texture = _validate_finite_f64(texture_size, "texture_size", (2,))

    if np.any(texture < 2.0):
        raise ValueError("Texture width and height must both be at least 2.")

    return float(texture[0]), float(texture[1])


def _resolve_proj_axes(
    proj_plane: ProjPlaneLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    zero = np.zeros(3, dtype=np.float64)

    if proj_plane is EProjPlane.XY:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))

    if proj_plane is EProjPlane.YZ:
        return zero, np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0))

    if proj_plane is EProjPlane.XZ:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0))

    if isinstance(proj_plane, EProjPlane):
        raise ValueError(f"Unsupported projection plane: {proj_plane}.")

    if isinstance(proj_plane, ProjPlane):
        normal_in = proj_plane.normal
        origin_in = proj_plane.origin
    else:
        try:
            normal_in, origin_in = proj_plane
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A custom projection plane must be (normal, origin).",
            ) from error

    normal = _validate_finite_f64(normal_in, "Projection normal", (3,))
    origin = _validate_finite_f64(origin_in, "Projection origin", (3,))
    normal_norm = np.linalg.norm(normal)

    if normal_norm == 0.0:
        raise ValueError("Projection normal must be nonzero.")

    normal = normal / normal_norm

    if abs(normal[2]) < 0.999:
        u_axis = np.cross(np.array((0.0, 0.0, 1.0)), normal)
    else:
        u_axis = np.cross(normal, np.array((0.0, 1.0, 0.0)))

    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    return origin, u_axis, v_axis


def _project_coords(
    coords: np.ndarray,
    proj_plane: ProjPlaneLike,
) -> np.ndarray:

    if proj_plane is EProjPlane.XY:
        return coords[:, :2]

    if proj_plane is EProjPlane.YZ:
        return coords[:, 1:3]

    if proj_plane is EProjPlane.XZ:
        return coords[:, (0, 2)]

    origin, u_axis, v_axis = _resolve_proj_axes(proj_plane)
    difference = coords - origin

    return np.column_stack((difference @ u_axis, difference @ v_axis))


def _proj_bounds(
    projected: np.ndarray,
) -> tuple[float, float, float, float]:
    minimum = np.min(projected, axis=0)
    maximum = np.max(projected, axis=0)
    extent = maximum - minimum
    if np.any(extent <= 0.0):
        raise ValueError("Projected mesh has zero area in the chosen plane.")
    return (
        float(minimum[0]), float(maximum[0]),
        float(minimum[1]), float(maximum[1]),
    )


def _uvs_from_proj(
    projected: np.ndarray,
    texture_size: tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    mode: EPlanarProjMode,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = _proj_bounds(projected)
    px_bounds = _validate_finite_f64(px_bbox, "px_bbox", (4,))
    px_x_lower, px_y_lower, px_x_upper, px_y_upper = px_bounds
    if px_x_upper <= px_x_lower or px_y_upper <= px_y_lower:
        raise ValueError("px_bbox upper bounds must exceed lower bounds.")
    scale_x = (px_x_upper - px_x_lower) / (x_max - x_min)
    scale_y = (px_y_upper - px_y_lower) / (y_max - y_min)
    if mode is EPlanarProjMode.FIT_X:
        scale = scale_x
    elif mode is EPlanarProjMode.FIT_Y:
        scale = scale_y
    elif mode is EPlanarProjMode.BEST:
        scale = min(scale_x, scale_y)
    else:
        raise ValueError(f"Unsupported planar projection mode: {mode}.")
    pixel_center = 0.5 * np.array(
        (px_x_lower + px_x_upper, px_y_lower + px_y_upper),
    )
    mesh_center = 0.5 * np.array((x_min + x_max, y_min + y_max))
    pixels = pixel_center + (projected - mesh_center) * scale
    texture_width, texture_height = texture_size
    uvs = np.empty((projected.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = pixels[:, 0] / (texture_width - 1.0)
    uvs[:, 1] = 1.0 - pixels[:, 1] / (texture_height - 1.0)
    return uvs


def project_uvs_planar_bbox(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    proj_plane: ProjPlaneLike,
    mode: EPlanarProjMode = EPlanarProjMode.BEST,
) -> np.ndarray:
    coords_in = _validate_coords(coords, contiguous_f64=True)
    texture_size_in = _validate_texture_size(texture_size)
    projected = _project_coords(coords_in, proj_plane)
    return _uvs_from_proj(projected, texture_size_in, px_bbox, mode)


def project_uvs_planar_centered(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    uv_span_max: float = 1.0,
    proj_plane: ProjPlaneLike = EProjPlane.XY,
) -> np.ndarray:
    coords_in = _validate_coords(coords, contiguous_f64=True)
    texture_width, texture_height = _validate_texture_size(texture_size)
    if not np.isfinite(uv_span_max) or not 0.0 < uv_span_max <= 1.0:
        raise ValueError(
            "uv_span_max must be finite and in the interval (0, 1].",
        )
    projected = _project_coords(coords_in, proj_plane)
    x_min, x_max, y_min, y_max = _proj_bounds(projected)
    aspect_ratio_ratio = (
        (x_max - x_min) / (y_max - y_min)
        / (texture_width / texture_height)
    )
    if aspect_ratio_ratio > 1.0:
        u_span = uv_span_max
        v_span = u_span / aspect_ratio_ratio
        mode = EPlanarProjMode.FIT_X
    else:
        v_span = uv_span_max
        u_span = v_span * aspect_ratio_ratio
        mode = EPlanarProjMode.FIT_Y
    u_min = 0.5 * (1.0 - u_span)
    v_min = 0.5 * (1.0 - v_span)
    px_bbox = (
        u_min * (texture_width - 1.0),
        v_min * (texture_height - 1.0),
        (1.0 - u_min) * (texture_width - 1.0),
        (1.0 - v_min) * (texture_height - 1.0),
    )
    return _uvs_from_proj(
        projected, (texture_width, texture_height), px_bbox, mode,
    )


__all__ = [
    "EPlanarProjMode", "EProjPlane", "ProjPlane",
    "project_uvs_planar_bbox", "project_uvs_planar_centered",
]

# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Planar UV projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EProjectionPlane(Enum):
    """Axis-aligned projection plane."""
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EPlanarProjectionMode(Enum):
    """Rule used to scale a projection into a pixel bounding box."""
    BEST = "best"
    FIT_X = "fit_x"
    FIT_Y = "fit_y"


@dataclass(frozen=True, slots=True)
class ProjectionPlane:
    """Arbitrary plane described by its normal and origin."""
    normal: np.ndarray
    origin: np.ndarray


ProjectionPlaneLike = EProjectionPlane | ProjectionPlane | tuple[
    np.ndarray, np.ndarray
]


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    """Return validated three-dimensional coordinates."""
    coords_in = np.ascontiguousarray(coords, dtype=np.float64)
    if coords_in.ndim != 2 or coords_in.shape[1] != 3 or not coords_in.shape[0]:
        raise ValueError("coords must have shape (nodes, 3) and not be empty.")
    if not np.all(np.isfinite(coords_in)):
        raise ValueError("coords must contain only finite values.")
    return coords_in


def _validate_texture_size(
    texture_size: tuple[int, int] | tuple[float, float],
) -> tuple[float, float]:
    """Return a validated texture width and height."""
    texture = np.asarray(texture_size, dtype=np.float64)
    if texture.shape != (2,) or not np.all(np.isfinite(texture)):
        raise ValueError("texture_size must contain two finite values.")
    if np.any(texture < 2.0):
        raise ValueError("Texture width and height must both be at least 2.")
    return float(texture[0]), float(texture[1])


def _resolve_projection_axes(
    projection_plane: ProjectionPlaneLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a projection plane to an origin and orthonormal basis."""
    zero = np.zeros(3, dtype=np.float64)
    if projection_plane is EProjectionPlane.XY:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    if projection_plane is EProjectionPlane.YZ:
        return zero, np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0))
    if projection_plane is EProjectionPlane.XZ:
        return zero, np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0))
    if isinstance(projection_plane, EProjectionPlane):
        raise ValueError(f"Unsupported projection plane: {projection_plane}.")

    if isinstance(projection_plane, ProjectionPlane):
        normal_in = projection_plane.normal
        origin_in = projection_plane.origin
    else:
        try:
            normal_in, origin_in = projection_plane
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A custom projection plane must be (normal, origin).",
            ) from error
    normal = np.asarray(normal_in, dtype=np.float64)
    origin = np.asarray(origin_in, dtype=np.float64)
    if normal.shape != (3,) or origin.shape != (3,):
        raise ValueError("Projection normal and origin must have shape (3,).")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(origin)):
        raise ValueError("Projection normal and origin must be finite.")
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
    projection_plane: ProjectionPlaneLike,
) -> np.ndarray:
    """Project coordinates onto a two-dimensional plane."""
    if projection_plane is EProjectionPlane.XY:
        return coords[:, :2]
    if projection_plane is EProjectionPlane.YZ:
        return coords[:, 1:3]
    if projection_plane is EProjectionPlane.XZ:
        return coords[:, (0, 2)]
    origin, u_axis, v_axis = _resolve_projection_axes(projection_plane)
    difference = coords - origin
    return np.column_stack((difference @ u_axis, difference @ v_axis))


def _projection_bounds(
    projected: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return finite projection bounds, rejecting zero-area projections."""
    minimum = np.min(projected, axis=0)
    maximum = np.max(projected, axis=0)
    extent = maximum - minimum
    if np.any(extent <= 0.0):
        raise ValueError("Projected mesh has zero area in the chosen plane.")
    return (
        float(minimum[0]), float(maximum[0]),
        float(minimum[1]), float(maximum[1]),
    )


def _uvs_from_projection(
    projected: np.ndarray,
    texture_size: tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    mode: EPlanarProjectionMode,
) -> np.ndarray:
    """Map projected coordinates into a pixel bounding box."""
    x_min, x_max, y_min, y_max = _projection_bounds(projected)
    px_bounds = np.asarray(px_bbox, dtype=np.float64)
    if px_bounds.shape != (4,) or not np.all(np.isfinite(px_bounds)):
        raise ValueError("px_bbox must contain four finite values.")
    px_x_lower, px_y_lower, px_x_upper, px_y_upper = px_bounds
    if px_x_upper <= px_x_lower or px_y_upper <= px_y_lower:
        raise ValueError("px_bbox upper bounds must exceed lower bounds.")
    scale_x = (px_x_upper - px_x_lower) / (x_max - x_min)
    scale_y = (px_y_upper - px_y_lower) / (y_max - y_min)
    if mode is EPlanarProjectionMode.FIT_X:
        scale = scale_x
    elif mode is EPlanarProjectionMode.FIT_Y:
        scale = scale_y
    elif mode is EPlanarProjectionMode.BEST:
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
    projection_plane: ProjectionPlaneLike,
    mode: EPlanarProjectionMode = EPlanarProjectionMode.BEST,
) -> np.ndarray:
    """Project coordinates into a texture-space pixel bounding box."""
    coords_in = _validate_coords(coords)
    texture_size_in = _validate_texture_size(texture_size)
    projected = _project_coords(coords_in, projection_plane)
    return _uvs_from_projection(projected, texture_size_in, px_bbox, mode)


def project_uvs_planar_centered(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    uv_span_max: float = 1.0,
    projection_plane: ProjectionPlaneLike = EProjectionPlane.XY,
) -> np.ndarray:
    """Project coordinates into a centered, aspect-preserving UV region."""
    coords_in = _validate_coords(coords)
    texture_width, texture_height = _validate_texture_size(texture_size)
    if not np.isfinite(uv_span_max) or not 0.0 < uv_span_max <= 1.0:
        raise ValueError(
            "uv_span_max must be finite and in the interval (0, 1].",
        )
    projected = _project_coords(coords_in, projection_plane)
    x_min, x_max, y_min, y_max = _projection_bounds(projected)
    aspect_ratio_ratio = (
        (x_max - x_min) / (y_max - y_min)
        / (texture_width / texture_height)
    )
    if aspect_ratio_ratio > 1.0:
        u_span = uv_span_max
        v_span = u_span / aspect_ratio_ratio
        mode = EPlanarProjectionMode.FIT_X
    else:
        v_span = uv_span_max
        u_span = v_span * aspect_ratio_ratio
        mode = EPlanarProjectionMode.FIT_Y
    u_min = 0.5 * (1.0 - u_span)
    v_min = 0.5 * (1.0 - v_span)
    px_bbox = (
        u_min * (texture_width - 1.0),
        v_min * (texture_height - 1.0),
        (1.0 - u_min) * (texture_width - 1.0),
        (1.0 - v_min) * (texture_height - 1.0),
    )
    return _uvs_from_projection(
        projected, (texture_width, texture_height), px_bbox, mode,
    )


__all__ = [
    "EPlanarProjectionMode", "EProjectionPlane", "ProjectionPlane",
    "project_uvs_planar_bbox", "project_uvs_planar_centered",
]

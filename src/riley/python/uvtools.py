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

from riley.python._verifio import _validate_coords, _validate_finite_f64


class EProjPlane(Enum):
    """Axis-aligned plane used for planar UV projection.

    Attributes
    ----------
    XY : str
        Project using the x and y coordinate axes.
    YZ : str
        Project using the y and z coordinate axes.
    XZ : str
        Project using the x and z coordinate axes.
    """
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EPlanarProjMode(Enum):
    """Scaling rule for fitting projected coordinates into pixel bounds.

    Attributes
    ----------
    BEST : str
        Use the smaller axis scale so the complete projection fits.
    FIT_X : str
        Fit the projected x extent to the horizontal pixel bounds.
    FIT_Y : str
        Fit the projected y extent to the vertical pixel bounds.
    """
    BEST = "best"
    FIT_X = "fit_x"
    FIT_Y = "fit_y"


@dataclass(frozen=True, slots=True)
class ProjPlane:
    """Arbitrary plane used for planar UV projection.

    Parameters
    ----------
    normal : numpy.ndarray
        Nonzero three-component plane normal.
    origin : numpy.ndarray
        Three-component point defining the projection-plane origin.

    Notes
    -----
    Riley calculates a deterministic orthonormal basis from ``normal``. The
    normal does not need to be normalized by the caller.
    """
    normal: np.ndarray
    origin: np.ndarray


ProjPlaneLike = EProjPlane | ProjPlane | tuple[
    np.ndarray, np.ndarray
]


def _validate_texture_size(
    texture_size: tuple[int, int] | tuple[float, float],
) -> tuple[float, float]:
    """Return a validated texture width and height."""

    texture = _validate_finite_f64(texture_size, "texture_size", (2,))

    if np.any(texture < 2.0):
        raise ValueError("Texture width and height must both be at least 2.")

    return float(texture[0]), float(texture[1])


def _resolve_proj_axes(
    proj_plane: ProjPlaneLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a projection plane to an origin and orthonormal basis."""

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
    """Project coordinates onto a two-dimensional plane."""

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


def _uvs_from_proj(
    projected: np.ndarray,
    texture_size: tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    mode: EPlanarProjMode,
) -> np.ndarray:
    """Map projected coordinates into a pixel bounding box."""
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
    """Project mesh coordinates into a texture-space pixel bounding box.

    Parameters
    ----------
    coords : numpy.ndarray
        Finite mesh coordinates with shape ``(nodes, 3)``.
    texture_size : tuple[int, int] or tuple[float, float]
        Texture width and height in pixels. Both dimensions must be at least
        two pixels.
    px_bbox : tuple[float, float, float, float]
        Lower x, lower y, upper x and upper y pixel coordinates. Upper bounds
        must exceed their corresponding lower bounds.
    proj_plane : EProjPlane, ProjPlane or tuple[numpy.ndarray, numpy.ndarray]
        Axis-aligned plane or a custom ``(normal, origin)`` plane.
    mode : EPlanarProjMode, optional
        Rule used to fit the projected mesh into ``px_bbox``. The default is
        :attr:`EPlanarProjMode.BEST`.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 UV coordinates with shape ``(nodes, 2)``.

    Raises
    ------
    ValueError
        If an input has an invalid shape, contains non-finite values, defines
        a degenerate projection, or specifies invalid pixel bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from riley.python.uvtools import EProjPlane
    >>> from riley.python.uvtools import project_uvs_planar_bbox
    >>> coords = np.array(((0., 0., 0.), (2., 0., 0.),
    ...                    (2., 1., 0.), (0., 1., 0.)))
    >>> uvs = project_uvs_planar_bbox(
    ...     coords, (201, 101), (0., 0., 200., 100.), EProjPlane.XY,
    ... )
    >>> uvs.shape
    (4, 2)
    >>> np.all((uvs >= 0.) & (uvs <= 1.))
    np.True_
    """
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
    """Project coordinates into a centered, aspect-preserving UV region.

    Parameters
    ----------
    coords : numpy.ndarray
        Finite mesh coordinates with shape ``(nodes, 3)``.
    texture_size : tuple[int, int] or tuple[float, float]
        Texture width and height in pixels. Both dimensions must be at least
        two pixels.
    uv_span_max : float, optional
        Maximum normalized span used by either UV axis. Must lie in ``(0, 1]``.
        The default is ``1.0``.
    proj_plane : EProjPlane, ProjPlane or tuple of numpy.ndarray, optional
        Axis-aligned plane or a custom ``(normal, origin)`` plane. The default
        is :attr:`EProjPlane.XY`.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 UV coordinates with shape ``(nodes, 2)``.

    Raises
    ------
    ValueError
        If an input is invalid or the selected projection has zero area.

    Examples
    --------
    >>> import numpy as np
    >>> from riley.python.uvtools import project_uvs_planar_centered
    >>> coords = np.array(((0., 0., 0.), (2., 0., 0.),
    ...                    (2., 1., 0.), (0., 1., 0.)))
    >>> uvs = project_uvs_planar_centered(coords, (201, 101), 0.8)
    >>> np.round(np.ptp(uvs, axis=0), 2)
    array([0.8, 0.8])
    """
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

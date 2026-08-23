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

from enum import Enum

import numpy as np


class EProjectionPlane(Enum):
    XY = "xy"
    YZ = "yz"
    XZ = "xz"


class EPlanarProjectionMode(Enum):
    BEST = "best"
    FIT_X = "fit_x"
    FIT_Y = "fit_y"


def project_uvs_planar_bbox(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    projection_plane: EProjectionPlane | tuple[np.ndarray, np.ndarray],
    *,
    mode: EPlanarProjectionMode = EPlanarProjectionMode.BEST,
) -> np.ndarray:
    coords_in = np.ascontiguousarray(coords, dtype=np.float64)
    origin, u_axis, v_axis = _resolve_projection_axes(projection_plane)

    diff = coords_in - origin
    x_proj = diff @ u_axis
    y_proj = diff @ v_axis

    x_min = np.min(x_proj)
    x_max = np.max(x_proj)
    y_min = np.min(y_proj)
    y_max = np.max(y_proj)

    mesh_w = x_max - x_min
    mesh_h = y_max - y_min

    px_x_l, px_y_l, px_x_u, px_y_u = px_bbox
    px_w = px_x_u - px_x_l
    px_h = px_y_u - px_y_l

    scale_x = px_w / mesh_w if mesh_w > 0.0 else 1.0
    scale_y = px_h / mesh_h if mesh_h > 0.0 else 1.0

    if mode == EPlanarProjectionMode.FIT_X:
        scale = scale_x
    elif mode == EPlanarProjectionMode.FIT_Y:
        scale = scale_y
    elif mode == EPlanarProjectionMode.BEST:
        scale = 0.5 * (scale_x + scale_y)
    else:
        raise ValueError(f"Unsupported planar projection mode: {mode}.")

    mesh_cx = 0.5 * (x_min + x_max)
    mesh_cy = 0.5 * (y_min + y_max)
    px_cx = 0.5 * (px_x_l + px_x_u)
    px_cy = 0.5 * (px_y_l + px_y_u)

    px_x = px_cx + (x_proj - mesh_cx) * scale
    px_y = px_cy + (y_proj - mesh_cy) * scale

    tex_w, tex_h = texture_size
    uvs = np.zeros((coords_in.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = px_x / float(tex_w - 1.0)
    uvs[:, 1] = 1.0 - (px_y / float(tex_h - 1.0))
    return np.ascontiguousarray(uvs, dtype=np.float64)


def project_uvs_planar_centered(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    *,
    uv_span_max: float = 1.0,
    projection_plane: EProjectionPlane | tuple[np.ndarray, np.ndarray] = (
        EProjectionPlane.XY
    ),
) -> np.ndarray:
    coords_in = np.ascontiguousarray(coords, dtype=np.float64)
    tex_w = float(texture_size[0])
    tex_h = float(texture_size[1])

    if isinstance(projection_plane, EProjectionPlane):
        if projection_plane == EProjectionPlane.XY:
            proj_coords = coords_in[:, :2]
        elif projection_plane == EProjectionPlane.YZ:
            proj_coords = coords_in[:, 1:3]
        elif projection_plane == EProjectionPlane.XZ:
            proj_coords = coords_in[:, (0, 2)]
        else:
            raise ValueError(f"Unsupported projection plane: {projection_plane}.")
    else:
        origin, u_axis, v_axis = _resolve_projection_axes(projection_plane)
        diff = coords_in - origin
        proj_coords = np.column_stack((diff @ u_axis, diff @ v_axis))

    x_min = np.min(proj_coords[:, 0])
    x_max = np.max(proj_coords[:, 0])
    y_min = np.min(proj_coords[:, 1])
    y_max = np.max(proj_coords[:, 1])

    mesh_w = x_max - x_min
    mesh_h = y_max - y_min
    if mesh_w <= 0.0 or mesh_h <= 0.0:
        raise ValueError("Projected mesh has zero area in the chosen plane.")

    mesh_ar = mesh_w / mesh_h
    tex_ar = tex_w / tex_h
    aspect_ratio_ratio = mesh_ar / tex_ar

    if aspect_ratio_ratio > 1.0:
        d_u = uv_span_max
        d_v = d_u / aspect_ratio_ratio
        mode = EPlanarProjectionMode.FIT_X
    else:
        d_v = uv_span_max
        d_u = d_v * aspect_ratio_ratio
        mode = EPlanarProjectionMode.FIT_Y

    u_min = 0.5 * (1.0 - d_u)
    u_max = 1.0 - u_min
    v_min = 0.5 * (1.0 - d_v)
    v_max = 1.0 - v_min

    px_bbox = (
        u_min * (tex_w - 1.0),
        (1.0 - v_max) * (tex_h - 1.0),
        u_max * (tex_w - 1.0),
        (1.0 - v_min) * (tex_h - 1.0),
    )
    return project_uvs_planar_bbox(
        coords_in,
        texture_size,
        px_bbox,
        projection_plane,
        mode=mode,
    )


def _resolve_projection_axes(
    projection_plane: EProjectionPlane | tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(projection_plane, EProjectionPlane):
        if projection_plane == EProjectionPlane.XY:
            origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
            u_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
            v_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        elif projection_plane == EProjectionPlane.YZ:
            origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
            u_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
            v_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        elif projection_plane == EProjectionPlane.XZ:
            origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
            u_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
            v_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported projection plane: {projection_plane}.")
        return origin, u_axis, v_axis

    normal, origin_in = projection_plane
    normal_vec = np.asarray(normal, dtype=np.float64)
    origin = np.asarray(origin_in, dtype=np.float64)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    if np.abs(normal_vec[2]) < 0.999:
        u_axis = np.cross(
            np.array((0.0, 0.0, 1.0), dtype=np.float64),
            normal_vec,
        )
    else:
        u_axis = np.cross(
            normal_vec,
            np.array((0.0, 1.0, 0.0), dtype=np.float64),
        )
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal_vec, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    return origin, u_axis, v_axis


__all__ = [
    "EProjectionPlane",
    "EPlanarProjectionMode",
    "project_uvs_planar_bbox",
    "project_uvs_planar_centered",
]
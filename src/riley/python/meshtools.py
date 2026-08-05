# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np

from riley.python.enums import (
    ConnectIndexing,
    PlanarProjectionMode,
    ProjectionPlane,
)


_SURFACE_NODE_COUNTS = frozenset((3, 4, 6, 7, 8, 9))
_TOL = 1.0e-12


def enforce_mesh_convention(
    coords: np.ndarray,
    connect: np.ndarray,
    *,
    indexing: ConnectIndexing = ConnectIndexing.auto,
) -> tuple[np.ndarray, np.ndarray]:
    coords_out = np.ascontiguousarray(coords, dtype=np.float64)
    connect_out = np.ascontiguousarray(connect, dtype=np.int64)

    if indexing == ConnectIndexing.one_based:
        connect_out = connect_out - 1
    elif indexing == ConnectIndexing.auto and _needs_zero_based_shift(
        connect_out,
        coords_out.shape[0],
    ):
        connect_out = connect_out - 1

    if not _check_indices_zero_based(connect_out, coords_out.shape[0]):
        raise ValueError("Connectivity contains indices outside the coordinate array.")

    connect_out = _enforce_right_handed_table(
        _enforce_ccw_winding_table(connect_out, coords_out),
        coords_out,
    )

    return coords_out, np.ascontiguousarray(connect_out, dtype=np.uintp)


def is_mesh_2d(coords: np.ndarray, connect: np.ndarray) -> bool:
    # 1. Check coordinate flatness
    coord_ranges = np.ptp(coords, axis=0)
    if np.any(coord_ranges < 1e-12):
        return True

    # 2. Check element nodes
    nodes_per_elem = connect.shape[1]
    if nodes_per_elem in (3, 6, 7, 9):
        return True
    if nodes_per_elem in (10, 20, 27):
        return False

    if nodes_per_elem == 4:
        num_check = min(10, connect.shape[0])
        is_tet = False
        for i in range(num_check):
            elem = connect[i]
            v = coords[elem]
            vol = np.abs(
                np.dot(v[1] - v[0], np.cross(v[2] - v[0], v[3] - v[0]))
            )
            if vol > 1e-10:
                is_tet = True
                break
        if not is_tet:
            return True

    if nodes_per_elem == 8:
        num_check = min(10, connect.shape[0])
        is_hex = False
        for i in range(num_check):
            elem = connect[i]
            v = coords[elem]
            vol = np.abs(
                np.dot(v[1] - v[0], np.cross(v[2] - v[0], v[4] - v[0]))
            )
            if vol > 1e-10:
                is_hex = True
                break
        if not is_hex:
            return True

    return False


def extract_surface_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
    *,
    indexing: ConnectIndexing = ConnectIndexing.auto,
    enforce_convention: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    coords_norm, connect_norm = enforce_mesh_convention(
        coords,
        connect,
        indexing=indexing,
    )
    connect_work = np.ascontiguousarray(connect_norm, dtype=np.int64)

    if is_mesh_2d(coords_norm, connect_norm):
        raise ValueError(
            "Surface extraction is only supported for 3D meshes. "
            "The provided mesh appears to be 2D."
        )

    if connect_work.shape[1] not in (4, 8, 10, 20, 27):
        raise NotImplementedError(
            "Surface extraction is only implemented for tet and hex element "
            "families.",
        )

    surf_faces, _ = _extract_surface_faces_from_table(connect_work, coords_norm)
    surf_node_inds = np.unique(surf_faces)
    surf_coords = np.ascontiguousarray(coords_norm[surf_node_inds], dtype=np.float64)

    coord_remap = np.full(coords_norm.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_inds] = np.arange(surf_node_inds.shape[0], dtype=np.int64)
    surf_connect = coord_remap[surf_faces]

    if enforce_convention:
        surf_coords, surf_connect = enforce_mesh_convention(
            surf_coords,
            surf_connect,
            indexing=ConnectIndexing.zero_based,
        )

    return surf_coords, np.ascontiguousarray(surf_connect, dtype=np.uintp)


def project_uvs_planar_bbox(
    coords: np.ndarray,
    texture_size: tuple[int, int] | tuple[float, float],
    px_bbox: tuple[float, float, float, float],
    projection_plane: ProjectionPlane | tuple[np.ndarray, np.ndarray],
    *,
    mode: PlanarProjectionMode = PlanarProjectionMode.best,
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

    if mode == PlanarProjectionMode.fit_x:
        scale = scale_x
    elif mode == PlanarProjectionMode.fit_y:
        scale = scale_y
    elif mode == PlanarProjectionMode.best:
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
    projection_plane: ProjectionPlane | tuple[np.ndarray, np.ndarray] = (
        ProjectionPlane.xy
    ),
) -> np.ndarray:
    coords_in = np.ascontiguousarray(coords, dtype=np.float64)
    tex_w = float(texture_size[0])
    tex_h = float(texture_size[1])

    if isinstance(projection_plane, ProjectionPlane):
        if projection_plane == ProjectionPlane.xy:
            proj_coords = coords_in[:, :2]
        elif projection_plane == ProjectionPlane.yz:
            proj_coords = coords_in[:, 1:3]
        elif projection_plane == ProjectionPlane.xz:
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
        mode = PlanarProjectionMode.fit_x
    else:
        d_v = uv_span_max
        d_u = d_v * aspect_ratio_ratio
        mode = PlanarProjectionMode.fit_y

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
    projection_plane: ProjectionPlane | tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(projection_plane, ProjectionPlane):
        if projection_plane == ProjectionPlane.xy:
            origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
            u_axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
            v_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        elif projection_plane == ProjectionPlane.yz:
            origin = np.array((0.0, 0.0, 0.0), dtype=np.float64)
            u_axis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
            v_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        elif projection_plane == ProjectionPlane.xz:
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


def _needs_zero_based_shift(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return False
    if np.any(connect < 0):
        return False
    if np.any(connect == 0):
        return False
    return bool(np.any(connect >= num_coords))


def _check_indices_zero_based(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return True
    return bool(np.all((connect >= 0) & (connect < num_coords)))


def _get_corner_indices(nodes_per_elem: int) -> np.ndarray:
    if nodes_per_elem in (3, 6, 7):
        return np.array((0, 1, 2), dtype=np.int64)
    if nodes_per_elem in (4, 8, 9):
        return np.array((0, 1, 2, 3), dtype=np.int64)
    if nodes_per_elem == 10:
        return np.array((0, 1, 2, 3), dtype=np.int64)
    if nodes_per_elem in (20, 27):
        return np.array((0, 1, 2, 3, 4, 5, 6, 7), dtype=np.int64)
    raise NotImplementedError(
        f"Unsupported element type with {nodes_per_elem} nodes.",
    )


def _get_volume_corner_indices(nodes_per_elem: int) -> np.ndarray:
    if nodes_per_elem in (4, 10):
        return np.array((0, 1, 2, 3), dtype=np.int64)
    if nodes_per_elem in (8, 20, 27):
        return np.array((0, 1, 2, 3, 4, 5, 6, 7), dtype=np.int64)
    raise NotImplementedError(
        f"Unsupported volume element with {nodes_per_elem} nodes.",
    )


def _active_coord_axes(coords: np.ndarray) -> np.ndarray:
    axis_range = np.ptp(coords, axis=0)
    active = np.flatnonzero(axis_range > _TOL)
    if active.shape[0] < 2:
        raise ValueError("At least two active coordinate axes are required.")
    return active[:2]


def _polygon_signed_area(coords_elem: np.ndarray) -> float:
    axes = _active_coord_axes(coords_elem)
    xy = coords_elem[:, axes]
    rolled = np.roll(xy, -1, axis=0)
    return float(
        0.5 * np.sum(xy[:, 0] * rolled[:, 1] - rolled[:, 0] * xy[:, 1]),
    )


def _is_coplanar(coords_elem: np.ndarray) -> bool:
    centred = coords_elem - np.mean(coords_elem, axis=0)
    return np.linalg.matrix_rank(centred, tol=_TOL) <= 2


def _tet_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[2] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
            )),
        ),
    )


def _hex_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
                coords_elem[4] - coords_elem[0],
            )),
        ),
    )


def _winding_metric(connect_row: np.ndarray, coords: np.ndarray) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem not in _SURFACE_NODE_COUNTS:
        return None
    corner_inds = _get_corner_indices(nodes_per_elem)
    coords_elem = coords[connect_row[corner_inds]]
    if not _is_coplanar(coords_elem):
        return None
    return _polygon_signed_area(coords_elem)


def _handedness_metric(connect_row: np.ndarray, coords: np.ndarray) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem in _SURFACE_NODE_COUNTS:
        metric = _winding_metric(connect_row, coords)
        if metric is not None:
            return metric

    corner_inds = _get_corner_indices(nodes_per_elem)
    coords_elem = coords[connect_row[corner_inds]]
    if nodes_per_elem in (4, 10):
        return _tet_signed_volume(coords_elem)
    if nodes_per_elem in (8, 20, 27):
        return _hex_signed_volume(coords_elem)

    raise NotImplementedError(
        f"Unsupported handedness check for {nodes_per_elem}-node elements.",
    )


def _reverse_surface_row(connect_row: np.ndarray) -> np.ndarray:
    perms = {
        3: np.array((0, 2, 1)),
        4: np.array((0, 3, 2, 1)),
        6: np.array((0, 2, 1, 5, 4, 3)),
        7: np.array((0, 2, 1, 5, 4, 3, 6)),
        8: np.array((0, 3, 2, 1, 7, 6, 5, 4)),
        9: np.array((0, 3, 2, 1, 7, 6, 5, 4, 8)),
    }
    return connect_row[perms[connect_row.shape[0]]]


def _reverse_handedness_row(connect_row: np.ndarray) -> np.ndarray:
    perms = {
        4: np.array((0, 2, 1, 3)),
        10: np.array((0, 2, 1, 3, 6, 5, 4, 7, 9, 8)),
        8: np.array((0, 3, 2, 1, 4, 7, 6, 5)),
        20: np.array((
            0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
        )),
        27: np.array((
            0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
            20, 21, 22, 23, 24, 25, 26,
        )),
    }
    return connect_row[perms[connect_row.shape[0]]]


def _enforce_ccw_winding_table(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    connect_out = np.copy(connect)
    for row_ind, row in enumerate(connect_out):
        metric = _winding_metric(row, coords)
        if metric is not None and metric < 0.0:
            connect_out[row_ind, :] = _reverse_surface_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    connect_out = np.copy(connect)
    for row_ind, row in enumerate(connect_out):
        metric = _handedness_metric(row, coords)
        if metric is None or metric >= 0.0:
            continue

        row_corners = coords[row[_get_corner_indices(row.shape[0])]]
        if row.shape[0] in _SURFACE_NODE_COUNTS and _is_coplanar(row_corners):
            connect_out[row_ind, :] = _reverse_surface_row(row)
        else:
            connect_out[row_ind, :] = _reverse_handedness_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _get_face_corner_coords(face_coords: np.ndarray) -> np.ndarray:
    if face_coords.shape[0] in (3, 6, 7):
        return face_coords[:3, :]
    if face_coords.shape[0] in (4, 8, 9):
        return face_coords[:4, :]
    raise NotImplementedError(
        f"Unsupported surface face with {face_coords.shape[0]} nodes.",
    )


def _calc_face_normal(face_coords: np.ndarray) -> np.ndarray:
    face_corners = _get_face_corner_coords(face_coords)
    face_normal = np.cross(
        face_corners[1] - face_corners[0],
        face_corners[2] - face_corners[0],
    )
    normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL and face_corners.shape[0] == 4:
        face_normal = np.cross(
            face_corners[2] - face_corners[0],
            face_corners[3] - face_corners[0],
        )
        normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL:
        raise ValueError("Degenerate face detected while extracting a surface.")

    return face_normal / normal_mag


def _orient_surface_face_outward(
    face_connect: np.ndarray,
    parent_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    face_coords = coords[face_connect]
    face_centroid = np.mean(_get_face_corner_coords(face_coords), axis=0)

    parent_corners = _get_volume_corner_indices(parent_connect.shape[0])
    parent_centroid = np.mean(coords[parent_connect[parent_corners]], axis=0)

    face_normal = _calc_face_normal(face_coords)
    outward_dir = face_centroid - parent_centroid

    if np.dot(face_normal, outward_dir) < 0.0:
        return _reverse_surface_row(face_connect)
    return face_connect


def _normalise_surface_face_node_order(
    face_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    nodes_per_face = face_connect.shape[0]
    face_out = np.copy(face_connect)
    face_coords = coords[face_out]

    if nodes_per_face in (6, 7):
        corner_inds = np.array((0, 1, 2), dtype=np.int64)
        midside_pool = np.arange(3, nodes_per_face, dtype=np.int64)
        edge_corner_pairs = ((0, 1), (1, 2), (2, 0))
    elif nodes_per_face in (8, 9):
        corner_inds = np.array((0, 1, 2, 3), dtype=np.int64)
        midside_pool = np.arange(4, nodes_per_face, dtype=np.int64)
        edge_corner_pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    else:
        return face_out

    face_centroid = np.mean(face_coords[corner_inds, :], axis=0)
    mid_pool_coords = face_coords[midside_pool, :]

    if nodes_per_face in (7, 9):
        centroid_dists = np.linalg.norm(mid_pool_coords - face_centroid, axis=1)
        center_pool_ind = int(np.argmin(centroid_dists))
        center_local_ind = int(midside_pool[center_pool_ind])
        edge_pool_mask = np.ones(midside_pool.shape[0], dtype=bool)
        edge_pool_mask[center_pool_ind] = False
        edge_pool_local_inds = midside_pool[edge_pool_mask]
        edge_pool_coords = face_coords[edge_pool_local_inds, :]
    else:
        center_local_ind = -1
        edge_pool_local_inds = midside_pool
        edge_pool_coords = mid_pool_coords

    edge_midpoints = np.array(
        [
            0.5 * (face_coords[start_ind, :] + face_coords[end_ind, :])
            for start_ind, end_ind in edge_corner_pairs
        ],
        dtype=np.float64,
    )
    edge_dists = np.linalg.norm(
        edge_pool_coords[:, None, :] - edge_midpoints[None, :, :],
        axis=2,
    )
    edge_order = np.argmin(edge_dists, axis=0)
    reordered_edge_inds = edge_pool_local_inds[edge_order]

    if nodes_per_face == 6:
        face_out[3:6] = face_out[reordered_edge_inds]
    elif nodes_per_face == 7:
        face_out[3:6] = face_out[reordered_edge_inds]
        face_out[6] = face_out[center_local_ind]
    elif nodes_per_face == 8:
        face_out[4:8] = face_out[reordered_edge_inds]
    elif nodes_per_face == 9:
        face_out[4:8] = face_out[reordered_edge_inds]
        face_out[8] = face_out[center_local_ind]

    return face_out


def _get_surface_map(nodes_per_elem: int) -> np.ndarray:
    if nodes_per_elem == 4:
        return np.array(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)))
    if nodes_per_elem == 8:
        return np.array((
            (0, 1, 2, 3),
            (0, 3, 7, 4),
            (4, 7, 6, 5),
            (1, 5, 6, 2),
            (0, 4, 5, 1),
            (2, 6, 7, 3),
        ))
    if nodes_per_elem == 10:
        return np.array((
            (0, 1, 2, 4, 5, 6),
            (0, 3, 1, 7, 8, 4),
            (0, 2, 3, 6, 9, 7),
            (1, 3, 2, 8, 9, 5),
        ))
    if nodes_per_elem == 20:
        return np.array((
            (0, 1, 2, 3, 8, 9, 10, 11),
            (0, 3, 7, 4, 11, 15, 19, 16),
            (4, 7, 6, 5, 15, 14, 13, 12),
            (1, 5, 6, 2, 17, 13, 18, 9),
            (0, 4, 5, 1, 16, 12, 17, 8),
            (2, 6, 7, 3, 18, 14, 19, 10),
        ))
    if nodes_per_elem == 27:
        return np.array((
            (0, 1, 2, 3, 8, 9, 10, 11, 24),
            (0, 3, 7, 4, 11, 15, 19, 16, 26),
            (4, 7, 6, 5, 15, 14, 13, 12, 25),
            (1, 5, 6, 2, 17, 13, 18, 9, 21),
            (0, 4, 5, 1, 16, 12, 17, 8, 22),
            (2, 6, 7, 3, 18, 14, 19, 10, 20),
        ))
    raise NotImplementedError(
        "Surface extraction is only implemented for tet and hex element families.",
    )


def _extract_surface_faces_from_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodes_per_elem = connect.shape[1]
    face_map = _get_surface_map(nodes_per_elem)
    faces_wound = connect[:, face_map]
    faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
    faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

    _, unique_inds, unique_counts = np.unique(
        faces_flat_sorted,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    ext_face_inds = unique_inds[unique_counts == 1]
    ext_parent_elem_inds = np.ascontiguousarray(
        ext_face_inds // face_map.shape[0],
        dtype=np.int64,
    )
    ext_faces = np.copy(faces_flat_wound[ext_face_inds])

    for face_ind, parent_elem_ind in enumerate(ext_parent_elem_inds):
        ext_faces[face_ind, :] = _orient_surface_face_outward(
            ext_faces[face_ind, :],
            connect[parent_elem_ind, :],
            coords,
        )
        ext_faces[face_ind, :] = _normalise_surface_face_node_order(
            ext_faces[face_ind, :],
            coords,
        )

    return np.ascontiguousarray(ext_faces, dtype=np.int64), ext_parent_elem_inds


__all__ = [
    "enforce_mesh_convention",
    "extract_surface_mesh",
    "project_uvs_planar_bbox",
    "project_uvs_planar_centered",
]

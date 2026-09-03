# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Implementation of Riley's mesh convention tools.

The public interface lives in :mod:`riley.python.meshconv`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import cache
from itertools import permutations as perms, product
from numbers import Integral
from types import MappingProxyType
import numpy as np

from riley.python.meshio import EConnectIndexing


@dataclass(frozen=True, slots=True)
class _Tolerances:
    """Store numerical tolerances used by mesh-convention checks."""

    ref: float = 1.0e-12
    geom: float = 1.0e-12

    role_match: float = 3.5e-1


_TOL = _Tolerances()

# Fixed, irregular directions avoid rays aligned with the axis-aligned faces,
# edges and vertices common in FE meshes. The values are otherwise arbitrary;
# three well-separated directions provide a deterministic majority vote when
# one ray passes through a numerically ambiguous feature.
_POINT_IN_SURF_RAY_DIRECTS = np.array((
    (0.745, 0.371, 0.553),
    (-0.299, 0.877, 0.376),
    (0.461, -0.314, 0.830),
))


def _calc_ref_node_perm(
    ref: np.ndarray,
    transformed: np.ndarray,
) -> tuple[int, ...]:
    """Map transformed reference coordinates back to their source slots."""
    slots: list[int] = []
    for point in transformed:
        matches = np.flatnonzero(
            np.all(np.isclose(ref, point, atol=_TOL.ref), axis=1)
        )
        if matches.shape[0] != 1:
            raise ValueError(
                "Reference transformation does not preserve element roles."
            )
        slots.append(int(matches[0]))
    if len(set(slots)) != ref.shape[0]:
        raise ValueError("Reference transformation is not a node permutation.")
    return tuple(slots)


def _calc_surf_orient_flips(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Return row reversals required by the std surface convention."""

    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    representatives: dict[tuple[int, ...], int] = {}
    duplicate_of = np.arange(connect.shape[0], dtype=np.int64)
    for row_idx, row in enumerate(connect):
        key = tuple(sorted(int(node) for node in row[corner_idxs]))
        representative = representatives.setdefault(key, row_idx)
        duplicate_of[row_idx] = representative

    try:
        topology = _build_surf_topology(connect, np.unique(duplicate_of), spec)
    except ValueError as error:
        if "Non-manifold surface edge" not in str(error):
            raise
        # Surface slices can deliberately contain non-manifold face sets. They
        # have no single orientable shell, so retain the historical per-face
        # behaviour rather than applying a false cavity interpretation.
        return _calc_indep_surf_orient_flips(connect, coords, spec)

    flips = np.zeros(connect.shape[0], dtype=bool)
    closed_comps: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for rows, edge_keys in topology:
        rel = _calc_comp_rel_flips(rows, edge_keys)
        flips[rows] = rel
        oriented = _apply_surf_flips(connect[rows], rel, spec)

        if not all(len(edge_keys[key]) == 2 for key in edge_keys):
            # An open non-planar sheet has no intrinsic exterior. Preserve a
            # coherent input orientation; planar sheets retain std CCW.
            comp_coords = coords[np.unique(oriented[:, corner_idxs])]
            if _check_coplanar(comp_coords):
                metric = _calc_first_surf_metric(
                    oriented,
                    comp_coords,
                    coords,
                    spec,
                )
                if metric is not None and metric < 0.0:
                    flips[rows] = ~flips[rows]
            continue

        vol = _calc_surf_signed_vol(oriented, coords, spec)
        if abs(vol) <= _TOL.geom:
            raise ValueError(
                "Closed surface component has zero signed volume; cannot "
                "select a material exterior."
            )
        if vol < 0.0:
            flips[rows] = ~flips[rows]
            oriented = _apply_surf_flips(
                oriented,
                np.ones(rows.shape[0], dtype=bool),
                spec,
            )
        comp_points = coords[np.unique(oriented[:, corner_idxs])]
        closed_comps.append((
            rows,
            oriented,
            np.min(comp_points, axis=0),
            np.max(comp_points, axis=0),
        ))

    # A disconnected closed shell contained by another shell is a cavity. Its
    # material-outward normal must point into the void, so its signed volume is
    # negative after local edge consistency has been established.
    for comp_idx, comp in enumerate(closed_comps):
        rows, oriented, _, _ = comp
        point = _calc_comp_probe_point(oriented, coords, spec)
        depth = 0
        for other_idx, other_comp in enumerate(closed_comps):
            _, other_oriented, bounds_min, bounds_max = other_comp
            if other_idx == comp_idx:
                continue
            if not np.all(point >= bounds_min - _TOL.geom):
                continue
            if not np.all(point <= bounds_max + _TOL.geom):
                continue
            depth += _check_point_in_closed_surf(
                point,
                other_oriented,
                coords,
                spec,
            )
        if depth % 2:
            flips[rows] = ~flips[rows]

    for row_idx, representative in enumerate(duplicate_of):
        if row_idx == representative:
            continue
        flips[row_idx] = flips[representative] ^ (
            not _check_surf_rows_same_orient(
                connect[row_idx], connect[representative], spec
            )
        )

    return flips


def _calc_indep_surf_orient_flips(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Calculate each surface row's orientation flip independently."""
    flips = np.zeros(connect.shape[0], dtype=bool)
    for row_idx, row in enumerate(connect):
        metric = _calc_winding_metric(row, coords, spec)
        flips[row_idx] = metric is not None and metric < 0.0
    return flips


def _build_surf_topology(
    connect: np.ndarray,
    active_rows: np.ndarray,
    spec: ElementSpec,
) -> list[tuple[np.ndarray, dict]]:
    """Build connected surface components keyed by their corner-node edges."""

    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    edge_map: dict = {}
    for row_idx in active_rows:
        row = connect[row_idx]
        corners = row[corner_idxs]
        for node_a, node_b in zip(corners, np.roll(corners, -1)):
            directed = (int(node_a), int(node_b))
            key = tuple(sorted(directed))
            edge_map.setdefault(key, []).append((row_idx, directed))

    for key, uses in edge_map.items():
        if len(uses) > 2:
            raise ValueError(
                f"Non-manifold surface edge {key} has {len(uses)} "
                "incident faces."
            )

    neighbours = {int(row): set() for row in active_rows}
    for uses in edge_map.values():
        if len(uses) == 2:
            row_a, _ = uses[0]
            row_b, _ = uses[1]
            neighbours[row_a].add(row_b)
            neighbours[row_b].add(row_a)

    comps: list[tuple[np.ndarray, dict]] = []
    unseen = {int(row) for row in active_rows}
    while unseen:
        seed = unseen.pop()
        rows = {seed}
        stack = [seed]
        while stack:
            row = stack.pop()
            for neighbour in neighbours[row]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    rows.add(neighbour)
                    stack.append(neighbour)
        rows_arr = np.asarray(sorted(rows), dtype=np.int64)
        comp_edges = {
            key: uses for key, uses in edge_map.items() if uses[0][0] in rows
        }
        comps.append((rows_arr, comp_edges))
    return comps


def _check_surf_rows_same_orient(
    row_a: np.ndarray,
    row_b: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Return whether duplicate surface rows have the same directed boundary."""

    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    corners_a = row_a[corner_idxs]
    corners_b = row_b[corner_idxs]
    edges_b = {
        (int(node_a), int(node_b))
        for node_a, node_b in zip(corners_b, np.roll(corners_b, -1))
    }
    for node_a, node_b in zip(corners_a, np.roll(corners_a, -1)):
        if (int(node_a), int(node_b)) in edges_b:
            return True
        if (int(node_b), int(node_a)) in edges_b:
            return False
    raise ValueError("Duplicate surface faces do not share a boundary edge.")


def _calc_comp_rel_flips(
    rows: np.ndarray,
    edge_keys: dict,
) -> np.ndarray:
    """Find local reversals making shared edges traverse opposite ways."""

    row_set = set(rows.tolist())
    constraints: dict[int, list[tuple[int, bool]]] = {
        row: [] for row in row_set
    }
    for uses in edge_keys.values():
        if len(uses) != 2:
            continue
        row_a, direct_a = uses[0]
        row_b, direct_b = uses[1]
        same_direct = direct_a == direct_b
        constraints[row_a].append((row_b, same_direct))
        constraints[row_b].append((row_a, same_direct))

    assigned: dict[int, bool] = {}
    for seed in rows:
        seed_int = int(seed)
        if seed_int in assigned:
            continue
        assigned[seed_int] = False
        stack = [seed_int]
        while stack:
            row = stack.pop()
            for neighbour, xor_flip in constraints[row]:
                expected = assigned[row] ^ xor_flip
                if neighbour in assigned:
                    if assigned[neighbour] != expected:
                        raise ValueError("Surface component is not orientable.")
                else:
                    assigned[neighbour] = expected
                    stack.append(neighbour)
    return np.asarray([assigned[int(row)] for row in rows], dtype=bool)


def _apply_surf_flips(
    connect: np.ndarray,
    flips: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Apply the requested winding reversal to each surface row."""
    out = np.array(connect, copy=True)
    for row_idx in np.flatnonzero(flips):
        out[row_idx] = _reverse_surf_row(out[row_idx], spec)
    return out


def _calc_first_surf_metric(
    connect: np.ndarray,
    comp_coords: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> float | None:
    """Calculate the first non-degenerate metric in a surface component."""
    normal = _calc_std_plane_normal(comp_coords)
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    for row in connect:
        metric = _calc_loc_polygon_signed_area(
            coords[row[corner_idxs]],
            normal,
        )
        if metric is not None and abs(metric) > _TOL.geom:
            return metric
    return None


def _calc_surf_signed_vol(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> float:
    """Calculate the signed volume enclosed by a triangulated surface."""
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    vol = 0.0
    for row in connect:
        points = coords[row[corner_idxs]]
        for point_idx in range(1, points.shape[0] - 1):
            vol += float(np.dot(
                points[0],
                np.cross(points[point_idx], points[point_idx + 1]),
            )) / 6.0
    return vol


def _calc_comp_probe_point(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Calculate an interior probe point near a surface component."""
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    points = coords[connect[0, corner_idxs]]
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= _TOL.geom:
        return np.mean(points, axis=0)
    extent = np.ptp(coords, axis=0)
    epsilon = max(float(np.linalg.norm(extent)) * 1.0e-9, _TOL.geom * 10.0)
    return np.mean(points, axis=0) + epsilon * normal / normal_norm


def _check_point_in_closed_surf(
    point: np.ndarray,
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Classify a point with parity ray casting against a closed surface."""

    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    votes: list[bool] = []
    for direct in _POINT_IN_SURF_RAY_DIRECTS:
        direct = direct / np.linalg.norm(direct)
        hits = 0

        for row in connect:
            corners = coords[row[corner_idxs]]

            for point_idx in range(1, corners.shape[0] - 1):
                if _check_ray_intersects_triangle(
                    point,
                    direct,
                    corners[0],
                    corners[point_idx],
                    corners[point_idx + 1],
                ):
                    hits += 1

        votes.append(bool(hits % 2))

    return sum(votes) >= 2


def _check_ray_intersects_triangle(
    origin: np.ndarray,
    direct: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> bool:
    """Return whether a forward ray intersects a triangle."""

    edge_ab = point_b - point_a
    edge_ac = point_c - point_a
    perpendicular = np.cross(direct, edge_ac)
    determinant = float(np.dot(edge_ab, perpendicular))
    if abs(determinant) <= _TOL.geom:
        return False

    inv_determinant = 1.0 / determinant
    offset = origin - point_a
    barycentric_u = inv_determinant * float(np.dot(offset, perpendicular))
    if barycentric_u <= _TOL.geom or barycentric_u >= 1.0 - _TOL.geom:
        return False

    cross_offset_edge = np.cross(offset, edge_ab)
    barycentric_v = inv_determinant * float(
        np.dot(direct, cross_offset_edge)
    )
    if (
        barycentric_v <= _TOL.geom
        or barycentric_u + barycentric_v >= 1.0 - _TOL.geom
    ):
        return False

    return (
        inv_determinant * float(np.dot(edge_ac, cross_offset_edge))
        > _TOL.geom
    )


def _enforce_surf_orient_table(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Return a table with std surface-component orientations."""
    flips = _calc_surf_orient_flips(connect, coords, spec)
    return _apply_surf_flips(connect, flips, spec)


def _check_perms_equiv(
    perm: tuple[int, ...],
    ref_perm: tuple[int, ...],
    symmetries: tuple[tuple[int, ...], ...],
) -> bool:
    """Return whether permutations differ only by a proper symmetry."""
    return any(
        perm == tuple(ref_perm[slot] for slot in symmetry)
        for symmetry in symmetries
    )


def infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer geometric local-slot mappings for canonical typed blocks."""
    inferred: dict[EElementType, tuple[int, ...]] = {}
    for name, block in mesh_in.blocks.items():
        elem_type = block.element_type
        spec = ELEMENT_SPECS[elem_type]
        try:
            normalised = _enforce_node_order_from_geom(
                block.connect,
                mesh_in.coords,
                elem_type,
                spec,
            )
        except ValueError as error:
            raise MeshConvErr(
                f"Could not infer '{name}' ({elem_type.value}); supply "
                "MeshConvention explicitly."
            ) from error

        perms: set[tuple[int, ...]] = set()
        for row, target in zip(block.connect, normalised, strict=True):
            perms.add(tuple(
                int(np.flatnonzero(row == node_id)[0]) for node_id in target
            ))

        representative = min(perms)
        symmetries = _get_elem_symmetries(elem_type)
        if any(
            not _check_perms_equiv(candidate, representative, symmetries)
            for candidate in perms
        ):
            raise MeshConvErr(
                f"Connectivity block '{name}' contains multiple source "
                f"layouts for {elem_type.value}; supply MeshConvention."
            )

        previous = inferred.get(elem_type)
        if previous is not None and not _check_perms_equiv(
            representative,
            previous,
            symmetries,
        ):
            raise MeshConvErr(
                f"Multiple blocks disagree on the {elem_type.value} source "
                "layout; supply MeshConvention."
            )
        inferred[elem_type] = representative

    return MeshConvention(inferred, standardise_equiv_orients=True)


def _enforce_node_order_table(
    connect: np.ndarray,
    elem_type: EElementType,
    src_convention: MeshConvention | None,
) -> np.ndarray:
    """Apply an explicitly declared source-to-Riley slot permutation."""
    if src_convention is None:
        return connect

    perm = src_convention.get_target_to_source_perm(elem_type)
    if perm is None:
        return connect

    normalised = np.ascontiguousarray(
        connect[:, np.asarray(perm, dtype=np.int64)],
        dtype=np.int64,
    )
    if src_convention.standardise_equiv_orients:
        return _enforce_std_orients(normalised, elem_type)
    return normalised


def _enforce_node_order_high_order(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Rebuild higher-order rows from inferred geometric roles."""

    connect_out = np.empty_like(connect)
    for row_idx, row in enumerate(connect):
        connect_out[row_idx] = _enforce_node_order_high_order_row(
            row,
            coords,
            spec,
        )

    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_node_order_from_geom(
    connect: np.ndarray,
    coords: np.ndarray,
    elem_type: EElementType,
    spec: ElementSpec,
) -> np.ndarray:
    """Infer node roles, then select the unique ID-stable proper orientation."""

    if connect.shape[1] == len(spec.corner_idxs):
        connect_out = np.empty_like(connect)
        src_slots = np.arange(connect.shape[1], dtype=np.int64)

        for row_idx, row in enumerate(connect):
            if spec.is_surf:
                ordered = _order_surf_corners(coords[row], src_slots)
            elif len(spec.corner_idxs) == 4:
                ordered = _order_tet_corners(coords[row], src_slots)
            else:
                ordered = _order_hex_corners(coords[row], src_slots)
            connect_out[row_idx] = row[ordered]
    else:
        connect_out = _enforce_node_order_high_order(connect, coords, spec)

    return _enforce_std_orients(connect_out, elem_type)


def _enforce_std_orients(
    connect: np.ndarray,
    elem_type: EElementType,
) -> np.ndarray:
    """Anchor each role-correct row at the lowest valid global-node sequence."""
    perms_arr = _get_elem_symmetry_arrs(elem_type)
    out = np.empty_like(connect)
    for row_idx, row in enumerate(connect):
        out[row_idx] = min(
            (row[perm] for perm in perms_arr),
            key=lambda candidate: tuple(int(node) for node in candidate),
        )
    return np.ascontiguousarray(out, dtype=np.int64)


def _enforce_node_order_high_order_row(
    connect_row: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Infer std node roles for one higher-order element row."""
    elem_coords = coords[connect_row]
    if _check_std_node_roles(elem_coords, spec):
        return connect_row

    corner_count = len(spec.corner_idxs)
    corner_loc = _infer_corner_nodes(elem_coords, corner_count)

    if spec.is_surf:
        corner_loc = _order_surf_corners(elem_coords, corner_loc)
    elif corner_count == 4:
        corner_loc = _order_tet_corners(elem_coords, corner_loc)
    else:
        corner_loc = _order_hex_corners(elem_coords, corner_loc)

    corner_coords = elem_coords[corner_loc]
    remaining = [
        idx for idx in range(connect_row.shape[0]) if idx not in corner_loc
    ]
    edge_pairs = spec.edge_pairs or tuple(
        (idx, (idx + 1) % corner_count) for idx in range(corner_count)
    )
    edge_targets = np.asarray([
        0.5 * (corner_coords[start] + corner_coords[end])
        for start, end in edge_pairs
    ], dtype=np.float64)
    edge_loc = _match_role_nodes(
        elem_coords,
        remaining,
        edge_targets,
        "edge",
    )
    remaining = [idx for idx in remaining if idx not in edge_loc]

    if spec.face_centre_idxs:
        face_targets = np.asarray([
            np.mean(corner_coords[list(face)], axis=0)
            for face in spec.face_corner_idxs
        ], dtype=np.float64)
        face_loc = _match_role_nodes(
            elem_coords,
            remaining,
            face_targets,
            "face centre",
        )

        remaining = [idx for idx in remaining if idx not in face_loc]

    if spec.centre_idx is not None or spec.cell_centre_idx is not None:
        centre_loc = _match_role_nodes(
            elem_coords,
            remaining,
            np.mean(corner_coords, axis=0, keepdims=True),
            "centre",
        )

        remaining = [idx for idx in remaining if idx not in centre_loc]

    if remaining:
        raise ValueError(
            "Could not assign every higher-order node to an element role."
        )

    out_loc = np.empty(connect_row.shape[0], dtype=np.int64)
    out_loc[np.asarray(spec.corner_idxs, dtype=np.int64)] = corner_loc
    edge_slots = np.arange(
        corner_count, corner_count + len(edge_pairs), dtype=np.int64
    )
    out_loc[edge_slots] = edge_loc

    if spec.face_centre_idxs:
        out_loc[
            np.asarray(spec.face_centre_idxs, dtype=np.int64)
        ] = face_loc

    if spec.centre_idx is not None:
        out_loc[spec.centre_idx] = centre_loc[0]

    if spec.cell_centre_idx is not None:
        out_loc[spec.cell_centre_idx] = centre_loc[0]

    return connect_row[out_loc]


def _check_std_node_roles(
    elem_coords: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Accept an already coherent row, including curved and seam elements."""
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    corner_coords = elem_coords[corner_idxs]

    corner_center = np.mean(corner_coords, axis=0)
    if np.linalg.matrix_rank(
        corner_coords - corner_center,
        tol=_TOL.geom,
    ) < (2 if spec.is_surf else 3):
        return False

    inferred_corners = _infer_corner_nodes(elem_coords, len(corner_idxs))
    if (set(inferred_corners) == set(corner_idxs)
            and inferred_corners.shape[0] == len(corner_idxs)):
        pass
    elif inferred_corners.shape[0] == len(corner_idxs):
        return False

    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
    edge_pairs = spec.edge_pairs or tuple(
        (idx, (idx + 1) % len(corner_idxs))
        for idx in range(len(corner_idxs))
    )
    edge_nodes = elem_coords[
        len(corner_idxs):len(corner_idxs) + len(edge_pairs)
    ]
    edge_distances = np.asarray([
        _calc_point_segment_distance(
            node,
            corner_coords[start],
            corner_coords[end],
        )
        for node, (start, end) in zip(edge_nodes, edge_pairs, strict=True)
    ])

    if np.any(edge_distances > _TOL.role_match * scale):
        return False

    if spec.face_centre_idxs:
        for centre_idx, face_corner_idxs in zip(
            spec.face_centre_idxs,
            spec.face_corner_idxs,
            strict=True,
        ):
            face_target = np.mean(
                corner_coords[np.asarray(face_corner_idxs)],
                axis=0,
            )
            if np.linalg.norm(
                elem_coords[centre_idx] - face_target
            ) > _TOL.role_match * scale:
                return False

    if spec.centre_idx is not None and np.linalg.norm(
        elem_coords[spec.centre_idx] - corner_center
    ) > _TOL.role_match * scale:
        return False

    if spec.cell_centre_idx is not None and np.linalg.norm(
        elem_coords[spec.cell_centre_idx] - corner_center
    ) > _TOL.role_match * scale:
        return False

    return True


def _infer_corner_nodes(
    elem_coords: np.ndarray,
    corner_count: int,
) -> np.ndarray:
    """Find affine-element vertices without depending on their source slots."""

    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
    midpoint_tol = _TOL.geom * scale * 100.0
    midpoint_nodes: set[int] = set()

    for node_idx, point in enumerate(elem_coords):
        for first in range(elem_coords.shape[0]):
            for second in range(first + 1, elem_coords.shape[0]):

                if node_idx in (first, second):
                    continue

                midpoint = 0.5 * (elem_coords[first] + elem_coords[second])
                if np.linalg.norm(point - midpoint) <= midpoint_tol:
                    midpoint_nodes.add(node_idx)
                    break

            if node_idx in midpoint_nodes:
                break

    candidates = np.asarray([
        idx for idx in range(elem_coords.shape[0]) if idx not in midpoint_nodes
    ], dtype=np.int64)
    if candidates.shape[0] == corner_count:
        return candidates

    distances = np.linalg.norm(
        elem_coords - np.mean(elem_coords, axis=0),
        axis=1,
    )
    return np.sort(np.argsort(distances)[-corner_count:])


def _calc_point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Calculate the shortest distance from a point to a line segment."""
    direct = end - start
    length_sq = float(np.dot(direct, direct))

    if length_sq <= _TOL.geom:
        return np.inf

    param = np.clip(
        float(np.dot(point - start, direct) / length_sq), 0.0, 1.0
    )

    return float(np.linalg.norm(point - (start + param * direct)))


def _order_surf_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return surface-corner slots in counter-clockwise order."""

    corner_coords = elem_coords[corner_loc]
    centred = corner_coords - np.mean(corner_coords, axis=0)
    if np.linalg.matrix_rank(centred, tol=_TOL.geom) < 2:
        raise ValueError(
            "Degenerate surface element has collinear corner nodes."
        )

    _, _, vecs = np.linalg.svd(centred, full_matrices=False)
    normal = vecs[-1]
    axis_u = vecs[0]
    axis_v = np.cross(normal, axis_u)
    angles = np.arctan2(centred @ axis_v, centred @ axis_u)

    ordered = corner_loc[np.argsort(angles)]
    ordered_coords = elem_coords[ordered]

    signed = np.dot(
        np.cross(
            ordered_coords[1] - ordered_coords[0],
            ordered_coords[2] - ordered_coords[0],
        ),
        normal,
    )

    if signed < 0.0:
        ordered = ordered[[0, *range(len(ordered) - 1, 0, -1)]]

    return ordered


def _order_tet_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return tetrahedron-corner slots in right-handed order."""
    ordered = corner_loc[np.lexsort(elem_coords[corner_loc].T[::-1])]
    corner_coords = elem_coords[ordered]
    metric = _calc_tet_signed_vol(corner_coords)

    if abs(metric) <= _TOL.geom:
        raise ValueError("Degenerate tetrahedron has zero signed volume.")

    if metric < 0.0:
        ordered[[1, 2]] = ordered[[2, 1]]

    return ordered


def _order_hex_corners(
    elem_coords: np.ndarray,
    corner_loc: np.ndarray,
) -> np.ndarray:
    """Return hexahedron-corner slots in right-handed Riley order."""

    corner_coords = elem_coords[corner_loc]
    corner_center = np.mean(corner_coords, axis=0)
    centered_corners = corner_coords - corner_center
    corner_rank = np.linalg.matrix_rank(centered_corners, tol=_TOL.geom)

    if corner_rank < 3:
        raise ValueError("Degenerate hexahedron has coplanar corner nodes.")

    origin = corner_loc[np.lexsort(corner_coords.T[::-1])[0]]
    candidates = [idx for idx in corner_loc if idx != origin]

    best_order: np.ndarray | None = None
    best_error = np.inf
    origin_coord = elem_coords[origin]

    for first in candidates:
        for second in candidates:
            for third in candidates:
                if len({first, second, third}) != 3:
                    continue

                directs = np.array((
                    elem_coords[first] - origin_coord,
                    elem_coords[second] - origin_coord,
                    elem_coords[third] - origin_coord,
                ))

                determinant = np.linalg.det(directs)
                if determinant <= _TOL.geom:
                    continue

                targets = np.array((
                    origin_coord,
                    origin_coord + directs[0],
                    origin_coord + directs[0] + directs[1],
                    origin_coord + directs[1],
                    origin_coord + directs[2],
                    origin_coord + directs[0] + directs[2],
                    origin_coord
                    + directs[0]
                    + directs[1]
                    + directs[2],
                    origin_coord + directs[1] + directs[2],
                ))

                assigned = _match_role_nodes(
                    elem_coords,
                    list(corner_loc),
                    targets,
                    "corner",
                    validate=False,
                )

                error = float(np.sum(
                    np.linalg.norm(elem_coords[assigned] - targets, axis=1)
                ))

                if error < best_error:
                    best_error = error
                    best_order = np.asarray(assigned, dtype=np.int64)

    if best_order is None:
        raise ValueError(
            "Could not determine a right-handed hexahedron corner order."
        )

    scale = float(np.max(
        np.linalg.norm(corner_coords - origin_coord, axis=1)
    ))

    if best_error > _TOL.role_match * scale:
        raise ValueError("Hexahedron corner nodes do not form a valid element.")

    return best_order


def _match_role_nodes(
    elem_coords: np.ndarray,
    candidate_loc: list[int],
    targets: np.ndarray,
    role_name: str,
    validate: bool = True,
) -> list[int]:
    """Match unused local nodes to target geometric roles."""

    if len(candidate_loc) < targets.shape[0]:
        raise ValueError(
            f"Not enough nodes available to assign {role_name} roles."
        )

    candidates = np.asarray(candidate_loc, dtype=np.int64)
    distances = np.linalg.norm(
        elem_coords[candidates, None, :] - targets[None, :, :],
        axis=2,
    )

    assigned: list[int] = []
    used: set[int] = set()
    for target_idx in range(targets.shape[0]):
        match_idx = None
        for idx in np.argsort(distances[:, target_idx]):
            if int(idx) not in used:
                match_idx = idx
                break

        if match_idx is None:
            raise ValueError(f"Could not assign a unique {role_name} node.")

        used.add(int(match_idx))
        assigned.append(int(candidates[match_idx]))

    if validate:
        scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geom)
        if np.any(
            np.linalg.norm(elem_coords[assigned] - targets, axis=1)
            > _TOL.role_match * scale
        ):
            raise ValueError(
                f"Could not match {role_name} nodes to the element geometry."
            )

    return assigned


def _check_coplanar(coords_elem: np.ndarray) -> bool:
    """Return whether coordinates lie on one plane."""
    centred = coords_elem - np.mean(coords_elem, axis=0)
    return np.linalg.matrix_rank(centred, tol=_TOL.geom) <= 2


def _calc_tet_signed_vol(coords_elem: np.ndarray) -> float:
    """Calculate a tetrahedron's signed volume metric."""
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[2] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
            ))
        )
    )


def _calc_hex_signed_vol(coords_elem: np.ndarray) -> float:
    """Calculate a hexahedron's signed corner metric."""
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
                coords_elem[4] - coords_elem[0],
            ))
        )
    )


def _calc_vol_signed_metric(corner_coords: np.ndarray) -> float:
    """Calculate the signed metric for tetrahedral or hexahedral corners."""
    if corner_coords.shape[0] == 4:
        return _calc_tet_signed_vol(corner_coords)
    return _calc_hex_signed_vol(corner_coords)


def _calc_winding_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> float | None:
    """Calculate a signed winding metric for an explicit surface topology."""
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    coords_elem = coords[connect_row[corner_idxs]]

    if _check_coplanar(coords):
        ref_normal = _calc_std_plane_normal(coords)
    else:
        outward = np.mean(coords_elem, axis=0) - np.mean(coords, axis=0)
        outward_norm = np.linalg.norm(outward)
        if outward_norm <= _TOL.geom:
            return None
        ref_normal = outward / outward_norm

    return _calc_loc_polygon_signed_area(coords_elem, ref_normal)


def _calc_std_plane_normal(coords: np.ndarray) -> np.ndarray:
    """Calculate a deterministically signed normal for planar coordinates."""
    centred = coords - np.mean(coords, axis=0)
    _, _, vecs = np.linalg.svd(centred, full_matrices=False)
    normal = vecs[-1]
    if normal[int(np.argmax(np.abs(normal)))] < 0.0:
        normal = -normal

    return normal / np.linalg.norm(normal)


def _calc_loc_polygon_signed_area(
    coords_elem: np.ndarray,
    ref_normal: np.ndarray,
) -> float | None:
    """Calculate polygon area signed against a reference normal."""
    origin = coords_elem[0]
    axis_u = None
    for point in coords_elem[1:]:
        edge = point - origin
        edge -= np.dot(edge, ref_normal) * ref_normal
        edge_norm = np.linalg.norm(edge)
        if edge_norm > _TOL.geom:
            axis_u = edge / edge_norm
            break

    if axis_u is None:
        return None

    axis_v = np.cross(ref_normal, axis_u)
    projected = np.column_stack((
        (coords_elem - origin) @ axis_u,
        (coords_elem - origin) @ axis_v,
    ))
    rolled = np.roll(projected, -1, axis=0)

    return float(0.5 * np.sum(
        projected[:, 0] * rolled[:, 1]
        - rolled[:, 0] * projected[:, 1],
    ))


def _calc_handedness_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> float | None:
    """Calculate the signed metric for an explicit element topology."""
    if spec.is_surf:
        return _calc_winding_metric(connect_row, coords, spec)

    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    return _calc_vol_signed_metric(coords[connect_row[corner_idxs]])


def _check_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Return whether all rows have positive handedness."""
    for row in connect:
        metric = _calc_handedness_metric(row, coords, spec)

        if metric is None:
            continue

        if abs(metric) <= _TOL.geom:
            continue

        if metric <= 0.0:
            return False

    return True


def _reverse_surf_row(
    connect_row: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Reverse winding while preserving every surface-node role."""
    if spec.surf_reverse_perm is None:
        raise NotImplementedError(
            f"Surface reversal is not implemented for "
            f"{spec.nodes_per_elem}-node elements."
        )

    return connect_row[np.asarray(spec.surf_reverse_perm, dtype=np.int64)]


def _reverse_handedness_row(
    connect_row: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Reverse handedness while preserving every volume-node role."""

    if spec.handedness_reverse_perm is None:
        raise NotImplementedError(
            f"Handedness reversal is not implemented for "
            f"{spec.nodes_per_elem}-node elements."
        )

    return connect_row[np.asarray(
        spec.handedness_reverse_perm,
        dtype=np.int64,
    )]


def _get_face_corner_coords(
    face_coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Return only the corner coordinates from a surface face."""
    return face_coords[np.asarray(spec.corner_idxs, dtype=np.int64)]


def _calc_face_normal(
    face_coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Calculate a unit normal from a non-degenerate surface face."""
    face_corners = _get_face_corner_coords(face_coords, spec)
    face_normal = np.cross(
        face_corners[1] - face_corners[0],
        face_corners[2] - face_corners[0],
    )
    normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geom and face_corners.shape[0] == 4:
        face_normal = np.cross(
            face_corners[2] - face_corners[0],
            face_corners[3] - face_corners[0],
        )
        normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geom:
        raise ValueError(
            "Degenerate face detected while extracting the surface mesh."
        )

    return face_normal / normal_mag


def _enforce_surf_face_outward(
    face_connect: np.ndarray,
    parent_connect: np.ndarray,
    coords: np.ndarray,
    face_spec: ElementSpec,
    parent_spec: ElementSpec,
) -> np.ndarray:
    """Orient an extracted face away from its parent element."""
    face_coords = coords[face_connect]
    face_corners = _get_face_corner_coords(face_coords, face_spec)
    face_centroid = np.mean(face_corners, axis=0)

    parent_corners = np.asarray(parent_spec.corner_idxs, dtype=np.int64)
    parent_centroid = np.mean(coords[parent_connect[parent_corners]], axis=0)

    face_normal = _calc_face_normal(face_coords, face_spec)
    outward_dir = face_centroid - parent_centroid

    if np.dot(face_normal, outward_dir) < 0.0:
        return _reverse_surf_row(face_connect, face_spec)

    return face_connect


def _enforce_surf_face_node_order(
    face_connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Restore high-order node roles on an extracted surface face."""
    nodes_per_face = spec.nodes_per_elem
    face_out = np.copy(face_connect)
    face_coords = coords[face_out]
    corner_idxs = np.asarray(spec.corner_idxs, dtype=np.int64)
    num_corners = corner_idxs.shape[0]
    if nodes_per_face == num_corners:
        return face_out

    midside_pool = np.arange(num_corners, nodes_per_face, dtype=np.int64)
    face_centroid = np.mean(face_coords[corner_idxs, :], axis=0)
    mid_pool_coords = face_coords[midside_pool, :]

    if spec.centre_idx is not None:
        centroid_dists = np.linalg.norm(mid_pool_coords - face_centroid, axis=1)
        center_pool_idx = int(np.argmin(centroid_dists))
        center_loc_idx = int(midside_pool[center_pool_idx])
        edge_pool_mask = np.ones(midside_pool.shape[0], dtype=bool)
        edge_pool_mask[center_pool_idx] = False
        edge_pool_loc_idxs = midside_pool[edge_pool_mask]
        edge_pool_coords = face_coords[edge_pool_loc_idxs, :]
    else:
        center_loc_idx = -1
        edge_pool_loc_idxs = midside_pool
        edge_pool_coords = mid_pool_coords

    edge_midpoints = np.asarray([
        0.5 * (
            face_coords[corner_idx, :]
            + face_coords[(corner_idx + 1) % num_corners, :]
        )
        for corner_idx in range(num_corners)
    ], dtype=np.float64)
    edge_dists = np.linalg.norm(
        edge_pool_coords[:, None, :] - edge_midpoints[None, :, :],
        axis=2,
    )
    reordered_edge_idxs = edge_pool_loc_idxs[np.argmin(edge_dists, axis=0)]
    face_out[
        num_corners:num_corners + num_corners
    ] = face_out[reordered_edge_idxs]

    if spec.centre_idx is not None:
        face_out[spec.centre_idx] = face_out[center_loc_idx]

    return face_out


def _enforce_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Return volume connectivity with positive handedness."""
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _calc_handedness_metric(row, coords, spec)
        if metric is not None and metric < 0.0:
            connect_out[idx, :] = _reverse_handedness_row(row, spec)

    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _extract_surf_faces_from_table(
    connect: np.ndarray,
    coords: np.ndarray,
    parent_spec: ElementSpec,
    face_spec: ElementSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract boundary faces and their parent rows from a volume block."""
    if parent_spec.surf_faces is None:
        raise NotImplementedError(
            "Surface extraction is not implemented for this element type."
        )
    face_map = np.asarray(parent_spec.surf_faces, dtype=np.int64)
    faces_flat = connect[:, face_map].reshape((-1, face_map.shape[1]))
    (_, unique_idxs, unique_counts) = np.unique(
        np.sort(faces_flat, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )

    ext_face_idxs = unique_idxs[unique_counts == 1]
    ext_parent_elem_idxs = np.ascontiguousarray(
        ext_face_idxs // face_map.shape[0],
        dtype=np.int64,
    )
    ext_faces = np.copy(faces_flat[ext_face_idxs])

    for ff, parent_elem_idx in enumerate(ext_parent_elem_idxs):
        ext_faces[ff, :] = _enforce_surf_face_outward(
            ext_faces[ff, :],
            connect[parent_elem_idx, :],
            coords,
            face_spec,
            parent_spec,
        )

        ext_faces[ff, :] = _enforce_surf_face_node_order(
            ext_faces[ff, :],
            coords,
            face_spec,
        )

    ext_faces = np.ascontiguousarray(ext_faces, dtype=np.int64)
    return ext_faces, ext_parent_elem_idxs


def _conv_to_vec3(
    values: np.ndarray | list[float] | tuple[float, ...],
    name: str,
) -> np.ndarray:
    """Return the first three finite components of an array-like value."""
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)

    if values_arr.size < 3:
        raise ValueError(f"'{name}' must contain at least three components.")

    vec = np.ascontiguousarray(values_arr[:3])
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"'{name}' must contain only finite values.")

    return vec


def _normalise_integer_connect(
    connect: np.ndarray,
    name: str,
    *,
    allow_integral_float: bool = False,
) -> np.ndarray:
    """Validate connectivity and return owned contiguous ``int64`` storage."""
    arr = np.asarray(connect)
    if arr.ndim != 2:
        raise ValueError(f"'{name}' must be a 2D array, got shape {arr.shape}.")

    is_float = np.issubdtype(arr.dtype, np.floating)
    if not np.issubdtype(arr.dtype, np.integer):
        if not is_float:
            raise TypeError(f"'{name}' must use an integer dtype.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"'{name}' must contain only finite integers.")
        if np.any(arr != np.trunc(arr)):
            raise ValueError(f"'{name}' contains fractional indices.")
        if not allow_integral_float:
            raise TypeError(f"'{name}' must use an integer dtype.")

    if np.any(arr < 0):
        raise ValueError(f"'{name}' cannot contain negative indices.")
    int64_upper = float(1 << 63) if is_float else np.iinfo(np.int64).max
    if np.any(arr >= int64_upper if is_float else arr > int64_upper):
        raise ValueError(f"'{name}' contains indices outside the int64 range.")

    return np.array(arr, dtype=np.int64, order="C", copy=True)


class MeshCheckCode(StrEnum):
    """A geometric mesh-convention condition that a block failed."""

    CCW_WINDING = "ccw_winding"
    RIGHT_HANDED_GEOMETRY = "right_handed_geometry"
    SURFACE_TOPOLOGY = "surface_topology"
    NODE_ORDER = "node_order"


MeshConvCheck = dict[str, list[MeshCheckCode]]


class EElementType(Enum):
    """Supported finite-element topologies."""

    TRI3 = "tri3"
    TRI6 = "tri6"
    TRI7 = "tri7"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"
    HEX20 = "hex20"
    HEX27 = "hex27"

    def calc_ref_coords(self) -> np.ndarray:
        """Return reference-node coordinates in Riley slot order.

        These coordinates are topology data, not geometry used to inspect a
        user mesh.  They let us derive every orientation-preserving
        automorphism of an element once, instead of maintaining hand-written
        permutations in several places.
        """
        if self in (EElementType.TRI3, EElementType.TRI6,
                    EElementType.TRI7):

            points = ((0., 0.), (1., 0.), (0., 1.), (.5, 0.), (.5, .5),
                      (0., .5))

            if self is EElementType.TRI3:
                points = points[:3]
            elif self is EElementType.TRI7:
                points += ((1. / 3., 1. / 3.),)

            return np.asarray(points, dtype=np.float64)

        if self in (EElementType.QUAD4, EElementType.QUAD8,
                    EElementType.QUAD9):

            points = ((0., 0.), (1., 0.), (1., 1.), (0., 1.),
                      (.5, 0.), (1., .5), (.5, 1.), (0., .5))

            if self is EElementType.QUAD4:
                points = points[:4]
            elif self is EElementType.QUAD9:
                points += ((.5, .5),)

            return np.asarray(points, dtype=np.float64)

        if self in (EElementType.TET4, EElementType.TET10):
            points = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                      (.5, 0., 0.), (.5, .5, 0.), (0., .5, 0.),
                      (0., 0., .5), (.5, 0., .5), (0., .5, .5))

            return np.asarray(
                points[:4] if self is EElementType.TET4 else points,
                dtype=np.float64,
            )

        if self in (EElementType.HEX8, EElementType.HEX20,
                    EElementType.HEX27):
            points = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
                      (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.),
                      (.5, 0., 0.), (1., .5, 0.), (.5, 1., 0.), (0., .5, 0.),
                      (.5, 0., 1.), (1., .5, 1.), (.5, 1., 1.), (0., .5, 1.),
                      (0., 0., .5), (1., 0., .5), (1., 1., .5), (0., 1., .5),
                      (.5, 0., .5), (1., .5, .5), (.5, 1., .5), (0., .5, .5),
                      (.5, .5, 0.), (.5, .5, 1.), (.5, .5, .5))

            count = ELEMENT_SPECS[self].nodes_per_elem

            if self is EElementType.HEX20:
                return np.asarray(points[:20], dtype=np.float64)

            return np.asarray(points[:count], dtype=np.float64)

        raise ValueError(f"No reference coordinates for {self.value}.")

    def calc_orient_preserving_perms(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        """Derive all proper topology symmetries in std-slot notation.

        A returned permutation maps a target Riley slot to a source Riley
        slot.  Applying one preserves all corner, edge, face-centre and
        volume-centre roles.  Surface reflections and volume inversions are
        deliberately absent.
        """
        spec = ELEMENT_SPECS[self]
        ref = self.calc_ref_coords()
        dims = 2 if spec.is_surf else 3
        corners = np.asarray(spec.corner_idxs, dtype=np.int64)
        candidates: tuple[tuple[int, ...], ...]

        if len(corners) == 8:
            # The proper rotational group of a cube: 3! axis orderings and sign
            # changes with positive determinant, for 24 transformations.
            candidate_rows: list[tuple[int, ...]] = []

            for axes in perms(range(3)):
                inversion_count = 0
                for idx in range(3):
                    for next_idx in range(idx + 1, 3):
                        inversion_count += axes[idx] > axes[next_idx]

                parity = 1 if inversion_count % 2 == 0 else -1

                for signs in product((-1., 1.), repeat=3):
                    sign_product = int(np.prod(signs))
                    if parity * sign_product < 0:
                        continue
                    transformed = (
                        (2.0 * ref - 1.0)[:, axes]
                        * np.asarray(signs)
                    )

                    transformed = 0.5 * (transformed + 1.0)
                    candidate_rows.append(
                        _calc_ref_node_perm(ref, transformed)
                    )

            candidates = tuple(candidate_rows)
        else:
            rows: list[tuple[int, ...]] = []
            src_corners = ref[corners, :dims]
            homogeneous = np.column_stack((src_corners,
                                           np.ones(len(corners))))

            for corner_perm in perms(range(len(corners))):
                target_corners = src_corners[np.asarray(corner_perm)]
                transform, _, _, _ = np.linalg.lstsq(
                    homogeneous,
                    target_corners,
                    rcond=None,
                )

                if np.linalg.det(transform[:dims]) <= 0.0:
                    continue

                transformed = np.column_stack(
                    (ref[:, :dims], np.ones(ref.shape[0]))
                ) @ transform
                try:
                    rows.append(_calc_ref_node_perm(
                        ref[:, :dims],
                        transformed,
                    ))
                except ValueError:
                    continue

            candidates = tuple(rows)

        return tuple(sorted(set(candidates)))


def _normalise_permutation(
    elem_type: EElementType,
    perm_raw: tuple[int, ...],
    owner: str,
) -> tuple[int, ...]:
    """Return a validated target-slot-to-source-slot permutation."""
    perm = tuple(perm_raw)
    node_count = ELEMENT_SPECS[elem_type].nodes_per_elem
    if len(perm) != node_count:
        raise ValueError(
            f"{owner} for {elem_type.value} requires {node_count} slots."
        )
    if any(
        not isinstance(slot, Integral) or isinstance(slot, (bool, np.bool_))
        for slot in perm
    ):
        raise TypeError(f"{owner} slots must be integers.")

    normalised = tuple(int(slot) for slot in perm)
    if set(normalised) != set(range(node_count)):
        raise ValueError(
            f"{owner} for {elem_type.value} must contain every slot from 0 "
            f"to {node_count - 1} exactly once."
        )
    return normalised


@dataclass(frozen=True, slots=True)
class MeshConvention:
    """Caller-declared source ordering for otherwise ambiguous elements.

    Each permutation maps a Riley std slot to the corresponding slot in
    the source connectivity row. Omitted element types are assumed to already
    use Riley ordering.
    """

    target_to_source_perms: Mapping[EElementType, tuple[int, ...]]
    standardise_equiv_orients: bool = False

    def __post_init__(self) -> None:
        """Validate permutations and retain an immutable defensive copy."""
        if not isinstance(self.target_to_source_perms, Mapping):
            raise TypeError("MeshConvention permutations must be a mapping.")

        perms_out: dict[EElementType, tuple[int, ...]] = {}
        for elem_type, perm_raw in self.target_to_source_perms.items():
            if not isinstance(elem_type, EElementType):
                raise TypeError(
                    "MeshConvention keys must be EElementType members."
                )
            perms_out[elem_type] = _normalise_permutation(
                elem_type,
                perm_raw,
                "MeshConvention permutation",
            )

        object.__setattr__(
            self,
            "target_to_source_perms",
            MappingProxyType(perms_out),
        )

    def get_target_to_source_perm(
        self,
        elem_type: EElementType,
    ) -> tuple[int, ...] | None:
        """Return the declared source permutation for an element type."""
        return self.target_to_source_perms.get(elem_type)


class MeshConvErr(ValueError):
    """Raised when source node roles cannot be inferred unambiguously."""


@dataclass(frozen=True, slots=True)
class ElementSpec:
    """Static node-ordering metadata for one supported element topology."""

    nodes_per_elem: int
    is_surf: bool
    corner_idxs: tuple[int, ...]
    surf_reverse_perm: tuple[int, ...] | None = None
    handedness_reverse_perm: tuple[int, ...] | None = None
    surf_faces: tuple[tuple[int, ...], ...] | None = None
    centre_idx: int | None = None
    edge_pairs: tuple[tuple[int, int], ...] = ()
    face_corner_idxs: tuple[tuple[int, ...], ...] = ()
    face_centre_idxs: tuple[int, ...] = ()
    cell_centre_idx: int | None = None


class EConnectLayout(Enum):
    """Declared source connectivity table layout."""

    ROW_MAJOR = "row_major"
    NODE_MAJOR = "node_major"


@dataclass(frozen=True, slots=True)
class SourceBlockSpec:
    """Explicit instructions for converting one source connectivity block."""

    element_type: EElementType
    indexing: EConnectIndexing
    layout: EConnectLayout
    target_to_source_perm: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element_type, EElementType):
            raise TypeError("'element_type' must be an EElementType member.")
        if not isinstance(self.indexing, EConnectIndexing):
            raise TypeError("'indexing' must be an EConnectIndexing member.")
        if self.indexing is EConnectIndexing.AUTO:
            raise ValueError(
                "EConnectIndexing.AUTO is not explicit and is rejected."
            )
        if not isinstance(self.layout, EConnectLayout):
            raise TypeError("'layout' must be an EConnectLayout member.")
        if self.target_to_source_perm is not None:
            object.__setattr__(
                self,
                "target_to_source_perm",
                _normalise_permutation(
                    self.element_type,
                    self.target_to_source_perm,
                    "SourceBlockSpec permutation",
                ),
            )


@dataclass(slots=True)
class ElementBlock:
    """One explicitly typed canonical connectivity block."""

    element_type: EElementType
    connect: np.ndarray
    elem_vars: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element_type, EElementType):
            raise TypeError("'element_type' must be an EElementType member.")

        connect = _normalise_integer_connect(self.connect, "connect")
        expected_cols = ELEMENT_SPECS[self.element_type].nodes_per_elem
        if connect.shape[1] != expected_cols:
            raise ValueError(
                f"{self.element_type.value} connectivity requires exactly "
                f"{expected_cols} columns, got {connect.shape[1]}."
            )
        if connect.shape[0] == 0:
            raise ValueError("An ElementBlock must contain at least one element.")
        self.connect = connect

        if self.elem_vars is None:
            return
        if not isinstance(self.elem_vars, Mapping):
            raise TypeError("'elem_vars' must be a mapping or None.")

        elem_vars_out: dict[str, np.ndarray] = {}
        for name, values in self.elem_vars.items():
            if not isinstance(name, str):
                raise TypeError("Element-variable names must be strings.")
            values_arr = np.asarray(values)
            if values_arr.ndim == 0:
                raise ValueError(
                    f"Element variable '{name}' must have an element row axis."
                )
            elem_vars_out[name] = np.array(values_arr, copy=True)
        self.elem_vars = elem_vars_out


@dataclass(slots=True)
class SimData:
    """Canonical mesh data with explicit homogeneous-topology blocks.

    Coordinate input is copied. Validated ``ElementBlock`` objects are retained
    directly and become owned by the mesh.
    """

    coords: np.ndarray
    blocks: dict[str, ElementBlock]
    time: np.ndarray | None = None
    side_sets: dict[tuple[str, str], np.ndarray] | None = None
    node_vars: dict[str, np.ndarray] | None = None
    glob_vars: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        coords_raw = np.asarray(self.coords)
        if np.iscomplexobj(coords_raw):
            raise TypeError("'coords' must contain real numeric values.")
        if coords_raw.ndim != 2 or coords_raw.shape[1] != 3:
            raise ValueError(
                f"'coords' must have shape (N, 3), got {coords_raw.shape}."
            )
        if coords_raw.shape[0] == 0:
            raise ValueError("'coords' must contain at least one node.")
        try:
            coords = np.array(
                coords_raw,
                dtype=np.float64,
                order="C",
                copy=True,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("'coords' must contain numeric values.") from error
        if not np.all(np.isfinite(coords)):
            raise ValueError("'coords' must contain only finite values.")

        if not isinstance(self.blocks, Mapping):
            raise TypeError("'blocks' must be a mapping.")
        if not self.blocks:
            raise ValueError("'blocks' must not be empty.")

        blocks_out: dict[str, ElementBlock] = {}
        topology_classes: set[bool] = set()
        for name, block in self.blocks.items():
            if not isinstance(name, str):
                raise TypeError("Block names must be strings.")
            if not isinstance(block, ElementBlock):
                raise TypeError(f"Block '{name}' must be an ElementBlock.")

            if np.any(block.connect >= coords.shape[0]):
                raise ValueError(
                    f"Block '{name}' contains connectivity indices outside "
                    f"[0, {coords.shape[0] - 1}]."
                )

            if block.elem_vars is not None:
                elem_count = block.connect.shape[0]
                for var_name, values in block.elem_vars.items():
                    if values.shape[0] != elem_count:
                        raise ValueError(
                            f"Element variable '{var_name}' in block '{name}' "
                            f"has {values.shape[0]} rows; expected {elem_count}."
                        )

            topology_classes.add(ELEMENT_SPECS[block.element_type].is_surf)
            blocks_out[name] = block

        if len(topology_classes) != 1:
            raise ValueError(
                "A SimData mesh cannot mix surface and volume blocks."
            )

        self.coords = coords
        self.blocks = blocks_out


ELEMENT_SPECS = MappingProxyType({
    EElementType.TRI3: ElementSpec(3, True, (0, 1, 2), (0, 2, 1)),
    EElementType.TRI6: ElementSpec(6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3)),
    EElementType.TRI7: ElementSpec(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6), centre_idx=6,
    ),
    EElementType.QUAD4: ElementSpec(4, True, (0, 1, 2, 3), (0, 3, 2, 1)),
    EElementType.QUAD8: ElementSpec(
        8,
        True,
        (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4),
    ),
    EElementType.QUAD9: ElementSpec(
        9, True, (0, 1, 2, 3), (0, 3, 2, 1, 7, 6, 5, 4, 8), centre_idx=8,
    ),
    EElementType.TET4: ElementSpec(
        4, False, (0, 1, 2, 3), None, (0, 2, 1, 3),
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
        edge_pairs=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    ),
    EElementType.TET10: ElementSpec(
        10, False, (0, 1, 2, 3), None, (0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
        ((0, 1, 2, 4, 5, 6), (0, 3, 1, 7, 8, 4),
         (0, 2, 3, 6, 9, 7), (1, 3, 2, 8, 9, 5)),
        edge_pairs=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    ),
    EElementType.HEX8: ElementSpec(
        8, False, (0, 1, 2, 3, 4, 5, 6, 7), None, (0, 3, 2, 1, 4, 7, 6, 5),
        ((0, 1, 2, 3), (0, 3, 7, 4), (4, 7, 6, 5), (1, 5, 6, 2),
         (0, 4, 5, 1), (2, 6, 7, 3)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    ),
    EElementType.HEX20: ElementSpec(
        20, False, (0, 1, 2, 3, 4, 5, 6, 7), None,
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17),
        ((0, 1, 2, 3, 8, 9, 10, 11), (0, 3, 7, 4, 11, 15, 19, 16),
         (4, 7, 6, 5, 15, 14, 13, 12), (1, 5, 6, 2, 17, 13, 18, 9),
         (0, 4, 5, 1, 16, 12, 17, 8), (2, 6, 7, 3, 18, 14, 19, 10)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    ),
    EElementType.HEX27: ElementSpec(
        27, False, (0, 1, 2, 3, 4, 5, 6, 7), None,
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
         20, 21, 22, 23, 24, 25, 26),
        ((0, 1, 2, 3, 8, 9, 10, 11, 24), (0, 3, 7, 4, 11, 15, 19, 16, 23),
         (4, 7, 6, 5, 15, 14, 13, 12, 25), (1, 5, 6, 2, 17, 13, 18, 9, 21),
         (0, 4, 5, 1, 16, 12, 17, 8, 20), (2, 6, 7, 3, 18, 14, 19, 10, 22)),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
        face_corner_idxs=((0, 1, 2, 3), (0, 3, 7, 4), (4, 7, 6, 5),
                             (1, 5, 6, 2), (0, 4, 5, 1), (2, 6, 7, 3)),
        face_centre_idxs=(24, 23, 25, 21, 20, 22),
        cell_centre_idx=26,
    ),
})


_VOLUME_FACE_TYPES = MappingProxyType({
    EElementType.TET4: EElementType.TRI3,
    EElementType.TET10: EElementType.TRI6,
    EElementType.HEX8: EElementType.QUAD4,
    EElementType.HEX20: EElementType.QUAD8,
    EElementType.HEX27: EElementType.QUAD9,
})


def convert_source_block(
    connect: np.ndarray,
    node_count: int,
    spec: SourceBlockSpec,
) -> ElementBlock:
    """Convert explicitly described source connectivity to canonical form."""
    if not isinstance(spec, SourceBlockSpec):
        raise TypeError("'spec' must be a SourceBlockSpec.")
    if not isinstance(node_count, Integral) or isinstance(
        node_count,
        (bool, np.bool_),
    ):
        raise TypeError("'node_count' must be an integer.")
    if node_count <= 0:
        raise ValueError("'node_count' must be positive.")
    node_count = int(node_count)

    source = _normalise_integer_connect(
        connect,
        "connect",
        allow_integral_float=True,
    )
    nodes_per_elem = ELEMENT_SPECS[spec.element_type].nodes_per_elem
    if spec.layout is EConnectLayout.NODE_MAJOR:
        if source.shape[0] != nodes_per_elem:
            raise ValueError(
                f"Node-major {spec.element_type.value} connectivity requires "
                f"exactly {nodes_per_elem} rows, got {source.shape[0]}."
            )
        canonical = np.ascontiguousarray(source.T)
    else:
        if source.shape[1] != nodes_per_elem:
            raise ValueError(
                f"Row-major {spec.element_type.value} connectivity requires "
                f"exactly {nodes_per_elem} columns, got {source.shape[1]}."
            )
        canonical = source

    if spec.indexing is EConnectIndexing.ONE_BASED:
        if np.any(canonical < 1) or np.any(canonical > node_count):
            raise ValueError(
                f"One-based connectivity indices must be in [1, {node_count}]."
            )
        canonical = canonical - 1
    else:
        if np.any(canonical >= node_count):
            raise ValueError(
                f"Zero-based connectivity indices must be in "
                f"[0, {node_count - 1}]."
            )

    if spec.target_to_source_perm is not None:
        canonical = canonical[:, np.asarray(spec.target_to_source_perm, dtype=np.int64)]

    return ElementBlock(spec.element_type, canonical)


@cache
def _get_elem_symmetries(
    elem_type: EElementType,
) -> tuple[tuple[int, ...], ...]:
    """Return lazily calculated proper symmetries for an element type."""
    return elem_type.calc_orient_preserving_perms()


@cache
def _get_elem_symmetry_arrs(
    elem_type: EElementType,
) -> tuple[np.ndarray, ...]:
    """Return cached array forms of an element's proper symmetries."""
    return tuple(
        np.asarray(perm, dtype=np.int64)
        for perm in _get_elem_symmetries(elem_type)
    )


def _check_or_enforce_block(
    block: ElementBlock,
    coords: np.ndarray,
    src_convention: MeshConvention | None,
    enforce: bool,
) -> tuple[np.ndarray, list[MeshCheckCode]]:
    """Check one typed block and optionally enforce geometric conventions."""
    elem_type = block.element_type
    spec = ELEMENT_SPECS[elem_type]
    connect = block.connect
    failures: list[MeshCheckCode] = []

    ordered = _enforce_node_order_table(
        connect,
        elem_type,
        src_convention,
    )
    if not np.array_equal(ordered, connect):
        failures.append(MeshCheckCode.NODE_ORDER)
    connect = ordered

    if spec.is_surf:
        try:
            flips = _calc_surf_orient_flips(connect, coords, spec)
        except ValueError:
            failures.append(MeshCheckCode.SURFACE_TOPOLOGY)
            if enforce:
                connect = _enforce_surf_orient_table(connect, coords, spec)
        else:
            if np.any(flips):
                failures.extend((
                    MeshCheckCode.CCW_WINDING,
                    MeshCheckCode.RIGHT_HANDED_GEOMETRY,
                ))
                if enforce:
                    connect = _apply_surf_flips(connect, flips, spec)
    elif not _check_right_handed_table(connect, coords, spec):
        failures.append(MeshCheckCode.RIGHT_HANDED_GEOMETRY)
        if enforce:
            connect = _enforce_right_handed_table(connect, coords, spec)

    return np.ascontiguousarray(connect, dtype=np.int64), failures


def check_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Return failed geometric checks for each non-conforming block."""
    per_block: MeshConvCheck = {}
    for name, block in mesh_in.blocks.items():
        _, failures = _check_or_enforce_block(
            block,
            mesh_in.coords,
            src_convention,
            enforce=False,
        )
        if failures:
            per_block[name] = failures
    return per_block


def enforce_mesh_convention(
    mesh_in: SimData,
    src_convention: MeshConvention | None = None,
) -> SimData:
    """Return a mesh with Riley local-node and orientation conventions."""
    blocks_out: dict[str, ElementBlock] = {}
    changed = False
    for name, block in mesh_in.blocks.items():
        connect, failures = _check_or_enforce_block(
            block,
            mesh_in.coords,
            src_convention,
            enforce=True,
        )
        changed = changed or bool(failures)
        blocks_out[name] = ElementBlock(
            block.element_type,
            connect,
            block.elem_vars,
        )

    if not changed:
        return mesh_in

    return SimData(
        coords=mesh_in.coords,
        blocks=blocks_out,
        time=mesh_in.time,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        glob_vars=mesh_in.glob_vars,
    )


_ExtractedRecord = tuple[EElementType, np.ndarray, np.ndarray, ElementBlock]
_ExtractedBlocks = dict[str, _ExtractedRecord]


def _build_extracted_mesh(
    mesh_in: SimData,
    extracted: _ExtractedBlocks,
) -> SimData:
    """Remap extracted global connectivity and associated block fields."""
    surf_node_idxs = np.unique(np.concatenate([
        faces.reshape(-1) for _, faces, _, _ in extracted.values()
    ]))
    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0],
        dtype=np.int64,
    )

    blocks_out: dict[str, ElementBlock] = {}
    for name, (elem_type, faces, parent_idxs, source_block) in extracted.items():
        elem_vars = None
        if source_block.elem_vars is not None:
            elem_vars = {
                var_name: values[parent_idxs, ...]
                for var_name, values in source_block.elem_vars.items()
            }
        blocks_out[name] = ElementBlock(
            elem_type,
            coord_remap[faces],
            elem_vars,
        )

    node_vars = None
    if mesh_in.node_vars is not None:
        node_vars = {
            name: values[surf_node_idxs, ...]
            for name, values in mesh_in.node_vars.items()
        }

    return SimData(
        coords=mesh_in.coords[surf_node_idxs],
        blocks=blocks_out,
        time=mesh_in.time,
        side_sets=None,
        node_vars=node_vars,
        glob_vars=mesh_in.glob_vars,
    )


def extract_surf_mesh(mesh_in: SimData) -> SimData:
    """Extract the external surface of explicit volume blocks."""
    if any(
        ELEMENT_SPECS[block.element_type].is_surf
        for block in mesh_in.blocks.values()
    ):
        raise ValueError(
            "extract_surf_mesh requires volume ElementBlock types; surface "
            "input is not supported."
        )

    mesh_std = enforce_mesh_convention(mesh_in)
    extracted: _ExtractedBlocks = {}
    for name, block in mesh_std.blocks.items():
        face_type = _VOLUME_FACE_TYPES[block.element_type]
        faces, parent_idxs = _extract_surf_faces_from_table(
            block.connect,
            mesh_std.coords,
            ELEMENT_SPECS[block.element_type],
            ELEMENT_SPECS[face_type],
        )
        if faces.size:
            extracted[name] = (face_type, faces, parent_idxs, block)

    if not extracted:
        raise ValueError("No external surface faces were found.")
    return _build_extracted_mesh(mesh_std, extracted)


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
) -> SimData:
    """Extract surface faces between two parallel planes."""
    point_arr = _conv_to_vec3(point, "point")
    normal_arr = _conv_to_vec3(normal, "normal")
    normal_magnitude = np.linalg.norm(normal_arr)
    if normal_magnitude < _TOL.geom:
        raise ValueError("Normal vector cannot be zero.")
    normal_arr = normal_arr / normal_magnitude

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("'tolerance' must be a finite non-negative value.")
    if distance is not None and not np.isfinite(distance):
        raise ValueError("'distance' must be finite when provided.")

    mesh_std = enforce_mesh_convention(mesh_in)
    projs = (mesh_std.coords - point_arr) @ normal_arr
    if distance is None:
        min_bound = -tolerance
        max_bound = tolerance
    else:
        plane_distance = float(distance)
        min_bound = min(0.0, plane_distance) - tolerance
        max_bound = max(0.0, plane_distance) + tolerance

    extracted: _ExtractedBlocks = {}
    for name, block in mesh_std.blocks.items():
        elem_spec = ELEMENT_SPECS[block.element_type]
        if elem_spec.is_surf:
            face_type = block.element_type
            candidate_faces = block.connect
            parent_idxs = np.arange(block.connect.shape[0], dtype=np.int64)
        else:
            face_type = _VOLUME_FACE_TYPES[block.element_type]
            if elem_spec.surf_faces is None:
                raise NotImplementedError(
                    f"Surface extraction is not implemented for "
                    f"{block.element_type.value}."
                )
            face_map = np.asarray(elem_spec.surf_faces, dtype=np.int64)
            faces_flat = block.connect[:, face_map].reshape((-1, face_map.shape[1]))
            _, unique_idxs = np.unique(
                np.sort(faces_flat, axis=1),
                axis=0,
                return_index=True,
            )
            candidate_faces = faces_flat[unique_idxs]
            parent_idxs = unique_idxs // face_map.shape[0]

        face_projs = projs[candidate_faces]
        in_bounds = np.all(
            (face_projs >= min_bound) & (face_projs <= max_bound),
            axis=1,
        )
        filtered_faces = candidate_faces[in_bounds]
        if filtered_faces.size:
            extracted[name] = (
                face_type,
                filtered_faces,
                parent_idxs[in_bounds],
                block,
            )

    if not extracted:
        raise ValueError(
            "No elements/faces found between the specified planes."
        )

    return enforce_mesh_convention(_build_extracted_mesh(mesh_std, extracted))

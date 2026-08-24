# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Private implementation of Riley's mesh-convention tools.

The public interface lives in :mod:`riley.python.meshconv`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from itertools import permutations, product
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True, slots=True)
class _Tolerances:
    """Numerical tolerances used by Riley's mesh-convention tools.

    reference: matching coordinates to the reference element layout.
    geometry: general degeneracy and zero-metric epsilon for geometry checks.
    quad8_edge: QUAD8 midside-on-edge fit as a fraction of edge length.
    role_match: higher-order node role match as a fraction of element scale.
    """

    reference: float = 1.0e-12
    geometry: float = 1.0e-12
    quad8_edge: float = 5.0e-2
    role_match: float = 3.5e-1


_TOL = _Tolerances()


def _reference_node_permutation(
    reference: np.ndarray,
    transformed: np.ndarray,
) -> tuple[int, ...]:
    slots: list[int] = []
    for point in transformed:
        matches = np.flatnonzero(
            np.all(np.isclose(reference, point, atol=_TOL.reference), axis=1)
        )
        if matches.shape[0] != 1:
            raise ValueError("Reference transformation does not preserve element roles.")
        slots.append(int(matches[0]))
    if len(set(slots)) != reference.shape[0]:
        raise ValueError("Reference transformation is not a node permutation.")
    return tuple(slots)


def _check_cw_winding(mesh_in: SimData) -> bool:
    """Checks whether all supported surface/2D elements are wound clockwise."""

    if mesh_in.connect is None:
        return True
    if mesh_in.coords is None:
        raise ValueError("Clockwise winding checks require 'coords' to be set.")

    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])
    for connect in mesh_in.connect.values():
        connect_row_major = _enforce_row_major_zero_based_table(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        if not _check_cw_winding_table(connect_row_major, mesh_in.coords):
            return False
    return True


def _check_ccw_winding(mesh_in: SimData) -> bool:
    """Checks whether all supported surface/2D elements are wound counter-clockwise."""

    if mesh_in.connect is None:
        return True
    if mesh_in.coords is None:
        raise ValueError("CCW winding checks require 'coords' to be set.")

    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])
    for connect in mesh_in.connect.values():
        connect_row_major = _enforce_row_major_zero_based_table(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        if not _check_ccw_winding_table(connect_row_major, mesh_in.coords):
            return False
    return True


def _enforce_cw_winding(mesh_in: SimData) -> SimData:
    """Returns a copy of ``mesh_in`` with supported 2D/surface elements wound CW."""

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("Clockwise winding enforcement requires 'coords' to be set.")

    connect_out: dict[str, np.ndarray] = {}
    changed = False
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect in mesh_in.connect.items():
        connect_row_major = _enforce_row_major_zero_based_table(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        connect_out[name] = _enforce_cw_winding_table(connect_row_major, mesh_in.coords)
        changed = changed or not np.array_equal(connect_out[name], connect_row_major)

    if not changed:
        return mesh_in

    return _copy_sim_data(mesh_in, connect=connect_out)


def _enforce_ccw_winding(mesh_in: SimData) -> SimData:
    """Returns a copy of ``mesh_in`` with supported 2D/surface elements wound CCW."""

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("CCW winding enforcement requires 'coords' to be set.")

    connect_out: dict[str, np.ndarray] = {}
    changed = False
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect in mesh_in.connect.items():
        connect_row_major = _enforce_row_major_zero_based_table(
            connect,
            mesh_in.coords.shape[0],
            shift_all=shift_all,
        )
        connect_out[name] = _enforce_ccw_winding_table(connect_row_major, mesh_in.coords)
        changed = changed or not np.array_equal(connect_out[name], connect_row_major)

    if not changed:
        return mesh_in

    return _copy_sim_data(mesh_in, connect=connect_out)


def _check_mesh_2d(mesh_in: SimData) -> bool:
    if mesh_in.coords is None or mesh_in.connect is None:
        return False

    if _check_volume_mesh(mesh_in):
        return False

    # 1. Check coordinate flatness
    coord_ranges = np.ptp(mesh_in.coords, axis=0)
    if np.any(coord_ranges < 1e-12):
        return True

    # 2. Check elements in connectivity tables
    for name, connect_raw in mesh_in.connect.items():
        connect = np.asarray(connect_raw, dtype=np.int64)
        if _check_transpose_needed(connect, name, mesh_in):
            connect = connect.T

        # Normalize 1-based indexing if present to avoid out-of-bounds errors
        shift_all = _check_mesh_needs_zero_based_shift(
            mesh_in, mesh_in.coords.shape[0]
        )
        legacy_connect = _check_table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            connect = connect - 1

        nodes_per_elem = connect.shape[1]
        if nodes_per_elem in _SURFACE_ONLY_NODE_COUNTS:
            return True
        if nodes_per_elem in _VOLUME_ONLY_NODE_COUNTS:
            return False

        try:
            volume_spec = _get_volume_spec(nodes_per_elem)
            for elem in connect[:10]:
                corner_indices = np.asarray(
                    volume_spec.corner_indices,
                    dtype=np.int64,
                )
                cell_coords = mesh_in.coords[elem[corner_indices]]
                if abs(_calc_volume_signed_metric(cell_coords)) > _TOL.geometry:
                    break
            else:
                return True
        except IndexError:
            pass

    return False


def _check_volume_mesh(mesh_in: SimData) -> bool:
    """Return whether every connectivity table describes volume elements.

    Element node count usually establishes the topology. Four-node and
    eight-node tables are ambiguous (TET4/QUAD4 and HEX8/QUAD8), so they are
    classified from their signed cell volume. A mesh containing both surface
    and volume tables is rejected because it has no single ``EMeshType``.
    """
    if mesh_in.coords is None or mesh_in.connect is None:
        return False

    num_coords = mesh_in.coords.shape[0]
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, num_coords)
    table_types: set[bool] = set()

    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_array_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            connect = connect.T

        if _check_table_needs_zero_based_shift(connect, num_coords, shift_all):
            connect = connect - 1

        if not _check_indices_zero_based(connect, num_coords):
            raise ValueError(
                f"Connectivity table '{name}' has invalid indices."
            )

        table_types.add(_check_volume_connectivity_table(connect, mesh_in.coords))

    if len(table_types) > 1:
        raise ValueError(
            "A SimData mesh cannot mix surface and volume connectivity tables."
        )

    return bool(table_types and table_types.pop())


def _check_volume_connectivity_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> bool:
    nodes_per_elem = connect.shape[1]
    _supported_nodes_per_elem(nodes_per_elem)

    if nodes_per_elem in _VOLUME_ONLY_NODE_COUNTS:
        return True
    if nodes_per_elem in _SURFACE_ONLY_NODE_COUNTS:
        return False

    if _get_surface_spec(nodes_per_elem) is ELEMENT_SPECS[EElementType.QUAD8]:
        if all(_check_quad8_surface_row(row, coords) for row in connect):
            return False

    volume_spec = _get_volume_spec(nodes_per_elem)
    corner_indices = np.asarray(volume_spec.corner_indices, dtype=np.int64)
    for row in connect:
        cell_coords = coords[row[corner_indices]]
        metric = _calc_volume_signed_metric(cell_coords)

        if abs(metric) > _TOL.geometry:
            return True

    return False


def _check_quad8_surface_row(
    connect_row: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Return whether an eight-node row has the QUAD8 midside layout."""
    elem_coords = coords[connect_row]
    corners = elem_coords[:4]
    midsides = elem_coords[4:]
    edge_starts = corners
    edge_ends = np.roll(corners, -1, axis=0)
    edges = edge_ends - edge_starts
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= _TOL.geometry):
        return False

    edge_params = np.sum((midsides - edge_starts) * edges, axis=1)
    edge_params /= edge_lengths**2
    closest = edge_starts + edge_params[:, None] * edges
    distances = np.linalg.norm(midsides - closest, axis=1)
    on_edges = distances <= _TOL.quad8_edge * edge_lengths
    between_corners = np.logical_and(edge_params >= 0.0, edge_params <= 1.0)
    return bool(np.all(np.logical_and(on_edges, between_corners)))


def _check_surface_connectivity_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> bool:
    """Return whether a connectivity table represents surface elements."""

    if surface_only:
        return True
    return not _check_volume_connectivity_table(connect, coords)


def _surface_orientation_flips(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Return the row reversals required by the canonical surface convention.

    Surface winding is a property of the complete surface, rather than of an
    individual face.  In particular, the boundary of a bore has material
    normals directed *towards* the bore axis.  A per-face test against the mesh
    centroid cannot represent that topology.  This routine first makes each
    connected component edge-consistent, then selects the component orientation
    from signed enclosed volume.  Nested closed components alternate between
    material exterior and cavity boundaries.
    """

    corner_inds = _get_corner_indices(connect.shape[1])
    representatives: dict[tuple[int, ...], int] = {}
    duplicate_of = np.arange(connect.shape[0], dtype=np.int64)
    for row_ind, row in enumerate(connect):
        key = tuple(sorted(int(node) for node in row[corner_inds]))
        representative = representatives.setdefault(key, row_ind)
        duplicate_of[row_ind] = representative

    try:
        topology = _surface_topology(connect, np.unique(duplicate_of))
    except ValueError as error:
        if "Non-manifold surface edge" not in str(error):
            raise
        # Surface slices can deliberately contain non-manifold face sets. They
        # have no single orientable shell, so retain the historical per-face
        # behaviour rather than applying a false cavity interpretation.
        return _legacy_surface_orientation_flips(connect, coords)
    flips = np.zeros(connect.shape[0], dtype=bool)
    closed_components: list[tuple[np.ndarray, np.ndarray]] = []

    for rows, edge_keys in topology:
        relative = _component_relative_flips(rows, edge_keys)
        flips[rows] = relative
        oriented = _apply_surface_flips(connect[rows], relative)

        is_closed = all(len(edge_keys[key]) == 2 for key in edge_keys)
        if not is_closed:
            # An open non-planar sheet has no intrinsic exterior. Preserve a
            # coherent input orientation; planar sheets retain canonical CCW.
            component_nodes = np.unique(oriented[:, _get_corner_indices(
                oriented.shape[1]
            )])
            component_coords = coords[component_nodes]
            if _check_coplanar(component_coords):
                metric = _calc_first_surface_metric(oriented, component_coords, coords)
                if metric is not None and metric < 0.0:
                    flips[rows] = ~flips[rows]
            continue

        volume = _calc_surface_signed_volume(oriented, coords)
        if abs(volume) <= _TOL.geometry:
            raise ValueError(
                "Closed surface component has zero signed volume; cannot "
                "select a material exterior."
            )
        if volume < 0.0:
            flips[rows] = ~flips[rows]
            oriented = _apply_surface_flips(oriented, np.ones(rows.shape[0], dtype=bool))
        closed_components.append((rows, oriented))

    # A disconnected closed shell contained by another shell is a cavity. Its
    # material-outward normal must point into the void, so its signed volume is
    # negative after local edge consistency has been established.
    for component_ind, (rows, oriented) in enumerate(closed_components):
        point = _calc_component_probe_point(oriented, coords)
        depth = sum(
            _check_point_in_closed_surface(point, other_oriented, coords)
            for other_ind, (_, other_oriented) in enumerate(closed_components)
            if other_ind != component_ind
        )
        if depth % 2:
            flips[rows] = ~flips[rows]

    for row_ind, representative in enumerate(duplicate_of):
        if row_ind == representative:
            continue
        same_orientation = _check_surface_rows_same_orientation(
            connect[row_ind],
            connect[representative],
            coords,
        )
        flips[row_ind] = flips[representative] ^ (not same_orientation)

    return flips


def _legacy_surface_orientation_flips(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    flips = np.zeros(connect.shape[0], dtype=bool)
    for row_ind, row in enumerate(connect):
        metric = _calc_winding_metric(row, coords, surface_only=True)
        flips[row_ind] = metric is not None and metric < 0.0
    return flips


def _surface_topology(
    connect: np.ndarray,
    active_rows: np.ndarray,
) -> list[tuple[np.ndarray, dict]]:
    """Build connected surface components keyed by their corner-node edges."""

    corner_inds = _get_corner_indices(connect.shape[1])
    edge_map: dict = {}
    for row_ind in active_rows:
        row = connect[row_ind]
        corners = row[corner_inds]
        for node_a, node_b in zip(corners, np.roll(corners, -1)):
            directed = (int(node_a), int(node_b))
            key = tuple(sorted(directed))
            edge_map.setdefault(key, []).append((row_ind, directed))

    for key, uses in edge_map.items():
        if len(uses) > 2:
            raise ValueError(
                f"Non-manifold surface edge {key} has {len(uses)} incident faces."
            )

    neighbours: dict[int, set[int]] = {int(row): set() for row in active_rows}
    for uses in edge_map.values():
        if len(uses) == 2:
            row_a, _ = uses[0]
            row_b, _ = uses[1]
            neighbours[row_a].add(row_b)
            neighbours[row_b].add(row_a)

    components: list[tuple[np.ndarray, dict]] = []
    unseen = set(int(row) for row in active_rows)
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
        rows_array = np.asarray(sorted(rows), dtype=np.int64)
        row_set = set(rows_array.tolist())
        component_edges = {
            key: uses for key, uses in edge_map.items()
            if uses[0][0] in row_set
        }
        components.append((rows_array, component_edges))
    return components


def _check_surface_rows_same_orientation(
    row_a: np.ndarray,
    row_b: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Return whether duplicate surface rows have the same directed boundary."""

    corner_inds = _get_corner_indices(row_a.shape[0])
    corners_a = row_a[corner_inds]
    corners_b = row_b[corner_inds]
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


def _component_relative_flips(
    rows: np.ndarray,
    edge_keys: dict,
) -> np.ndarray:
    """Find local reversals making shared edges traverse opposite ways."""

    row_set = set(rows.tolist())
    constraints: dict[int, list[tuple[int, bool]]] = {row: [] for row in row_set}
    for uses in edge_keys.values():
        if len(uses) != 2:
            continue
        row_a, direction_a = uses[0]
        row_b, direction_b = uses[1]
        same_direction = direction_a == direction_b
        constraints[row_a].append((row_b, same_direction))
        constraints[row_b].append((row_a, same_direction))

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


def _apply_surface_flips(connect: np.ndarray, flips: np.ndarray) -> np.ndarray:
    out = np.array(connect, copy=True)
    for row_ind in np.flatnonzero(flips):
        out[row_ind] = _reverse_surface_row(out[row_ind])
    return out


def _calc_first_surface_metric(
    connect: np.ndarray,
    component_coords: np.ndarray,
    coords: np.ndarray,
) -> float | None:
    normal = _calc_canonical_plane_normal(component_coords)
    corner_inds = _get_corner_indices(connect.shape[1])
    for row in connect:
        metric = _calc_local_polygon_signed_area(coords[row[corner_inds]], normal)
        if metric is not None and abs(metric) > _TOL.geometry:
            return metric
    return None


def _calc_surface_signed_volume(connect: np.ndarray, coords: np.ndarray) -> float:
    corner_inds = _get_corner_indices(connect.shape[1])
    volume = 0.0
    for row in connect:
        points = coords[row[corner_inds]]
        for point_ind in range(1, points.shape[0] - 1):
            volume += float(np.dot(
                points[0],
                np.cross(points[point_ind], points[point_ind + 1]),
            )) / 6.0
    return volume


def _calc_component_probe_point(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    corner_inds = _get_corner_indices(connect.shape[1])
    points = coords[connect[0, corner_inds]]
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= _TOL.geometry:
        return np.mean(points, axis=0)
    extent = np.ptp(coords, axis=0)
    epsilon = max(float(np.linalg.norm(extent)) * 1.0e-9, _TOL.geometry * 10.0)
    return np.mean(points, axis=0) + epsilon * normal / normal_norm


def _check_point_in_closed_surface(
    point: np.ndarray,
    connect: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Classify a point with parity ray casting against a closed surface."""

    corner_inds = _get_corner_indices(connect.shape[1])
    directions = np.array((
        (0.745, 0.371, 0.553),
        (-0.299, 0.877, 0.376),
        (0.461, -0.314, 0.830),
    ))
    votes: list[bool] = []
    for direction in directions:
        direction = direction / np.linalg.norm(direction)
        hits = 0
        for row in connect:
            corners = coords[row[corner_inds]]
            for point_ind in range(1, corners.shape[0] - 1):
                if _check_ray_intersects_triangle(
                    point,
                    direction,
                    corners[0],
                    corners[point_ind],
                    corners[point_ind + 1],
                ):
                    hits += 1
        votes.append(bool(hits % 2))
    return sum(votes) >= 2


def _check_ray_intersects_triangle(
    origin: np.ndarray,
    direction: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> bool:
    edge_ab = point_b - point_a
    edge_ac = point_c - point_a
    perpendicular = np.cross(direction, edge_ac)
    determinant = float(np.dot(edge_ab, perpendicular))
    if abs(determinant) <= _TOL.geometry:
        return False
    inv_determinant = 1.0 / determinant
    offset = origin - point_a
    u = inv_determinant * float(np.dot(offset, perpendicular))
    if u <= _TOL.geometry or u >= 1.0 - _TOL.geometry:
        return False
    q_vec = np.cross(offset, edge_ab)
    v = inv_determinant * float(np.dot(direction, q_vec))
    if v <= _TOL.geometry or u + v >= 1.0 - _TOL.geometry:
        return False
    distance = inv_determinant * float(np.dot(edge_ac, q_vec))
    return distance > _TOL.geometry


def _enforce_surface_orientation_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    return _apply_surface_flips(
        connect, _surface_orientation_flips(connect, coords)
    )


def _copy_sim_data(
    mesh_in: SimData,
    connect: dict[str, np.ndarray] | None = None,
) -> SimData:
    mesh_out = SimData(
        mesh_type=mesh_in.mesh_type,
        time=mesh_in.time,
        coords=mesh_in.coords,
        connect=mesh_in.connect if connect is None else connect,
        side_sets=mesh_in.side_sets,
        node_vars=mesh_in.node_vars,
        elem_vars=mesh_in.elem_vars,
        glob_vars=mesh_in.glob_vars,
    )
    return mesh_out


def _infer_mesh_convention(mesh_in: SimData) -> MeshConvention:
    """Infer one source-to-Riley ordering for each supported element family.

    A convention is only inferred when every row of a table establishes the
    same permutation.  This deliberately rejects mixtures of local layouts:
    callers must declare those explicitly rather than receive a guessed mesh.
    """
    if mesh_in.coords is None or mesh_in.connect is None:
        raise MeshConvError(
            "Mesh convention inference requires coordinates and connectivity."
        )

    inferred: dict[EElementType, tuple[int, ...]] = {}
    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_array_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            connect = connect.T
        if _check_zero_based_shift_needed(connect, mesh_in.coords.shape[0]):
            connect = connect - 1
        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise MeshConvError(
                f"Connectivity table '{name}' has invalid indices."
            )
        nodes_per_element = connect.shape[1]
        if nodes_per_element in _SURFACE_ONLY_NODE_COUNTS:
            spec = _get_surface_spec(nodes_per_element)
        elif nodes_per_element in _VOLUME_ONLY_NODE_COUNTS:
            spec = _get_volume_spec(nodes_per_element)
        elif _check_surface_mesh_type(mesh_in.mesh_type):
            spec = _get_surface_spec(nodes_per_element)
        elif _check_volume_connectivity_table(connect, mesh_in.coords):
            spec = _get_volume_spec(nodes_per_element)
        else:
            spec = _get_surface_spec(nodes_per_element)
        element_type = _get_element_type_from_spec(spec)
        try:
            normalised = _enforce_node_order_from_geometry(
                connect, mesh_in.coords, spec,
            )
        except ValueError as error:
            raise MeshConvError(
                f"Could not infer '{name}' ({element_type.value}); supply "
                "MeshConvention explicitly."
            ) from error
        permutations = {
            tuple(int(np.flatnonzero(row == node_id)[0]) for node_id in target)
            for row, target in zip(connect, normalised, strict=True)
        }
        representative = min(permutations)
        symmetries = ELEMENT_SYMMETRIES[element_type]
        if any(
            not any(
                permutation == tuple(
                    representative[index] for index in symmetry
                )
                for symmetry in symmetries
            )
            for permutation in permutations
        ):
            raise MeshConvError(
                f"Connectivity table '{name}' contains multiple source "
                f"layouts for {element_type.value}; supply MeshConvention."
            )
        permutation = representative
        previous = inferred.get(element_type)
        if previous is not None and not any(
            permutation == tuple(
                previous[index] for index in symmetry
            )
            for symmetry in symmetries
        ):
            raise MeshConvError(
                (
                    f"Multiple connectivity tables disagree on the "
                    f"{element_type.value} source layout; supply MeshConvention."
                )
            )
        inferred[element_type] = permutation
    return MeshConvention(
        MappingProxyType(inferred),
        canonicalise_equivalent_orientations=True,
    )


def _check_surface_mesh_type(mesh_type: EMeshType | None) -> bool:
    return mesh_type is EMeshType.SURF


def _enforce_connect_array_format(connect: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(connect)
    if array.ndim != 2:
        raise ValueError(
            f"Connectivity table '{name}' must be 2D, got shape {array.shape}."
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def _supported_nodes_per_elem(nodes_per_elem: int) -> None:
    if nodes_per_elem not in _SUPPORTED_NODE_COUNTS:
        raise NotImplementedError(
            "Mesh convention tools do not support elements with "
            f"{nodes_per_elem} nodes."
        )


def _get_surface_spec(nodes_per_elem: int) -> ElementSpec:
    for spec in _ELEMENT_SPECS_BY_NODE_COUNT.get(nodes_per_elem, ()):
        if spec.is_surface and spec.nodes_per_element == nodes_per_elem:
            return spec
    raise NotImplementedError(
        (
            f"Surface metadata is not implemented for "
            f"{nodes_per_elem}-node elements."
        )
    )


def _get_volume_spec(nodes_per_elem: int) -> ElementSpec:
    for spec in _ELEMENT_SPECS_BY_NODE_COUNT.get(nodes_per_elem, ()):
        if not spec.is_surface and spec.nodes_per_element == nodes_per_elem:
            return spec
    raise NotImplementedError(
        (
            f"Volume metadata is not implemented for "
            f"{nodes_per_elem}-node elements."
        )
    )


def _check_transpose_needed(
    connect: np.ndarray,
    connect_name: str | None = None,
    mesh_in: SimData | None = None,
) -> bool:
    rows_supported = connect.shape[0] in _SUPPORTED_NODE_COUNTS
    cols_supported = connect.shape[1] in _SUPPORTED_NODE_COUNTS

    if rows_supported and not cols_supported:
        return True
    if cols_supported and not rows_supported:
        return False
    if cols_supported and rows_supported:
        block_rows = _get_elem_var_block_rows(connect_name, mesh_in)
        if block_rows is not None:
            row_match = connect.shape[0] in block_rows
            col_match = connect.shape[1] in block_rows
            if row_match and not col_match:
                return False
            if col_match and not row_match:
                return True
        if mesh_in is not None and mesh_in.coords is not None:
            return (
                _calc_orientation_score(connect.T, mesh_in.coords)
                > _calc_orientation_score(connect, mesh_in.coords)
            )
        return False
    raise NotImplementedError(
        "Could not infer connectivity orientation from shape "
        f"{connect.shape}. Expected a supported nodes-per-element dimension."
    )


def _check_zero_based_shift_needed(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return False
    if np.any(connect < 0):
        return False
    if np.any(connect == 0):
        return False
    return bool(np.any(connect >= num_coords))


def _check_ambiguously_positive(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return False
    return bool(
        np.all(connect >= 1)
        and np.all(connect < num_coords)
        and not np.any(connect == 0)
    )


def _check_mesh_needs_zero_based_shift(mesh_in: SimData, num_coords: int) -> bool:
    any_zero_based = False
    any_definitely_one_based = False

    for connect_name, connect_raw in mesh_in.connect.items():
        connect = np.asarray(connect_raw, dtype=np.int64)
        if _check_transpose_needed(connect, connect_name, mesh_in):
            connect = connect.T
        if np.any(connect == 0):
            any_zero_based = True
        if _check_zero_based_shift_needed(connect, num_coords):
            any_definitely_one_based = True

    if any_zero_based and any_definitely_one_based:
        raise ValueError(
            (
                "Mixed zero-based and one-based connectivity tables "
                "detected in the same mesh."
            )
        )

    return any_definitely_one_based and not any_zero_based


def _check_table_needs_zero_based_shift(
    connect: np.ndarray,
    num_coords: int,
    shift_all: bool,
) -> bool:
    if _check_zero_based_shift_needed(connect, num_coords):
        return True
    return shift_all and _check_ambiguously_positive(connect, num_coords)


def _get_elem_var_block_rows(
    connect_name: str | None,
    mesh_in: SimData | None,
) -> set[int] | None:
    if connect_name is None or mesh_in is None or mesh_in.elem_vars is None:
        return None

    try:
        block_id = int(connect_name.replace("connect", ""))
    except ValueError:
        return None

    row_counts = {
        values.shape[0]
        for (_field_name, field_block), values in mesh_in.elem_vars.items()
        if field_block == block_id
    }
    return row_counts or None


def _check_indices_zero_based(connect: np.ndarray, num_coords: int) -> bool:
    if connect.size == 0:
        return True
    valid_mask = np.logical_and(connect >= 0, connect < num_coords)
    return bool(np.all(valid_mask))


def _enforce_row_major_zero_based_table(
    connect: np.ndarray,
    num_coords: int,
    shift_all: bool = False,
) -> np.ndarray:
    connect_out = np.asarray(connect, dtype=np.int64)
    if connect_out.ndim != 2:
        raise ValueError(
        f"Connectivity table must be 2D, got shape {connect_out.shape}."
    )
    if _check_transpose_needed(connect_out):
        connect_out = connect_out.T
    legacy_connect = _check_table_needs_zero_based_shift(
        connect_out, num_coords, shift_all
    )
    if legacy_connect:
        connect_out = connect_out - 1
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _calc_orientation_score(connect_row_major: np.ndarray, coords: np.ndarray) -> int:
    num_coords = coords.shape[0]
    connect_eval = np.asarray(connect_row_major, dtype=np.int64)

    if connect_eval.ndim != 2 or connect_eval.shape[1] not in _SUPPORTED_NODE_COUNTS:
        return -1

    if _check_zero_based_shift_needed(connect_eval, num_coords):
        connect_eval = connect_eval - 1

    if not _check_indices_zero_based(connect_eval, num_coords):
        return -1

    score = 0
    for row in connect_eval:
        if np.unique(row).shape[0] != row.shape[0]:
            continue

        try:
            metric = _calc_row_orientation_metric(row, coords)
        except ValueError:
            continue

        if metric is not None and abs(metric) > _TOL.geometry:
            score += 1

    return score


def _calc_row_orientation_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
) -> float | None:
    """Return a metric only for resolving row-major connectivity ambiguity."""
    nodes_per_elem = connect_row.shape[0]

    if nodes_per_elem in _SURFACE_NODE_COUNTS:
        corner_coords = coords[connect_row[_get_corner_indices(nodes_per_elem)]]
        if _check_coplanar(corner_coords):
            return _calc_polygon_signed_area(corner_coords)

    if nodes_per_elem in _VOLUME_NODE_COUNTS:
        volume_spec = _get_volume_spec(nodes_per_elem)
        volume_coords = coords[
            connect_row[np.asarray(volume_spec.corner_indices, dtype=np.int64)]
        ]
        metric = _calc_volume_signed_metric(volume_coords)
        if nodes_per_elem in _VOLUME_ONLY_NODE_COUNTS or abs(metric) > _TOL.geometry:
            return metric

    return None


def _enforce_node_order_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
    source_convention: MeshConvention | None = None,
) -> np.ndarray:
    """Rebuild higher-order rows from geometry, independent of source order."""
    if connect.shape[1] in _SURFACE_ONLY_NODE_COUNTS:
        spec = _get_surface_spec(connect.shape[1])
    elif connect.shape[1] in _VOLUME_ONLY_NODE_COUNTS:
        spec = _get_volume_spec(connect.shape[1])
    elif surface_only or not _check_volume_connectivity_table(connect, coords):
        spec = _get_surface_spec(connect.shape[1])
    else:
        spec = _get_volume_spec(connect.shape[1])

    if source_convention is not None:
        permutation = source_convention.permutation_for(
            _get_element_type_from_spec(spec)
        )
        if permutation is not None:
            if len(permutation) != connect.shape[1] or set(permutation) != set(
                range(connect.shape[1])
            ):
                raise ValueError(
                    "MeshConvention permutation does not match the element "
                    f"topology {_get_element_type_from_spec(spec).value}."
                )
            normalised = np.ascontiguousarray(
                connect[:, np.asarray(permutation, dtype=np.int64)],
                dtype=np.int64,
            )
            if source_convention.canonicalise_equivalent_orientations:
                return _enforce_canonical_orientations(normalised, spec)
            return normalised

    return connect


def _get_element_type_from_spec(spec: ElementSpec) -> EElementType:
    for element_type, registered_spec in ELEMENT_SPECS.items():
        if registered_spec is spec:
            return element_type
    raise ValueError("Element specification is not registered.")


def _enforce_node_order_high_order(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Rebuild higher-order rows from inferred geometric roles."""

    connect_out = np.empty_like(connect)
    for row_index, row in enumerate(connect):
        connect_out[row_index] = _enforce_node_order_high_order_row(row, coords, spec)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_node_order_from_geometry(
    connect: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Infer node roles, then select the unique ID-stable proper orientation."""
    if connect.shape[1] == len(spec.corner_indices):
        connect_out = np.empty_like(connect)
        source_slots = np.arange(connect.shape[1], dtype=np.int64)
        for row_index, row in enumerate(connect):
            if spec.is_surface:
                ordered = _order_surface_corners(coords[row], source_slots)
            elif len(spec.corner_indices) == 4:
                ordered = _order_tet_corners(coords[row], source_slots)
            else:
                ordered = _order_hex_corners(coords[row], source_slots)
            connect_out[row_index] = row[ordered]
    else:
        connect_out = _enforce_node_order_high_order(connect, coords, spec)
    return _enforce_canonical_orientations(connect_out, spec)


def _enforce_canonical_orientations(
    connect: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    """Anchor each role-correct row at the lowest valid global-node sequence."""
    symmetries = ELEMENT_SYMMETRIES[_get_element_type_from_spec(spec)]
    permutations_array = tuple(
        np.asarray(permutation, dtype=np.int64) for permutation in symmetries
    )
    out = np.empty_like(connect)
    for row_index, row in enumerate(connect):
        out[row_index] = min(
            (row[permutation] for permutation in permutations_array),
            key=lambda candidate: tuple(int(node) for node in candidate),
        )
    return np.ascontiguousarray(out, dtype=np.int64)


def _enforce_node_order_high_order_row(
    connect_row: np.ndarray,
    coords: np.ndarray,
    spec: ElementSpec,
) -> np.ndarray:
    elem_coords = coords[connect_row]
    if _check_canonical_node_roles(elem_coords, spec):
        return connect_row
    if _check_coincident_nodes(elem_coords):
        return connect_row

    corner_count = len(spec.corner_indices)
    centre = np.mean(elem_coords, axis=0)
    corner_local = _infer_corner_nodes(elem_coords, corner_count)

    if spec.is_surface:
        corner_local = _order_surface_corners(elem_coords, corner_local)
    elif corner_count == 4:
        corner_local = _order_tet_corners(elem_coords, corner_local)
    else:
        corner_local = _order_hex_corners(elem_coords, corner_local)

    corner_coords = elem_coords[corner_local]
    role_locals = list(corner_local)
    remaining = [
        index for index in range(connect_row.shape[0])
        if index not in corner_local
    ]

    edge_pairs = spec.edge_pairs or tuple(
        (index, (index + 1) % corner_count)
        for index in range(corner_count)
    )
    edge_targets = np.array(
        [
            0.5 * (corner_coords[start] + corner_coords[end])
            for start, end in edge_pairs
        ],
        dtype=np.float64,
    )
    edge_local = _match_role_nodes(
        elem_coords,
        remaining,
        edge_targets,
        "edge",
    )
    role_locals.extend(edge_local)
    remaining = [index for index in remaining if index not in edge_local]

    if spec.face_centre_indices:
        face_targets = np.array(
            [
                np.mean(corner_coords[list(face)], axis=0)
                for face in spec.face_corner_indices
            ],
            dtype=np.float64,
        )
        face_local = _match_role_nodes(
            elem_coords,
            remaining,
            face_targets,
            "face centre",
        )
        role_locals.extend(face_local)
        remaining = [index for index in remaining if index not in face_local]

    if spec.centre_index is not None or spec.cell_centre_index is not None:
        centre_local = _match_role_nodes(
            elem_coords,
            remaining,
            np.mean(corner_coords, axis=0, keepdims=True),
            "centre",
        )
        role_locals.extend(centre_local)
        remaining = [index for index in remaining if index not in centre_local]

    if remaining:
        raise ValueError(
            "Could not assign every higher-order node to an element role."
        )

    out_local = np.empty(connect_row.shape[0], dtype=np.int64)
    out_local[np.asarray(spec.corner_indices, dtype=np.int64)] = corner_local
    edge_slots = np.arange(
        corner_count, corner_count + len(edge_pairs), dtype=np.int64
    )
    out_local[edge_slots] = edge_local

    if spec.face_centre_indices:
        out_local[
            np.asarray(spec.face_centre_indices, dtype=np.int64)
        ] = face_local
    if spec.centre_index is not None:
        out_local[spec.centre_index] = centre_local[0]
    if spec.cell_centre_index is not None:
        out_local[spec.cell_centre_index] = centre_local[0]

    return connect_row[out_local]


def _check_canonical_node_roles(
    elem_coords: np.ndarray,
    spec: ElementSpec,
) -> bool:
    """Accept an already coherent row, including curved and seam elements."""
    corner_indices = np.asarray(spec.corner_indices, dtype=np.int64)
    corner_coords = elem_coords[corner_indices]
    if _check_coincident_nodes(elem_coords):
        return True

    rank_required = 2 if spec.is_surface else 3
    if np.linalg.matrix_rank(
        corner_coords - np.mean(corner_coords, axis=0), tol=_TOL.geometry
    ) < rank_required:
        return False

    inferred_corners = _infer_corner_nodes(elem_coords, len(corner_indices))
    if (set(inferred_corners) == set(corner_indices)
            and inferred_corners.shape[0] == len(corner_indices)):
        pass
    elif inferred_corners.shape[0] == len(corner_indices):
        return False

    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geometry)
    edge_pairs = spec.edge_pairs or tuple(
        (index, (index + 1) % len(corner_indices))
        for index in range(len(corner_indices))
    )
    edge_nodes = elem_coords[len(corner_indices):len(corner_indices) + len(edge_pairs)]
    edge_distances = np.array(
        [
            [
                _calc_point_segment_distance(
                    node, corner_coords[start], corner_coords[end]
                )
                for start, end in edge_pairs
            ]
            for node in edge_nodes
        ]
    )
    if np.any(edge_distances > _TOL.role_match * scale):
        return False

    if spec.centre_index is not None:
        if np.linalg.norm(
            elem_coords[spec.centre_index] - np.mean(corner_coords, axis=0)
        ) > _TOL.role_match * scale:
            return False
    if spec.cell_centre_index is not None:
        if np.linalg.norm(
            elem_coords[spec.cell_centre_index] - np.mean(corner_coords, axis=0)
        ) > _TOL.role_match * scale:
            return False
    return True


def _check_coincident_nodes(elem_coords: np.ndarray) -> bool:
    scale = max(float(np.ptp(elem_coords, axis=0).max()), 1.0)
    differences = elem_coords[:, None, :] - elem_coords[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    upper_triangle = distances[np.triu_indices(elem_coords.shape[0], k=1)]
    return bool(np.any(upper_triangle <= _TOL.geometry * scale))


def _infer_corner_nodes(elem_coords: np.ndarray, corner_count: int) -> np.ndarray:
    """Find affine-element vertices without depending on their source slots."""
    scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geometry)
    midpoint_tol = _TOL.geometry * scale * 100.0
    midpoint_nodes: set[int] = set()
    for node_index, point in enumerate(elem_coords):
        for first in range(elem_coords.shape[0]):
            for second in range(first + 1, elem_coords.shape[0]):
                if node_index in (first, second):
                    continue
                midpoint = 0.5 * (elem_coords[first] + elem_coords[second])
                if np.linalg.norm(point - midpoint) <= midpoint_tol:
                    midpoint_nodes.add(node_index)
                    break
            if node_index in midpoint_nodes:
                break
    candidates = np.array(
        [
            index for index in range(elem_coords.shape[0])
            if index not in midpoint_nodes
        ],
        dtype=np.int64,
    )
    if candidates.shape[0] == corner_count:
        return candidates
    distances = np.linalg.norm(elem_coords - np.mean(elem_coords, axis=0), axis=1)
    return np.sort(np.argsort(distances)[-corner_count:])


def _calc_point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= _TOL.geometry:
        return np.inf
    parameter = np.clip(
        float(np.dot(point - start, direction) / length_sq), 0.0, 1.0
    )
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _order_surface_corners(
    elem_coords: np.ndarray,
    corner_local: np.ndarray,
) -> np.ndarray:
    corner_coords = elem_coords[corner_local]
    centred = corner_coords - np.mean(corner_coords, axis=0)
    if np.linalg.matrix_rank(centred, tol=_TOL.geometry) < 2:
        raise ValueError("Degenerate surface element has collinear corner nodes.")
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    normal = vectors[-1]
    axis_u = vectors[0]
    axis_v = np.cross(normal, axis_u)
    angles = np.arctan2(centred @ axis_v, centred @ axis_u)
    ordered = corner_local[np.argsort(angles)]
    ordered_coords = elem_coords[ordered]
    if len(ordered) == 3:
        signed = np.dot(
            np.cross(
                ordered_coords[1] - ordered_coords[0],
                ordered_coords[2] - ordered_coords[0]
            ),
            normal,
        )
    else:
        signed = np.dot(
            np.cross(
                ordered_coords[1] - ordered_coords[0],
                ordered_coords[2] - ordered_coords[0]
            ),
            normal,
        )
    if signed < 0.0:
        ordered = ordered[[0, *range(len(ordered) - 1, 0, -1)]]
    return ordered


def _order_tet_corners(
    elem_coords: np.ndarray,
    corner_local: np.ndarray,
) -> np.ndarray:
    ordered = corner_local[np.lexsort(elem_coords[corner_local].T[::-1])]
    corner_coords = elem_coords[ordered]
    metric = _calc_tet_signed_volume(corner_coords)
    if abs(metric) <= _TOL.geometry:
        raise ValueError("Degenerate tetrahedron has zero signed volume.")
    if metric < 0.0:
        ordered[[1, 2]] = ordered[[2, 1]]
    return ordered


def _order_hex_corners(
    elem_coords: np.ndarray,
    corner_local: np.ndarray,
) -> np.ndarray:
    corner_coords = elem_coords[corner_local]
    if np.linalg.matrix_rank(
        corner_coords - np.mean(corner_coords, axis=0), tol=_TOL.geometry
    ) < 3:
        raise ValueError("Degenerate hexahedron has coplanar corner nodes.")

    origin = corner_local[np.lexsort(corner_coords.T[::-1])[0]]
    candidates = [index for index in corner_local if index != origin]
    best_order: np.ndarray | None = None
    best_error = np.inf
    origin_coord = elem_coords[origin]
    for first in candidates:
        for second in candidates:
            for third in candidates:
                if len({first, second, third}) != 3:
                    continue
                directions = np.array((
                    elem_coords[first] - origin_coord,
                    elem_coords[second] - origin_coord,
                    elem_coords[third] - origin_coord,
                ))
                determinant = np.linalg.det(directions)
                if determinant <= _TOL.geometry:
                    continue
                targets = np.array((
                    origin_coord,
                    origin_coord + directions[0],
                    origin_coord + directions[0] + directions[1],
                    origin_coord + directions[1],
                    origin_coord + directions[2],
                    origin_coord + directions[0] + directions[2],
                    origin_coord + directions[0] + directions[1] + directions[2],
                    origin_coord + directions[1] + directions[2],
                ))
                assigned = _match_role_nodes(
                    elem_coords,
                    list(corner_local),
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
        raise ValueError("Could not determine a right-handed hexahedron corner order.")
    scale = float(np.max(
        np.linalg.norm(corner_coords - origin_coord, axis=1)
    ))
    if best_error > _TOL.role_match * scale:
        raise ValueError("Hexahedron corner nodes do not form a valid element.")
    return best_order


def _match_role_nodes(
    elem_coords: np.ndarray,
    candidate_local: list[int],
    targets: np.ndarray,
    role_name: str,
    validate: bool = True,
) -> list[int]:
    if len(candidate_local) < targets.shape[0]:
        raise ValueError(f"Not enough nodes available to assign {role_name} roles.")
    candidates = np.asarray(candidate_local, dtype=np.int64)
    distances = np.linalg.norm(
        elem_coords[candidates, None, :] - targets[None, :, :],
        axis=2,
    )
    assigned: list[int] = []
    used: set[int] = set()
    for target_index in range(targets.shape[0]):
        ranked = np.argsort(distances[:, target_index])
        match = next((index for index in ranked if int(index) not in used), None)
        if match is None:
            raise ValueError(f"Could not assign a unique {role_name} node.")
        used.add(int(match))
        assigned.append(int(candidates[match]))

    if validate:
        scale = max(float(np.ptp(elem_coords, axis=0).max()), _TOL.geometry)
        errors = np.linalg.norm(elem_coords[assigned] - targets, axis=1)
        if np.any(errors > _TOL.role_match * scale):
            raise ValueError(
                f"Could not match {role_name} nodes to the element geometry."
            )
    return assigned


def _get_corner_indices(nodes_per_elem: int) -> np.ndarray:
    _supported_nodes_per_elem(nodes_per_elem)
    if nodes_per_elem in _SURFACE_NODE_COUNTS:
        return np.asarray(
            _get_surface_spec(nodes_per_elem).corner_indices, dtype=np.int64
        )
    return np.asarray(
        _get_volume_spec(nodes_per_elem).corner_indices, dtype=np.int64
    )


def _get_volume_corner_indices(nodes_per_elem: int) -> np.ndarray:
    return np.asarray(
        _get_volume_spec(nodes_per_elem).corner_indices, dtype=np.int64
    )


def _calc_active_coord_axes(coords: np.ndarray) -> np.ndarray:
    axis_range = np.ptp(coords, axis=0)
    active = np.flatnonzero(axis_range > _TOL.geometry)
    if active.shape[0] < 2:
        raise ValueError("At least two active coordinate axes are required.")
    return active[:2]


def _calc_polygon_signed_area(coords_elem: np.ndarray) -> float:
    axes = _calc_active_coord_axes(coords_elem)
    xy = coords_elem[:, axes]
    rolled = np.roll(xy, -1, axis=0)
    return 0.5 * np.sum(xy[:, 0] * rolled[:, 1] - rolled[:, 0] * xy[:, 1])


def _check_coplanar(coords_elem: np.ndarray) -> bool:
    centred = coords_elem - np.mean(coords_elem, axis=0)
    return np.linalg.matrix_rank(centred, tol=_TOL.geometry) <= 2


def _calc_tet_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[2] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
            ))
        )
    )


def _calc_hex_signed_volume(coords_elem: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack((
                coords_elem[1] - coords_elem[0],
                coords_elem[3] - coords_elem[0],
                coords_elem[4] - coords_elem[0],
            ))
        )
    )


def _calc_volume_signed_metric(corner_coords: np.ndarray) -> float:
    if corner_coords.shape[0] == 4:
        return _calc_tet_signed_volume(corner_coords)
    return _calc_hex_signed_volume(corner_coords)


def _calc_winding_metric(
    connect_row: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if nodes_per_elem not in _SURFACE_NODE_COUNTS:
        return None

    if not surface_only and nodes_per_elem not in _SURFACE_ONLY_NODE_COUNTS:
        volume_spec = _get_volume_spec(nodes_per_elem)
        cell_coords = coords[
            connect_row[np.asarray(volume_spec.corner_indices, dtype=np.int64)]
        ]
        if abs(_calc_volume_signed_metric(cell_coords)) > _TOL.geometry:
            return None

    corner_inds = _get_corner_indices(nodes_per_elem)
    coords_elem = coords[connect_row[corner_inds]]
    if _check_coplanar(coords):
        reference_normal = _calc_canonical_plane_normal(coords)
    else:
        face_centroid = np.mean(coords_elem, axis=0)
        outward = face_centroid - np.mean(coords, axis=0)
        outward_norm = np.linalg.norm(outward)
        if outward_norm <= _TOL.geometry:
            return None
        reference_normal = outward / outward_norm

    return _calc_local_polygon_signed_area(coords_elem, reference_normal)


def _calc_canonical_plane_normal(coords: np.ndarray) -> np.ndarray:
    centred = coords - np.mean(coords, axis=0)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    normal = vectors[-1]
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0.0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _calc_local_polygon_signed_area(
    coords_elem: np.ndarray,
    reference_normal: np.ndarray,
) -> float | None:
    origin = coords_elem[0]
    axis_u = None
    for point in coords_elem[1:]:
        edge = point - origin
        edge -= np.dot(edge, reference_normal) * reference_normal
        edge_norm = np.linalg.norm(edge)
        if edge_norm > _TOL.geometry:
            axis_u = edge / edge_norm
            break

    if axis_u is None:
        return None

    axis_v = np.cross(reference_normal, axis_u)
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
    surface_only: bool = False,
) -> float | None:
    nodes_per_elem = connect_row.shape[0]
    if surface_only:
        return _calc_winding_metric(connect_row, coords, surface_only=True)
    if nodes_per_elem in _VOLUME_NODE_COUNTS:
        volume_spec = _get_volume_spec(nodes_per_elem)
        volume_coords = coords[
            connect_row[np.asarray(volume_spec.corner_indices, dtype=np.int64)]
        ]
        volume_metric = _calc_volume_signed_metric(volume_coords)
        if nodes_per_elem in _VOLUME_ONLY_NODE_COUNTS:
            return volume_metric
        if abs(volume_metric) > _TOL.geometry:
            return volume_metric

    if nodes_per_elem in _SURFACE_NODE_COUNTS:
        metric = _calc_winding_metric(connect_row, coords)
        if metric is not None:
            return metric
    raise NotImplementedError(
        (
            f"Handedness checks are not implemented for "
            f"{nodes_per_elem}-node elements."
        )
    )


def _check_ccw_winding_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> bool:
    for row in connect:
        metric = _calc_winding_metric(row, coords, surface_only=surface_only)
        if metric is None:
            continue
        if abs(metric) <= _TOL.geometry:
            continue
        if metric <= 0.0:
            return False
    return True


def _check_cw_winding_table(connect: np.ndarray, coords: np.ndarray) -> bool:
    for row in connect:
        metric = _calc_winding_metric(row, coords)
        if metric is None:
            continue
        if abs(metric) <= _TOL.geometry:
            continue
        if metric >= 0.0:
            return False
    return True


def _check_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> bool:
    for row in connect:
        metric = _calc_handedness_metric(row, coords, surface_only=surface_only)
        if metric is None:
            continue
        if abs(metric) <= _TOL.geometry:
            continue
        if metric <= 0.0:
            return False
    return True


def _reverse_surface_row(connect_row: np.ndarray) -> np.ndarray:
    spec = _get_surface_spec(connect_row.shape[0])
    if spec.surface_reverse_permutation is None:
        raise NotImplementedError(
            (
                f"Surface reversal is not implemented for "
                f"{spec.nodes_per_element}-node elements."
            )
        )
    return connect_row[np.asarray(spec.surface_reverse_permutation, dtype=np.int64)]


def _reverse_handedness_row(connect_row: np.ndarray) -> np.ndarray:
    spec = _get_volume_spec(connect_row.shape[0])
    if spec.handedness_reverse_permutation is None:
        raise NotImplementedError(
        (
            f"Handedness reversal is not implemented for "
            f"{spec.nodes_per_element}-node elements."
        )
    )
    return connect_row[np.asarray(spec.handedness_reverse_permutation, dtype=np.int64)]


def _get_face_corner_coords(face_coords: np.ndarray) -> np.ndarray:
    nodes_per_face = face_coords.shape[0]
    corner_indices = np.asarray(
        _get_surface_spec(nodes_per_face).corner_indices,
        dtype=np.int64,
    )
    return face_coords[corner_indices]


def _calc_face_normal(face_coords: np.ndarray) -> np.ndarray:
    face_corners = _get_face_corner_coords(face_coords)
    face_normal = np.cross(
        face_corners[1] - face_corners[0],
        face_corners[2] - face_corners[0],
    )
    normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geometry and face_corners.shape[0] == 4:
        face_normal = np.cross(
            face_corners[2] - face_corners[0],
            face_corners[3] - face_corners[0],
        )
        normal_mag = np.linalg.norm(face_normal)

    if normal_mag <= _TOL.geometry:
        raise ValueError("Degenerate face detected while extracting the surface mesh.")

    return face_normal / normal_mag


def _enforce_surface_face_outward(
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


def _enforce_surface_face_node_order(
    face_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    nodes_per_face = face_connect.shape[0]
    face_out = np.copy(face_connect)
    face_coords = coords[face_out]

    spec = _get_surface_spec(nodes_per_face)
    corner_inds = np.asarray(spec.corner_indices, dtype=np.int64)
    num_corners = corner_inds.shape[0]
    if nodes_per_face == num_corners:
        return face_out

    midside_pool = np.arange(num_corners, nodes_per_face, dtype=np.int64)
    edge_corner_pairs = tuple(
        (corner_index, (corner_index + 1) % num_corners)
        for corner_index in range(num_corners)
    )

    face_centroid = np.mean(face_coords[corner_inds, :], axis=0)
    mid_pool_coords = face_coords[midside_pool, :]

    if spec.centre_index is not None:
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
            0.5 * (
                face_coords[start_ind, :] +
                face_coords[end_ind, :]
            )
            for (start_ind, end_ind) in edge_corner_pairs
        ],
        dtype=np.float64,
    )
    edge_dists = np.linalg.norm(
        edge_pool_coords[:, None, :] - edge_midpoints[None, :, :],
        axis=2,
    )
    edge_order = np.argmin(edge_dists, axis=0)
    reordered_edge_inds = edge_pool_local_inds[edge_order]

    face_out[
        num_corners:num_corners + num_corners
    ] = face_out[reordered_edge_inds]
    if spec.centre_index is not None:
        face_out[spec.centre_index] = face_out[center_local_ind]

    return face_out


def _enforce_ccw_winding_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _calc_winding_metric(row, coords, surface_only=surface_only)
        if metric is not None and metric < 0.0:
            connect_out[idx, :] = _reverse_surface_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_cw_winding_table(connect: np.ndarray, coords: np.ndarray) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _calc_winding_metric(row, coords)
        if metric is not None and metric > 0.0:
            connect_out[idx, :] = _reverse_surface_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _enforce_right_handed_table(
    connect: np.ndarray,
    coords: np.ndarray,
    surface_only: bool = False,
) -> np.ndarray:
    connect_out = np.copy(connect)
    for idx, row in enumerate(connect_out):
        metric = _calc_handedness_metric(row, coords, surface_only=surface_only)
        if metric is not None and metric < 0.0:
            if surface_only or (
                row.shape[0] in _SURFACE_NODE_COUNTS
                and _check_coplanar(
                coords[row[_get_corner_indices(row.shape[0])]]
            )
            ):
                connect_out[idx, :] = _reverse_surface_row(row)
            else:
                connect_out[idx, :] = _reverse_handedness_row(row)
    return np.ascontiguousarray(connect_out, dtype=np.int64)


def _get_surface_map(nodes_per_elem: int) -> np.ndarray:
    spec = _get_volume_spec(nodes_per_elem)
    if spec.surface_faces is None:
        raise NotImplementedError(
            (
                f"Surface extraction is not implemented for "
                f"{spec.nodes_per_element}-node elements."
            )
        )
    return np.asarray(spec.surface_faces, dtype=np.int64)


def _extract_surface_faces_from_table(
    connect: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodes_per_elem = connect.shape[1]
    face_map = _get_surface_map(nodes_per_elem)
    faces_wound = connect[:, face_map]
    faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
    faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

    (_, unique_inds, unique_counts) = np.unique(
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

    for ff, parent_elem_ind in enumerate(ext_parent_elem_inds):
        ext_faces[ff, :] = _enforce_surface_face_outward(
            ext_faces[ff, :],
            connect[parent_elem_ind, :],
            coords,
        )
        ext_faces[ff, :] = _enforce_surface_face_node_order(
            ext_faces[ff, :],
            coords,
        )

    ext_faces = np.ascontiguousarray(ext_faces, dtype=np.int64)
    return ext_faces, ext_parent_elem_inds


def _enforce_source_connectivity_style(
    mesh_in: SimData,
    one_based: bool,
    transposed: bool,
) -> SimData:
    if mesh_in.connect is None:
        return mesh_in

    connect_out: dict[str, np.ndarray] = {}
    for name, connect in mesh_in.connect.items():
        connect_fmt = np.copy(connect)
        if one_based:
            connect_fmt = connect_fmt + 1
        if transposed:
            connect_fmt = connect_fmt.T
        connect_out[name] = np.ascontiguousarray(connect_fmt, dtype=np.int64)

    return _copy_sim_data(mesh_in, connect=connect_out)


class MeshCheckCode(StrEnum):
    """A single mesh-convention condition that a connectivity table failed."""

    ROW_MAJOR_CONNECTIVITY = "row_major_connectivity"
    ZERO_BASED_INDEXING = "zero_based_indexing"
    CONNECTIVITY_INDICES = "connectivity_indices"
    CCW_WINDING = "ccw_winding"
    RIGHT_HANDED_GEOMETRY = "right_handed_geometry"
    SURFACE_TOPOLOGY = "surface_topology"
    NODE_ORDER = "node_order"


MeshConvCheck = dict[str, list[MeshCheckCode]]


class EMeshType(Enum):
    """Topological class of a Riley simulation mesh."""

    VOL = "volume"
    SURF = "surface"


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

    def reference_coordinates(self) -> np.ndarray:
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

            count = ELEMENT_SPECS[self].nodes_per_element

            if self is EElementType.HEX20:
                return np.asarray(points[:20], dtype=np.float64)
            return np.asarray(points[:count], dtype=np.float64)

        raise ValueError(f"No reference coordinates for {self.value}.")

    def orientation_preserving_permutations(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        """Derive all proper topology symmetries in canonical-slot notation.

        A returned permutation maps a target Riley slot to a source Riley
        slot.  Applying one preserves all corner, edge, face-centre and
        volume-centre roles.  Surface reflections and volume inversions are
        deliberately absent.
        """
        spec = ELEMENT_SPECS[self]
        reference = self.reference_coordinates()
        dimensions = 2 if spec.is_surface else 3
        corners = np.asarray(spec.corner_indices, dtype=np.int64)
        candidates: tuple[tuple[int, ...], ...]

        if len(corners) == 8:
            # The proper rotational group of a cube: 3! axis orderings and sign
            # changes with positive determinant, for 24 transformations.
            candidate_rows: list[tuple[int, ...]] = []
            for axes in permutations(range(3)):
                parity = 1 if (
                    sum(axes[index] > axes[next_index]
                        for index in range(3)
                        for next_index in range(index + 1, 3)) % 2 == 0
                ) else -1

                for signs in product((-1., 1.), repeat=3):
                    if parity * int(np.prod(signs)) < 0:
                        continue
                    transformed = (2.0 * reference - 1.0)[:, axes] * np.asarray(signs)
                    transformed = 0.5 * (transformed + 1.0)
                    candidate_rows.append(
                        _reference_node_permutation(reference, transformed)
                    )

            candidates = tuple(candidate_rows)
        else:
            rows: list[tuple[int, ...]] = []
            source_corners = reference[corners, :dimensions]
            homogeneous = np.column_stack((source_corners,
                                           np.ones(len(corners))))

            for corner_permutation in permutations(range(len(corners))):
                target_corners = source_corners[np.asarray(corner_permutation)]
                transform, _, _, _ = np.linalg.lstsq(homogeneous, target_corners,
                                                     rcond=None)

                if np.linalg.det(transform[:dimensions]) <= 0.0:
                    continue
                transformed = np.column_stack(
                    (reference[:, :dimensions], np.ones(reference.shape[0]))
                ) @ transform
                try:
                    rows.append(_reference_node_permutation(
                        reference[:, :dimensions],
                        transformed,
                    ))
                except ValueError:
                    continue
            candidates = tuple(rows)

        return tuple(sorted(set(candidates)))


@dataclass(frozen=True, slots=True)
class MeshConvention:
    """Caller-declared source ordering for otherwise ambiguous elements.

    Each permutation maps a Riley canonical slot to the corresponding slot in
    the source connectivity row.  Omitted element types are inferred from the
    mesh-wide topology and coordinates.
    """

    source_to_riley_permutations: Mapping[EElementType, tuple[int, ...]]
    canonicalise_equivalent_orientations: bool = False

    def permutation_for(self, element_type: EElementType) -> tuple[int, ...] | None:
        return self.source_to_riley_permutations.get(element_type)


class MeshConvError(ValueError):
    """Raised when source node roles cannot be inferred unambiguously."""


class MeshConventionInferenceError(ValueError):
    """Raised when source node roles cannot be inferred unambiguously."""


@dataclass(frozen=True, slots=True)
class ElementSpec:
    """Static node-ordering metadata for one supported element topology."""

    nodes_per_element: int
    is_surface: bool
    corner_indices: tuple[int, ...]
    surface_reverse_permutation: tuple[int, ...] | None = None
    handedness_reverse_permutation: tuple[int, ...] | None = None
    surface_faces: tuple[tuple[int, ...], ...] | None = None
    centre_index: int | None = None
    edge_pairs: tuple[tuple[int, int], ...] = ()
    face_corner_indices: tuple[tuple[int, ...], ...] = ()
    face_centre_indices: tuple[int, ...] = ()
    cell_centre_index: int | None = None


@dataclass(slots=True)
class SimData:
    """Mesh data used by Riley's mesh-convention tools.

    Connectivity tables may initially use either indexing convention and either
    table orientation; :func:`enforce_mesh_convention` canonicalises them.
    The optional metadata fields allow external adapters to preserve associated
    data without making Riley depend on their data-model types.
    """

    coords: np.ndarray | None = None
    connect: dict[str, np.ndarray] | None = None
    mesh_type: EMeshType | None = None
    time: np.ndarray | None = None
    side_sets: dict[tuple[str, str], np.ndarray] | None = None
    node_vars: dict[str, np.ndarray] | None = None
    elem_vars: dict[tuple[str, int], np.ndarray] | None = None
    glob_vars: dict[str, np.ndarray] | None = None

    def refresh_mesh_type(self) -> None:
        if self.coords is None or self.connect is None:
            self.mesh_type = None
            return
        self.mesh_type = EMeshType.VOL if _check_volume_mesh(self) else EMeshType.SURF


ELEMENT_SPECS = MappingProxyType({
    EElementType.TRI3: ElementSpec(3, True, (0, 1, 2), (0, 2, 1)),
    EElementType.TRI6: ElementSpec(6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3)),
    EElementType.TRI7: ElementSpec(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6), centre_index=6,
    ),
    EElementType.QUAD4: ElementSpec(4, True, (0, 1, 2, 3), (0, 3, 2, 1)),
    EElementType.QUAD8: ElementSpec(8, True, (0, 1, 2, 3), (0, 3, 2, 1, 7, 6, 5, 4)),
    EElementType.QUAD9: ElementSpec(
        9, True, (0, 1, 2, 3), (0, 3, 2, 1, 7, 6, 5, 4, 8), centre_index=8,
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
        face_corner_indices=((0, 1, 2, 3), (0, 3, 7, 4), (4, 7, 6, 5),
                             (1, 5, 6, 2), (0, 4, 5, 1), (2, 6, 7, 3)),
        face_centre_indices=(24, 23, 25, 21, 20, 22),
        cell_centre_index=26,
    ),
})


ELEMENT_SYMMETRIES = MappingProxyType({
    element_type: element_type.orientation_preserving_permutations()
    for element_type in EElementType
})


_ELEMENT_SPECS_BY_NODE_COUNT = MappingProxyType({
    nodes_per_element: tuple(
        spec for spec in ELEMENT_SPECS.values()
        if spec.nodes_per_element == nodes_per_element
    )
    for nodes_per_element in {
        spec.nodes_per_element for spec in ELEMENT_SPECS.values()
    }
})


_SURFACE_NODE_COUNTS = frozenset(
    spec.nodes_per_element for spec in ELEMENT_SPECS.values() if spec.is_surface
)


_VOLUME_NODE_COUNTS = frozenset(
    spec.nodes_per_element for spec in ELEMENT_SPECS.values() if not spec.is_surface
)


_SUPPORTED_NODE_COUNTS = _SURFACE_NODE_COUNTS | _VOLUME_NODE_COUNTS


_SURFACE_ONLY_NODE_COUNTS = frozenset(
    nodes_per_element
    for nodes_per_element, specs in _ELEMENT_SPECS_BY_NODE_COUNT.items()
    if all(spec.is_surface for spec in specs)
)


_VOLUME_ONLY_NODE_COUNTS = frozenset(
    nodes_per_element
    for nodes_per_element, specs in _ELEMENT_SPECS_BY_NODE_COUNT.items()
    if all(not spec.is_surface for spec in specs)
)


def check_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> MeshConvCheck:
    """Return failed checks for each non-conforming connectivity table.

    An empty dictionary means the mesh conforms.  This mapping is the only
    convention-check result: it preserves every failure without duplicating
    aggregate flags or summaries.
    """

    if mesh_in.connect is None:
        return {}
    if mesh_in.coords is None:
        raise ValueError("Mesh convention checks require 'coords' to be set.")

    per_table: MeshConvCheck = {}
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])
    surface_only = _check_surface_mesh_type(mesh_in.mesh_type)

    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_array_format(connect_raw, name)
        failures: list[MeshCheckCode] = []

        if _check_transpose_needed(connect, name, mesh_in):
            failures.append(MeshCheckCode.ROW_MAJOR_CONNECTIVITY)
            connect = connect.T

        legacy_connect = _check_table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )

        if legacy_connect:
            failures.append(MeshCheckCode.ZERO_BASED_INDEXING)

        connect_check = connect
        if legacy_connect:
            connect_check = connect - 1

        if not _check_indices_zero_based(connect_check, mesh_in.coords.shape[0]):
            failures.append(MeshCheckCode.CONNECTIVITY_INDICES)
        else:
            normalised_connect = _enforce_node_order_table(
                connect_check,
                mesh_in.coords,
                surface_only=surface_only,
                source_convention=source_convention,
            )
            if not np.array_equal(normalised_connect, connect_check):
                failures.append(MeshCheckCode.NODE_ORDER)
            connect_check = normalised_connect

            if _check_surface_connectivity_table(
                connect_check,
                mesh_in.coords,
                surface_only=surface_only,
            ):
                try:
                    surface_flips = _surface_orientation_flips(
                        connect_check,
                        mesh_in.coords,
                    )
                except ValueError:
                    failures.append(MeshCheckCode.SURFACE_TOPOLOGY)
                else:
                    if np.any(surface_flips):
                        failures.extend((
                            MeshCheckCode.CCW_WINDING,
                            MeshCheckCode.RIGHT_HANDED_GEOMETRY,
                        ))
            else:
                if not _check_ccw_winding_table(
                    connect_check,
                    mesh_in.coords,
                    surface_only=surface_only,
                ):
                    failures.append(MeshCheckCode.CCW_WINDING)

                if not _check_right_handed_table(
                    connect_check,
                    mesh_in.coords,
                    surface_only=surface_only,
                ):
                    failures.append(MeshCheckCode.RIGHT_HANDED_GEOMETRY)

        if failures:
            per_table[name] = failures

    return per_table


def enforce_mesh_convention(
    mesh_in: SimData,
    source_convention: MeshConvention | None = None,
) -> SimData:
    """Normalises a mesh to the mesh convention:
    - 0-based indexing
    - CCW node ordering when viewed from the outward/visible side
    - Right-handed geometry conventions
    - Check that all indices in the connectivity table map to a row in coords

    Only the conditions flagged by :func:`check_mesh_convention` are fixed,
    and they are fixed in canonical condition order so the result never
    depends on the report ordering.
    """

    report = check_mesh_convention(mesh_in, source_convention)
    if not report:
        return mesh_in

    if mesh_in.connect is None:
        return mesh_in
    if mesh_in.coords is None:
        raise ValueError("Mesh convention enforcement requires coords.")

    num_coords = mesh_in.coords.shape[0]
    surface_only = _check_surface_mesh_type(mesh_in.mesh_type)
    orientation_codes = frozenset((
        MeshCheckCode.CCW_WINDING,
        MeshCheckCode.RIGHT_HANDED_GEOMETRY,
        MeshCheckCode.SURFACE_TOPOLOGY,
    ))

    connect_out: dict[str, np.ndarray] = {}
    for name, connect_raw in mesh_in.connect.items():
        failures = frozenset(report.get(name, ()))
        connect = _enforce_connect_array_format(connect_raw, name)

        if MeshCheckCode.CONNECTIVITY_INDICES in failures:
            raise ValueError(
                "Connectivity table "
                f"'{name}' contains indices outside the coordinate array after "
                "0-based normalization."
            )

        if MeshCheckCode.ROW_MAJOR_CONNECTIVITY in failures:
            connect = connect.T

        if MeshCheckCode.ZERO_BASED_INDEXING in failures:
            connect = connect - 1

        if MeshCheckCode.NODE_ORDER in failures:
            connect = _enforce_node_order_table(
                connect,
                mesh_in.coords,
                surface_only=surface_only,
                source_convention=source_convention,
            )

        if failures & orientation_codes:
            if _check_surface_connectivity_table(
                connect,
                mesh_in.coords,
                surface_only=surface_only,
            ):
                connect = _enforce_surface_orientation_table(
                    connect,
                    mesh_in.coords,
                )
            else:
                if MeshCheckCode.CCW_WINDING in failures:
                    connect = _enforce_ccw_winding_table(
                        connect,
                        mesh_in.coords,
                        surface_only=surface_only,
                    )
                if MeshCheckCode.RIGHT_HANDED_GEOMETRY in failures:
                    connect = _enforce_right_handed_table(
                        connect,
                        mesh_in.coords,
                        surface_only=surface_only,
                    )

        if not _check_indices_zero_based(connect, num_coords):
            raise ValueError(
                "Connectivity table "
                f"'{name}' became invalid during mesh convention enforcement."
            )

        connect_out[name] = np.ascontiguousarray(connect, dtype=np.int64)

    return _copy_sim_data(mesh_in, connect=connect_out)


def extract_surf_mesh(
    mesh_in: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts the external surface mesh from supported 3D volume elements."""

    if _check_mesh_2d(mesh_in):
        raise ValueError(
            "Surface extraction is only supported for 3D meshes. "
            "The provided mesh appears to be 2D."
        )

    if mesh_in.connect is None:
        raise ValueError("Surface extraction requires connectivity tables.")
    if mesh_in.coords is None:
        raise ValueError("Surface extraction requires coords.")

    connect_norm: dict[str, np.ndarray] = {}
    source_zero_based = True
    source_row_major = True
    shift_all = _check_mesh_needs_zero_based_shift(mesh_in, mesh_in.coords.shape[0])

    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_array_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            source_row_major = False
            connect = connect.T
        legacy_connect = _check_table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            source_zero_based = False
            connect = connect - 1
        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise ValueError(
                (
                    f"Connectivity table '{name}' contains invalid indices "
                    f"for surface extraction."
                )
            )
        connect = _enforce_node_order_table(connect, mesh_in.coords)
        connect_norm[name] = _enforce_right_handed_table(
            _enforce_ccw_winding_table(connect, mesh_in.coords),
            mesh_in.coords,
        )

    surf_connect_global: dict[str, np.ndarray] = {}
    surf_elem_sources: dict[str, np.ndarray] = {}
    surf_node_inds = np.array([], dtype=np.int64)

    for name, connect in connect_norm.items():
        (surf_faces, surf_parent_elem_inds) = _extract_surface_faces_from_table(
            connect,
            mesh_in.coords,
        )
        surf_connect_global[name] = surf_faces
        surf_elem_sources[name] = surf_parent_elem_inds
        if surf_faces.size:
            surf_node_inds = np.union1d(surf_node_inds, np.unique(surf_faces))

    surf_coords = np.ascontiguousarray(mesh_in.coords[surf_node_inds], 
                                       dtype=mesh_in.coords.dtype)
    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_inds] = np.arange(surf_node_inds.shape[0], dtype=np.int64)

    surf_connect_local: dict[str, np.ndarray] = {}
    for name, surf_faces in surf_connect_global.items():
        surf_connect_local[name] = coord_remap[surf_faces]

    surf_mesh = _copy_sim_data(mesh_in)
    surf_mesh.coords = surf_coords
    surf_mesh.connect = surf_connect_local
    surf_mesh.refresh_mesh_type()

    if mesh_in.node_vars is not None:
        surf_mesh.node_vars = {
            name: values[surf_node_inds, :]
            for name, values in mesh_in.node_vars.items()
        }

    if mesh_in.elem_vars is not None:
        surf_mesh.elem_vars = {}
        for (name, block_id), values in mesh_in.elem_vars.items():
            connect_key = f"connect{block_id}"
            if connect_key in surf_elem_sources:
                surf_mesh.elem_vars[(name, block_id)] = values[
                    surf_elem_sources[connect_key], :
                ]

    if not enforce_convention:
        surf_mesh = _enforce_source_connectivity_style(
            surf_mesh,
            one_based=not source_zero_based,
            transposed=not source_row_major,
        )
        return surf_mesh

    return surf_mesh


def extract_surf_between(
    mesh_in: SimData,
    point: np.ndarray | list[float] | tuple[float, ...],
    normal: np.ndarray | list[float] | tuple[float, ...],
    distance: float | None = None,
    tolerance: float = 1.0e-6,
    enforce_convention: bool = True,
) -> SimData:
    """Extracts a surface mesh between two planes defined by point, normal,
    and distance.

    Parameters
    ----------
    mesh_in : SimData
        The input simulation data/mesh.
    point : np.ndarray | list[float] | tuple[float, ...]
        A point on the first plane.
    normal : np.ndarray | list[float] | tuple[float, ...]
        The normal vector of the planes.
    distance : float | None, optional
        The distance along the normal to the second plane. If None, the
        surface is extracted at +/- tolerance about the first plane.
    tolerance : float, optional
        Numerical tolerance for checking if nodes lie between the planes.
        Defaults to 1.0e-6.
    enforce_convention : bool, optional
        If True, normalizes the output mesh to the pyvale convention.
        Defaults to True.

    Returns
    -------
    SimData
        The extracted surface mesh.
    """
    if mesh_in.connect is None:
        raise ValueError("Surface extraction requires connectivity tables.")
    if mesh_in.coords is None:
        raise ValueError("Surface extraction requires coords.")

    # Format point and normal to 3D arrays
    point_arr = np.zeros(3, dtype=np.float64)
    point_arr[:len(point)] = point

    normal_arr = np.zeros(3, dtype=np.float64)
    normal_arr[:len(normal)] = normal
    norm_val = np.linalg.norm(normal_arr)
    if norm_val < _TOL.geometry:
        raise ValueError("Normal vector cannot be zero.")
    normal_arr = normal_arr / norm_val

    # Project coordinates along normal relative to point
    diff = mesh_in.coords - point_arr
    proj = diff @ normal_arr

    # Determine bounds
    if distance is not None:
        d = float(distance)
        min_bound = min(0.0, d) - tolerance
        max_bound = max(0.0, d) + tolerance
    else:
        min_bound = -tolerance
        max_bound = tolerance

    connect_norm: dict[str, np.ndarray] = {}
    source_zero_based = True
    source_row_major = True
    shift_all = _check_mesh_needs_zero_based_shift(
        mesh_in, mesh_in.coords.shape[0]
    )

    for name, connect_raw in mesh_in.connect.items():
        connect = _enforce_connect_array_format(connect_raw, name)
        if _check_transpose_needed(connect, name, mesh_in):
            source_row_major = False
            connect = connect.T
        legacy_connect = _check_table_needs_zero_based_shift(
            connect,
            mesh_in.coords.shape[0],
            shift_all,
        )
        if legacy_connect:
            source_zero_based = False
            connect = connect - 1
        if not _check_indices_zero_based(connect, mesh_in.coords.shape[0]):
            raise ValueError(
                f"Connectivity table '{name}' contains invalid indices "
                "for surface extraction."
            )
        connect = _enforce_node_order_table(connect, mesh_in.coords)
        connect_norm[name] = _enforce_right_handed_table(
            _enforce_ccw_winding_table(connect, mesh_in.coords),
            mesh_in.coords,
        )

    surf_connect_global: dict[str, np.ndarray] = {}
    surf_elem_sources: dict[str, np.ndarray] = {}
    surf_node_inds = np.array([], dtype=np.int64)

    for name, connect in connect_norm.items():
        nodes_per_elem = connect.shape[1]
        is_vol = _check_volume_connectivity_table(connect, mesh_in.coords)
        if is_vol:
            face_map = _get_surface_map(nodes_per_elem)
            faces_per_elem = face_map.shape[0]
            faces_wound = connect[:, face_map]
            faces_flat_wound = faces_wound.reshape((-1, face_map.shape[1]))
            faces_flat_sorted = np.sort(faces_flat_wound, axis=1)

            _, unique_inds = np.unique(
                faces_flat_sorted,
                axis=0,
                return_index=True,
            )
            candidate_faces = faces_flat_wound[unique_inds]
            parent_inds = unique_inds // faces_per_elem
        else:
            candidate_faces = connect
            parent_inds = np.arange(connect.shape[0], dtype=np.int64)

        if candidate_faces.size == 0:
            continue

        face_projs = proj[candidate_faces]
        in_bounds = np.all(
            (face_projs >= min_bound) & (face_projs <= max_bound),
            axis=1
        )

        filtered_faces = candidate_faces[in_bounds]
        filtered_parents = parent_inds[in_bounds]

        if filtered_faces.size > 0:
            surf_connect_global[name] = filtered_faces
            surf_elem_sources[name] = filtered_parents
            surf_node_inds = np.union1d(
                surf_node_inds, np.unique(filtered_faces)
            )

    if len(surf_connect_global) == 0 or surf_node_inds.size == 0:
        raise ValueError(
            "No elements/faces found between the specified planes."
        )

    surf_coords = np.ascontiguousarray(
        mesh_in.coords[surf_node_inds],
        dtype=mesh_in.coords.dtype
    )
    coord_remap = np.full(mesh_in.coords.shape[0], -1, dtype=np.int64)
    coord_remap[surf_node_inds] = np.arange(
        surf_node_inds.shape[0], dtype=np.int64
    )

    surf_connect_local: dict[str, np.ndarray] = {}
    for name, surf_faces in surf_connect_global.items():
        surf_connect_local[name] = coord_remap[surf_faces]

    surf_mesh = _copy_sim_data(mesh_in)
    surf_mesh.coords = surf_coords
    surf_mesh.connect = surf_connect_local
    surf_mesh.side_sets = None

    if mesh_in.node_vars is not None:
        surf_mesh.node_vars = {
            name: values[surf_node_inds, :]
            for name, values in mesh_in.node_vars.items()
        }

    if mesh_in.elem_vars is not None:
        surf_mesh.elem_vars = {}
        for (var_name, block_id), values in mesh_in.elem_vars.items():
            connect_key = f"connect{block_id}"
            if connect_key in surf_elem_sources:
                surf_mesh.elem_vars[(var_name, block_id)] = values[
                    surf_elem_sources[connect_key], :
                ]

    if not enforce_convention:
        surf_mesh = _enforce_source_connectivity_style(
            surf_mesh,
            one_based=not source_zero_based,
            transposed=not source_row_major,
        )
        return surf_mesh

    return enforce_mesh_convention(surf_mesh)

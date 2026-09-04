# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Convert, verify and extract meshes in Riley's standard convention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np


class MeshConvErr(ValueError):
    """Raised when an explicit mesh conversion cannot be completed."""


class EElementType(Enum):
    """Finite-element topologies supported by Riley's mesh tools."""

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
        """Return reference-node coordinates in Riley slot order."""
        return np.asarray(_REF_COORDS[self], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ElementSpec:
    """Static topology needed by explicit conversion and extraction."""

    node_count: int
    is_surf: bool
    corner_slots: tuple[int, ...]
    reverse_slots: tuple[int, ...]
    edge_corners: tuple[tuple[int, int], ...] = ()
    surface_faces: tuple[tuple[int, ...], ...] = ()
    face_corners: tuple[tuple[int, ...], ...] = ()
    face_slots: tuple[int, ...] = ()
    centre_slot: int | None = None


_TRI_COORDS = (
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
    (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
)
_QUAD_COORDS = (
    (-1.0, -1.0), (1.0, -1.0),
    (1.0, 1.0), (-1.0, 1.0),
    (0.0, -1.0), (1.0, 0.0),
    (0.0, 1.0), (-1.0, 0.0),
)
_TET_COORDS = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0), (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
)
_HEX_COORDS = (
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    (0.5, 0.0, 0.0), (1.0, 0.5, 0.0),
    (0.5, 1.0, 0.0), (0.0, 0.5, 0.0),
    (0.5, 0.0, 1.0), (1.0, 0.5, 1.0),
    (0.5, 1.0, 1.0), (0.0, 0.5, 1.0),
    (0.0, 0.0, 0.5), (1.0, 0.0, 0.5),
    (1.0, 1.0, 0.5), (0.0, 1.0, 0.5),
    (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
    (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
    (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
    (0.5, 0.5, 0.5),
)
_REF_COORDS = MappingProxyType({
    EElementType.TRI3: _TRI_COORDS[:3],
    EElementType.TRI6: _TRI_COORDS,
    EElementType.TRI7: _TRI_COORDS + ((1.0 / 3.0, 1.0 / 3.0),),
    EElementType.QUAD4: _QUAD_COORDS[:4],
    EElementType.QUAD8: _QUAD_COORDS,
    EElementType.QUAD9: _QUAD_COORDS + ((0.0, 0.0),),
    EElementType.TET4: _TET_COORDS[:4],
    EElementType.TET10: _TET_COORDS,
    EElementType.HEX8: _HEX_COORDS[:8],
    EElementType.HEX20: _HEX_COORDS[:20],
    EElementType.HEX27: _HEX_COORDS,
})

_TRI_EDGES = ((0, 1), (1, 2), (2, 0))
_QUAD_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_TET4_FACES = (
    (0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1),
)
_TET10_FACES = (
    (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8),
    (2, 0, 3, 6, 7, 9), (0, 2, 1, 6, 5, 4),
)
_HEX8_FACES = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)
_HEX20_FACES = (
    (0, 4, 7, 3, 16, 15, 19, 11),
    (1, 2, 6, 5, 9, 18, 13, 17),
    (0, 1, 5, 4, 8, 17, 12, 16),
    (3, 7, 6, 2, 19, 14, 18, 10),
    (0, 3, 2, 1, 11, 10, 9, 8),
    (4, 5, 6, 7, 12, 13, 14, 15),
)
_HEX27_FACES = (
    _HEX20_FACES[0] + (20,), _HEX20_FACES[1] + (21,),
    _HEX20_FACES[2] + (22,), _HEX20_FACES[3] + (23,),
    _HEX20_FACES[4] + (24,), _HEX20_FACES[5] + (25,),
)
_HEX_FACE_CORNERS = (
    (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
    (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
)

_ELEMENT_SPECS = MappingProxyType({
    EElementType.TRI3: _ElementSpec(
        3, True, (0, 1, 2), (0, 2, 1),
    ),
    EElementType.TRI6: _ElementSpec(
        6, True, (0, 1, 2), (0, 2, 1, 5, 4, 3), _TRI_EDGES,
    ),
    EElementType.TRI7: _ElementSpec(
        7, True, (0, 1, 2), (0, 2, 1, 5, 4, 3, 6),
        _TRI_EDGES, centre_slot=6,
    ),
    EElementType.QUAD4: _ElementSpec(
        4, True, (0, 1, 2, 3), (0, 3, 2, 1),
    ),
    EElementType.QUAD8: _ElementSpec(
        8, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4), _QUAD_EDGES,
    ),
    EElementType.QUAD9: _ElementSpec(
        9, True, (0, 1, 2, 3),
        (0, 3, 2, 1, 7, 6, 5, 4, 8),
        _QUAD_EDGES, centre_slot=8,
    ),
    EElementType.TET4: _ElementSpec(
        4, False, (0, 1, 2, 3), (0, 2, 1, 3),
        _TET_EDGES, _TET4_FACES,
    ),
    EElementType.TET10: _ElementSpec(
        10, False, (0, 1, 2, 3),
        (0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
        _TET_EDGES, _TET10_FACES,
    ),
    EElementType.HEX8: _ElementSpec(
        8, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5), _HEX_EDGES, _HEX8_FACES,
    ),
    EElementType.HEX20: _ElementSpec(
        20, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17),
        _HEX_EDGES, _HEX20_FACES,
    ),
    EElementType.HEX27: _ElementSpec(
        27, False, tuple(range(8)),
        (0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8,
         15, 14, 13, 12, 16, 19, 18, 17,
         22, 23, 20, 21, 24, 25, 26),
        _HEX_EDGES, _HEX27_FACES, _HEX_FACE_CORNERS,
        (20, 21, 22, 23, 24, 25), 26,
    ),
})

_GEOM_TOL = 1.0e-12
_INSIDE_PROBE_MIN_MULTIPLIER = 10.0
_INSIDE_PROBE_SCALE = 1.0e-8
_TET_VOL_SCALE = 6.0
_RAY_DIRECT = np.asarray((0.87287156, 0.43643578, 0.21821789))
_SURF_TYPES_BY_NODE_COUNT = MappingProxyType({
    3: EElementType.TRI3,
    4: EElementType.QUAD4,
    6: EElementType.TRI6,
    7: EElementType.TRI7,
    8: EElementType.QUAD8,
    9: EElementType.QUAD9,
})

_EdgeKey: TypeAlias = tuple[int, int]
_EdgeUse: TypeAlias = tuple[int, tuple[int, int]]
_EdgeUses: TypeAlias = dict[_EdgeKey, list[_EdgeUse]]
_SurfComponent: TypeAlias = tuple[np.ndarray, _EdgeUses]


def _get_surf_corner_count(node_count: int) -> int:
    """Get the corner count for a supported surface node count."""
    elem_type = _SURF_TYPES_BY_NODE_COUNT[node_count]
    return len(_ELEMENT_SPECS[elem_type].corner_slots)


class EConnectAxis(Enum):
    """Axis containing elements in a source connectivity array."""

    ROW = "row"
    COLUMN = "column"


class ENodeOrder(Enum):
    """Built-in local-node ordering adapters."""

    RILEY = "riley"
    VTK = "vtk"
    EXODUS = "exodus"


@dataclass(frozen=True, slots=True)
class EdgeNode:
    """Identify a source edge-node slot by its source corner slots."""

    corners: tuple[int, int]
    node_slot: int


@dataclass(frozen=True, slots=True)
class FaceNode:
    """Identify a source face-node slot by its source corner slots."""

    corners: tuple[int, ...]
    node_slot: int


@dataclass(frozen=True, slots=True)
class UserTopology:
    """Describe local topology using slots in a source connectivity row.

    ``corner_slots`` must follow a valid source topological ordering. Edge and
    face nodes are associated with the source corner slots bounding their
    topological entity. No Riley slot numbers are required.
    """

    corner_slots: tuple[int, ...]
    edge_nodes: tuple[EdgeNode, ...] = ()
    face_nodes: tuple[FaceNode, ...] = ()
    centre_slot: int | None = None


@dataclass(frozen=True, slots=True)
class ConnectConvention:
    """Describe how to interpret one source connectivity array.

    ``elem_axis`` says whether each element occupies a row or a column, and
    ``index_base`` must explicitly be zero or one. ``node_order`` selects a
    named source adapter or a ``UserTopology``; Riley does not guess it.

    For an open surface, source winding declares the material-facing side.
    Set ``reverse_open_surface`` to choose its opposite, or provide a
    non-zero ``material_normal_hint`` to choose the side whose component
    normal has a positive dot product with that vector. These options do not
    alter the exterior/cavity orientation of closed shells.

    One convention describes one connectivity table. Reuse the same instance
    for multiple tables with the same representation, or provide a mapping of
    table names to conventions in calling code when their representations
    differ.
    """

    elem_type: EElementType
    elem_axis: EConnectAxis
    index_base: Literal[0, 1]
    node_order: ENodeOrder | UserTopology = ENodeOrder.RILEY
    material_normal_hint: tuple[float, float, float] | None = None
    reverse_open_surface: bool = False

    def __post_init__(self) -> None:
        """Verify convention fields at construction time."""
        if not isinstance(self.elem_type, EElementType):
            raise TypeError("elem_type must be an EElementType member.")
        if not isinstance(self.elem_axis, EConnectAxis):
            raise TypeError("elem_axis must be an EConnectAxis member.")
        if isinstance(self.index_base, bool) or self.index_base not in (0, 1):
            raise ValueError("index_base must be the integer 0 or 1.")
        valid_order = isinstance(self.node_order, (ENodeOrder, UserTopology))
        if not valid_order:
            raise TypeError("node_order must be ENodeOrder or UserTopology.")


@dataclass(frozen=True, slots=True)
class MeshGeometry:
    """A single-topology mesh using Riley's standard convention."""

    elem_type: EElementType
    coords: np.ndarray
    connect: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshVerifyIssue:
    """One independently detected Riley mesh verification failure."""

    code: str
    message: str
    elem_idx: int | None = None


class MeshVerifyErr(ValueError):
    """Collect all mesh verification failures found in one pass."""

    def __init__(self, issues: list[MeshVerifyIssue]) -> None:
        self.issues = tuple(issues)
        lines = [f"Mesh verification failed with {len(issues)} issue(s):"]
        for issue in issues:
            location = ""
            if issue.elem_idx is not None:
                location = f" element {issue.elem_idx}:"
            lines.append(f"- [{issue.code}]{location} {issue.message}")
        super().__init__("\n".join(lines))


_vtk_to_riley_out: dict[EElementType, tuple[int, ...]] = {}
for _elem_type, _elem_spec in _ELEMENT_SPECS.items():
    _vtk_to_riley_out[_elem_type] = tuple(range(_elem_spec.node_count))
_VTK_TO_RILEY = MappingProxyType(_vtk_to_riley_out)

# Exodus uses bottom, vertical, then top HEX edge groups. Riley and VTK use
# bottom, top, then vertical edge groups. Other supported Exodus topology
# mappings are identical to Riley's standard local roles.
_EXODUS_TO_RILEY = MappingProxyType({
    **_VTK_TO_RILEY,
    EElementType.HEX20: (
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
    ),
})


def _get_user_topology_perm(
    elem_type: EElementType,
    topology: UserTopology,
) -> tuple[int, ...]:
    """Build a Riley-target-to-source permutation from source topology."""
    spec = _ELEMENT_SPECS[elem_type]
    node_count = spec.node_count
    corner_count = len(spec.corner_slots)
    if len(topology.corner_slots) != corner_count:
        raise MeshConvErr(
            f"{elem_type.value} requires {corner_count} corner slots."
        )

    perm = [-1] * node_count
    for target_slot, source_slot in zip(
        spec.corner_slots,
        topology.corner_slots,
        strict=True,
    ):
        perm[target_slot] = source_slot

    edge_lookup: dict[frozenset[int], int] = {}
    for edge in topology.edge_nodes:
        edge_lookup[frozenset(edge.corners)] = edge.node_slot

    edge_pairs = spec.edge_corners
    if not edge_pairs and node_count > corner_count:
        edge_pairs_out = []
        for idx in range(corner_count):
            edge_pairs_out.append((idx, (idx + 1) % corner_count))
        edge_pairs = tuple(edge_pairs_out)

    for target_slot, (start, end) in enumerate(
        edge_pairs,
        start=corner_count,
    ):
        source_edge = frozenset((
            topology.corner_slots[start],
            topology.corner_slots[end],
        ))
        if source_edge not in edge_lookup:
            raise MeshConvErr(
                f"{elem_type.value} topology is missing edge "
                f"{tuple(sorted(source_edge))}."
            )
        perm[target_slot] = edge_lookup[source_edge]

    face_lookup: dict[frozenset[int], int] = {}
    for face in topology.face_nodes:
        face_lookup[frozenset(face.corners)] = face.node_slot

    for target_slot, face_corners in zip(
        spec.face_slots,
        spec.face_corners,
        strict=True,
    ):
        source_face_slots = []
        for idx in face_corners:
            source_face_slots.append(topology.corner_slots[idx])
        source_face = frozenset(source_face_slots)
        if source_face not in face_lookup:
            raise MeshConvErr(
                f"{elem_type.value} topology is missing face "
                f"{tuple(sorted(source_face))}."
            )
        perm[target_slot] = face_lookup[source_face]

    centre_slot = spec.centre_slot
    if centre_slot is not None:
        if topology.centre_slot is None:
            raise MeshConvErr(
                f"{elem_type.value} topology requires a centre slot."
            )
        perm[centre_slot] = topology.centre_slot

    expected_slots = set(range(node_count))
    if set(perm) != expected_slots:
        raise MeshConvErr(
            f"{elem_type.value} topology must use every source slot exactly "
            "once."
        )
    return tuple(perm)


def _get_source_perm(convention: ConnectConvention) -> tuple[int, ...]:
    """Return the Riley-target-to-source permutation for a convention."""
    if isinstance(convention.node_order, UserTopology):
        return _get_user_topology_perm(
            convention.elem_type,
            convention.node_order,
        )
    if convention.node_order in (ENodeOrder.RILEY, ENodeOrder.VTK):
        return _VTK_TO_RILEY[convention.elem_type]
    if convention.node_order is ENodeOrder.EXODUS:
        if convention.elem_type is EElementType.HEX27:
            raise MeshConvErr(
                "The Exodus HEX27 adapter requires an authoritative fixture; "
                "supply UserTopology explicitly."
            )
        return _EXODUS_TO_RILEY[convention.elem_type]
    raise MeshConvErr(f"Unsupported node ordering: {convention.node_order}.")


def _build_surf_topology(
    connect: np.ndarray,
) -> list[_SurfComponent]:
    """Build connected surface components from corner-node edges."""
    corner_count = _get_surf_corner_count(connect.shape[1])
    edge_uses: _EdgeUses = {}
    neighbours: dict[int, set[int]] = {}
    for idx in range(connect.shape[0]):
        neighbours[idx] = set()

    for row_idx, row in enumerate(connect):
        corners = row[:corner_count]
        for corner_idx in range(corner_count):
            node_a = int(corners[corner_idx])
            node_b = int(corners[(corner_idx + 1) % corner_count])
            edge_key = (min(node_a, node_b), max(node_a, node_b))
            edge_uses.setdefault(edge_key, []).append(
                (row_idx, (node_a, node_b))
            )

    for edge_key, uses in edge_uses.items():
        if len(uses) > 2:
            raise ValueError(
                f"Non-manifold surface edge {edge_key} has {len(uses)} "
                "incident faces."
            )
        if len(uses) == 2:
            row_a = uses[0][0]
            row_b = uses[1][0]
            neighbours[row_a].add(row_b)
            neighbours[row_b].add(row_a)

    components: list[_SurfComponent] = []
    unseen = set(range(connect.shape[0]))
    while unseen:
        seed = unseen.pop()
        component_rows = {seed}
        pending = [seed]
        while pending:
            row_idx = pending.pop()
            for neighbour in neighbours[row_idx]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component_rows.add(neighbour)
                    pending.append(neighbour)

        rows = np.asarray(sorted(component_rows), dtype=np.int64)
        component_edges = {}
        for edge_key, uses in edge_uses.items():
            if uses[0][0] in component_rows:
                component_edges[edge_key] = uses
        components.append((rows, component_edges))

    return components


def _calc_comp_rel_flips(
    rows: np.ndarray,
    edge_uses: _EdgeUses,
) -> np.ndarray:
    """Calculate reversals that make neighbouring faces coherent."""
    constraints: dict[int, list[tuple[int, bool]]] = {}
    for row in rows:
        constraints[int(row)] = []
    for uses in edge_uses.values():
        if len(uses) != 2:
            continue
        row_a, direct_a = uses[0]
        row_b, direct_b = uses[1]
        same_direct = direct_a == direct_b
        constraints[row_a].append((row_b, same_direct))
        constraints[row_b].append((row_a, same_direct))

    assigned: dict[int, bool] = {}
    for seed_raw in rows:
        seed = int(seed_raw)
        if seed in assigned:
            continue
        assigned[seed] = False
        pending = [seed]
        while pending:
            row_idx = pending.pop()
            for neighbour, toggle in constraints[row_idx]:
                expected = assigned[row_idx] ^ toggle
                if neighbour in assigned and assigned[neighbour] != expected:
                    raise ValueError("Surface component is not orientable.")
                if neighbour not in assigned:
                    assigned[neighbour] = expected
                    pending.append(neighbour)

    return np.asarray([assigned[int(row)] for row in rows], dtype=bool)


def _check_comp_closed(edge_uses: _EdgeUses) -> bool:
    """Check whether every component edge has two incident faces."""
    for uses in edge_uses.values():
        if len(uses) != 2:
            return False
    return True


def _reverse_surf_rows(
    connect: np.ndarray,
    flips: np.ndarray,
) -> np.ndarray:
    """Reverse selected surface rows while preserving local-node roles."""
    connect_out = np.array(connect, copy=True)
    spec = _ELEMENT_SPECS[_SURF_TYPES_BY_NODE_COUNT[connect.shape[1]]]
    reverse_slots = np.asarray(spec.reverse_slots, dtype=np.int64)
    connect_out[flips] = connect_out[flips][:, reverse_slots]
    return connect_out


def _calc_face_normal(
    face_connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Calculate a unit normal from a non-degenerate surface face."""
    corner_count = _get_surf_corner_count(face_connect.shape[0])
    corners = coords[face_connect[:corner_count]]
    face_normal = np.cross(corners[1] - corners[0],
                           corners[2] - corners[0])
    normal_magnitude = float(np.linalg.norm(face_normal))
    if normal_magnitude <= _GEOM_TOL and corner_count == 4:
        face_normal = np.cross(corners[2] - corners[0],
                               corners[3] - corners[0])
        normal_magnitude = float(np.linalg.norm(face_normal))
    if normal_magnitude <= _GEOM_TOL:
        raise ValueError("Surface face is degenerate.")
    return face_normal / normal_magnitude


def _calc_vol_metric(corner_coords: np.ndarray) -> float:
    """Calculate a signed tetrahedron or hexahedron corner metric."""
    if corner_coords.shape[0] == 4:
        edge_matrix = np.column_stack((
            corner_coords[1] - corner_coords[0],
            corner_coords[2] - corner_coords[0],
            corner_coords[3] - corner_coords[0],
        ))
    else:
        edge_matrix = np.column_stack((
            corner_coords[1] - corner_coords[0],
            corner_coords[3] - corner_coords[0],
            corner_coords[4] - corner_coords[0],
        ))
    return float(np.linalg.det(edge_matrix))


def _calc_surf_signed_vol(connect: np.ndarray, coords: np.ndarray) -> float:
    """Calculate signed volume enclosed by a coherent surface."""
    corner_count = _get_surf_corner_count(connect.shape[1])
    signed_vol = 0.0
    for row in connect:
        corners = coords[row[:corner_count]]
        for point_idx in range(1, corner_count - 1):
            signed_vol += float(np.dot(
                corners[0],
                np.cross(corners[point_idx], corners[point_idx + 1]),
            )) / _TET_VOL_SCALE
    return signed_vol


def _check_ray_triangle(
    origin: np.ndarray,
    direct: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    """Check whether a forward ray intersects a triangle."""
    edge_ab = triangle[1] - triangle[0]
    edge_ac = triangle[2] - triangle[0]
    perpendicular = np.cross(direct, edge_ac)
    determinant = float(np.dot(edge_ab, perpendicular))
    if abs(determinant) <= _GEOM_TOL:
        return False
    inverse = 1.0 / determinant
    offset = origin - triangle[0]
    barycentric_u = inverse * float(np.dot(offset, perpendicular))
    if barycentric_u <= _GEOM_TOL or barycentric_u >= 1.0 - _GEOM_TOL:
        return False
    cross_offset = np.cross(offset, edge_ab)
    barycentric_v = inverse * float(np.dot(direct, cross_offset))
    outside = (
        barycentric_v <= _GEOM_TOL
        or barycentric_u + barycentric_v >= 1.0 - _GEOM_TOL
    )
    if outside:
        return False
    distance = inverse * float(np.dot(edge_ac, cross_offset))
    return distance > _GEOM_TOL


def _check_point_in_surf(
    point: np.ndarray,
    connect: np.ndarray,
    coords: np.ndarray,
) -> bool:
    """Check whether a point lies inside a closed surface."""
    corner_count = _get_surf_corner_count(connect.shape[1])
    intersections = 0
    for row in connect:
        corners = coords[row[:corner_count]]
        for point_idx in range(1, corner_count - 1):
            triangle = corners[[0, point_idx, point_idx + 1]]
            intersects = _check_ray_triangle(point, _RAY_DIRECT, triangle)
            intersections += intersects
    return bool(intersections % 2)


def _calc_inside_probe(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Calculate a point immediately inside a closed surface."""
    first_row = connect[0]
    corner_count = _get_surf_corner_count(first_row.shape[0])
    face_centre = np.mean(coords[first_row[:corner_count]], axis=0)
    face_normal = _calc_face_normal(first_row, coords)
    extent = float(np.linalg.norm(np.ptp(coords, axis=0)))
    offset = max(
        extent * _INSIDE_PROBE_SCALE,
        _GEOM_TOL * _INSIDE_PROBE_MIN_MULTIPLIER,
    )
    candidate_a = face_centre + offset * face_normal
    if _check_point_in_surf(candidate_a, connect, coords):
        return candidate_a
    return face_centre - offset * face_normal


def _calc_closed_comp_flips(
    connect: np.ndarray,
    coords: np.ndarray,
    components: list[_SurfComponent],
) -> np.ndarray:
    """Calculate exterior and cavity orientation reversals."""
    flips = np.zeros(connect.shape[0], dtype=bool)
    closed_components: list[tuple[np.ndarray, np.ndarray]] = []
    for rows, edge_uses in components:
        is_closed = _check_comp_closed(edge_uses)
        if not is_closed:
            continue
        rel_flips = _calc_comp_rel_flips(rows, edge_uses)
        flips[rows] = rel_flips
        coherent = _reverse_surf_rows(connect[rows], rel_flips)
        closed_components.append((rows, coherent))

    for rows, coherent in closed_components:
        probe = _calc_inside_probe(coherent, coords)
        nesting_depth = 0
        for other_rows, other_connect in closed_components:
            if np.array_equal(rows, other_rows):
                continue
            nesting_depth += _check_point_in_surf(
                probe, other_connect, coords
            )
        expect_positive = nesting_depth % 2 == 0
        is_positive = _calc_surf_signed_vol(coherent, coords) > 0.0
        if is_positive != expect_positive:
            flips[rows] = np.logical_not(flips[rows])
    return flips


def _enforce_surface_orient(
    connect: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Make faces coherent while retaining each open component's side."""
    topology = _build_surf_topology(connect)
    flips = _calc_closed_comp_flips(connect, coords, topology)
    for rows, edge_keys in topology:
        is_closed = _check_comp_closed(edge_keys)
        if not is_closed:
            flips[rows] = _calc_comp_rel_flips(rows, edge_keys)
    return _reverse_surf_rows(connect, flips)


def _orient_surface_to_hint(
    connect: np.ndarray,
    coords: np.ndarray,
    hint: np.ndarray,
) -> np.ndarray:
    """Orient each connected surface component towards a normal hint."""
    topology = _build_surf_topology(connect)
    flips = np.zeros(connect.shape[0], dtype=bool)
    for rows, edge_keys in topology:
        is_closed = _check_comp_closed(edge_keys)
        if is_closed:
            continue
        normal = _calc_face_normal(connect[rows[0]], coords)
        if np.dot(normal, hint) < 0.0:
            flips[rows] = True
    return _reverse_surf_rows(connect, flips)


def _reverse_open_surface(
    connect: np.ndarray,
) -> np.ndarray:
    """Reverse open components without changing closed-shell orientation."""
    topology = _build_surf_topology(connect)
    flips = np.zeros(connect.shape[0], dtype=bool)
    for rows, edge_keys in topology:
        is_closed = _check_comp_closed(edge_keys)
        if not is_closed:
            flips[rows] = True
    return _reverse_surf_rows(connect, flips)


def convert_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
    convention: ConnectConvention,
) -> MeshGeometry:
    """Convert NumPy mesh arrays to Riley's standard convention.

    Parameters
    ----------
    coords : np.ndarray
        Source coordinates with shape ``(nodes, 3)``.
    connect : np.ndarray
        One connectivity table containing exactly one element topology.
    convention : ConnectConvention
        Explicit source element type, element axis, index base and local-node
        ordering. A built-in adapter or explicit ``UserTopology`` is required.

    Returns
    -------
    MeshGeometry
        A new mesh in Riley's standard convention. Coordinate rows and global
        node IDs are not compacted or otherwise reordered.
    """
    coords_in = np.asarray(coords)
    connect_in = np.asarray(connect)
    if coords_in.ndim != 2 or coords_in.shape[1] != 3:
        raise MeshConvErr("coords must have shape (nodes, 3).")
    if coords_in.shape[0] == 0:
        raise MeshConvErr("coords must contain at least one node.")
    if not np.issubdtype(coords_in.dtype, np.floating):
        raise MeshConvErr("coords must have a floating-point dtype.")
    if not np.all(np.isfinite(coords_in)):
        raise MeshConvErr("coords must contain only finite values.")
    if connect_in.ndim != 2:
        raise MeshConvErr("connect must be a two-dimensional array.")
    if connect_in.size == 0 or connect_in.shape[0] == 0:
        raise MeshConvErr("connect must contain at least one element.")
    if np.issubdtype(connect_in.dtype, np.bool_) or not np.issubdtype(
        connect_in.dtype,
        np.integer,
    ):
        raise MeshConvErr("connect must have an integer dtype.")

    if convention.elem_axis is EConnectAxis.COLUMN:
        connect_in = connect_in.T
    spec = _ELEMENT_SPECS[convention.elem_type]
    if connect_in.shape[1] != spec.node_count:
        raise MeshConvErr(
            f"{convention.elem_type.value} connectivity requires "
            f"{spec.node_count} nodes per element."
        )

    if np.issubdtype(connect_in.dtype, np.unsignedinteger):
        max_int64 = np.iinfo(np.int64).max
        if connect_in.size and int(np.max(connect_in)) > max_int64:
            raise MeshConvErr("connect contains an index outside int64 range.")
    connect_std = np.ascontiguousarray(connect_in, dtype=np.int64)
    connect_std = connect_std - convention.index_base
    if np.any(connect_std < 0) or np.any(connect_std >= coords_in.shape[0]):
        raise MeshConvErr(
            "connect contains an index outside the coordinate array."
        )

    source_perm = np.asarray(_get_source_perm(convention), dtype=np.int64)
    connect_std = np.ascontiguousarray(
        connect_std[:, source_perm],
        dtype=np.int64,
    )
    coords_std = np.ascontiguousarray(coords_in, dtype=np.float64)
    if spec.is_surf:
        connect_std = _enforce_surface_orient(connect_std, coords_std)
    else:
        corner_slots = np.asarray(spec.corner_slots, dtype=np.int64)
        reverse_slots = np.asarray(spec.reverse_slots, dtype=np.int64)
        for elem_idx, row in enumerate(connect_std):
            metric = _calc_vol_metric(coords_std[row[corner_slots]])
            if metric < -_GEOM_TOL:
                connect_std[elem_idx] = row[reverse_slots]

    if spec.is_surf and convention.reverse_open_surface:
        connect_std = _reverse_open_surface(connect_std)
    if spec.is_surf and convention.material_normal_hint is not None:
        hint = np.asarray(convention.material_normal_hint, dtype=np.float64)
        if hint.shape != (3,) or not np.all(np.isfinite(hint)):
            raise MeshConvErr("material_normal_hint must be a finite vector3.")
        if np.linalg.norm(hint) <= _GEOM_TOL:
            raise MeshConvErr("material_normal_hint must be non-zero.")
        connect_std = _orient_surface_to_hint(
            connect_std,
            coords_std,
            hint,
        )

    mesh_out = MeshGeometry(
        elem_type=convention.elem_type,
        coords=coords_std,
        connect=np.ascontiguousarray(connect_std, dtype=np.int64),
    )
    verify_mesh(mesh_out)
    return mesh_out


def verify_mesh(mesh: MeshGeometry) -> None:
    """Verify Riley's standard mesh convention and report all found issues.

    Independent structural and per-element failures are collected into one
    ``MeshVerifyErr`` so a user can correct several input problems at once.
    Checks that depend on unsafe or malformed arrays are skipped.
    """
    issues: list[MeshVerifyIssue] = []
    if not isinstance(mesh.elem_type, EElementType):
        issues.append(MeshVerifyIssue(
            "element_type", "elem_type must be an EElementType member."
        ))
        raise MeshVerifyErr(issues)

    spec = _ELEMENT_SPECS[mesh.elem_type]
    coords = np.asarray(mesh.coords)
    connect = np.asarray(mesh.connect)
    coords_safe = coords.ndim == 2 and coords.shape[1:] == (3,)
    connect_safe = connect.ndim == 2
    connect_values_safe = connect_safe

    if not coords_safe:
        issues.append(MeshVerifyIssue(
            "coordinate_shape", "coords must have shape (nodes, 3)."
        ))
    else:
        if coords.shape[0] == 0:
            issues.append(MeshVerifyIssue(
                "empty_coordinates", "coords must contain at least one node."
            ))
        if coords.dtype != np.float64:
            issues.append(MeshVerifyIssue(
                "coordinate_dtype", "coords must have dtype float64."
            ))
        if not coords.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "coordinate_layout", "coords must be C-contiguous."
            ))
        if not np.all(np.isfinite(coords)):
            issues.append(MeshVerifyIssue(
                "coordinate_values", "coords must contain only finite values."
            ))

    if not connect_safe:
        issues.append(MeshVerifyIssue(
            "connectivity_shape", "connect must be two-dimensional."
        ))
    else:
        if connect.shape[0] == 0:
            issues.append(MeshVerifyIssue(
                "empty_connectivity", "connect must contain an element."
            ))
        if connect.shape[1] != spec.node_count:
            issues.append(MeshVerifyIssue(
                "connectivity_width",
                f"{mesh.elem_type.value} requires {spec.node_count} "
                "nodes per element.",
            ))
            connect_safe = False
        if connect.dtype != np.int64:
            issues.append(MeshVerifyIssue(
                "connectivity_dtype", "connect must have dtype int64."
            ))
        connect_is_integer = np.issubdtype(connect.dtype, np.integer)
        if np.issubdtype(connect.dtype, np.bool_) or not connect_is_integer:
            connect_values_safe = False
        if not connect.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "connectivity_layout", "connect must be C-contiguous."
            ))

    arrays_safe = coords_safe and connect_safe and connect_values_safe
    if arrays_safe and connect.size:
        valid_idxs = np.logical_and(connect >= 0, connect < coords.shape[0])
        if not np.all(valid_idxs):
            issues.append(MeshVerifyIssue(
                "connectivity_indices",
                "connect contains indices outside the coordinate array.",
            ))
        else:
            for elem_idx, row in enumerate(connect):
                if np.unique(row).shape[0] != row.shape[0]:
                    issues.append(MeshVerifyIssue(
                        "duplicate_node",
                        "connectivity contains a duplicate node ID.",
                        elem_idx,
                    ))
                    continue
            if spec.is_surf:
                try:
                    topology = _build_surf_topology(connect)
                    expected = _enforce_surface_orient(connect, coords)
                except (ValueError, NotImplementedError) as error:
                    issues.append(MeshVerifyIssue(
                        "surface_topology",
                        str(error),
                    ))
                else:
                    if not np.array_equal(expected, connect):
                        issues.append(MeshVerifyIssue(
                            "surface_orientation",
                            "surface faces are inconsistent or a closed shell "
                            "is not material-facing.",
                        ))
                    if not topology:
                        issues.append(MeshVerifyIssue(
                            "surface_topology", "surface has no components."
                        ))
            else:
                for elem_idx, row in enumerate(connect):
                    corner_idxs = np.asarray(
                        spec.corner_slots, dtype=np.int64
                    )
                    metric = _calc_vol_metric(
                        coords[row[corner_idxs]]
                    )
                    if metric <= _GEOM_TOL:
                        issues.append(MeshVerifyIssue(
                            "volume_orientation",
                            "volume element is degenerate or not right-handed.",
                            elem_idx,
                        ))

    if issues:
        raise MeshVerifyErr(issues)


_VOLUME_SURFACE_TYPES = MappingProxyType({
    EElementType.TET4: EElementType.TRI3,
    EElementType.TET10: EElementType.TRI6,
    EElementType.HEX8: EElementType.QUAD4,
    EElementType.HEX20: EElementType.QUAD8,
    EElementType.HEX27: EElementType.QUAD9,
})


def extract_surface(mesh: MeshGeometry) -> MeshGeometry:
    """Extract a compact Riley-standard surface from a volume mesh."""
    verify_mesh(mesh)
    if mesh.elem_type not in _VOLUME_SURFACE_TYPES:
        raise MeshConvErr("extract_surface requires a volume mesh.")

    spec = _ELEMENT_SPECS[mesh.elem_type]
    surf_type = _VOLUME_SURFACE_TYPES[mesh.elem_type]
    surf_spec = _ELEMENT_SPECS[surf_type]
    face_uses: dict[tuple[int, ...], list[np.ndarray]] = {}
    corner_count = len(surf_spec.corner_slots)
    for elem_row in mesh.connect:
        parent_centre = np.mean(
            mesh.coords[elem_row[np.asarray(spec.corner_slots)]], axis=0
        )
        for face_slots in spec.surface_faces:
            face = elem_row[np.asarray(face_slots, dtype=np.int64)]
            face_corners = mesh.coords[face[:corner_count]]
            face_centre = np.mean(face_corners, axis=0)
            face_normal = _calc_face_normal(face, mesh.coords)
            points_outward = np.dot(
                face_normal, face_centre - parent_centre
            ) > 0.0
            if not points_outward:
                reverse_slots = np.asarray(
                    surf_spec.reverse_slots, dtype=np.int64
                )
                face = face[reverse_slots]
            face_nodes = []
            for node in face[:corner_count]:
                face_nodes.append(int(node))
            face_key = tuple(sorted(face_nodes))
            face_uses.setdefault(face_key, []).append(face)

    surf_faces = []
    for face_key, uses in face_uses.items():
        if len(uses) > 2:
            raise MeshConvErr(
                f"Non-manifold volume face {face_key} has {len(uses)} "
                "incident elements."
            )
        if len(uses) == 1:
            surf_faces.append(uses[0])
    if not surf_faces:
        raise MeshConvErr("Volume mesh has no boundary faces.")

    surf_connect_glob = np.ascontiguousarray(
        np.vstack(surf_faces), dtype=np.int64
    )
    surf_node_idxs = np.unique(surf_connect_glob)
    node_remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    node_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0], dtype=np.int64
    )
    mesh_out = MeshGeometry(
        elem_type=surf_type,
        coords=np.ascontiguousarray(mesh.coords[surf_node_idxs]),
        connect=np.ascontiguousarray(node_remap[surf_connect_glob]),
    )
    verify_mesh(mesh_out)
    return mesh_out


__all__ = [
    "ConnectConvention",
    "EConnectAxis",
    "EElementType",
    "ENodeOrder",
    "EdgeNode",
    "FaceNode",
    "MeshConvErr",
    "MeshGeometry",
    "MeshVerifyErr",
    "MeshVerifyIssue",
    "UserTopology",
    "convert_mesh",
    "extract_surface",
    "verify_mesh",
]

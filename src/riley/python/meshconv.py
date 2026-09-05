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
from typing import Literal

import numpy as np

from riley.python.meshconstants import (
    EXODUS_TO_RILEY_MAP,
    RILEY_ELEM_TOP_MAP,
    RILEY_VOL_SURF_TYPE_MAP,
    VTK_TO_RILEY_MAP,
    EElemType,
)


class EConnectAxis(Enum):

    ROW = "row"
    COLUMN = "column"


class ENodeOrder(Enum):

    RILEY = "riley"
    VTK = "vtk"
    EXODUS = "exodus"


@dataclass(frozen=True, slots=True)
class EdgeNode:

    corners: tuple[int, int]
    node_slot: int


@dataclass(frozen=True, slots=True)
class FaceNode:

    corners: tuple[int, ...]
    node_slot: int


@dataclass(frozen=True, slots=True)
class UserTopology:

    corner_slots: tuple[int, ...]
    edge_nodes: tuple[EdgeNode, ...] = ()
    face_nodes: tuple[FaceNode, ...] = ()
    centre_slot: int | None = None


@dataclass(frozen=True, slots=True)
class ConnectConvention:

    elem_type: EElemType
    elem_axis: EConnectAxis
    index_base: Literal[0, 1]
    node_order: ENodeOrder | UserTopology
    material_normal_hint: tuple[float, float, float] | None = None
    reverse_open_surface: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.elem_type, EElemType):
            raise TypeError("elem_type must be an EElemType member.")
        if not isinstance(self.elem_axis, EConnectAxis):
            raise TypeError("elem_axis must be an EConnectAxis member.")
        if isinstance(self.index_base, bool) or self.index_base not in (0, 1):
            raise ValueError("index_base must be the integer 0 or 1.")
        valid_order = isinstance(self.node_order, (ENodeOrder, UserTopology))
        if not valid_order:
            raise TypeError("node_order must be ENodeOrder or UserTopology.")


@dataclass(frozen=True, slots=True)
class MeshGeometry:

    elem_type: EElemType
    coords: np.ndarray
    connect: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshVerifyIssue:

    code: str
    message: str
    elem_idx: int | None = None


class MeshError(ValueError):

    def __init__(
        self,
        message: str | None = None,
        issues: (
            list[MeshVerifyIssue] | tuple[MeshVerifyIssue, ...] | None
        ) = None,
    ) -> None:
        self.issues: tuple[MeshVerifyIssue, ...] = (
            tuple(issues) if issues is not None else ()
        )
        if message is not None:
            super().__init__(message)
        else:
            lines = [
                f"Mesh verification failed with {len(self.issues)} issue(s):"
            ]
            for issue in self.issues:
                location = ""
                if issue.elem_idx is not None:
                    location = f" elem {issue.elem_idx}:"
                lines.append(f"- [{issue.code}]{location} {issue.message}")
            super().__init__("\n".join(lines))


def _get_user_topology_perm(
    elem_type: EElemType,
    topology: UserTopology,
) -> tuple[int, ...]:
    spec = RILEY_ELEM_TOP_MAP[elem_type]
    node_count = spec.node_count
    corner_count = len(spec.corner_slots)
    if len(topology.corner_slots) != corner_count:
        raise MeshError(
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
            raise MeshError(
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
            raise MeshError(
                f"{elem_type.value} topology is missing face "
                f"{tuple(sorted(source_face))}."
            )
        perm[target_slot] = face_lookup[source_face]

    centre_slot = spec.centre_slot
    if centre_slot is not None:
        if topology.centre_slot is None:
            raise MeshError(
                f"{elem_type.value} topology requires a centre slot."
            )
        perm[centre_slot] = topology.centre_slot

    expected_slots = set(range(node_count))
    if set(perm) != expected_slots:
        raise MeshError(
            f"{elem_type.value} topology must use every source slot exactly "
            "once."
        )
    return tuple(perm)


def _get_source_perm(convention: ConnectConvention) -> tuple[int, ...]:
    if isinstance(convention.node_order, UserTopology):
        return _get_user_topology_perm(
            convention.elem_type,
            convention.node_order,
        )
    if convention.node_order in (ENodeOrder.RILEY, ENodeOrder.VTK):
        return VTK_TO_RILEY_MAP[convention.elem_type]
    if convention.node_order is ENodeOrder.EXODUS:
        return EXODUS_TO_RILEY_MAP[convention.elem_type]
    raise MeshError(f"Unsupported node ordering: {convention.node_order}.")

def convert_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
    convention: ConnectConvention,
) -> MeshGeometry:
    if not isinstance(convention, ConnectConvention):
        raise TypeError(
            "convention must be an instance of ConnectConvention."
        )

    coords_in = np.asarray(coords)
    connect_in = np.asarray(connect)

    if convention.elem_axis is EConnectAxis.COLUMN and connect_in.ndim == 2:
        connect_in = connect_in.T

    source_perm = np.asarray(_get_source_perm(convention), dtype=np.uintp)
    if (
        connect_in.ndim == 2
        and connect_in.shape[1] == source_perm.shape[0]
        and np.issubdtype(connect_in.dtype, np.integer)
        and not np.issubdtype(connect_in.dtype, np.bool_)
    ):
        connect_shifted = (
            np.ascontiguousarray(connect_in, dtype=np.int64)
            - convention.index_base
        )
        connect_std = np.ascontiguousarray(
            connect_shifted[:, source_perm],
            dtype=np.uintp,
        )
    else:
        connect_std = connect_in

    if np.issubdtype(coords_in.dtype, np.floating):
        coords_std = np.ascontiguousarray(coords_in, dtype=np.float64)
    else:
        coords_std = coords_in

    mesh_out = MeshGeometry(
        elem_type=convention.elem_type,
        coords=coords_std,
        connect=connect_std,
    )
    verify_mesh(mesh_out)
    return mesh_out


def verify_mesh(mesh: MeshGeometry) -> None:
    if not isinstance(mesh, MeshGeometry):
        raise TypeError("mesh must be an instance of MeshGeometry.")

    issues: list[MeshVerifyIssue] = []
    if not isinstance(mesh.elem_type, EElemType):
        issues.append(MeshVerifyIssue(
            "elem_type", "elem_type must be an EElemType member."
        ))
        raise MeshError(issues=issues)

    spec = RILEY_ELEM_TOP_MAP[mesh.elem_type]
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
        if connect.shape[0] == 0 or connect.size == 0:
            issues.append(MeshVerifyIssue(
                "empty_connectivity", "connect must contain an elem."
            ))
        if connect.shape[1] != spec.node_count:
            issues.append(MeshVerifyIssue(
                "connectivity_width",
                f"{mesh.elem_type.value} requires {spec.node_count} "
                "nodes per elem.",
            ))
            connect_safe = False
        if connect.dtype != np.uintp:
            issues.append(MeshVerifyIssue(
                "connectivity_dtype", "connect must have dtype uintp."
            ))
        if np.issubdtype(connect.dtype, np.bool_) or not np.issubdtype(
            connect.dtype,
            np.integer,
        ):
            connect_values_safe = False
        if not connect.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "connectivity_layout", "connect must be C-contiguous."
            ))

    arrays_safe = coords_safe and connect_safe and connect_values_safe
    if arrays_safe and connect.size and coords.shape[0] > 0:
        valid_idxs = np.logical_and(connect >= 0, connect < coords.shape[0])
        if not np.all(valid_idxs):
            issues.append(MeshVerifyIssue(
                "connectivity_indices",
                "connect contains indices outside the coordinate array.",
            ))
        else:
            for ee, row in enumerate(connect):
                if np.unique(row).shape[0] != row.shape[0]:
                    issues.append(MeshVerifyIssue(
                        "duplicate_node",
                        "connectivity contains a duplicate node ID.",
                        ee,
                    ))

    if issues:
        raise MeshError(issues=issues)


def _extract_surface_with_node_idxs(
    mesh: MeshGeometry,
) -> tuple[MeshGeometry, np.ndarray]:

    verify_mesh(mesh)

    if mesh.elem_type not in RILEY_VOL_SURF_TYPE_MAP:
        raise MeshError("extract_surface requires a volume mesh.")

    spec = RILEY_ELEM_TOP_MAP[mesh.elem_type]
    surf_type = RILEY_VOL_SURF_TYPE_MAP[mesh.elem_type]
    surf_spec = RILEY_ELEM_TOP_MAP[surf_type]
    face_uses: dict[tuple[int, ...], list[np.ndarray]] = {}
    corner_count = len(surf_spec.corner_slots)

    for elem_row in mesh.connect:
        for face_slots in spec.surf_faces:
            face = elem_row[np.asarray(face_slots, dtype=np.uintp)]
            face_nodes = [int(node) for node in face[:corner_count]]
            face_key = tuple(sorted(face_nodes))
            face_uses.setdefault(face_key, []).append(face)

    surf_faces = []
    for face_key, uses in face_uses.items():
        if len(uses) > 2:
            raise MeshError(
                f"Non-manifold volume face {face_key} has {len(uses)} "
                "incident elems."
            )
        if len(uses) == 1:
            surf_faces.append(uses[0])

    if not surf_faces:
        raise MeshError("Volume mesh has no boundary faces.")

    surf_connect_glob = np.ascontiguousarray(
        np.vstack(surf_faces), dtype=np.uintp
    )
    surf_node_idxs = np.unique(surf_connect_glob)

    node_remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    node_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0], dtype=np.int64
    )

    mesh_out = MeshGeometry(
        elem_type=surf_type,
        coords=np.ascontiguousarray(mesh.coords[surf_node_idxs]),
        connect=np.ascontiguousarray(
            node_remap[surf_connect_glob], dtype=np.uintp
        ),
    )

    verify_mesh(mesh_out)

    return mesh_out, surf_node_idxs


def extract_surface(mesh: MeshGeometry) -> MeshGeometry:
    mesh_out, _ = _extract_surface_with_node_idxs(mesh)
    return mesh_out


__all__ = [
    "ConnectConvention",
    "EConnectAxis",
    "EElemType",
    "ENodeOrder",
    "EdgeNode",
    "FaceNode",
    "MeshError",
    "MeshGeometry",
    "MeshVerifyIssue",
    "UserTopology",
    "convert_mesh",
    "extract_surface",
    "verify_mesh",
]

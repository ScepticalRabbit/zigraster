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
from typing import Literal

import numpy as np

from riley.python.meshconstants import (
    EXODUS_TO_RILEY,
    RILEY_ELEMENT_SPECS,
    RILEY_VOLUME_SURFACE_TYPES,
    VTK_TO_RILEY,
    EElementType,
)


class MeshConvErr(ValueError):
    """Raised when an explicit mesh conversion cannot be completed."""

    def __init__(
        self,
        message: str,
        issues: (
            list[MeshVerifyIssue] | tuple[MeshVerifyIssue, ...] | None
        ) = None,
    ) -> None:
        self.issues: tuple[MeshVerifyIssue, ...] = (
            tuple(issues) if issues is not None else ()
        )
        super().__init__(message)


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
    node_order: ENodeOrder | UserTopology
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


def _get_user_topology_perm(
    elem_type: EElementType,
    topology: UserTopology,
) -> tuple[int, ...]:
    """Build a Riley-target-to-source permutation from source topology."""
    spec = RILEY_ELEMENT_SPECS[elem_type]
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
        return VTK_TO_RILEY[convention.elem_type]
    if convention.node_order is ENodeOrder.EXODUS:
        return EXODUS_TO_RILEY[convention.elem_type]
    raise MeshConvErr(f"Unsupported node ordering: {convention.node_order}.")

def _verify_mesh_core(
    coords: np.ndarray,
    connect: np.ndarray,
    elem_type: EElementType,
    *,
    strict_std_types: bool = False,
    check_c_contiguous: bool = False,
) -> list[MeshVerifyIssue]:
    """Perform core structural checks on coordinates and connectivity.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array candidate.
    connect : np.ndarray
        Connectivity array candidate (already index-base shifted).
    elem_type : EElementType
        Target or source element type.
    strict_std_types : bool, default=False
        If True, require float64 coords and uintp connect. If False, accept
        any floating dtype for coords and any integer dtype for connect.
    check_c_contiguous : bool, default=False
        If True, verify both arrays are C-contiguous in memory.

    Returns
    -------
    list[MeshVerifyIssue]
        Independent validation issues discovered during inspection.
    """
    issues: list[MeshVerifyIssue] = []
    if not isinstance(elem_type, EElementType):
        issues.append(MeshVerifyIssue(
            "element_type", "elem_type must be an EElementType member."
        ))
        return issues

    spec = RILEY_ELEMENT_SPECS[elem_type]
    coords_arr = np.asarray(coords)
    connect_arr = np.asarray(connect)

    coords_safe = coords_arr.ndim == 2 and coords_arr.shape[1:] == (3,)
    connect_safe = connect_arr.ndim == 2
    connect_values_safe = connect_safe

    if not coords_safe:
        issues.append(MeshVerifyIssue(
            "coordinate_shape", "coords must have shape (nodes, 3)."
        ))
    else:
        if coords_arr.shape[0] == 0:
            issues.append(MeshVerifyIssue(
                "empty_coordinates", "coords must contain at least one node."
            ))
        if strict_std_types:
            if coords_arr.dtype != np.float64:
                issues.append(MeshVerifyIssue(
                    "coordinate_dtype", "coords must have dtype float64."
                ))
        else:
            if not np.issubdtype(coords_arr.dtype, np.floating):
                issues.append(MeshVerifyIssue(
                    "coordinate_dtype",
                    "coords must have a floating-point dtype.",
                ))
        if check_c_contiguous and not coords_arr.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "coordinate_layout", "coords must be C-contiguous."
            ))
        if not np.all(np.isfinite(coords_arr)):
            issues.append(MeshVerifyIssue(
                "coordinate_values", "coords must contain only finite values."
            ))

    if not connect_safe:
        issues.append(MeshVerifyIssue(
            "connectivity_shape", "connect must be two-dimensional."
        ))
    else:
        if connect_arr.shape[0] == 0 or connect_arr.size == 0:
            issues.append(MeshVerifyIssue(
                "empty_connectivity", "connect must contain an element."
            ))
        if connect_arr.shape[1] != spec.node_count:
            issues.append(MeshVerifyIssue(
                "connectivity_width",
                f"{elem_type.value} requires {spec.node_count} "
                "nodes per element.",
            ))
            connect_safe = False
        if strict_std_types:
            if connect_arr.dtype != np.uintp:
                issues.append(MeshVerifyIssue(
                    "connectivity_dtype", "connect must have dtype uintp."
                ))
            if np.issubdtype(connect_arr.dtype, np.bool_) or not np.issubdtype(
                connect_arr.dtype,
                np.integer,
            ):
                connect_values_safe = False
        else:
            if np.issubdtype(connect_arr.dtype, np.bool_) or not np.issubdtype(
                connect_arr.dtype,
                np.integer,
            ):
                issues.append(MeshVerifyIssue(
                    "connectivity_dtype",
                    "connect must have an integer dtype.",
                ))
                connect_values_safe = False
        if check_c_contiguous and not connect_arr.flags.c_contiguous:
            issues.append(MeshVerifyIssue(
                "connectivity_layout", "connect must be C-contiguous."
            ))

    arrays_safe = coords_safe and connect_safe and connect_values_safe
    if arrays_safe and connect_arr.size and coords_arr.shape[0] > 0:
        if not strict_std_types and np.issubdtype(
            connect_arr.dtype,
            np.unsignedinteger,
        ):
            max_int64 = np.iinfo(np.int64).max
            if int(np.max(connect_arr)) > max_int64:
                issues.append(MeshVerifyIssue(
                    "connectivity_indices",
                    "connect contains an index outside int64 range.",
                ))
                return issues
        valid_idxs = np.logical_and(
            connect_arr >= 0,
            connect_arr < coords_arr.shape[0],
        )
        if not np.all(valid_idxs):
            issues.append(MeshVerifyIssue(
                "connectivity_indices",
                "connect contains indices outside the coordinate array.",
            ))

    return issues


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
    if not isinstance(convention, ConnectConvention):
        raise TypeError(
            "convention must be an instance of ConnectConvention."
        )

    coords_in = np.asarray(coords)
    connect_in = np.asarray(connect)

    if convention.elem_axis is EConnectAxis.COLUMN and connect_in.ndim == 2:
        connect_in = connect_in.T

    if (
        connect_in.ndim == 2
        and np.issubdtype(connect_in.dtype, np.integer)
        and not np.issubdtype(connect_in.dtype, np.bool_)
    ):
        if np.issubdtype(connect_in.dtype, np.unsignedinteger):
            max_int64 = np.iinfo(np.int64).max
            if connect_in.size and int(np.max(connect_in)) > max_int64:
                connect_shifted = connect_in
            else:
                connect_shifted = (
                    connect_in.astype(np.int64) - convention.index_base
                )
        else:
            connect_shifted = (
                connect_in.astype(np.int64) - convention.index_base
            )
    else:
        connect_shifted = connect_in

    issues = _verify_mesh_core(
        coords=coords_in,
        connect=connect_shifted,
        elem_type=convention.elem_type,
        strict_std_types=False,
        check_c_contiguous=False,
    )
    if issues:
        lines = [
            f"Mesh conversion failed with {len(issues)} validation issue(s):"
        ]
        for issue in issues:
            location = ""
            if issue.elem_idx is not None:
                location = f" element {issue.elem_idx}:"
            lines.append(f"- [{issue.code}]{location} {issue.message}")
        raise MeshConvErr("\n".join(lines), issues=issues)

    connect_std = np.ascontiguousarray(connect_shifted, dtype=np.int64)
    source_perm = np.asarray(_get_source_perm(convention), dtype=np.uintp)
    connect_std = np.ascontiguousarray(
        connect_std[:, source_perm],
        dtype=np.uintp,
    )
    coords_std = np.ascontiguousarray(coords_in, dtype=np.float64)

    mesh_out = MeshGeometry(
        elem_type=convention.elem_type,
        coords=coords_std,
        connect=connect_std,
    )
    verify_mesh(mesh_out)
    return mesh_out


def verify_mesh(mesh: MeshGeometry) -> None:
    """Verify Riley's standard mesh convention and report all found issues.

    Independent structural and per-element failures are collected into one
    ``MeshVerifyErr`` so a user can correct several input problems at once.
    Checks that depend on unsafe or malformed arrays are skipped.
    """
    if not isinstance(mesh, MeshGeometry):
        raise TypeError("mesh must be an instance of MeshGeometry.")

    issues = _verify_mesh_core(
        coords=mesh.coords,
        connect=mesh.connect,
        elem_type=mesh.elem_type,
        strict_std_types=True,
        check_c_contiguous=True,
    )
    if not isinstance(mesh.elem_type, EElementType):
        raise MeshVerifyErr(issues)

    coords = np.asarray(mesh.coords)
    connect = np.asarray(mesh.connect)

    coords_safe = (
        coords.ndim == 2
        and coords.shape[1:] == (3,)
        and coords.shape[0] > 0
        and np.all(np.isfinite(coords))
    )
    spec = RILEY_ELEMENT_SPECS[mesh.elem_type]
    connect_safe = (
        connect.ndim == 2
        and connect.shape[0] > 0
        and connect.shape[1] == spec.node_count
        and np.issubdtype(connect.dtype, np.integer)
        and not np.issubdtype(connect.dtype, np.bool_)
    )

    if coords_safe and connect_safe and connect.size:
        valid_idxs = np.logical_and(connect >= 0, connect < coords.shape[0])
        if np.all(valid_idxs):
            for ee, row in enumerate(connect):
                if np.unique(row).shape[0] != row.shape[0]:
                    issues.append(MeshVerifyIssue(
                        "duplicate_node",
                        "connectivity contains a duplicate node ID.",
                        ee,
                    ))

    if issues:
        raise MeshVerifyErr(issues)


def _extract_surface_with_node_idxs(
    mesh: MeshGeometry,
) -> tuple[MeshGeometry, np.ndarray]:
    """Extract a surface and return its original global node indices."""
    verify_mesh(mesh)
    if mesh.elem_type not in RILEY_VOLUME_SURFACE_TYPES:
        raise MeshConvErr("extract_surface requires a volume mesh.")

    spec = RILEY_ELEMENT_SPECS[mesh.elem_type]
    surf_type = RILEY_VOLUME_SURFACE_TYPES[mesh.elem_type]
    surf_spec = RILEY_ELEMENT_SPECS[surf_type]
    face_uses: dict[tuple[int, ...], list[np.ndarray]] = {}
    corner_count = len(surf_spec.corner_slots)

    for elem_row in mesh.connect:
        for face_slots in spec.surface_faces:
            face = elem_row[np.asarray(face_slots, dtype=np.uintp)]
            face_nodes = [int(node) for node in face[:corner_count]]
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
    """Extract a compact Riley-standard surface from a volume mesh."""
    mesh_out, _ = _extract_surface_with_node_idxs(mesh)
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

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

from riley.python.meshconst import (
    ELEM_NODE_COUNT_MAP,
    EXODUS_TO_RILEY_MAP,
    RILEY_ELEM_TOP_MAP,
    RILEY_MAP,
    RILEY_TRI_STENCIL_MAP,
    RILEY_VOL_SURF_TYPE_MAP,
    VTK_TO_RILEY_MAP,
    EElemType,
)


class EConnectAxis(Enum):
    """Array dimension orientation of elements in a connectivity table.

    Members
    -------
    ROW
        Each row represents an element: shape `(num_elems, nodes_per_elem)`.
    COLUMN
        Each column represents an element: shape `(nodes_per_elem, num_elems)`.
    """

    ROW = "row"
    COLUMN = "column"


class ENodeOrder(Enum):
    """Standard element node numbering and winding conventions.

    Members
    -------
    RILEY
        Canonical Riley node winding order.
    VTK
        Visualization Toolkit (VTK) standard node order.
    EXODUS
        Exodus II standard node order.
    """

    RILEY = "riley"
    VTK = "vtk"
    EXODUS = "exodus"


@dataclass(frozen=True, slots=True)
class EdgeNode:
    """Mid-edge node slot mapping for custom user topology.

    Attributes
    ----------
    corners : tuple of int
        Pair of corner vertex local indices `(corner_a, corner_b)`.
    node_slot : int
        Zero-based index of the mid-edge node in the source element.
    """

    corners: tuple[int, int]
    node_slot: int


@dataclass(frozen=True, slots=True)
class FaceNode:
    """Face node slot mapping for custom user topology.

    Attributes
    ----------
    corners : tuple of int
        Tuple of corner vertex local indices bounding the face.
    node_slot : int
        Zero-based index of the face node in the source element.
    """

    corners: tuple[int, ...]
    node_slot: int


@dataclass(frozen=True, slots=True)
class UserTopology:
    """Custom node numbering layout specification for element conversion.

    Attributes
    ----------
    corner_slots : tuple of int
        Indices of corner vertices in source element ordering.
    edge_nodes : tuple of EdgeNode, default=()
        Mid-edge node definitions.
    face_nodes : tuple of FaceNode, default=()
        Face node definitions.
    centre_slot : int or None, default=None
        Index of the center/bubble node, if present.
    """

    corner_slots: tuple[int, ...]
    edge_nodes: tuple[EdgeNode, ...] = ()
    face_nodes: tuple[FaceNode, ...] = ()
    centre_slot: int | None = None


@dataclass(frozen=True, slots=True)
class ConnectConvention:
    """Specification of connectivity table formatting and ordering.

    Attributes
    ----------
    elem_type : EElemType
        Target finite element geometry type.
    elem_axis : EConnectAxis
        Orientation of elements in the connectivity table (row or column).
    index_base : {0, 1}
        Indexing base for node IDs (0-based or 1-based).
    node_order : ENodeOrder or UserTopology
        Standard convention or custom topology mapping for node winding.
    material_normal_hint : tuple of float or None, default=None
        Optional direction `(nx, ny, nz)` indicating the material side.
    reverse_open_surface : bool, default=False
        Whether to reverse the surface normal winding for open surfaces.
    """

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


@dataclass(slots=True)
class MeshGeometry:
    """Standardized mesh geometry consisting of nodes and connectivity.

    Attributes
    ----------
    elem_type : EElemType
        Finite element geometry type.
    coords : numpy.ndarray
        Node coordinates array of shape `(N, 3)` and dtype `np.float64`,
        where `N` is the number of nodes and columns represent `(x, y, z)`.
    connect : numpy.ndarray
        Element connectivity table of shape `(E, M)` and dtype `np.uintp`,
        where `E` is the number of elements and `M` is the number of nodes
        per element in canonical Riley winding order.
    """

    elem_type: EElemType
    coords: np.ndarray
    connect: np.ndarray


@dataclass(frozen=True, slots=True)
class MeshVerifyIssue:
    """Individual issue or violation detected during mesh verification.

    Attributes
    ----------
    code : str
        Machine-readable error classification code.
    message : str
        Human-readable description of the issue.
    elem_idx : int or None, default=None
        Index of the offending element, if applicable.
    """

    code: str
    message: str
    elem_idx: int | None = None


class MeshError(ValueError):
    """Exception raised when mesh data, conversion, or verification fails."""

    def __init__(
        self,
        message: str | None = None,
        issues: Sequence[MeshVerifyIssue] | None = None,
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
    """Compute index permutation from a custom UserTopology to Riley ordering.

    Parameters
    ----------
    elem_type : EElemType
        Target finite element geometry type.
    topology : UserTopology
        User-specified node layout and topological definitions.

    Returns
    -------
    tuple of int
        Permutation tuple mapping canonical Riley slots to source slots.

    Raises
    ------
    MeshError
        If `topology` has incorrect corners, missing edges/faces, or
        incomplete node mappings.
    """
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
    """Retrieve the node index permutation tuple for a ConnectConvention.

    Parameters
    ----------
    convention : ConnectConvention
        Input mesh connectivity convention.

    Returns
    -------
    tuple of int
        Permutation tuple from source ordering to Riley standard ordering.

    Raises
    ------
    MeshError
        If `convention.node_order` is unsupported.
    """
    if isinstance(convention.node_order, UserTopology):
        return _get_user_topology_perm(
            convention.elem_type,
            convention.node_order,
        )

    if convention.node_order is ENodeOrder.RILEY:
        return RILEY_MAP[convention.elem_type]
    elif convention.node_order is ENodeOrder.VTK:
        return VTK_TO_RILEY_MAP[convention.elem_type]
    elif convention.node_order is ENodeOrder.EXODUS:
        return EXODUS_TO_RILEY_MAP[convention.elem_type]

    raise MeshError(f"Unsupported node ordering: {convention.node_order}.")


def convert_mesh(
    coords: np.ndarray,
    connect: np.ndarray,
    convention: ConnectConvention,
) -> MeshGeometry:
    """Standardize raw mesh arrays into canonical Riley MeshGeometry.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of node coordinates of shape `(N, 3)` and floating-point dtype
        `np.float64`, where `N` is the number of nodes and columns represent
        spatial coordinates `(x, y, z)`.
    connect : numpy.ndarray
        Connectivity table of shape `(E, M)` if `convention.elem_axis` is ROW,
        or `(M, E)` if COLUMN, with integer dtype, where `E` is the number of
        elements and `M` is the number of nodes per element.
    convention : ConnectConvention
        Description of element type, axis orientation, base index, and
        node winding order of the input `connect` table.

    Returns
    -------
    MeshGeometry
        Verified standardized mesh containing C-contiguous `coords` of shape
        `(N, 3)` with dtype `np.float64` and `connect` of shape `(E, M)` with
        dtype `np.uintp`.

    Raises
    ------
    TypeError
        If `convention` is not a `ConnectConvention`.
    MeshError
        If conversion fails or the resulting mesh fails validation.
    """
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
    """Validate that a MeshGeometry satisfies Riley integrity requirements.

    Parameters
    ----------
    mesh : MeshGeometry
        Mesh geometry to verify. `coords` must have shape `(N, 3)` and dtype
        `np.float64`. `connect` must have shape `(E, M)` and dtype `np.uintp`.

    Raises
    ------
    TypeError
        If `mesh` is not a `MeshGeometry` instance.
    MeshError
        If coordinates or connectivity are non-contiguous, out of range,
        empty, contain duplicates, or have mismatched element shapes.
    """
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
            sorted_connect = np.sort(connect, axis=1)
            has_duplicate = np.any(
                sorted_connect[:, :-1] == sorted_connect[:, 1:],
                axis=1,
            )
            if np.any(has_duplicate):
                for ee in np.flatnonzero(has_duplicate):
                    issues.append(MeshVerifyIssue(
                        "duplicate_node",
                        "connectivity contains a duplicate node ID.",
                        int(ee),
                    ))

    if issues:
        raise MeshError(issues=issues)


def _extract_surface_with_node_idxs(
    mesh: MeshGeometry,
) -> tuple[MeshGeometry, np.ndarray]:
    """Extract exterior boundary surface and track original node indices.

    Parameters
    ----------
    mesh : MeshGeometry
        Input 3D volume mesh with coordinates and connectivity.

    Returns
    -------
    MeshGeometry
        2D surface mesh containing only boundary faces and referenced nodes.
    numpy.ndarray
        Array of shape `(K,)` and dtype `np.uintp` mapping each surface node
        back to its original node index in `mesh.coords`.

    Raises
    ------
    MeshError
        If `mesh` is not a volume mesh, has no boundary, or has non-manifold
        topology.
    """
    # Check the input mesh before relying on its connectivity and coordinates.
    verify_mesh(mesh)

    # Surface extraction requires a supported volume element type.
    if mesh.elem_type not in RILEY_VOL_SURF_TYPE_MAP:
        raise MeshError("extract_surface requires a volume mesh.")

    # Get the topology needed to find volume faces and build surface elements.
    spec = RILEY_ELEM_TOP_MAP[mesh.elem_type]
    surf_type = RILEY_VOL_SURF_TYPE_MAP[mesh.elem_type]
    surf_spec = RILEY_ELEM_TOP_MAP[surf_type]
    face_uses: dict[tuple[int, ...], list[np.ndarray]] = {}
    corner_count = len(surf_spec.corner_slots)

    # Collect every face so we can distinguish shared from boundary faces.
    for elem_row in mesh.connect:
        for face_slots in spec.surf_faces:
            # Preserve prescribed face ordering, including higher-order nodes.
            face = elem_row[np.asarray(face_slots, dtype=np.uintp)]

            # Use corner node IDs to identify which geometric face this is.
            face_nodes: list[int] = []
            for node in face[:corner_count]:
                face_nodes.append(int(node))

            # Ignore corner ordering when matching adjacent element faces.
            face_key = tuple(sorted(face_nodes))
            face_uses.setdefault(face_key, []).append(face)

    # Keep faces used once; faces shared by two elements are internal.
    surf_faces = []
    for face_key, uses in face_uses.items():
        # More than two incident elements makes the face non-manifold.
        if len(uses) > 2:
            raise MeshError(
                f"Non-manifold volume face {face_key} has {len(uses)} "
                "incident elems."
            )

        if len(uses) == 1:
            surf_faces.append(uses[0])

    # Fail if there is no boundary from which to build a surface mesh.
    if not surf_faces:
        raise MeshError("Volume mesh has no boundary faces.")

    # Assemble boundary connectivity and find all referenced nodes.
    surf_connect_glob = np.ascontiguousarray(
        np.vstack(surf_faces), dtype=np.uintp
    )
    surf_node_idxs = np.unique(surf_connect_glob)

    # Map original node indices to consecutive surface indices starting at zero.
    # Nodes absent from the surface retain -1 and are never referenced below.
    node_remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    node_remap[surf_node_idxs] = np.arange(
        surf_node_idxs.shape[0], dtype=np.int64
    )

    # Discard unused coordinates and rebase connectivity to the reduced array.
    # Store both arrays contiguously for subsequent mesh processing.
    mesh_out = MeshGeometry(
        elem_type=surf_type,
        coords=np.ascontiguousarray(mesh.coords[surf_node_idxs]),
        connect=np.ascontiguousarray(
            node_remap[surf_connect_glob], dtype=np.uintp
        ),
    )

    # Check that extraction and node remapping produced a valid surface mesh.
    # We catch stray -1 left over from the remapping here.
    verify_mesh(mesh_out)

    # Return original node indices too, so callers can subset associated fields.
    return mesh_out, surf_node_idxs


def extract_surface(mesh: MeshGeometry) -> MeshGeometry:
    """Extract exterior boundary 2D surface elements from a 3D volume mesh.

    Parameters
    ----------
    mesh : MeshGeometry
        Input 3D volume mesh (`TET4`, `TET10`, `HEX8`, `HEX20`, `HEX27`).

    Returns
    -------
    MeshGeometry
        Extracted 2D boundary surface mesh with matching element order.

    Raises
    ------
    MeshError
        If `mesh` is not a volume mesh or has non-manifold topology.
    """
    mesh_out, _ = _extract_surface_with_node_idxs(mesh)
    return mesh_out


def _reduce_elem_order_with_node_idxs(
    mesh: MeshGeometry,
    source_node_idxs: np.ndarray,
    target: EElemType,
) -> tuple[MeshGeometry, np.ndarray]:
    """Reduce element interpolation order and track original node indices.

    Parameters
    ----------
    mesh : MeshGeometry
        Input higher-order mesh geometry.
    source_node_idxs : numpy.ndarray
        Array of shape `(N,)` and dtype `np.uintp` of source node indices.
    target : EElemType
        Lower-order target element type.

    Returns
    -------
    MeshGeometry
        Reduced order mesh with unused higher-order nodes stripped.
    numpy.ndarray
        Array of shape `(K,)` and dtype `np.uintp` of retained node indices.

    Raises
    ------
    MeshError
        If `target` has more nodes than `mesh.elem_type`.
    """
    verify_mesh(mesh)

    if target not in ELEM_NODE_COUNT_MAP:
        raise MeshError(f"Unsupported reduction target elem type: {target}.")

    target_count = ELEM_NODE_COUNT_MAP[target]
    if mesh.connect.shape[1] < target_count:
        raise MeshError(
            f"Cannot reduce {mesh.elem_type.value} to {target.value}: "
            f"source has {mesh.connect.shape[1]} nodes, target needs "
            f"{target_count}."
        )

    connect = mesh.connect[:, :target_count]
    retained = np.unique(connect)
    remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    remap[retained] = np.arange(retained.size, dtype=np.int64)

    result = MeshGeometry(
        target,
        np.ascontiguousarray(mesh.coords[retained]),
        np.ascontiguousarray(remap[connect], dtype=np.uintp),
    )

    verify_mesh(result)

    return result, np.ascontiguousarray(source_node_idxs[retained])


def reduce_mesh_order(
    mesh: MeshGeometry,
    target: EElemType,
) -> MeshGeometry:
    """Down-sample a higher-order mesh to a lower-order element topology.

    Parameters
    ----------
    mesh : MeshGeometry
        Input higher-order mesh geometry.
    target : EElemType
        Target element type with fewer nodes (e.g., `QUAD8` to `QUAD4`).

    Returns
    -------
    MeshGeometry
        Reduced mesh containing only nodes referenced by the target topology.

    Raises
    ------
    MeshError
        If `target` requires more nodes than the source element type.
    """
    node_idxs = np.arange(mesh.coords.shape[0], dtype=np.uintp)
    result, _ = _reduce_elem_order_with_node_idxs(mesh, node_idxs, target)

    return result


def _triangulate_with_node_idxs(
    mesh: MeshGeometry,
    source_node_idxs: np.ndarray,
    source: EElemType,
) -> tuple[MeshGeometry, np.ndarray]:
    """Subdivide 2D elements into linear triangles tracking node indices.

    Parameters
    ----------
    mesh : MeshGeometry
        Input 2D surface mesh geometry.
    source_node_idxs : numpy.ndarray
        Array of shape `(N,)` and dtype `np.uintp` of source node indices.
    source : EElemType
        Element type of the input mesh.

    Returns
    -------
    MeshGeometry
        Subdivided `TRI3` mesh geometry.
    numpy.ndarray
        Array of shape `(K,)` and dtype `np.uintp` of retained node indices.

    Raises
    ------
    MeshError
        If `source` element type does not have a triangulation stencil.
    """
    verify_mesh(mesh)

    if source not in RILEY_TRI_STENCIL_MAP:
        raise MeshError(
            f"Cannot triangulate unsupported element type: {source.value}."
        )

    stencil = np.asarray(RILEY_TRI_STENCIL_MAP[source], dtype=np.uintp)
    tri_connect = mesh.connect[:, stencil].reshape(-1, 3)
    retained = np.unique(tri_connect)

    remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    remap[retained] = np.arange(retained.size, dtype=np.int64)

    result = MeshGeometry(
        EElemType.TRI3,
        np.ascontiguousarray(mesh.coords[retained]),
        np.ascontiguousarray(remap[tri_connect], dtype=np.uintp),
    )

    verify_mesh(result)

    return result, np.ascontiguousarray(source_node_idxs[retained])


def triangulate_mesh(mesh: MeshGeometry) -> MeshGeometry:
    """Subdivide a 2D polygonal or quadratic mesh into 3-node linear triangles.

    Parameters
    ----------
    mesh : MeshGeometry
        Input 2D surface mesh (`TRI3`, `TRI6`, `TRI7`, `QUAD4`,
        `QUAD8`, `QUAD9`).

    Returns
    -------
    MeshGeometry
        Triangulated `TRI3` mesh geometry.

    Raises
    ------
    MeshError
        If `mesh.elem_type` is not a 2D surface element.
    """
    node_idxs = np.arange(mesh.coords.shape[0], dtype=np.uintp)

    result, _ = _triangulate_with_node_idxs(
        mesh, node_idxs, mesh.elem_type
    )

    return result


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
    "reduce_mesh_order",
    "triangulate_mesh",
    "verify_mesh",
]

# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Tests for the explicit Riley standard mesh conversion API."""

from __future__ import annotations

import numpy as np
import pytest

from riley.python import meshconv
from riley.python.meshconv import (
    ConnectConvention,
    EConnectAxis,
    EdgeNode,
    EElemType,
    ENodeOrder,
    MeshError,
    MeshGeometry,
    UserTopology,
)

@pytest.mark.parametrize(
    "connect",
    (
        np.empty((0, 3), dtype=np.int64),
        np.array(((0, 1, 4),), dtype=np.int64),
        np.array(((0.0, 1.5, 2.0),)),
    ),
)
def test_convert_mesh_rejects_invalid_connectivity(
    connect: np.ndarray,
) -> None:
    coords = np.eye(3, dtype=np.float64)
    convention = meshconv.ConnectConvention(
        meshconv.EElemType.TRI3, meshconv.EConnectAxis.ROW, 0,
        meshconv.ENodeOrder.RILEY,
    )

    with pytest.raises(meshconv.MeshError):
        meshconv.convert_mesh(coords, connect, convention)


def test_connect_convention_rejects_invalid_elem_type() -> None:
    with pytest.raises(TypeError, match="elem_type must be an EElemType"):
        ConnectConvention(
            elem_type="tri3",  # type: ignore[arg-type]
            elem_axis=EConnectAxis.ROW,
            index_base=0,
            node_order=ENodeOrder.RILEY,
        )


def test_connect_convention_rejects_invalid_elem_axis() -> None:
    with pytest.raises(TypeError, match="elem_axis must be an EConnectAxis"):
        ConnectConvention(
            elem_type=EElemType.TRI3,
            elem_axis="row",  # type: ignore[arg-type]
            index_base=0,
            node_order=ENodeOrder.RILEY,
        )


def test_connect_convention_rejects_invalid_index_base() -> None:
    with pytest.raises(
        ValueError, match="index_base must be the integer 0 or 1"
    ):
        ConnectConvention(
            elem_type=EElemType.TRI3,
            elem_axis=EConnectAxis.ROW,
            index_base=2,  # type: ignore[arg-type]
            node_order=ENodeOrder.RILEY,
        )
    with pytest.raises(
        ValueError, match="index_base must be the integer 0 or 1"
    ):
        ConnectConvention(
            elem_type=EElemType.TRI3,
            elem_axis=EConnectAxis.ROW,
            index_base=True,  # type: ignore[arg-type]
            node_order=ENodeOrder.RILEY,
        )


def test_connect_convention_rejects_invalid_node_order() -> None:
    with pytest.raises(
        TypeError, match="node_order must be ENodeOrder or UserTopology"
    ):
        ConnectConvention(
            elem_type=EElemType.TRI3,
            elem_axis=EConnectAxis.ROW,
            index_base=0,
            node_order="invalid_order",  # type: ignore[arg-type]
        )


def test_user_topology_rejects_wrong_corner_count() -> None:
    topology = UserTopology(
        corner_slots=(0, 1),
    )
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.uintp)

    with pytest.raises(MeshError, match="tri3 requires 3 corner slots"):
        meshconv.convert_mesh(coords, connect, convention)


def test_user_topology_rejects_non_permutation_corners() -> None:
    topology = UserTopology(
        corner_slots=(0, 0, 1),
    )
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.uintp)

    with pytest.raises(
        MeshError, match="must use every source slot exactly once"
    ):
        meshconv.convert_mesh(coords, connect, convention)


def test_user_topology_rejects_wrong_edge_node_count() -> None:
    topology = UserTopology(
        corner_slots=(0, 1, 2),
        edge_nodes=(
            EdgeNode((0, 1), 3),
            EdgeNode((1, 2), 4),
        ),
    )
    convention = ConnectConvention(
        elem_type=EElemType.TRI6,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.zeros((6, 3), dtype=np.float64)
    connect = np.arange(6, dtype=np.uintp)[None, :]

    with pytest.raises(MeshError, match="tri6 topology is missing edge"):
        meshconv.convert_mesh(coords, connect, convention)


def test_user_topology_rejects_invalid_edge_node_definition() -> None:
    topology = UserTopology(
        corner_slots=(0, 1, 2),
        edge_nodes=(
            EdgeNode((0, 0), 3),
            EdgeNode((1, 2), 4),
            EdgeNode((2, 0), 5),
        ),
    )
    convention = ConnectConvention(
        elem_type=EElemType.TRI6,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.zeros((6, 3), dtype=np.float64)
    connect = np.arange(6, dtype=np.uintp)[None, :]

    with pytest.raises(MeshError, match="tri6 topology is missing edge"):
        meshconv.convert_mesh(coords, connect, convention)


def test_user_topology_rejects_wrong_face_node_count() -> None:
    topology = UserTopology(
        corner_slots=tuple(range(8)),
        edge_nodes=tuple(
            EdgeNode((c0, c1), 8 + ii)
            for ii, (c0, c1) in enumerate(
                ((0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
            )
        ),
        face_nodes=(),
        centre_slot=26,
    )
    convention = ConnectConvention(
        elem_type=EElemType.HEX27,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.zeros((27, 3), dtype=np.float64)
    connect = np.arange(27, dtype=np.uintp)[None, :]

    with pytest.raises(MeshError, match="hex27 topology is missing face"):
        meshconv.convert_mesh(coords, connect, convention)


def test_user_topology_rejects_duplicate_or_incomplete_slots() -> None:
    topology = UserTopology(
        corner_slots=(0, 1, 2),
        edge_nodes=(
            EdgeNode((0, 1), 3),
            EdgeNode((1, 2), 4),
            EdgeNode((2, 0), 3),  # duplicate slot 3
        ),
    )
    convention = ConnectConvention(
        elem_type=EElemType.TRI6,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )
    coords = np.zeros((6, 3), dtype=np.float64)
    connect = np.arange(6, dtype=np.uintp)[None, :]

    with pytest.raises(
        MeshError, match="must use every source slot exactly once"
    ):
        meshconv.convert_mesh(coords, connect, convention)


def test_convert_mesh_requires_connect_convention() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.uintp)

    with pytest.raises(
        TypeError, match="convention must be an instance of ConnectConvention"
    ):
        meshconv.convert_mesh(
            coords, connect, convention="invalid",  # type: ignore[arg-type]
        )


def test_convert_mesh_rejects_column_axis_wrong_dim() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.zeros((4, 1), dtype=np.uintp)
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.COLUMN,
        index_base=0,
        node_order=ENodeOrder.RILEY,
    )

    with pytest.raises(MeshError, match="requires 3 nodes per elem"):
        meshconv.convert_mesh(coords, connect, convention)


def test_convert_mesh_rejects_row_axis_wrong_dim() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.zeros((1, 4), dtype=np.uintp)
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=ENodeOrder.RILEY,
    )

    with pytest.raises(MeshError, match="requires 3 nodes per elem"):
        meshconv.convert_mesh(coords, connect, convention)


def test_convert_mesh_rejects_non_integer_connect() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0.5, 1.2, 2.3),))
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=ENodeOrder.RILEY,
    )

    with pytest.raises(MeshError, match="connectivity_dtype"):
        meshconv.convert_mesh(coords, connect, convention)


def test_convert_mesh_rejects_negative_node_indices() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((-1, 0, 1),))
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=ENodeOrder.RILEY,
    )

    with pytest.raises(MeshError, match="connectivity_indices"):
        meshconv.convert_mesh(coords, connect, convention)


def test_convert_mesh_rejects_out_of_bounds_node_indices() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 99),))
    convention = ConnectConvention(
        elem_type=EElemType.TRI3,
        elem_axis=EConnectAxis.ROW,
        index_base=0,
        node_order=ENodeOrder.RILEY,
    )

    with pytest.raises(MeshError, match="outside the coordinate array"):
        meshconv.convert_mesh(coords, connect, convention)


def test_verify_mesh_requires_mesh_geometry_instance() -> None:
    with pytest.raises(
        TypeError, match="mesh must be an instance of MeshGeometry"
    ):
        meshconv.verify_mesh("invalid_mesh")  # type: ignore[arg-type]


def test_verify_mesh_rejects_invalid_elem_type() -> None:
    mesh = MeshGeometry(
        elem_type="tri3",  # type: ignore[arg-type]
        coords=np.eye(3, dtype=np.float64),
        connect=np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="elem_type must be an EElemType"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_2d_or_wrong_dim_coords() -> None:
    # 1D coords
    mesh_1d = MeshGeometry(
        EElemType.TRI3,
        np.zeros(9, dtype=np.float64),
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="coordinate_shape"):
        meshconv.verify_mesh(mesh_1d)

    # 2D coords with 2 components instead of 3
    mesh_2d = MeshGeometry(
        EElemType.TRI3,
        np.zeros((3, 2), dtype=np.float64),
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="coordinate_shape"):
        meshconv.verify_mesh(mesh_2d)


def test_verify_mesh_rejects_empty_coords() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.zeros((0, 3), dtype=np.float64),
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="empty_coordinates"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_float64_coords() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float32),
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="coordinate_dtype"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_c_contiguous_coords() -> None:
    coords_fortran = np.asfortranarray(np.eye(3, dtype=np.float64))
    mesh = MeshGeometry(
        EElemType.TRI3,
        coords_fortran,
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="coordinate_layout"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_finite_coords() -> None:
    coords = np.eye(3, dtype=np.float64)
    coords[0, 0] = np.nan
    mesh = MeshGeometry(
        EElemType.TRI3,
        coords,
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="coordinate_values"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_2d_connect() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.array((0, 1, 2), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="connectivity_shape"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_empty_connect() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.zeros((0, 3), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="empty_connectivity"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_wrong_connect_width() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(4, dtype=np.float64),
        np.array(((0, 1, 2, 3),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="connectivity_width"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_uintp_connect() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.array(((0, 1, 2),), dtype=np.int32),
    )
    with pytest.raises(MeshError, match="connectivity_dtype"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_non_c_contiguous_connect() -> None:
    connect_f = np.asfortranarray(
        np.array(((0, 1, 2), (1, 2, 0)), dtype=np.uintp)
    )
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        connect_f,
    )
    with pytest.raises(MeshError, match="connectivity_layout"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_out_of_bounds_connect() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.array(((0, 1, 99),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="connectivity_indices"):
        meshconv.verify_mesh(mesh)


def test_verify_mesh_rejects_duplicate_node_ids() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.array(((0, 1, 1),), dtype=np.uintp),
    )
    with pytest.raises(MeshError, match="duplicate_node"):
        meshconv.verify_mesh(mesh)


def test_extract_surface_rejects_surface_mesh() -> None:
    mesh = MeshGeometry(
        EElemType.TRI3,
        np.eye(3, dtype=np.float64),
        np.array(((0, 1, 2),), dtype=np.uintp),
    )
    with pytest.raises(
        MeshError, match="extract_surface requires a volume mesh"
    ):
        meshconv.extract_surface(mesh)


def test_extract_surface_rejects_non_manifold_volume_faces() -> None:
    coords = np.zeros((6, 3), dtype=np.float64)
    # 3 tetrahedra sharing face (0, 1, 3)
    connect = np.array(
        ((0, 1, 2, 3), (0, 1, 4, 3), (0, 1, 5, 3)),
        dtype=np.uintp,
    )
    mesh = MeshGeometry(EElemType.TET4, coords, connect)

    with pytest.raises(MeshError, match="Non-manifold volume face"):
        meshconv.extract_surface(mesh)

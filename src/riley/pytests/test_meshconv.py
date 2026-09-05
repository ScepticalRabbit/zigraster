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

import netCDF4
import numpy as np
import pytest

from riley import data
from riley.python import meshconv, meshio
from riley.python.meshconv import (
    ConnectConvention,
    EConnectAxis,
    EdgeNode,
    EElemType,
    ENodeOrder,
    FaceNode,
    MeshError,
    MeshGeometry,
    UserTopology,
)

_CUBE_TYPES = {
    "tet4": meshconv.EElemType.TET4,
    "tet10": meshconv.EElemType.TET10,
    "hex8": meshconv.EElemType.HEX8,
    "hex20": meshconv.EElemType.HEX20,
    "hex27": meshconv.EElemType.HEX27,
}

_SPHERE_TYPES = {
    "tri3_sphere200": meshconv.EElemType.TRI3,
    "tri6_sphere200": meshconv.EElemType.TRI6,
    "quad4newton_sphere200": meshconv.EElemType.QUAD4,
    "quad8_sphere200": meshconv.EElemType.QUAD8,
    "quad9_sphere200": meshconv.EElemType.QUAD9,
}

_VTK_VOLUME_FACES = {
    meshconv.EElemType.TET4: (
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 2, 1),
    ),
    meshconv.EElemType.TET10: (
        (0, 1, 3, 4, 8, 7),
        (1, 2, 3, 5, 9, 8),
        (2, 0, 3, 6, 7, 9),
        (0, 2, 1, 6, 5, 4),
    ),
    meshconv.EElemType.HEX8: (
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
    ),
    meshconv.EElemType.HEX20: (
        (0, 4, 7, 3, 16, 15, 19, 11),
        (1, 2, 6, 5, 9, 18, 13, 17),
        (0, 1, 5, 4, 8, 17, 12, 16),
        (3, 7, 6, 2, 19, 14, 18, 10),
        (0, 3, 2, 1, 11, 10, 9, 8),
        (4, 5, 6, 7, 12, 13, 14, 15),
    ),
}


@pytest.mark.parametrize("elem_type", tuple(meshconv.EElemType))
@pytest.mark.parametrize("elem_axis", tuple(meshconv.EConnectAxis))
@pytest.mark.parametrize("index_base", (0, 1))
def test_vtk_adapter_handles_every_array_representation(
    elem_type: meshconv.EElemType,
    elem_axis: meshconv.EConnectAxis,
    index_base: int,
) -> None:
    coords = elem_type.get_para_coords()
    if coords.shape[1] == 2:
        coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    connect_std = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    connect_src = connect_std + index_base
    if elem_axis is meshconv.EConnectAxis.COLUMN:
        connect_src = connect_src.T
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=elem_axis,
        index_base=index_base,
        node_order=meshconv.ENodeOrder.VTK,
    )

    mesh = meshconv.convert_mesh(coords, connect_src, convention)

    meshconv.verify_mesh(mesh)
    assert mesh.elem_type is elem_type
    assert np.array_equal(mesh.connect, connect_std)
    assert mesh.connect.dtype == np.uintp
    assert mesh.connect.flags.c_contiguous


def test_exodus_hex20_adapter_reorders_edge_groups() -> None:
    elem_type = meshconv.EElemType.HEX20
    coords = elem_type.get_para_coords()
    connect_std = np.arange(20, dtype=np.int64)[None, :]
    riley_from_exodus = np.array((
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
    ))
    connect_src = np.empty_like(connect_std)
    connect_src[:, riley_from_exodus] = connect_std
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=meshconv.ENodeOrder.EXODUS,
    )

    mesh = meshconv.convert_mesh(coords, connect_src, convention)

    assert np.array_equal(mesh.connect, connect_std)


def test_exodus_hex27_adapter_reorders_edges_faces_and_centre() -> None:
    elem_type = meshconv.EElemType.HEX27
    coords = elem_type.get_para_coords()
    connect_std = np.arange(27, dtype=np.int64)[None, :]
    riley_from_exodus = np.array((
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
        23, 24, 25, 26, 21, 22, 20,
    ))
    connect_src = np.empty_like(connect_std)
    connect_src[:, riley_from_exodus] = connect_std
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=meshconv.ENodeOrder.EXODUS,
    )

    mesh = meshconv.convert_mesh(coords, connect_src, convention)

    assert np.array_equal(mesh.connect, connect_std)


@pytest.mark.parametrize(("case_name", "elem_type"), _CUBE_TYPES.items())
def test_moose_exodus_cube_converts_verifies_and_extracts(
    case_name: str,
    elem_type: meshconv.EElemType,
) -> None:
    with netCDF4.Dataset(data.cube_exodus_path(case_name)) as dataset:
        coords = np.column_stack((
            dataset.variables["coordx"][:],
            dataset.variables["coordy"][:],
            dataset.variables["coordz"][:],
        ))
        connect = np.asarray(dataset.variables["connect1"][:])
        node_count = len(dataset.dimensions["num_nodes"])
        for variable_idx in (1, 2, 3, 10):
            assert dataset.variables[
                f"vals_nod_var{variable_idx}"
            ].shape == (21, node_count)

    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=1,
        node_order=meshconv.ENodeOrder.EXODUS,
    )

    volume = meshconv.convert_mesh(coords, connect, convention)
    surface = meshconv.extract_surface(volume)

    meshconv.verify_mesh(volume)
    meshconv.verify_mesh(surface)
    assert volume.connect.shape == connect.shape
    assert surface.coords.shape[0] < volume.coords.shape[0]


@pytest.mark.parametrize(
    "case_name",
    ("tet10", "hex20", "hex27"),
)
def test_moose_exodus_high_order_nodes_have_standard_roles(
    case_name: str,
) -> None:
    elem_type = _CUBE_TYPES[case_name]
    with netCDF4.Dataset(data.cube_exodus_path(case_name)) as dataset:
        coords = np.column_stack((
            dataset.variables["coordx"][:],
            dataset.variables["coordy"][:],
            dataset.variables["coordz"][:],
        ))
        connect = np.asarray(dataset.variables["connect1"][:])

    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=1,
        node_order=meshconv.ENodeOrder.EXODUS,
    )
    volume = meshconv.convert_mesh(coords, connect, convention)
    if elem_type is meshconv.EElemType.TET10:
        edges = (
            (0, 1), (1, 2), (2, 0),
            (0, 3), (1, 3), (2, 3),
        )
        corner_count = 4
    else:
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        corner_count = 8

    for elem_connect in volume.connect:
        elem_coords = volume.coords[elem_connect]
        for edge_slot, edge_corners in enumerate(edges, corner_count):
            expected = np.mean(elem_coords[list(edge_corners)], axis=0)
            np.testing.assert_allclose(elem_coords[edge_slot], expected)

        if elem_type is meshconv.EElemType.HEX27:
            faces = (
                (0, 4, 7, 3), (1, 2, 6, 5),
                (0, 1, 5, 4), (3, 7, 6, 2),
                (0, 1, 2, 3), (4, 5, 6, 7),
            )
            for face_slot, face_corners in enumerate(faces, 20):
                expected = np.mean(elem_coords[list(face_corners)], axis=0)
                np.testing.assert_allclose(elem_coords[face_slot], expected)
            np.testing.assert_allclose(
                elem_coords[26],
                np.mean(elem_coords[:8], axis=0),
            )


def test_user_topology_describes_source_relationships() -> None:
    elem_type = meshconv.EElemType.QUAD8
    coords = elem_type.get_para_coords()
    coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    topology = meshconv.UserTopology(
        corner_slots=(2, 3, 0, 1),
        edge_nodes=(
            meshconv.EdgeNode((2, 3), 6),
            meshconv.EdgeNode((3, 0), 7),
            meshconv.EdgeNode((0, 1), 4),
            meshconv.EdgeNode((1, 2), 5),
        ),
    )
    source_perm = np.array((2, 3, 0, 1, 6, 7, 4, 5))
    connect_std = np.arange(8, dtype=np.int64)[None, :]
    connect_src = np.empty_like(connect_std)
    connect_src[:, source_perm] = connect_std
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=topology,
    )

    mesh = meshconv.convert_mesh(coords, connect_src, convention)

    assert np.array_equal(mesh.connect, connect_std)


def test_verify_mesh_collects_independent_failures() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, np.nan, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    connect = np.array(((0, 1, 2),), dtype=np.int32)
    mesh = meshconv.MeshGeometry(
        elem_type=meshconv.EElemType.TRI3,
        coords=coords,
        connect=connect,
    )

    with pytest.raises(meshconv.MeshError) as error_info:
        meshconv.verify_mesh(mesh)

    codes = {issue.code for issue in error_info.value.issues}
    assert codes == {
        "coordinate_dtype",
        "coordinate_values",
        "connectivity_dtype",
    }


def test_verify_mesh_accepts_collapsed_surface_face() -> None:
    coords = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    mesh = meshconv.MeshGeometry(
        meshconv.EElemType.TRI3,
        coords,
        np.array(((0, 1, 2),), dtype=np.uintp),
    )

    meshconv.verify_mesh(mesh)


def test_verify_mesh_accepts_coincident_uv_seam_faces() -> None:
    coords = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    mesh = meshconv.MeshGeometry(
        meshconv.EElemType.TRI3,
        coords,
        np.array(((0, 1, 2), (3, 4, 5)), dtype=np.uintp),
    )

    meshconv.verify_mesh(mesh)


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


def test_extract_surface_returns_standard_compact_hex27_surface() -> None:
    elem_type = meshconv.EElemType.HEX27
    coords = elem_type.get_para_coords()
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=meshconv.ENodeOrder.VTK,
    )
    volume = meshconv.convert_mesh(
        coords,
        np.arange(27, dtype=np.int64)[None, :],
        convention,
    )

    surface = meshconv.extract_surface(volume)

    assert surface.elem_type is meshconv.EElemType.QUAD9
    assert surface.coords.shape == (26, 3)
    assert surface.connect.shape == (6, 9)
    assert np.array_equal(
        np.unique(surface.connect),
        np.arange(26, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        surface.connect,
        (
            (0, 4, 7, 3, 16, 15, 19, 11, 20),
            (1, 2, 6, 5, 9, 18, 13, 17, 21),
            (0, 1, 5, 4, 8, 17, 12, 16, 22),
            (3, 7, 6, 2, 19, 14, 18, 10, 23),
            (0, 3, 2, 1, 11, 10, 9, 8, 24),
            (4, 5, 6, 7, 12, 13, 14, 15, 25),
        ),
    )
    np.testing.assert_array_equal(
        volume.coords[20:26],
        (
            (0.0, 0.5, 0.5),
            (1.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 1.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 1.0),
        ),
    )


@pytest.mark.parametrize(
    ("elem_type", "expected_faces"),
    _VTK_VOLUME_FACES.items(),
)
def test_extract_surface_uses_vtk_face_order(
    elem_type: meshconv.EElemType,
    expected_faces: tuple[tuple[int, ...], ...],
) -> None:
    coords = elem_type.get_para_coords()
    volume = meshconv.MeshGeometry(
        elem_type,
        coords,
        np.arange(coords.shape[0], dtype=np.uintp)[None, :],
    )

    surface = meshconv.extract_surface(volume)

    np.testing.assert_array_equal(surface.connect, expected_faces)


@pytest.mark.parametrize(("case_name", "elem_type"), _CUBE_TYPES.items())
def test_packaged_cube_converts_verifies_and_extracts(
    case_name: str,
    elem_type: meshconv.EElemType,
) -> None:
    case_path = data.cube_case_path(case_name)
    coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
    connect = np.loadtxt(
        case_path / "connectivity.csv",
        delimiter=",",
        dtype=np.int64,
        ndmin=2,
    )
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=meshconv.ENodeOrder.RILEY,
    )

    volume = meshconv.convert_mesh(coords, connect, convention)
    surface = meshconv.extract_surface(volume)

    meshconv.verify_mesh(volume)
    meshconv.verify_mesh(surface)


@pytest.mark.parametrize(("case_name", "elem_type"), _SPHERE_TYPES.items())
def test_packaged_sphere_converts_and_verifies(
    case_name: str,
    elem_type: meshconv.EElemType,
) -> None:
    case_path = data.sphere200_case_path(case_name)
    coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
    connect_path = case_path / "connectivity.csv"
    if not connect_path.is_file():
        connect_path = case_path / "connect.csv"
    connect = np.loadtxt(
        connect_path,
        delimiter=",",
        dtype=np.int64,
        ndmin=2,
    )
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
        node_order=meshconv.ENodeOrder.RILEY,
    )

    mesh = meshconv.convert_mesh(coords, connect, convention)

    meshconv.verify_mesh(mesh)


def test_packaged_square_donut_converts_and_verifies() -> None:
    case_path = data.platehole_csv_case_path()
    coords = np.loadtxt(case_path / "coords.csv", delimiter=",")
    connect = meshio.load_csv(case_path / "connect.csv", dtype=np.int64)
    convention = meshconv.ConnectConvention(
        meshconv.EElemType.QUAD8, meshconv.EConnectAxis.ROW, 0,
        meshconv.ENodeOrder.RILEY,
    )

    mesh = meshconv.convert_mesh(coords, connect, convention)

    meshconv.verify_mesh(mesh)
# ==========================================================================
# 1. ConnectConvention and UserTopology Construction Errors
# ==========================================================================


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


# ==========================================================================
# 2. convert_mesh Input Errors
# ==========================================================================


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


# ==========================================================================
# 3. verify_mesh Errors
# ==========================================================================


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


# ==========================================================================
# 4. extract_surface Errors
# ==========================================================================


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


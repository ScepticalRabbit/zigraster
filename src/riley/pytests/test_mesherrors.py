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
import pytest

import riley
from riley.cython.riley import (
    FunctionShader,
    FuncShaderBuiltin,
    MeshType,
    NodalShader,
    TextureShader,
)
from riley.python import exodusio, meshconv, meshio
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


# ==========================================================================
# 5. load_csv Errors
# ==========================================================================


def test_load_csv_rejects_negative_skip_rows() -> None:
    with pytest.raises(ValueError, match="skip_rows must be non-negative"):
        meshio.load_csv("some_file.csv", skip_rows=-1)


def test_load_csv_rejects_non_existent_file() -> None:
    with pytest.raises(OSError):
        meshio.load_csv("non_existent_file_12345.csv")


def test_load_csv_rejects_fractional_for_integer_dtype(tmp_path: Path) -> None:
    frac_csv = tmp_path / "fractional.csv"
    frac_csv.write_text("1.0, 2.5, 3.0\n")

    with pytest.raises(ValueError, match="contains non-integer values"):
        meshio.load_csv(frac_csv, dtype=np.int64)


def test_load_csv_rejects_non_finite_values(tmp_path: Path) -> None:
    nan_csv = tmp_path / "nan.csv"
    nan_csv.write_text("1.0, NaN, 3.0\n")

    with pytest.raises(ValueError, match="contains non-finite values"):
        meshio.load_csv(nan_csv)


# ==========================================================================
# 6. create_mesh and Shader/Disp/UV Preparation Errors
# ==========================================================================


def test_create_mesh_rejects_invalid_convention_arg() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.int64)
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    with pytest.raises(
        TypeError, match="convention must be a ConnectConvention"
    ):
        meshio.create_mesh(
            convention="not_a_convention",  # type: ignore[arg-type]
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_rejects_invalid_mesh_type_arg() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.int64)
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    with pytest.raises(
        TypeError, match="mesh_type must be a MeshType member"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type="tri3",  # type: ignore[arg-type]
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_rejects_invalid_shader_arg() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.array(((0, 1, 2),), dtype=np.int64)
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )

    with pytest.raises(
        TypeError, match="shader must be a Riley supported shader"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader="flat_shader",  # type: ignore[arg-type]
            coords=coords,
            connect=connect,
        )


def test_create_mesh_rejects_incompatible_element_family() -> None:
    coords = np.zeros((4, 3), dtype=np.float64)
    connect = np.arange(4, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.QUAD4, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 4, 1), dtype=np.float64))

    with pytest.raises(MeshError, match="Cannot change quad4 into tri6"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri6,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_rejects_elevating_element_order() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    with pytest.raises(MeshError, match="Cannot convert tri3 to tri6"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri6,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_disp_rejects_non_tuple_or_wrong_length() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    # disp is a list instead of a tuple
    with pytest.raises(TypeError, match="disp must be a tuple"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=[
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                np.zeros((3, 1)),
            ],  # type: ignore[arg-type]
        )

    # disp tuple length 2 instead of 3
    with pytest.raises(TypeError, match="disp must be a tuple"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=(np.zeros((3, 1)), np.zeros((3, 1))),  # type: ignore[arg-type]
        )


def test_create_mesh_disp_rejects_wrong_shape_component() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    # disp_x has 4 nodes instead of 3
    disp = (
        np.zeros((4, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="disp_x must have shape"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=disp,
        )


def test_create_mesh_disp_rejects_non_float_component() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    disp = (
        np.zeros((3, 1), dtype=np.int32),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
    )
    with pytest.raises(
        TypeError, match="disp_x must have a floating-point dtype"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=disp,
        )


def test_create_mesh_disp_rejects_non_finite_component() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    disp_x = np.zeros((3, 1), dtype=np.float64)
    disp_x[0, 0] = np.nan
    disp = (
        disp_x,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
    )
    with pytest.raises(
        ValueError, match="disp_x must contain only finite values"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=disp,
        )


def test_create_mesh_disp_rejects_mismatched_component_shapes() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((1, 3, 1), dtype=np.float64))

    disp = (
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 2), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
    )
    with pytest.raises(
        ValueError, match="Displacement components must have equal shapes"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
            disp=disp,
        )


def test_create_mesh_uvs_rejects_wrong_shape() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.zeros((1, 10, 10), dtype=np.uint8)
    shader = TextureShader(
        texture=texture,
        uvs=np.zeros((3, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="uvs must have shape"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_uvs_rejects_non_float_dtype() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.zeros((1, 10, 10), dtype=np.uint8)
    shader = TextureShader(
        texture=texture,
        uvs=np.zeros((3, 2), dtype=np.int32),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="uvs must have a floating-point dtype"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_uvs_rejects_non_finite_values() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.zeros((1, 10, 10), dtype=np.uint8)
    uvs = np.zeros((3, 2), dtype=np.float64)
    uvs[0, 0] = np.nan
    shader = TextureShader(texture=texture, uvs=uvs)

    with pytest.raises(ValueError, match="uvs must contain only finite values"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_nodal_shader_rejects_wrong_node_count() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((4, 1, 1), dtype=np.float64))

    with pytest.raises(ValueError, match="nodal field must have shape"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_nodal_shader_rejects_invalid_field_count() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    # 2 fields instead of 1 or 3
    shader = NodalShader(np.ones((3, 1, 2), dtype=np.float64))

    with pytest.raises(
        ValueError, match="nodal field must contain one or three fields"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_nodal_shader_rejects_non_float_dtype() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = NodalShader(np.ones((3, 1, 1), dtype=np.int32))

    with pytest.raises(
        TypeError, match="nodal field must have a floating-point dtype"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_nodal_shader_rejects_non_finite_values() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    field = np.ones((3, 1, 1), dtype=np.float64)
    field[0, 0, 0] = np.nan
    shader = NodalShader(field)

    with pytest.raises(
        ValueError, match="nodal field must contain only finite values"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_texture_shader_requires_uvs() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.zeros((1, 10, 10), dtype=np.uint8)
    shader = TextureShader(texture=texture, uvs=None)

    with pytest.raises(ValueError, match="uvs must have shape"):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_texture_shader_rejects_invalid_channels() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    # 4 channels instead of 1 or 3
    texture = np.zeros((4, 10, 10), dtype=np.uint8)
    uvs = np.zeros((3, 2), dtype=np.float64)
    shader = TextureShader(texture=texture, uvs=uvs)

    with pytest.raises(
        ValueError,
        match="texture must be a supported dtype with 1 or 3 channels",
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_texture_shader_rejects_invalid_dtype() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.zeros((1, 10, 10), dtype=np.int32)
    uvs = np.zeros((3, 2), dtype=np.float64)
    shader = TextureShader(texture=texture, uvs=uvs)

    with pytest.raises(
        ValueError,
        match="texture must be a supported dtype with 1 or 3 channels",
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_texture_shader_rejects_non_finite_floats() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    texture = np.full((1, 10, 10), np.nan, dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float64)
    shader = TextureShader(texture=texture, uvs=uvs)

    with pytest.raises(
        ValueError, match="floating texture must contain finite values"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_create_mesh_function_shader_rejects_invalid_channels() -> None:
    coords = np.eye(3, dtype=np.float64)
    connect = np.arange(3, dtype=np.int64)[None, :]
    convention = ConnectConvention(
        EElemType.TRI3, EConnectAxis.ROW, 0, ENodeOrder.RILEY,
    )
    shader = FunctionShader(
        builtin=FuncShaderBuiltin.constant,
        channels=2,  # invalid channels (must be 1 or 3)
    )

    with pytest.raises(
        ValueError, match="FunctionShader.channels must be one or three"
    ):
        meshio.create_mesh(
            convention=convention,
            mesh_type=MeshType.tri3,
            shader=shader,
            coords=coords,
            connect=connect,
        )


def test_prepare_shader_rejects_unsupported_shader_type() -> None:
    with pytest.raises(TypeError, match="shader must be a Riley shader object"):
        meshio._prepare_shader(
            "not_a_shader",  # type: ignore[arg-type]
            3,
            np.arange(3, dtype=np.uintp),
        )


# ==========================================================================
# 11. Exodus Reader and Parser Errors
# ==========================================================================


def test_parse_exodus_elem_type_rejects_non_positive_node_count() -> None:
    with pytest.raises(ValueError, match="Invalid node count"):
        exodusio.parse_exodus_elem_type("HEX8", 0)
    with pytest.raises(ValueError, match="Invalid node count"):
        exodusio.parse_exodus_elem_type("HEX8", -5)


def test_parse_exodus_elem_type_rejects_unknown_elem_string() -> None:
    with pytest.raises(
        MeshError, match="Cannot determine EElemType for Exodus element"
    ):
        exodusio.parse_exodus_elem_type("UNKNOWN_ELEM", 8)


def test_parse_exodus_elem_type_rejects_node_count_mismatch() -> None:
    with pytest.raises(
        MeshError, match="Cannot determine EElemType for Exodus element"
    ):
        exodusio.parse_exodus_elem_type("HEX20", 8)


def test_parse_exodus_elem_type_rejects_ambiguous_none_string() -> None:
    with pytest.raises(
        MeshError, match="Cannot determine EElemType for Exodus element"
    ):
        exodusio.parse_exodus_elem_type(None, 8)
    with pytest.raises(
        MeshError, match="Cannot determine EElemType for Exodus element"
    ):
        exodusio.parse_exodus_elem_type(None, 4)


def test_load_exodus_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Exodus file not found"):
        exodusio.load_exodus("non_existent_file.e")


def test_load_exodus_rejects_missing_connectivity_key() -> None:
    path = riley.data.platehole_exodus_path()
    with pytest.raises(
        KeyError, match="Connectivity table 'connect99' not found"
    ):
        exodusio.load_exodus(path, connect_keys=("connect99",))


def test_load_exodus_rejects_missing_nodal_key() -> None:
    path = riley.data.platehole_exodus_path()
    with pytest.raises(
        KeyError, match="Nodal variable 'unknown_var' not found"
    ):
        exodusio.load_exodus(path, nodal_keys=("unknown_var",))


def test_load_exodus_rejects_invalid_disp_keys_length() -> None:
    path = riley.data.platehole_exodus_path()
    with pytest.raises(ValueError, match="disp_keys must contain 2 or 3 keys"):
        exodusio.load_exodus(path, disp_keys=("disp_x",))
    with pytest.raises(ValueError, match="disp_keys must contain 2 or 3 keys"):
        exodusio.load_exodus(
            path, disp_keys=("disp_x", "disp_y", "disp_z", "disp_w")
        )


def test_load_exodus_rejects_missing_custom_disp_keys() -> None:
    path = riley.data.platehole_exodus_path()
    with pytest.raises(KeyError, match="Displacement variable.*not found"):
        exodusio.load_exodus(path, disp_keys=("u_x", "u_y", "u_z"))



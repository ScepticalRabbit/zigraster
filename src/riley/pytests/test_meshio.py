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
from riley.pytests.common import coords_3d, function_shader, save_csv


_SURF_CASES = (
    (riley.EElemType.TRI3, riley.MeshType.tri3),
    (riley.EElemType.TRI3, riley.MeshType.tri3opt),
    (riley.EElemType.TRI6, riley.MeshType.tri6),
    (riley.EElemType.QUAD4, riley.MeshType.quad4ibi),
    (riley.EElemType.QUAD4, riley.MeshType.quad4newton),
    (riley.EElemType.QUAD8, riley.MeshType.quad8),
    (riley.EElemType.QUAD9, riley.MeshType.quad9),
)


@pytest.mark.parametrize("shape", ("cube", "cylinder", "platewithhole"))
@pytest.mark.parametrize(
    "elem_type",
    ("tet4", "tet10", "hex8", "hex20", "hex27"),
)
def test_packaged_shape_data_paths_exist(
    shape: str, elem_type: str
) -> None:
    coords_path = riley.data.shape_coords_path(shape, elem_type)
    connect_path = riley.data.shape_connectivity_path(shape, elem_type)
    exodus_path = riley.data.shape_exodus_path(shape, elem_type)
    disp_x_path = riley.data.shape_disp_path(shape, elem_type, "x")
    disp_y_path = riley.data.shape_disp_path(shape, elem_type, "y")
    disp_z_path = riley.data.shape_disp_path(shape, elem_type, "z")
    temp_path = riley.data.shape_temperature_path(shape, elem_type)

    assert coords_path.is_file()
    assert connect_path.is_file()
    assert exodus_path.is_file()
    assert disp_x_path.is_file()
    assert disp_y_path.is_file()
    assert disp_z_path.is_file()
    assert temp_path.is_file()

    coords = np.loadtxt(coords_path, delimiter=",")
    disp_x = np.loadtxt(disp_x_path, delimiter=",")
    disp_y = np.loadtxt(disp_y_path, delimiter=",")
    disp_z = np.loadtxt(disp_z_path, delimiter=",")
    temp = np.loadtxt(temp_path, delimiter=",")

    num_nodes = coords.shape[0]
    assert disp_x.shape == (num_nodes, 5)
    assert disp_y.shape == (num_nodes, 5)
    assert disp_z.shape == (num_nodes, 5)
    assert temp.shape == (num_nodes, 5)


@pytest.mark.parametrize(
    "case_name",
    ("tet4", "tet10", "hex8", "hex20", "hex27"),
)
def test_packaged_pure_cube_data_paths_exist(case_name: str) -> None:
    coords_path = riley.data.cube_coords_path(case_name, pure=True)
    connect_path = riley.data.cube_connectivity_path(case_name, pure=True)
    exodus_path = riley.data.cube_exodus_path(case_name, pure=True)
    disp_x_path = riley.data.cube_disp_path(case_name, "x", pure=True)
    disp_y_path = riley.data.cube_disp_path(case_name, "y", pure=True)
    disp_z_path = riley.data.cube_disp_path(case_name, "z", pure=True)
    temp_path = riley.data.cube_temperature_path(case_name, pure=True)

    assert coords_path.is_file()
    assert connect_path.is_file()
    assert exodus_path.is_file()
    assert disp_x_path.is_file()
    assert disp_y_path.is_file()
    assert disp_z_path.is_file()
    assert temp_path.is_file()

    coords = np.loadtxt(coords_path, delimiter=",")
    disp_x = np.loadtxt(disp_x_path, delimiter=",")
    temp = np.loadtxt(temp_path, delimiter=",")
    num_nodes = coords.shape[0]
    assert disp_x.shape == (num_nodes, 5)
    assert temp.shape == (num_nodes, 5)


@pytest.mark.parametrize(
    "case_name",
    (
        "tri3_sphere200",
        "tri6_sphere200",
        "quad4newton_sphere200",
        "quad8_sphere200",
        "quad9_sphere200",
    ),
)
def test_packaged_sphere_data_paths_exist(case_name: str) -> None:
    case_path = riley.data.sphere200_case_path(case_name)
    for file_name in ("coords.csv", "connect.csv", "field.csv", "uvs.csv"):
        assert (case_path / file_name).is_file()


def test_other_packaged_data_paths_exist() -> None:
    assert riley.data.speckle_texture_path().is_file()
    assert riley.data.cal_target_texture_path().is_file()
    assert riley.data.platehole_csv_case_path().is_dir()
    assert riley.data.platehole_exodus_path().is_file()
    assert riley.data.stereocal_case_path().is_dir()
    assert riley.data.rabbit_case_path("riley", "tri3").is_dir()


def test_load_csv_preserves_table_orientation(tmp_path: Path) -> None:
    table = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    save_csv(tmp_path / "table.csv", table)

    loaded = riley.load_csv(tmp_path / "table.csv")

    assert loaded.dtype == np.float64
    assert loaded.flags.c_contiguous
    np.testing.assert_allclose(loaded, table)


def test_load_csv_preserves_integer_indices(tmp_path: Path) -> None:
    connect = np.array(((1, 2, 3), (3, 4, 1)), dtype=np.int64)
    np.savetxt(tmp_path / "connect.csv", connect, delimiter=",", fmt="%d")

    loaded = riley.load_csv(tmp_path / "connect.csv", dtype=np.int64)

    assert loaded.dtype == np.int64
    np.testing.assert_array_equal(loaded, connect)


@pytest.mark.parametrize(("elem_type", "mesh_type"), _SURF_CASES)
@pytest.mark.parametrize("axis", tuple(riley.EConnectAxis))
@pytest.mark.parametrize("index_base", (0, 1))
def test_create_mesh_identity_representations(
    elem_type: riley.EElemType,
    mesh_type: riley.MeshType,
    axis: riley.EConnectAxis,
    index_base: int,
) -> None:
    coords = coords_3d(elem_type)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :] + index_base
    if axis is riley.EConnectAxis.COLUMN:
        connect = connect.T
    convention = riley.ConnectConvention(
        elem_type, axis, index_base, riley.ENodeOrder.RILEY
    )
    mesh = riley.create_mesh(
        convention, mesh_type, coords, connect, shader=function_shader()
    )
    assert mesh.mesh_type is mesh_type
    assert mesh.coords.dtype == np.float64
    assert mesh.connect.dtype == np.uintp
    assert mesh.coords.flags.c_contiguous
    assert mesh.connect.flags.c_contiguous


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (riley.EElemType.TRI7, riley.MeshType.tri6),
        (riley.EElemType.TRI7, riley.MeshType.tri3),
        (riley.EElemType.TRI6, riley.MeshType.tri3),
        (riley.EElemType.QUAD4, riley.MeshType.tri3),
        (riley.EElemType.QUAD8, riley.MeshType.tri3),
        (riley.EElemType.QUAD9, riley.MeshType.tri3),
        (riley.EElemType.QUAD9, riley.MeshType.quad8),
        (riley.EElemType.QUAD9, riley.MeshType.quad4newton),
        (riley.EElemType.QUAD8, riley.MeshType.quad4ibi),
        (riley.EElemType.TET4, riley.MeshType.tri3),
        (riley.EElemType.TET10, riley.MeshType.tri6),
        (riley.EElemType.TET10, riley.MeshType.tri3),
        (riley.EElemType.HEX8, riley.MeshType.tri3),
        (riley.EElemType.HEX8, riley.MeshType.quad4newton),
        (riley.EElemType.HEX20, riley.MeshType.tri3),
        (riley.EElemType.HEX20, riley.MeshType.quad8),
        (riley.EElemType.HEX20, riley.MeshType.quad4ibi),
        (riley.EElemType.HEX27, riley.MeshType.tri3),
        (riley.EElemType.HEX27, riley.MeshType.quad9),
        (riley.EElemType.HEX27, riley.MeshType.quad8),
        (riley.EElemType.HEX27, riley.MeshType.quad4newton),
    ),
)
def test_create_mesh_supported_topology_transitions(source, target) -> None:
    coords = coords_3d(source)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        target, coords, connect, shader=function_shader(),
    )
    assert mesh.connect.shape[1] in (3, 4, 6, 8, 9)


@pytest.mark.parametrize(
    ("source", "expected_tri_count"),
    (
        (riley.EElemType.TRI3, 1),
        (riley.EElemType.TRI6, 4),
        (riley.EElemType.TRI7, 6),
        (riley.EElemType.QUAD4, 2),
        (riley.EElemType.QUAD8, 6),
        (riley.EElemType.QUAD9, 8),
        (riley.EElemType.TET4, 4),
        (riley.EElemType.TET10, 16),
        (riley.EElemType.HEX8, 12),
        (riley.EElemType.HEX20, 36),
        (riley.EElemType.HEX27, 48),
    ),
)
def test_create_mesh_triangulation_elem_counts(
    source: riley.EElemType,
    expected_tri_count: int,
) -> None:
    coords = coords_3d(source)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        riley.MeshType.tri3, coords, connect, shader=function_shader(),
    )
    assert mesh.mesh_type is riley.MeshType.tri3
    assert mesh.connect.shape == (expected_tri_count, 3)
    is_hex27 = source is riley.EElemType.HEX27
    expected_nodes = 26 if is_hex27 else coords.shape[0]
    assert mesh.coords.shape[0] == expected_nodes


def test_create_mesh_remaps_every_nodal_array_by_source_index() -> None:
    source = riley.EElemType.HEX27
    coords = source.get_para_coords()
    ids = np.arange(coords.shape[0], dtype=np.float64)
    uvs = np.column_stack((ids, -ids))
    texture = np.zeros((1, 2, 2), dtype=np.uint8)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        riley.MeshType.quad4newton,
        coords,
        np.arange(27, dtype=np.int64)[None, :],
        disp=(ids[:, None], (ids + 100)[:, None], (ids + 200)[:, None]),
        shader=riley.TextureShader(uvs, texture),
    )
    retained = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(mesh.shader.uvs[:, 0], retained)
    np.testing.assert_array_equal(mesh.disp[0, :, 0], retained)
    np.testing.assert_array_equal(mesh.disp[0, :, 1], retained + 100)


@pytest.mark.parametrize("channels", (1, 3))
def test_create_mesh_prepares_nodal_shader(channels: int) -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    field = np.ones((3, 2, channels), dtype=np.float32)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElemType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=riley.NodalShader(field),
    )
    assert mesh.shader.field.shape == (2, 3, channels)
    assert mesh.shader.field.dtype == np.float64


def test_create_mesh_result_crosses_cython_mesh_boundary() -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElemType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=function_shader(),
    )
    center = riley.roi_cent_over_meshes([mesh])
    np.testing.assert_allclose(center, (0.5, 0.5, 0))


@pytest.mark.parametrize(
    ("case_name", "elem_type", "mesh_type"),
    (
        ("tet4", riley.EElemType.TET4, riley.MeshType.tri3),
        ("tet10", riley.EElemType.TET10, riley.MeshType.tri6),
        ("hex8", riley.EElemType.HEX8, riley.MeshType.quad4newton),
        ("hex20", riley.EElemType.HEX20, riley.MeshType.quad8),
        ("hex27", riley.EElemType.HEX27, riley.MeshType.quad9),
    ),
)
def test_create_mesh_reuses_packaged_cube_fixtures(
    case_name: str,
    elem_type: riley.EElemType,
    mesh_type: riley.MeshType,
) -> None:
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            elem_type, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        mesh_type,
        riley.load_csv(riley.data.cube_coords_path(case_name)),
        riley.load_csv(
            riley.data.cube_connectivity_path(case_name), dtype=np.int64
        ),
        shader=function_shader(),
    )
    assert mesh.connect.shape[0] > 0
    riley.roi_cent_over_meshes([mesh])


@pytest.mark.parametrize("channels", (1, 3))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float64))
def test_create_mesh_supports_all_texture_storage_types(
    channels: int,
    dtype: type,
) -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    texture = np.ones((channels, 2, 2), dtype=dtype)
    shader = riley.TextureShader(coords[:, :2], texture)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElemType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=shader,
    )
    riley.roi_cent_over_meshes([mesh])


@pytest.mark.parametrize("builtin", tuple(riley.FuncShaderBuiltin))
@pytest.mark.parametrize("channels", (1, 3))
def test_create_mesh_supports_every_function_shader(builtin, channels) -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElemType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=riley.FunctionShader(builtin, channels=channels),
    )
    riley.roi_cent_over_meshes([mesh])

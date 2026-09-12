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
    (riley.EElemType.QUAD4, riley.MeshType.quad4),
    (riley.EElemType.QUAD8, riley.MeshType.quad8),
    (riley.EElemType.QUAD9, riley.MeshType.quad9),
)


@pytest.mark.parametrize(
    "shape", ("cube", "cylinder", "sphere", "platewithhole")
)
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

    msh_path = riley.data.shape_msh_path(shape, elem_type)
    geo_path = riley.data.shape_geo_path(shape, elem_type)
    moose_path = riley.data.shape_moose_input_path(shape, elem_type)

    assert coords_path.is_file()
    assert connect_path.is_file()
    assert exodus_path.is_file()
    assert disp_x_path.is_file()
    assert disp_y_path.is_file()
    assert disp_z_path.is_file()
    assert temp_path.is_file()
    assert msh_path.is_file()
    assert geo_path.is_file()
    assert moose_path.is_file()

    coords = np.loadtxt(coords_path, delimiter=",")
    disp_x = np.loadtxt(disp_x_path, delimiter=",")
    disp_y = np.loadtxt(disp_y_path, delimiter=",")
    disp_z = np.loadtxt(disp_z_path, delimiter=",")
    temp = np.loadtxt(temp_path, delimiter=",")

    num_nodes = coords.shape[0]
    assert disp_x.shape == (num_nodes, 4)
    assert disp_y.shape == (num_nodes, 4)
    assert disp_z.shape == (num_nodes, 4)
    assert temp.shape == (num_nodes, 4)


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
    assert disp_x.shape == (num_nodes, 4)
    assert temp.shape == (num_nodes, 4)


@pytest.mark.parametrize(
    "shape", ("cube", "cylinder", "sphere", "platewithhole")
)
@pytest.mark.parametrize(
    "surf_type",
    ("tri3", "tri6", "quad4", "quad8", "quad9"),
)
def test_packaged_surface_data_paths_exist(
    shape: str, surf_type: str
) -> None:
    surf_sub = riley.data.shape_surface_dataset_path(shape, surf_type)
    assert surf_sub.is_dir()
    for name in (
        "coords.csv",
        "connect.csv",
        "uvs.csv",
        "temperature.csv",
        "disp_x.csv",
        "disp_y.csv",
        "disp_z.csv",
    ):
        assert (surf_sub / name).is_file()


@pytest.mark.parametrize(
    "elem_type",
    ("quad4", "quad8", "quad9", "tri3", "tri6"),
)
def test_packaged_platewithhole2d_data_paths_exist(
    elem_type: str,
) -> None:
    coords_path = riley.data.platewithhole2d_coords_path(elem_type)
    connect_path = riley.data.platewithhole2d_connectivity_path(elem_type)
    exodus_path = riley.data.platewithhole2d_exodus_path(elem_type)
    disp_x_path = riley.data.platewithhole2d_disp_path(elem_type, "x")
    disp_y_path = riley.data.platewithhole2d_disp_path(elem_type, "y")
    disp_z_path = riley.data.platewithhole2d_disp_path(elem_type, "z")
    temp_path = riley.data.platewithhole2d_temperature_path(elem_type)

    msh_path = riley.data.platewithhole2d_msh_path(elem_type)
    geo_path = riley.data.platewithhole2d_geo_path(elem_type)
    moose_path = riley.data.platewithhole2d_moose_input_path(elem_type)

    assert coords_path.is_file()
    assert connect_path.is_file()
    assert exodus_path.is_file()
    assert disp_x_path.is_file()
    assert disp_y_path.is_file()
    assert disp_z_path.is_file()
    assert temp_path.is_file()
    assert msh_path.is_file()
    assert geo_path.is_file()
    assert moose_path.is_file()

    coords = np.loadtxt(coords_path, delimiter=",")
    disp_x = np.loadtxt(disp_x_path, delimiter=",")
    disp_y = np.loadtxt(disp_y_path, delimiter=",")
    disp_z = np.loadtxt(disp_z_path, delimiter=",")
    temp = np.loadtxt(temp_path, delimiter=",")

    num_nodes = coords.shape[0]
    assert disp_x.shape == (num_nodes, 4)
    assert disp_y.shape == (num_nodes, 4)
    assert disp_z.shape == (num_nodes, 4)
    assert np.all(disp_z == 0.0)
    assert temp.shape == (num_nodes, 4)


@pytest.mark.parametrize(
    "case_name",
    (
        "tri3_sphere200",
        "tri6_sphere200",
        "quad4_sphere200",
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


def test_remap_nodal_data_reorders_rows() -> None:
    source_values = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ]
    )
    geom = riley.MeshGeometry(
        elem_type=riley.EElemType.TRI3,
        coords=source_values.copy(),
        connect=np.array([[0, 1, 2]], dtype=np.uintp),
    )
    conv = riley.MeshConversion(
        mesh_type=riley.MeshType.tri3,
        geometry=geom,
        source_node_indices=np.array([2, 0, 1], dtype=np.uintp),
        source_node_count=3,
    )
    remapped = riley.remap_nodal_data(conv, source_values)

    expected = np.array(
        [
            [30.0, 31.0],
            [10.0, 11.0],
            [20.0, 21.0],
        ]
    )
    np.testing.assert_array_equal(remapped, expected)


def test_load_csv_loads_floats_and_integers(tmp_path: Path) -> None:
    float_path = tmp_path / "floats.csv"
    save_csv(float_path, np.array([[1.5, 2.5], [3.5, 4.5]]))
    int_path = tmp_path / "ints.csv"
    save_csv(int_path, np.array([[1.0, 2.0], [3.0, 4.0]]))

    loaded_floats = riley.load_csv(float_path)
    loaded_ints = riley.load_csv(int_path, dtype=np.int64)

    assert loaded_floats.dtype == np.float64
    assert loaded_ints.dtype == np.int64
    np.testing.assert_array_equal(
        loaded_floats, np.array([[1.5, 2.5], [3.5, 4.5]])
    )
    np.testing.assert_array_equal(loaded_ints, np.array([[1, 2], [3, 4]]))


@pytest.mark.parametrize(("source_elem", "mesh_type"), _SURF_CASES)
def test_create_mesh_builds_and_renders_for_all_supported_surface_types(
    source_elem: riley.EElemType,
    mesh_type: riley.MeshType,
) -> None:
    coords = coords_3d(source_elem)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        source_elem,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        mesh_type,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == mesh_type

    pixels_num = (64, 64)
    pixels_size = (10e-6, 10e-6)
    focal_length = 50e-3
    rotation = (0.0, 0.0, 0.0)
    target = riley.roi_cent_from_coords(coords)
    position = riley.pos_frame_mesh(
        mesh,
        pixels_num,
        pixels_size,
        focal_length,
        rotation,
        target=target,
    )
    camera = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=position,
        rot_world=rotation,
        roi_cent_world=target,
        focal_length=focal_length,
        sub_sample=2,
    )
    config = riley.create_raster_config(
        1,
        save_strategy=riley.SaveStrategy.memory,
    )
    riley.raster(mesh, camera, config)


def test_create_mesh_converts_volume_hex8_to_quad4() -> None:
    coords = coords_3d(riley.EElemType.HEX8)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.HEX8,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.quad4,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.quad4
    assert mesh.coords.shape[0] == 8
    assert mesh.connect.shape == (6, 4)


def test_create_mesh_converts_volume_tet4_to_tri3() -> None:
    coords = coords_3d(riley.EElemType.TET4)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.TET4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.tri3
    assert mesh.coords.shape[0] == 4
    assert mesh.connect.shape == (4, 3)


def test_create_mesh_maps_displacements_from_source_volume_to_surface() -> None:
    coords = coords_3d(riley.EElemType.HEX8)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.HEX8,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()
    disp_x = np.arange(coords.shape[0], dtype=np.float64) * 0.1
    disp_y = np.arange(coords.shape[0], dtype=np.float64) * 0.2
    disp_z = np.arange(coords.shape[0], dtype=np.float64) * 0.3

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.quad4,
        coords,
        connect,
        shader,
        disp=(disp_x, disp_y, disp_z),
    )

    assert mesh.disp is not None
    assert mesh.disp.shape == (1, mesh.coords.shape[0], 3)
    np.testing.assert_allclose(mesh.disp[0, :, 0], disp_x)
    np.testing.assert_allclose(mesh.disp[0, :, 1], disp_y)
    np.testing.assert_allclose(mesh.disp[0, :, 2], disp_z)


def test_create_mesh_reduces_order_from_quad8_to_quad4() -> None:
    coords = coords_3d(riley.EElemType.QUAD8)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.QUAD8,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.quad4,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.quad4
    assert mesh.coords.shape[0] == 4
    assert mesh.connect.shape == (1, 4)


def test_quad4_mesh_type_enum_contract() -> None:
    assert int(riley.MeshType.quad4) == 4
    assert riley.MeshType(4) is riley.MeshType.quad4
    assert not hasattr(riley.MeshType, "quad4ibi")
    assert not hasattr(riley.MeshType, "quad4newton")
    with pytest.raises(ValueError):
        riley.MeshType(3)


def test_create_mesh_triangulates_quad8_to_tri3() -> None:
    coords = coords_3d(riley.EElemType.QUAD8)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.QUAD8,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.tri3
    assert mesh.coords.shape[0] == 8
    assert mesh.connect.shape == (6, 3)


def test_create_mesh_triangulates_quad4_to_tri3() -> None:
    coords = coords_3d(riley.EElemType.QUAD4)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.QUAD4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.tri3
    assert mesh.coords.shape[0] == 4
    assert mesh.connect.shape == (2, 3)


def test_create_mesh_triangulates_tri6_to_tri3() -> None:
    coords = coords_3d(riley.EElemType.TRI6)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.TRI6,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = function_shader()

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.tri3
    assert mesh.coords.shape[0] == 6
    assert mesh.connect.shape == (4, 3)


def test_create_mesh_supports_temporal_fields() -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    nodal_field = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ]
    )
    shader = riley.NodalShader(nodal_field)
    disp_x = np.zeros((3, 3), dtype=np.float64)
    disp_y = np.zeros((3, 3), dtype=np.float64)
    disp_z = np.zeros((3, 3), dtype=np.float64)

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
        shader,
        disp=(disp_x, disp_y, disp_z),
    )
    assert mesh.disp is not None
    assert mesh.disp.shape == (3, 3, 3)
    assert mesh.shader.field.shape == (3, 3, 1)


def test_create_mesh_from_prepared_creates_mesh() -> None:
    coords = coords_3d(riley.EElemType.TRI3)
    connect = np.arange(coords.shape[0], dtype=np.uintp).reshape((1, -1))
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    conversion = riley.convert_mesh_for_render(
        convention,
        riley.MeshType.tri3,
        coords,
        connect,
    )
    shader = function_shader()

    mesh = riley.create_mesh_from_prepared(
        conversion,
        shader,
    )
    assert mesh.mesh_type == riley.MeshType.tri3
    assert mesh.coords.shape == (3, 3)
    assert mesh.connect.shape == (1, 3)

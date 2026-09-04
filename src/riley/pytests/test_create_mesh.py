"""Tests for the public mesh-construction boundary."""

import numpy as np
import pytest
import riley


_SURF_CASES = (
    (riley.EElementType.TRI3, riley.MeshType.tri3),
    (riley.EElementType.TRI3, riley.MeshType.tri3opt),
    (riley.EElementType.TRI6, riley.MeshType.tri6),
    (riley.EElementType.QUAD4, riley.MeshType.quad4ibi),
    (riley.EElementType.QUAD4, riley.MeshType.quad4newton),
    (riley.EElementType.QUAD8, riley.MeshType.quad8),
    (riley.EElementType.QUAD9, riley.MeshType.quad9),
)


def _coords_3d(elem_type: riley.EElementType) -> np.ndarray:
    coords = elem_type.calc_ref_coords()
    if coords.shape[1] == 2:
        coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    return coords


def _function_shader() -> riley.FunctionShader:
    return riley.FunctionShader(riley.FuncShaderBuiltin.constant)


@pytest.mark.parametrize(("elem_type", "mesh_type"), _SURF_CASES)
@pytest.mark.parametrize("axis", tuple(riley.EConnectAxis))
@pytest.mark.parametrize("index_base", (0, 1))
def test_create_mesh_identity_representations(
    elem_type: riley.EElementType,
    mesh_type: riley.MeshType,
    axis: riley.EConnectAxis,
    index_base: int,
) -> None:
    coords = _coords_3d(elem_type)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :] + index_base
    if axis is riley.EConnectAxis.COLUMN:
        connect = connect.T
    convention = riley.ConnectConvention(
        elem_type, axis, index_base, riley.ENodeOrder.RILEY
    )
    mesh = riley.create_mesh(
        convention, mesh_type, coords, connect, shader=_function_shader()
    )
    assert mesh.mesh_type is mesh_type
    assert mesh.coords.dtype == np.float64
    assert mesh.connect.dtype == np.uintp
    assert mesh.coords.flags.c_contiguous
    assert mesh.connect.flags.c_contiguous


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (riley.EElementType.TRI7, riley.MeshType.tri6),
        (riley.EElementType.TRI7, riley.MeshType.tri3),
        (riley.EElementType.TRI6, riley.MeshType.tri3),
        (riley.EElementType.QUAD9, riley.MeshType.quad8),
        (riley.EElementType.QUAD9, riley.MeshType.quad4newton),
        (riley.EElementType.QUAD8, riley.MeshType.quad4ibi),
        (riley.EElementType.TET4, riley.MeshType.tri3),
        (riley.EElementType.TET10, riley.MeshType.tri6),
        (riley.EElementType.TET10, riley.MeshType.tri3),
        (riley.EElementType.HEX8, riley.MeshType.quad4newton),
        (riley.EElementType.HEX20, riley.MeshType.quad8),
        (riley.EElementType.HEX20, riley.MeshType.quad4ibi),
        (riley.EElementType.HEX27, riley.MeshType.quad9),
        (riley.EElementType.HEX27, riley.MeshType.quad8),
        (riley.EElementType.HEX27, riley.MeshType.quad4newton),
    ),
)
def test_create_mesh_supported_topology_transitions(source, target) -> None:
    coords = _coords_3d(source)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        target, coords, connect, shader=_function_shader(),
    )
    assert mesh.connect.shape[1] in (3, 4, 6, 8, 9)


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (riley.EElementType.TRI3, riley.MeshType.tri6),
        (riley.EElementType.TRI6, riley.MeshType.quad4ibi),
        (riley.EElementType.QUAD8, riley.MeshType.tri3),
        (riley.EElementType.TET10, riley.MeshType.quad8),
        (riley.EElementType.HEX20, riley.MeshType.tri6),
    ),
)
def test_create_mesh_rejects_unsupported_topology_transitions(
    source,
    target,
) -> None:
    coords = _coords_3d(source)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    with pytest.raises(riley.MeshConvErr):
        riley.create_mesh(
            riley.ConnectConvention(
                source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
            ),
            target, coords, connect, shader=_function_shader(),
        )


def test_create_mesh_remaps_every_nodal_array_by_source_index() -> None:
    source = riley.EElementType.HEX27
    coords = source.calc_ref_coords()
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
    coords = _coords_3d(riley.EElementType.TRI3)
    field = np.ones((3, 2, channels), dtype=np.float32)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElementType.TRI3, riley.EConnectAxis.ROW, 0,
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
    coords = _coords_3d(riley.EElementType.TRI3)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElementType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=_function_shader(),
    )
    center = riley.roi_cent_over_meshes([mesh])
    np.testing.assert_allclose(center, (0.5, 0.5, 0))


@pytest.mark.parametrize(
    ("case_name", "elem_type", "mesh_type"),
    (
        ("tet4", riley.EElementType.TET4, riley.MeshType.tri3),
        ("tet10", riley.EElementType.TET10, riley.MeshType.tri6),
        ("hex8", riley.EElementType.HEX8, riley.MeshType.quad4newton),
        ("hex20", riley.EElementType.HEX20, riley.MeshType.quad8),
        ("hex27", riley.EElementType.HEX27, riley.MeshType.quad9),
    ),
)
def test_create_mesh_reuses_packaged_cube_fixtures(
    case_name: str,
    elem_type: riley.EElementType,
    mesh_type: riley.MeshType,
) -> None:
    case_path = riley.data.cube_case_path(case_name)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            elem_type, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
        ),
        mesh_type,
        riley.load_csv(case_path / "coords.csv"),
        riley.load_csv(case_path / "connectivity.csv", dtype=np.int64),
        shader=_function_shader(),
    )
    assert mesh.connect.shape[0] > 0
    riley.roi_cent_over_meshes([mesh])


@pytest.mark.parametrize("channels", (1, 3))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float64))
def test_create_mesh_supports_all_texture_storage_types(
    channels: int,
    dtype: type,
) -> None:
    coords = _coords_3d(riley.EElementType.TRI3)
    texture = np.ones((channels, 2, 2), dtype=dtype)
    shader = riley.TextureShader(coords[:, :2], texture)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElementType.TRI3, riley.EConnectAxis.ROW, 0,
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
    coords = _coords_3d(riley.EElementType.TRI3)
    mesh = riley.create_mesh(
        riley.ConnectConvention(
            riley.EElementType.TRI3, riley.EConnectAxis.ROW, 0,
            riley.ENodeOrder.RILEY,
        ),
        riley.MeshType.tri3,
        coords,
        np.array(((0, 1, 2),), dtype=np.int64),
        shader=riley.FunctionShader(builtin, channels=channels),
    )
    riley.roi_cent_over_meshes([mesh])

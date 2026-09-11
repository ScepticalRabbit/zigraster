from __future__ import annotations

import numpy as np

import riley


def test_conversion_preserves_source_node_identity() -> None:
    coords = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (9.0, 9.0, 9.0),
        ),
        dtype=np.float64,
    )
    connect = np.asarray(((0, 1, 2),), dtype=np.int64)
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )

    conversion = riley.convert_mesh_for_render(
        convention, riley.MeshType.tri3, coords, connect
    )

    np.testing.assert_array_equal(
        conversion.source_node_indices, np.arange(4, dtype=np.uintp)
    )
    assert conversion.source_node_count == 4


def test_remap_nodal_data_uses_requested_axis() -> None:
    coords = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 2.0, 2.0),
        ),
        dtype=np.float64,
    )
    connect = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    convention = riley.ConnectConvention(
        riley.EElemType.TET4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    conversion = riley.convert_mesh_for_render(
        convention, riley.MeshType.tri3, coords, connect
    )
    values = np.arange(10).reshape(2, 5)

    actual = riley.remap_nodal_data(conversion, values, node_axis=1)

    np.testing.assert_array_equal(actual, values[:, :4])
    assert actual.flags.c_contiguous


def test_two_stage_source_construction_matches_create_mesh() -> None:
    coords = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    connect = np.asarray(((0, 1, 2),), dtype=np.int64)
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = riley.FunctionShader(riley.FuncShaderBuiltin.checker)
    conversion = riley.convert_mesh_for_render(
        convention, riley.MeshType.tri3, coords, connect
    )

    direct = riley.create_mesh(
        convention, riley.MeshType.tri3, coords, connect, shader
    )
    staged = riley.create_mesh_from_conversion(conversion, shader)

    np.testing.assert_array_equal(staged.coords, direct.coords)
    np.testing.assert_array_equal(staged.connect, direct.connect)
    assert staged.mesh_type is direct.mesh_type


def test_prepared_construction_accepts_converted_uvs() -> None:
    coords = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    connect = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    convention = riley.ConnectConvention(
        riley.EElemType.TET4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    conversion = riley.convert_mesh_for_render(
        convention, riley.MeshType.tri3, coords, connect
    )
    uvs = np.zeros((conversion.geometry.coords.shape[0], 2))
    texture = np.zeros((1, 2, 2), dtype=np.uint8)

    mesh = riley.create_mesh_from_prepared(
        conversion, riley.TextureShader(uvs, texture)
    )

    np.testing.assert_array_equal(mesh.shader.uvs, uvs)

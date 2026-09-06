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
from riley.cython.riley import (
    FunctionShader,
    FuncShaderBuiltin,
    MeshType,
    NodalShader,
    TextureShader,
)
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


def test_packaged_data_paths_reject_unknown_cases() -> None:
    with pytest.raises(ValueError, match="Unsupported element type"):
        riley.data.cube_case_path("tet14")
    with pytest.raises(ValueError, match="Unsupported element type"):
        riley.data.cube_exodus_path("tet14")
    with pytest.raises(ValueError, match="Unsupported shape"):
        riley.data.shape_case_path("torus", "hex8")
    with pytest.raises(ValueError, match="Unsupported sphere200 data case"):
        riley.data.sphere200_case_path("unknown")


def test_load_csv_rejects_fractional_integer_input(tmp_path: Path) -> None:
    save_csv(tmp_path / "connect.csv", np.array(((0.0, 1.5, 2.0),)))

    with pytest.raises(ValueError):
        riley.load_csv(tmp_path / "connect.csv", dtype=np.int64)


def test_load_csv_rejects_non_finite_values(tmp_path: Path) -> None:
    save_csv(tmp_path / "coords.csv", np.array(((0.0, np.nan, 1.0),)))

    with pytest.raises(ValueError, match="non-finite"):
        riley.load_csv(tmp_path / "coords.csv")


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




@pytest.mark.parametrize(
    ("source", "target"),
    (
        (riley.EElemType.TRI3, riley.MeshType.tri6),
        (riley.EElemType.TRI6, riley.MeshType.quad4ibi),
        (riley.EElemType.QUAD4, riley.MeshType.tri6),
        (riley.EElemType.QUAD8, riley.MeshType.tri6),
        (riley.EElemType.TET10, riley.MeshType.quad8),
        (riley.EElemType.HEX20, riley.MeshType.tri6),
    ),
)
def test_create_mesh_rejects_unsupported_topology_transitions(
    source,
    target,
) -> None:
    coords = coords_3d(source)
    connect = np.arange(coords.shape[0], dtype=np.int64)[None, :]
    with pytest.raises(riley.MeshError):
        riley.create_mesh(
            riley.ConnectConvention(
                source, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
            ),
            target, coords, connect, shader=function_shader(),
        )

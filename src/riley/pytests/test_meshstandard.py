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

from riley import data
from riley.python import meshconv


_CUBE_TYPES = {
    "tet4": meshconv.EElementType.TET4,
    "tet10": meshconv.EElementType.TET10,
    "hex8": meshconv.EElementType.HEX8,
    "hex20": meshconv.EElementType.HEX20,
    "hex27": meshconv.EElementType.HEX27,
}

_SPHERE_TYPES = {
    "tri3_sphere200": meshconv.EElementType.TRI3,
    "tri6_sphere200": meshconv.EElementType.TRI6,
    "quad4newton_sphere200": meshconv.EElementType.QUAD4,
    "quad8_sphere200": meshconv.EElementType.QUAD8,
    "quad9_sphere200": meshconv.EElementType.QUAD9,
}


@pytest.mark.parametrize("elem_type", tuple(meshconv.EElementType))
@pytest.mark.parametrize("elem_axis", tuple(meshconv.EConnectAxis))
@pytest.mark.parametrize("index_base", (0, 1))
def test_vtk_adapter_handles_every_array_representation(
    elem_type: meshconv.EElementType,
    elem_axis: meshconv.EConnectAxis,
    index_base: int,
) -> None:
    coords = elem_type.calc_ref_coords()
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
    assert mesh.connect.dtype == np.int64
    assert mesh.connect.flags.c_contiguous


def test_exodus_hex20_adapter_reorders_edge_groups() -> None:
    elem_type = meshconv.EElementType.HEX20
    coords = elem_type.calc_ref_coords()
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


def test_user_topology_describes_source_relationships() -> None:
    elem_type = meshconv.EElementType.QUAD8
    coords = elem_type.calc_ref_coords()
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
        elem_type=meshconv.EElementType.TRI3,
        coords=coords,
        connect=connect,
    )

    with pytest.raises(meshconv.MeshVerifyErr) as error_info:
        meshconv.verify_mesh(mesh)

    codes = {issue.code for issue in error_info.value.issues}
    assert codes == {
        "coordinate_dtype",
        "coordinate_values",
        "connectivity_dtype",
    }


def test_extract_surface_returns_standard_compact_hex27_surface() -> None:
    elem_type = meshconv.EElementType.HEX27
    coords = elem_type.calc_ref_coords()
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

    assert surface.elem_type is meshconv.EElementType.QUAD9
    assert surface.coords.shape == (26, 3)
    assert surface.connect.shape == (6, 9)
    assert np.array_equal(
        np.unique(surface.connect),
        np.arange(26, dtype=np.int64),
    )


@pytest.mark.parametrize(("case_name", "elem_type"), _CUBE_TYPES.items())
def test_packaged_cube_converts_verifies_and_extracts(
    case_name: str,
    elem_type: meshconv.EElementType,
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
    )

    volume = meshconv.convert_mesh(coords, connect, convention)
    surface = meshconv.extract_surface(volume)

    meshconv.verify_mesh(volume)
    meshconv.verify_mesh(surface)


@pytest.mark.parametrize(("case_name", "elem_type"), _SPHERE_TYPES.items())
def test_packaged_sphere_converts_and_verifies(
    case_name: str,
    elem_type: meshconv.EElementType,
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
    if elem_type in (
        meshconv.EElementType.TRI3,
        meshconv.EElementType.TRI6,
    ):
        # These historical fixtures contain collapsed polar faces; TRI6 also
        # contains an exact duplicate. Riley deliberately rejects both, so
        # retain the valid faces for the curved-surface conversion test.
        corners = coords[connect[:, :3]]
        twice_area = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0],
                     corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        connect = connect[twice_area > 1.0e-12]
        _, unique_idxs = np.unique(connect, axis=0, return_index=True)
        connect = connect[np.sort(unique_idxs)]
    convention = meshconv.ConnectConvention(
        elem_type=elem_type,
        elem_axis=meshconv.EConnectAxis.ROW,
        index_base=0,
    )

    mesh = meshconv.convert_mesh(coords, connect, convention)

    meshconv.verify_mesh(mesh)

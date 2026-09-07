# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest

import riley
from riley import data
from riley.python.meshconst import (
    EElemType,
    MeshType,
)
from riley.python.meshconv import (
    ConnectConvention,
    EConnectAxis,
    ENodeOrder,
    MeshGeometry,
    convert_mesh,
    extract_surface,
    reduce_mesh_order,
    triangulate_mesh,
    verify_mesh,
)

_ALL_SHAPE_CASES = (
    ("cube", "hex8", True),
    ("cube", "hex20", True),
    ("cube", "hex27", True),
    ("cube", "tet4", True),
    ("cube", "tet10", True),
    ("cube", "hex8", False),
    ("cube", "hex20", False),
    ("cube", "hex27", False),
    ("cube", "tet4", False),
    ("cube", "tet10", False),
    ("cylinder", "hex8", False),
    ("cylinder", "hex20", False),
    ("cylinder", "hex27", False),
    ("cylinder", "tet4", False),
    ("cylinder", "tet10", False),
    ("platewithhole", "hex8", False),
    ("platewithhole", "hex20", False),
    ("platewithhole", "hex27", False),
    ("platewithhole", "tet4", False),
    ("platewithhole", "tet10", False),
)

_QUADRATIC_SHAPE_CASES = (
    ("cube", "hex20", True, EElemType.HEX8, EElemType.QUAD4),
    ("cube", "hex27", True, EElemType.HEX8, EElemType.QUAD4),
    ("cube", "tet10", True, EElemType.TET4, EElemType.TRI3),
    ("cube", "hex20", False, EElemType.HEX8, EElemType.QUAD4),
    ("cube", "hex27", False, EElemType.HEX8, EElemType.QUAD4),
    ("cube", "tet10", False, EElemType.TET4, EElemType.TRI3),
    ("cylinder", "hex20", False, EElemType.HEX8, EElemType.QUAD4),
    ("cylinder", "hex27", False, EElemType.HEX8, EElemType.QUAD4),
    ("cylinder", "tet10", False, EElemType.TET4, EElemType.TRI3),
    ("platewithhole", "hex20", False, EElemType.HEX8, EElemType.QUAD4),
    ("platewithhole", "hex27", False, EElemType.HEX8, EElemType.QUAD4),
    ("platewithhole", "tet10", False, EElemType.TET4, EElemType.TRI3),
)

_MULTISHAPE_CASES = (
    ("hex8_tet4", EElemType.HEX8, EElemType.TET4),
    ("hex20_tet10", EElemType.HEX20, EElemType.TET10),
    ("hex27_tet10", EElemType.HEX27, EElemType.TET10),
    ("tet4_hex8", EElemType.TET4, EElemType.HEX8),
    ("tet10_hex20", EElemType.TET10, EElemType.HEX20),
    ("tet10_hex27", EElemType.TET10, EElemType.HEX27),
)

_ELEM_ENUM_MAP = {
    "hex8": EElemType.HEX8,
    "hex20": EElemType.HEX20,
    "hex27": EElemType.HEX27,
    "tet4": EElemType.TET4,
    "tet10": EElemType.TET10,
}


def compute_surface_geometry(
    mesh: MeshGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = mesh.coords
    connect = mesh.connect
    elem_count = connect.shape[0]
    normals = np.zeros((elem_count, 3), dtype=np.float64)
    areas = np.zeros(elem_count, dtype=np.float64)
    centroids = np.zeros((elem_count, 3), dtype=np.float64)

    is_quad = mesh.elem_type in (
        EElemType.QUAD4,
        EElemType.QUAD8,
        EElemType.QUAD9,
    )
    for ii in range(elem_count):
        elem = connect[ii]
        c0 = coords[elem[0]]
        c1 = coords[elem[1]]
        c2 = coords[elem[2]]
        if is_quad:
            c3 = coords[elem[3]]
            na = np.cross(c1 - c0, c2 - c0)
            nb = np.cross(c2 - c0, c3 - c0)
            n = na + nb
            area = 0.5 * (np.linalg.norm(na) + np.linalg.norm(nb))
            centroid = 0.25 * (c0 + c1 + c2 + c3)
        else:
            n = np.cross(c1 - c0, c2 - c0)
            area = 0.5 * np.linalg.norm(n)
            centroid = (c0 + c1 + c2) / 3.0

        norm = np.linalg.norm(n)
        if norm > 0.0:
            normals[ii] = n / norm
        areas[ii] = area
        centroids[ii] = centroid

    return normals, areas, centroids


def check_cube_invariants(
    mesh: MeshGeometry,
    is_surface: bool = False,
) -> None:
    coords = mesh.coords
    assert np.all(coords >= -1e-7)
    assert np.all(coords <= 0.01 + 1e-7)
    if is_surface:
        normals, areas, centroids = compute_surface_geometry(mesh)
        total_area = np.sum(areas)
        assert np.isclose(total_area, 6.0 * (0.01**2), rtol=1e-3)
        canonical = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
        matched = set()
        for ii in range(normals.shape[0]):
            dots = [float(np.dot(normals[ii], d)) for d in canonical]
            best_idx = int(np.argmax(dots))
            assert dots[best_idx] > 0.99
            matched.add(best_idx)
            c = centroids[ii]
            if best_idx == 0:
                assert np.isclose(c[0], 0.01, atol=1e-6)
            elif best_idx == 1:
                assert np.isclose(c[0], 0.0, atol=1e-6)
            elif best_idx == 2:
                assert np.isclose(c[1], 0.01, atol=1e-6)
            elif best_idx == 3:
                assert np.isclose(c[1], 0.0, atol=1e-6)
            elif best_idx == 4:
                assert np.isclose(c[2], 0.01, atol=1e-6)
            elif best_idx == 5:
                assert np.isclose(c[2], 0.0, atol=1e-6)
        assert len(matched) == 6


def check_cylinder_invariants(
    mesh: MeshGeometry,
    is_surface: bool = False,
) -> None:
    coords = mesh.coords
    assert np.all(coords[:, 1] >= -1e-5)
    assert np.all(coords[:, 1] <= 0.012 + 1e-5)
    r_sq = coords[:, 0] ** 2 + coords[:, 2] ** 2
    assert np.all(r_sq <= 0.005**2 + 1e-6)
    if is_surface:
        normals, areas, centroids = compute_surface_geometry(mesh)
        total_area = np.sum(areas)
        expected = (2.0 * np.pi * 0.005 * 0.012) + (2.0 * np.pi * (0.005**2))
        assert np.isclose(total_area, expected, rtol=0.10)
        for ii in range(mesh.connect.shape[0]):
            elem_nodes = mesh.connect[ii]
            y_vals = coords[elem_nodes, 1]
            n = normals[ii]
            c = centroids[ii]
            if np.allclose(y_vals, 0.0, atol=1e-5):
                assert np.dot(n, [0.0, -1.0, 0.0]) > 0.95
            elif np.allclose(y_vals, 0.012, atol=1e-5):
                assert np.dot(n, [0.0, 1.0, 0.0]) > 0.95
            else:
                rad_dist = np.sqrt(c[0] ** 2 + c[2] ** 2)
                if rad_dist > 1e-6:
                    u_rad = np.array([c[0] / rad_dist, 0.0, c[2] / rad_dist])
                    assert np.dot(n, u_rad) > 0.80


def check_platewithhole_invariants(
    mesh: MeshGeometry,
    is_surface: bool = False,
) -> None:
    coords = mesh.coords
    assert np.all(coords[:, 0] >= -1e-5)
    assert np.all(coords[:, 0] <= 0.025 + 1e-5)
    assert np.all(coords[:, 1] >= -1e-5)
    assert np.all(coords[:, 1] <= 0.035 + 1e-5)
    assert np.all(coords[:, 2] >= -1e-5)
    assert np.all(coords[:, 2] <= 0.002 + 1e-5)
    r_hole_sq = (coords[:, 0] - 0.0125) ** 2 + (coords[:, 1] - 0.0175) ** 2
    assert np.all(r_hole_sq >= 0.003125**2 - 1e-6)
    if is_surface:
        normals, areas, centroids = compute_surface_geometry(mesh)
        total_area = np.sum(areas)
        expected = (
            2.0 * (0.025 * 0.035 - np.pi * (0.003125**2))
            + 2.0 * (0.025 + 0.035) * 0.002
            + 2.0 * np.pi * 0.003125 * 0.002
        )
        assert np.isclose(total_area, expected, rtol=0.10)
        for ii in range(mesh.connect.shape[0]):
            elem_nodes = mesh.connect[ii]
            z_vals = coords[elem_nodes, 2]
            x_vals = coords[elem_nodes, 0]
            y_vals = coords[elem_nodes, 1]
            n = normals[ii]
            c = centroids[ii]
            if np.allclose(z_vals, 0.0, atol=1e-5):
                assert np.dot(n, [0.0, 0.0, -1.0]) > 0.95
            elif np.allclose(z_vals, 0.002, atol=1e-5):
                assert np.dot(n, [0.0, 0.0, 1.0]) > 0.95
            elif np.allclose(x_vals, 0.0, atol=1e-5):
                assert np.dot(n, [-1.0, 0.0, 0.0]) > 0.95
            elif np.allclose(x_vals, 0.025, atol=1e-5):
                assert np.dot(n, [1.0, 0.0, 0.0]) > 0.95
            elif np.allclose(y_vals, 0.0, atol=1e-5):
                assert np.dot(n, [0.0, -1.0, 0.0]) > 0.95
            elif np.allclose(y_vals, 0.035, atol=1e-5):
                assert np.dot(n, [0.0, 1.0, 0.0]) > 0.95
            else:
                dist_hole = np.sqrt(
                    (c[0] - 0.0125) ** 2 + (c[1] - 0.0175) ** 2
                )
                if dist_hole < 0.0035:
                    u_hole = -np.array(
                        [
                            (c[0] - 0.0125) / dist_hole,
                            (c[1] - 0.0175) / dist_hole,
                            0.0,
                        ]
                    )
                    assert np.dot(n, u_hole) > 0.80


def check_shape_invariants(
    shape: str,
    mesh: MeshGeometry,
    is_surface: bool = False,
) -> None:
    if shape == "cube":
        check_cube_invariants(mesh, is_surface=is_surface)
    elif shape == "cylinder":
        check_cylinder_invariants(mesh, is_surface=is_surface)
    elif shape in ("platewithhole", "platehole"):
        check_platewithhole_invariants(mesh, is_surface=is_surface)
    else:
        raise ValueError(f"Unknown shape: {shape}")


@pytest.mark.parametrize(("shape", "elem_type", "pure"), _ALL_SHAPE_CASES)
def test_shape_exodus_loading_and_conversion(
    shape: str,
    elem_type: str,
    pure: bool,
) -> None:
    exodus_path = data.shape_exodus_path(shape, elem_type, pure=pure)
    sim = riley.load_exodus(
        exodus_path,
        disp_keys=("disp_x", "disp_y", "disp_z"),
        nodal_keys=("temperature",),
    )
    block = list(sim.elem_blocks.values())[0]
    expected_enum = _ELEM_ENUM_MAP[elem_type]
    assert block.elem_type is expected_enum

    conv = ConnectConvention(
        block.elem_type,
        EConnectAxis.ROW,
        1,
        ENodeOrder.EXODUS,
    )
    vol = convert_mesh(sim.coords, block.connect, conv)
    verify_mesh(vol)
    check_shape_invariants(shape, vol, is_surface=False)

    assert sim.disp is not None
    assert len(sim.disp) == 3
    assert all(comp.shape == (vol.coords.shape[0], 5) for comp in sim.disp)
    assert "temperature" in sim.nodal_vars
    assert sim.nodal_vars["temperature"].shape == (vol.coords.shape[0], 5)


@pytest.mark.parametrize(("shape", "elem_type", "pure"), _ALL_SHAPE_CASES)
def test_shape_csv_conversion(
    shape: str,
    elem_type: str,
    pure: bool,
) -> None:
    coords = riley.load_csv(
        data.shape_coords_path(shape, elem_type, pure=pure)
    )
    connect = riley.load_csv(
        data.shape_connectivity_path(shape, elem_type, pure=pure),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "x", pure=pure)
    )
    disp_y = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "y", pure=pure)
    )
    disp_z = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "z", pure=pure)
    )
    temp = riley.load_csv(
        data.shape_temperature_path(shape, elem_type, pure=pure)
    )

    elem_enum = _ELEM_ENUM_MAP[elem_type]
    conv = ConnectConvention(
        elem_enum,
        EConnectAxis.ROW,
        0,
        ENodeOrder.RILEY,
    )
    vol = convert_mesh(coords, connect, conv)
    verify_mesh(vol)
    check_shape_invariants(shape, vol, is_surface=False)

    assert coords.shape[0] == vol.coords.shape[0]
    assert disp_x.shape == (coords.shape[0], 5)
    assert disp_y.shape == (coords.shape[0], 5)
    assert disp_z.shape == (coords.shape[0], 5)
    assert temp.shape == (coords.shape[0], 5)


@pytest.mark.parametrize(("shape", "elem_type", "pure"), _ALL_SHAPE_CASES)
def test_shape_surface_extraction_invariants_and_fields(
    shape: str,
    elem_type: str,
    pure: bool,
) -> None:
    coords = riley.load_csv(
        data.shape_coords_path(shape, elem_type, pure=pure)
    )
    connect = riley.load_csv(
        data.shape_connectivity_path(shape, elem_type, pure=pure),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "x", pure=pure)
    )
    disp_y = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "y", pure=pure)
    )
    disp_z = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "z", pure=pure)
    )
    temp = riley.load_csv(
        data.shape_temperature_path(shape, elem_type, pure=pure)
    )

    elem_enum = _ELEM_ENUM_MAP[elem_type]
    conv = ConnectConvention(
        elem_enum,
        EConnectAxis.ROW,
        0,
        ENodeOrder.RILEY,
    )
    vol = convert_mesh(coords, connect, conv)
    surf = extract_surface(vol)
    verify_mesh(surf)
    check_shape_invariants(shape, surf, is_surface=True)

    for ii in range(surf.coords.shape[0]):
        c = surf.coords[ii]
        dists = np.linalg.norm(coords - c, axis=1)
        orig_idx = int(np.argmin(dists))
        assert dists[orig_idx] < 1e-12


@pytest.mark.parametrize(("shape", "elem_type", "pure"), _ALL_SHAPE_CASES)
def test_shape_triangulation_invariants_and_fields(
    shape: str,
    elem_type: str,
    pure: bool,
) -> None:
    coords = riley.load_csv(
        data.shape_coords_path(shape, elem_type, pure=pure)
    )
    connect = riley.load_csv(
        data.shape_connectivity_path(shape, elem_type, pure=pure),
        dtype=np.int64,
    )
    elem_enum = _ELEM_ENUM_MAP[elem_type]
    conv = ConnectConvention(
        elem_enum,
        EConnectAxis.ROW,
        0,
        ENodeOrder.RILEY,
    )
    vol = convert_mesh(coords, connect, conv)
    surf = extract_surface(vol)
    tri = triangulate_mesh(surf)
    verify_mesh(tri)
    assert tri.elem_type is EElemType.TRI3
    check_shape_invariants(shape, tri, is_surface=True)

    for ii in range(tri.coords.shape[0]):
        c = tri.coords[ii]
        dists = np.linalg.norm(coords - c, axis=1)
        orig_idx = int(np.argmin(dists))
        assert dists[orig_idx] < 1e-12


@pytest.mark.parametrize(
    ("shape", "elem_type", "pure", "target_vol", "target_surf"),
    _QUADRATIC_SHAPE_CASES,
)
def test_shape_order_reduction_invariants_and_fields(
    shape: str,
    elem_type: str,
    pure: bool,
    target_vol: EElemType,
    target_surf: EElemType,
) -> None:
    coords = riley.load_csv(
        data.shape_coords_path(shape, elem_type, pure=pure)
    )
    connect = riley.load_csv(
        data.shape_connectivity_path(shape, elem_type, pure=pure),
        dtype=np.int64,
    )
    elem_enum = _ELEM_ENUM_MAP[elem_type]
    conv = ConnectConvention(
        elem_enum,
        EConnectAxis.ROW,
        0,
        ENodeOrder.RILEY,
    )
    vol = convert_mesh(coords, connect, conv)
    vol_red = reduce_mesh_order(vol, target_vol)
    verify_mesh(vol_red)
    assert vol_red.elem_type is target_vol
    check_shape_invariants(shape, vol_red, is_surface=False)

    surf = extract_surface(vol)
    surf_red = reduce_mesh_order(surf, target_surf)
    verify_mesh(surf_red)
    assert surf_red.elem_type is target_surf
    check_shape_invariants(shape, surf_red, is_surface=True)


@pytest.mark.parametrize(("shape", "elem_type", "pure"), _ALL_SHAPE_CASES)
def test_shape_create_mesh_end_to_end(
    shape: str,
    elem_type: str,
    pure: bool,
) -> None:
    coords = riley.load_csv(
        data.shape_coords_path(shape, elem_type, pure=pure)
    )
    connect = riley.load_csv(
        data.shape_connectivity_path(shape, elem_type, pure=pure),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "x", pure=pure)
    )
    disp_y = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "y", pure=pure)
    )
    disp_z = riley.load_csv(
        data.shape_disp_path(shape, elem_type, "z", pure=pure)
    )
    temp = riley.load_csv(
        data.shape_temperature_path(shape, elem_type, pure=pure)
    )

    elem_enum = _ELEM_ENUM_MAP[elem_type]
    conv = ConnectConvention(
        elem_enum,
        EConnectAxis.ROW,
        0,
        ENodeOrder.RILEY,
    )
    shader = riley.NodalShader(temp)
    mesh = riley.create_mesh(
        conv,
        MeshType.tri3,
        coords,
        connect,
        shader,
        disp=(disp_x, disp_y, disp_z),
    )
    assert mesh.mesh_type == MeshType.tri3
    assert mesh.coords.ndim == 2
    assert mesh.connect.shape[1] == 3
    assert mesh.disp is not None
    assert mesh.disp.shape == (5, mesh.coords.shape[0], 3)
    assert mesh.shader is not None

    for ii in range(mesh.coords.shape[0]):
        c = mesh.coords[ii]
        dists = np.linalg.norm(coords - c, axis=1)
        orig_idx = int(np.argmin(dists))
        assert dists[orig_idx] < 1e-12
        for tt in range(5):
            assert np.isclose(mesh.disp[tt, ii, 0], disp_x[orig_idx, tt])
            assert np.isclose(mesh.disp[tt, ii, 1], disp_y[orig_idx, tt])
            assert np.isclose(mesh.disp[tt, ii, 2], disp_z[orig_idx, tt])
            assert np.isclose(mesh.shader.field[tt, ii, 0], temp[orig_idx, tt])


@pytest.mark.parametrize(
    ("case_name", "cube_type", "cyl_type"),
    _MULTISHAPE_CASES,
)
def test_multishape_exodus_loading_and_invariants(
    case_name: str,
    cube_type: EElemType,
    cyl_type: EElemType,
) -> None:
    exo_path = data.multishape_exodus_path(case_name)
    sim = riley.load_exodus(
        exo_path,
        connect_keys=("connect1", "connect2"),
        disp_keys=("disp_x", "disp_y", "disp_z"),
        nodal_keys=("temperature",),
    )
    b1 = sim.elem_blocks["connect1"]
    b2 = sim.elem_blocks["connect2"]
    assert b1.elem_type is cube_type
    assert b2.elem_type is cyl_type

    vol1 = convert_mesh(
        sim.coords,
        b1.connect,
        ConnectConvention(
            b1.elem_type, EConnectAxis.ROW, 1, ENodeOrder.EXODUS
        ),
    )
    vol2 = convert_mesh(
        sim.coords,
        b2.connect,
        ConnectConvention(
            b2.elem_type, EConnectAxis.ROW, 1, ENodeOrder.EXODUS
        ),
    )
    verify_mesh(vol1)
    verify_mesh(vol2)

    cube_nodes = np.unique(vol1.connect)
    cyl_nodes = np.unique(vol2.connect)
    c1 = vol1.coords[cube_nodes]
    c2 = vol2.coords[cyl_nodes]

    max_x_cube = np.max(c1[:, 0])
    min_x_cyl = np.min(c2[:, 0])
    gap = min_x_cyl - max_x_cube
    assert 0.0009 <= gap <= 0.0012

    top_nodes = np.where(sim.coords[:, 1] >= 0.0099)[0]
    assert np.all(sim.disp[1][top_nodes, 4] > 5e-5)

    bot_nodes = np.where(sim.coords[:, 1] <= 1e-5)[0]
    avg_top = float(np.mean(sim.nodal_vars["temperature"][top_nodes, 4]))
    avg_bot = float(np.mean(sim.nodal_vars["temperature"][bot_nodes, 4]))
    assert avg_top > avg_bot

    surf1 = extract_surface(vol1)
    surf2 = extract_surface(vol2)
    verify_mesh(surf1)
    verify_mesh(surf2)

    tri1 = triangulate_mesh(surf1)
    tri2 = triangulate_mesh(surf2)
    verify_mesh(tri1)
    verify_mesh(tri2)
    assert tri1.elem_type is EElemType.TRI3
    assert tri2.elem_type is EElemType.TRI3


@pytest.mark.parametrize(
    ("case_name", "cube_type", "cyl_type"),
    _MULTISHAPE_CASES,
)
def test_multishape_csv_loading_and_conversion(
    case_name: str,
    cube_type: EElemType,
    cyl_type: EElemType,
) -> None:
    coords = riley.load_csv(data.multishape_coords_path(case_name))
    cube_connect = riley.load_csv(
        data.multishape_connectivity_path(case_name, block="cube"),
        dtype=np.int64,
    )
    cyl_connect = riley.load_csv(
        data.multishape_connectivity_path(case_name, block="cylinder"),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(
        data.multishape_disp_path(case_name, "x")
    )
    disp_y = riley.load_csv(
        data.multishape_disp_path(case_name, "y")
    )
    disp_z = riley.load_csv(
        data.multishape_disp_path(case_name, "z")
    )
    temp = riley.load_csv(
        data.multishape_temperature_path(case_name)
    )

    conv1 = ConnectConvention(
        cube_type, EConnectAxis.ROW, 0, ENodeOrder.RILEY
    )
    conv2 = ConnectConvention(
        cyl_type, EConnectAxis.ROW, 0, ENodeOrder.RILEY
    )
    vol1 = convert_mesh(coords, cube_connect, conv1)
    vol2 = convert_mesh(coords, cyl_connect, conv2)
    verify_mesh(vol1)
    verify_mesh(vol2)

    assert coords.shape[0] == vol1.coords.shape[0]
    assert disp_x.shape == (coords.shape[0], 5)
    assert disp_y.shape == (coords.shape[0], 5)
    assert disp_z.shape == (coords.shape[0], 5)
    assert temp.shape == (coords.shape[0], 5)


_2D_PLATE_CASES = (
    ("quad4", EElemType.QUAD4),
    ("quad8", EElemType.QUAD8),
    ("quad9", EElemType.QUAD9),
    ("tri3", EElemType.TRI3),
    ("tri6", EElemType.TRI6),
)

_2D_QUADRATIC_CASES = (
    ("quad8", EElemType.QUAD4),
    ("quad9", EElemType.QUAD4),
    ("tri6", EElemType.TRI3),
)

_ELEM_ENUM_MAP_2D = {
    "quad4": EElemType.QUAD4,
    "quad8": EElemType.QUAD8,
    "quad9": EElemType.QUAD9,
    "tri3": EElemType.TRI3,
    "tri6": EElemType.TRI6,
}


@pytest.mark.parametrize(("case_name", "elem_type"), _2D_PLATE_CASES)
def test_platewithhole2d_exodus_loading_and_invariants(
    case_name: str,
    elem_type: EElemType,
) -> None:
    exo_path = data.platewithhole2d_exodus_path(case_name)
    sim = riley.load_exodus(
        exo_path,
        connect_keys=("connect1",),
        disp_keys=("disp_x", "disp_y"),
        nodal_keys=("temperature",),
    )
    b1 = sim.elem_blocks["connect1"]
    assert b1.elem_type is elem_type

    mesh = convert_mesh(
        sim.coords,
        b1.connect,
        ConnectConvention(
            b1.elem_type, EConnectAxis.ROW, 1, ENodeOrder.EXODUS
        ),
    )
    verify_mesh(mesh)

    coords = mesh.coords
    assert np.all(coords[:, 0] >= -1e-6)
    assert np.all(coords[:, 0] <= 0.025 + 1e-6)
    assert np.all(coords[:, 1] >= -1e-6)
    assert np.all(coords[:, 1] <= 0.035 + 1e-6)
    assert np.all(np.abs(coords[:, 2]) <= 1e-12)

    hole_center = np.array([0.0125, 0.0175, 0.0])
    hole_rad = 0.025 / 8.0
    dists = np.linalg.norm(coords - hole_center, axis=1)
    min_dist = np.min(dists)
    assert np.isclose(min_dist, hole_rad, atol=1e-4)

    top_nodes = np.where(coords[:, 1] >= 0.035 - 1e-5)[0]
    assert np.all(sim.disp[1][top_nodes, 4] > 5e-5)

    tri = triangulate_mesh(mesh)
    verify_mesh(tri)
    assert tri.elem_type is EElemType.TRI3


@pytest.mark.parametrize(("case_name", "elem_type"), _2D_PLATE_CASES)
def test_platewithhole2d_csv_loading_and_triangulation(
    case_name: str,
    elem_type: EElemType,
) -> None:
    coords = riley.load_csv(data.platewithhole2d_coords_path(case_name))
    connect = riley.load_csv(
        data.platewithhole2d_connectivity_path(case_name),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(data.platewithhole2d_disp_path(case_name, "x"))
    disp_y = riley.load_csv(data.platewithhole2d_disp_path(case_name, "y"))
    disp_z = riley.load_csv(data.platewithhole2d_disp_path(case_name, "z"))
    temp = riley.load_csv(data.platewithhole2d_temperature_path(case_name))

    conv = ConnectConvention(
        elem_type, EConnectAxis.ROW, 0, ENodeOrder.RILEY
    )
    mesh = convert_mesh(coords, connect, conv)
    verify_mesh(mesh)

    assert coords.shape[0] == mesh.coords.shape[0]
    assert disp_x.shape == (coords.shape[0], 5)
    assert disp_y.shape == (coords.shape[0], 5)
    assert disp_z.shape == (coords.shape[0], 5)
    assert np.all(disp_z == 0.0)
    assert temp.shape == (coords.shape[0], 5)

    tri = triangulate_mesh(mesh)
    verify_mesh(tri)
    assert tri.elem_type is EElemType.TRI3


@pytest.mark.parametrize(
    ("case_name", "target_type"), _2D_QUADRATIC_CASES
)
def test_platewithhole2d_order_reduction(
    case_name: str,
    target_type: EElemType,
) -> None:
    elem_t = _ELEM_ENUM_MAP_2D[case_name]
    coords = riley.load_csv(data.platewithhole2d_coords_path(case_name))
    connect = riley.load_csv(
        data.platewithhole2d_connectivity_path(case_name),
        dtype=np.int64,
    )
    conv = ConnectConvention(
        elem_t, EConnectAxis.ROW, 0, ENodeOrder.RILEY
    )
    mesh = convert_mesh(coords, connect, conv)
    verify_mesh(mesh)

    reduced = reduce_mesh_order(mesh, target_type)
    verify_mesh(reduced)
    assert reduced.elem_type is target_type


@pytest.mark.parametrize(("case_name", "elem_type"), _2D_PLATE_CASES)
def test_platewithhole2d_riley_mesh_field_mapping(
    case_name: str,
    elem_type: EElemType,
) -> None:
    coords = riley.load_csv(data.platewithhole2d_coords_path(case_name))
    connect = riley.load_csv(
        data.platewithhole2d_connectivity_path(case_name),
        dtype=np.int64,
    )
    disp_x = riley.load_csv(data.platewithhole2d_disp_path(case_name, "x"))
    disp_y = riley.load_csv(data.platewithhole2d_disp_path(case_name, "y"))
    disp_z = riley.load_csv(data.platewithhole2d_disp_path(case_name, "z"))
    temp = riley.load_csv(data.platewithhole2d_temperature_path(case_name))

    conv = ConnectConvention(
        elem_type, EConnectAxis.ROW, 0, ENodeOrder.RILEY
    )
    mesh_geom = convert_mesh(coords, connect, conv)
    verify_mesh(mesh_geom)

    tri = triangulate_mesh(mesh_geom)
    verify_mesh(tri)

    mesh_type_tri = MeshType.tri3
    shader = riley.NodalShader(temp)
    mesh = riley.create_mesh(
        conv,
        mesh_type_tri,
        coords,
        connect,
        shader,
        disp=(disp_x, disp_y, disp_z),
    )
    assert mesh.mesh_type == MeshType.tri3
    assert mesh.coords.ndim == 2
    assert mesh.connect.shape[1] == 3
    assert mesh.disp is not None
    assert mesh.disp.shape == (5, mesh.coords.shape[0], 3)
    assert mesh.shader is not None

    for ii in range(mesh.coords.shape[0]):
        c = mesh.coords[ii]
        dists = np.linalg.norm(coords - c, axis=1)
        orig_idx = int(np.argmin(dists))
        assert dists[orig_idx] < 1e-12
        for tt in range(5):
            assert np.isclose(mesh.disp[tt, ii, 0], disp_x[orig_idx, tt])
            assert np.isclose(mesh.disp[tt, ii, 1], disp_y[orig_idx, tt])
            assert np.isclose(mesh.disp[tt, ii, 2], disp_z[orig_idx, tt])
            assert np.isclose(
                mesh.shader.field[tt, ii, 0], temp[orig_idx, tt]
            )


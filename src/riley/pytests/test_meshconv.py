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

from riley.python import _meshconv, meshconv


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SUPPORTED_CUBES = ("tet4", "tet10", "hex8", "hex20", "hex27")
SPHERE_MESHES = (
    "tri3_sphere200",
    "tri6_sphere200",
    "quad4newton_sphere200",
    "quad8_sphere200",
    "quad9_sphere200",
)


def test_element_specs_are_complete_and_mapping_is_read_only() -> None:
    assert {
        spec.nodes_per_element for spec in _meshconv.ELEMENT_SPECS.values()
    } == {3, 4, 6, 7, 8, 9, 10, 20, 27}
    assert (
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.HEX27].centre_index
        is None
    )

    with pytest.raises(TypeError):
        _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3] = (
            _meshconv.ELEMENT_SPECS[meshconv.EElementType.TRI3]
        )


def test_check_mesh_convention_passes_for_canonical_quad() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
    )
    mesh.refresh_mesh_type()

    report = meshconv.check_mesh_convention(mesh)

    assert mesh.mesh_type is meshconv.EMeshType.SURF
    assert report == {}


def test_enforce_mesh_convention_corrects_legacy_connectivity() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (2,), (3,), (4,)))},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )
    assert not meshconv.check_mesh_convention(mesh_out)


def test_check_mesh_convention_reports_failed_checks() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (4,), (3,), (2,)))},
    )

    report = meshconv.check_mesh_convention(mesh)

    assert report["connect1"] == [
        meshconv.MeshCheckCode.ROW_MAJOR_CONNECTIVITY,
        meshconv.MeshCheckCode.ZERO_BASED_INDEXING,
        meshconv.MeshCheckCode.CCW_WINDING,
        meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY,
    ]


def test_enforce_mesh_convention_raises_for_invalid_indices() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="invalid|outside"):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_mesh_convention_fixes_tet_handedness() -> None:
    mesh = meshconv.SimData(
        coords=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        connect={"connect1": np.array(((0, 2, 1, 3),), dtype=np.int64)},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )


def test_enforce_returns_same_object_when_mesh_conforms() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
    )
    mesh.refresh_mesh_type()

    assert meshconv.enforce_mesh_convention(mesh) is mesh


def test_enforce_emits_conforming_sibling_tables_untouched() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (2.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    good_connect = np.array(((0, 1, 2, 3),), dtype=np.int64)
    bad_connect = np.array(((4, 7, 6, 5),), dtype=np.int64)
    fixed_connect = np.array(((4, 5, 6, 7),), dtype=np.int64)
    mesh = meshconv.SimData(
        coords=coords,
        connect={"connect_good": good_connect, "connect_bad": bad_connect},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out is not mesh
    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect_good"], good_connect)
    assert np.array_equal(mesh_out.connect["connect_bad"], fixed_connect)
    assert mesh.connect is not None
    assert np.array_equal(mesh.connect["connect_bad"], bad_connect)


def test_enforce_reports_indices_outside_coordinate_array() -> None:
    mesh = meshconv.SimData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
    )

    with pytest.raises(
        ValueError,
        match="contains indices outside the coordinate array",
    ):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_propagates_zero_volume_topology_errors() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.5, 0.0)),
        dtype=np.float64,
    )
    mesh = meshconv.SimData(
        coords=coords,
        connect={
            "connect1": np.array(
                ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int64,
            ),
        },
        mesh_type=meshconv.EMeshType.SURF,
    )

    with pytest.raises(ValueError, match="zero signed volume"):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_tolerates_nonmanifold_surface_slices() -> None:
    """Non-manifold slices have no orientable shell; consistently wound input
    falls back to the per-face behaviour and passes untouched."""

    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, -1.0, 0.5), (0.0, 0.0, -1.0)),
        dtype=np.float64,
    )
    mesh = meshconv.SimData(
        coords=coords,
        connect={
            "connect1": np.array(((0, 1, 2), (0, 3, 1), (0, 1, 4)), dtype=np.int64),
        },
        mesh_type=meshconv.EMeshType.SURF,
    )

    assert not meshconv.check_mesh_convention(mesh)
    assert meshconv.enforce_mesh_convention(mesh) is mesh


def test_enforce_fixes_mirrored_hex_handedness_and_is_idempotent() -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        dtype=np.float64,
    )
    mirrored_row = np.array((0, 3, 2, 1, 4, 7, 6, 5), dtype=np.int64)[None, :]
    report = meshconv.check_mesh_convention(
        meshconv.SimData(coords=coords, connect={"connect1": mirrored_row}),
    )
    assert set(report["connect1"]) == {meshconv.MeshCheckCode.RIGHT_HANDED_GEOMETRY}

    def hex_volume(row: np.ndarray) -> float:
        points = coords[row[0, :]]
        return float(np.linalg.det(np.column_stack((
            points[1] - points[0], points[3] - points[0], points[4] - points[0],
        ))))

    assert hex_volume(mirrored_row) < 0.0

    mesh = meshconv.SimData(coords=coords, connect={"connect1": mirrored_row})
    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert not meshconv.check_mesh_convention(mesh_out)
    assert hex_volume(mesh_out.connect["connect1"]) > 0.0
    assert np.array_equal(
        meshconv.enforce_mesh_convention(mesh_out).connect["connect1"],
        mesh_out.connect["connect1"],
    )


def test_explicit_mesh_convention_reorders_source_slots() -> None:
    mesh = _load_cube("hex20")
    assert mesh.connect is not None
    canonical = mesh.connect["connect1"]
    source_to_riley = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        16, 17, 18, 19, 12, 13, 14, 15,
    )
    mesh.connect["connect1"] = canonical[:, np.argsort(source_to_riley)]
    convention = meshconv.MeshConvention({
        meshconv.EElementType.HEX20: source_to_riley,
    })

    assert meshconv.MeshCheckCode.NODE_ORDER in meshconv.check_mesh_convention(
        mesh,
        convention,
    )["connect1"]
    mesh_out = meshconv.enforce_mesh_convention(mesh, convention)

    assert mesh_out.connect is not None
    assert np.array_equal(mesh_out.connect["connect1"], canonical)
    assert not meshconv.check_mesh_convention(mesh_out)


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_canonical_cube_meshes_pass_and_enforcement_is_idempotent(
    cube_name: str,
) -> None:
    mesh = _load_cube(cube_name)

    assert not meshconv.check_mesh_convention(mesh)
    enforced_once = meshconv.enforce_mesh_convention(mesh)
    enforced_twice = meshconv.enforce_mesh_convention(enforced_once)

    assert not meshconv.check_mesh_convention(enforced_once)
    assert enforced_once.connect is not None
    assert enforced_twice.connect is not None
    for name, connect in enforced_once.connect.items():
        assert np.array_equal(connect, enforced_twice.connect[name])


def test_tet14_cube_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="supported nodes-per-element"):
        meshconv.check_mesh_convention(
            meshconv.SimData(
                coords=np.zeros((14, 3), dtype=np.float64),
                connect={"connect1": np.arange(14, dtype=np.int64).reshape(1, 14)},
            )
        )


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_extracted_cube_surface_passes_convention_check(cube_name: str) -> None:
    surface = meshconv.extract_surf_mesh(
        meshconv.enforce_mesh_convention(_load_cube(cube_name)),
    )

    assert not meshconv.check_mesh_convention(surface)


@pytest.mark.parametrize("mesh_name", SPHERE_MESHES)
def test_native_sphere_meshes_normalize_to_an_idempotent_convention(
    mesh_name: str,
) -> None:
    mesh = _load_native_mesh(
        DATA_DIR / "min" / mesh_name,
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)
    mesh_twice = meshconv.enforce_mesh_convention(mesh_out)

    assert not meshconv.check_mesh_convention(mesh_out)
    assert np.array_equal(mesh.coords, mesh_out.coords)
    assert mesh_out.connect is not None
    assert mesh_twice.connect is not None
    for name, connect in mesh_out.connect.items():
        assert np.array_equal(connect, mesh_twice.connect[name])


def test_plate_with_hole_keeps_inward_bore_normals() -> None:
    """A closed plate surface must retain its material-facing bore wall."""

    mesh = _load_native_mesh(
        DATA_DIR / "FE" / "platehole3d_2mr_63f",
        mesh_type=meshconv.EMeshType.SURF,
    )

    assert not meshconv.check_mesh_convention(mesh)
    mesh_out = meshconv.enforce_mesh_convention(mesh)
    assert mesh_out.connect is not None
    assert mesh.connect is not None
    assert np.array_equal(mesh_out.connect["connect1"], mesh.connect["connect1"])

    connect = mesh_out.connect["connect1"]
    assert mesh_out.coords is not None
    corners = mesh_out.coords[connect[:, :4]]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    radial = np.mean(corners, axis=1)[:, :2] - np.array((0.0125, 0.0175))
    radial_norm = np.linalg.norm(radial, axis=1)
    wall_rows = np.abs(normals[:, 2]) < 1.0e-12
    bore_rows = wall_rows & np.isclose(radial_norm, radial_norm[wall_rows].min())
    outer_rows = wall_rows & ~bore_rows

    assert np.count_nonzero(bore_rows) == 64
    assert np.all(np.sum(normals[bore_rows, :2] * radial[bore_rows], axis=1) < 0.0)
    assert np.all(np.sum(normals[outer_rows, :2] * radial[outer_rows], axis=1) > 0.0)


def test_nested_closed_surface_orients_cavity_into_the_void() -> None:
    outer_coords, outer_connect = _cube_surface(2.0, 0)
    inner_coords, inner_connect = _cube_surface(1.0, 8)
    mesh = meshconv.SimData(
        coords=np.vstack((outer_coords, inner_coords)),
        connect={"connect1": np.vstack((outer_connect, inner_connect))},
        mesh_type=meshconv.EMeshType.SURF,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert not meshconv.check_mesh_convention(mesh_out)
    connect = mesh_out.connect["connect1"]
    assert _surface_volume(mesh_out.coords, connect[:6]) > 0.0
    assert _surface_volume(mesh_out.coords, connect[6:]) < 0.0


def _quad_coords() -> np.ndarray:
    return np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )


def _cube_surface(scale: float, node_offset: int) -> tuple[np.ndarray, np.ndarray]:
    coords = scale * np.array(
        (
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    connect = np.array(
        ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
        dtype=np.int64,
    )
    return coords, connect + node_offset


def _surface_volume(coords: np.ndarray, connect: np.ndarray) -> float:
    volume = 0.0
    for row in connect:
        points = coords[row]
        for point_ind in range(1, points.shape[0] - 1):
            volume += np.dot(
                points[0],
                np.cross(points[point_ind], points[point_ind + 1]),
            ) / 6.0
    return float(volume)


def _load_cube(name: str) -> meshconv.SimData:
    return _load_native_mesh(DATA_DIR / "cubes" / name)


def _load_native_mesh(
    mesh_dir: Path,
    *,
    mesh_type: meshconv.EMeshType | None = None,
) -> meshconv.SimData:
    coords = np.loadtxt(mesh_dir / "coords.csv", delimiter=",", dtype=np.float64)
    connect_path = mesh_dir / "connectivity.csv"
    if not connect_path.is_file():
        connect_path = mesh_dir / "connect.csv"
    connect = np.loadtxt(connect_path, delimiter=",", dtype=np.float64).astype(np.int64)
    return meshconv.SimData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=mesh_type,
    )

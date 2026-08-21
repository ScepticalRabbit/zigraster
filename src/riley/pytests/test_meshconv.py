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

from riley.python import meshconv


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SUPPORTED_CUBES = ("tet4", "tet10", "hex8", "hex20", "hex27")
SPHERE_MESHES = (
    "tri3_sphere200",
    "tri6_sphere200",
    "quad4newton_sphere200",
    "quad8_sphere200",
    "quad9_sphere200",
)


def test_check_mesh_convention_passes_for_canonical_quad() -> None:
    mesh = meshconv.MeshData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 3),), dtype=np.int64)},
        num_spat_dims=2,
    )

    report = meshconv.check_mesh_convention(mesh)

    assert report.is_valid
    assert report.connectivity_failures["connect1"] == tuple()


def test_enforce_mesh_convention_corrects_legacy_connectivity() -> None:
    mesh = meshconv.MeshData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (2,), (3,), (4,)))},
        num_spat_dims=2,
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert mesh_out.connect is not None
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )
    assert meshconv.check_mesh_convention(mesh_out).is_valid


def test_check_mesh_convention_reports_failed_checks() -> None:
    mesh = meshconv.MeshData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((1,), (4,), (3,), (2,)))},
        num_spat_dims=2,
    )

    report = meshconv.check_mesh_convention(mesh)

    assert not report.is_valid
    assert "zero_based_indexing" in report.failed_checks
    assert "row_major_connectivity" in report.failed_checks
    assert "ccw_winding" in report.failed_checks


def test_enforce_mesh_convention_raises_for_invalid_indices() -> None:
    mesh = meshconv.MeshData(
        coords=_quad_coords(),
        connect={"connect1": np.array(((0, 1, 2, 10),), dtype=np.int64)},
        num_spat_dims=2,
    )

    with pytest.raises(ValueError, match="invalid|outside"):
        meshconv.enforce_mesh_convention(mesh)


def test_enforce_mesh_convention_fixes_tet_handedness() -> None:
    mesh = meshconv.MeshData(
        coords=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        connect={"connect1": np.array(((0, 2, 1, 3),), dtype=np.int64)},
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)

    assert meshconv.check_mesh_convention(mesh_out).is_valid
    assert np.array_equal(
        mesh_out.connect["connect1"],
        np.array(((0, 1, 2, 3),), dtype=np.int64),
    )


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_cube_mesh_convention_enforcement_is_idempotent(cube_name: str) -> None:
    mesh = _load_cube(cube_name)

    raw_report = meshconv.check_mesh_convention(mesh)
    enforced_once = meshconv.enforce_mesh_convention(mesh)
    enforced_twice = meshconv.enforce_mesh_convention(enforced_once)

    assert not raw_report.is_valid
    assert meshconv.check_mesh_convention(enforced_once).is_valid
    assert enforced_once.connect is not None
    assert enforced_twice.connect is not None
    for name, connect in enforced_once.connect.items():
        assert np.array_equal(connect, enforced_twice.connect[name])


def test_tet14_cube_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="supported nodes-per-element"):
        meshconv.check_mesh_convention(_load_cube("tet14"))


@pytest.mark.parametrize("cube_name", SUPPORTED_CUBES)
def test_extracted_cube_surface_passes_convention_check(cube_name: str) -> None:
    surface = meshconv.extract_surf_mesh(
        meshconv.enforce_mesh_convention(_load_cube(cube_name)),
    )

    assert meshconv.check_mesh_convention(surface).is_valid


@pytest.mark.parametrize("mesh_name", SPHERE_MESHES)
def test_native_sphere_meshes_normalize_to_an_idempotent_convention(
    mesh_name: str,
) -> None:
    mesh = _load_native_mesh(
        DATA_DIR / "min" / mesh_name,
        mesh_type="surface",
    )

    mesh_out = meshconv.enforce_mesh_convention(mesh)
    mesh_twice = meshconv.enforce_mesh_convention(mesh_out)

    assert meshconv.check_mesh_convention(mesh_out).is_valid
    assert np.array_equal(mesh.coords, mesh_out.coords)
    assert mesh_out.connect is not None
    assert mesh_twice.connect is not None
    for name, connect in mesh_out.connect.items():
        assert np.array_equal(connect, mesh_twice.connect[name])


def _quad_coords() -> np.ndarray:
    return np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )


def _load_cube(name: str) -> meshconv.MeshData:
    return _load_native_mesh(DATA_DIR / "cubes" / name)


def _load_native_mesh(
    mesh_dir: Path,
    *,
    mesh_type: str | None = None,
) -> meshconv.MeshData:
    coords = np.loadtxt(mesh_dir / "coords.csv", delimiter=",", dtype=np.float64)
    connect_path = mesh_dir / "connectivity.csv"
    if not connect_path.is_file():
        connect_path = mesh_dir / "connect.csv"
    connect = np.loadtxt(connect_path, delimiter=",", dtype=np.int64)
    return meshconv.MeshData(
        coords=coords,
        connect={"connect1": connect},
        mesh_type=mesh_type,
    )

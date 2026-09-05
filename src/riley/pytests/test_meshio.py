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
from riley.python import meshio


@pytest.mark.parametrize(
    "case_name",
    ("tet4", "tet10", "hex8", "hex20", "hex27"),
)
def test_packaged_cube_data_paths_exist(case_name: str) -> None:
    case_path = riley.data.cube_case_path(case_name)
    assert (case_path / "coords.csv").is_file()
    assert (case_path / "connectivity.csv").is_file()
    assert riley.data.cube_exodus_path(case_name).is_file()


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


def test_packaged_data_paths_reject_unknown_cases() -> None:
    with pytest.raises(ValueError, match="Unsupported cube data case"):
        riley.data.cube_case_path("tet14")
    with pytest.raises(ValueError, match="Unsupported cube data case"):
        riley.data.cube_exodus_path("tet14")
    with pytest.raises(ValueError, match="Unsupported sphere200 data case"):
        riley.data.sphere200_case_path("unknown")


def test_other_packaged_data_paths_exist() -> None:
    assert riley.data.speckle_texture_path().is_file()
    assert riley.data.cal_target_texture_path().is_file()
    assert riley.data.platehole_csv_case_path().is_dir()
    assert riley.data.platehole_exodus_path().is_file()
    assert riley.data.stereocal_case_path().is_dir()
    assert riley.data.rabbit_case_path("riley", "tri3").is_dir()


def _save_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",", fmt="%.8f")


def test_load_csv_preserves_table_orientation(tmp_path: Path) -> None:
    table = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    _save_csv(tmp_path / "table.csv", table)

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


def test_load_csv_rejects_fractional_integer_input(tmp_path: Path) -> None:
    _save_csv(tmp_path / "connect.csv", np.array(((0.0, 1.5, 2.0),)))

    with pytest.raises(ValueError):
        riley.load_csv(tmp_path / "connect.csv", dtype=np.int64)


def test_load_csv_rejects_non_finite_values(tmp_path: Path) -> None:
    _save_csv(tmp_path / "coords.csv", np.array(((0.0, np.nan, 1.0),)))

    with pytest.raises(ValueError, match="non-finite"):
        riley.load_csv(tmp_path / "coords.csv")


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


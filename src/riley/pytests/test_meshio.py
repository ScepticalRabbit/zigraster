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


def test_packaged_data_paths_exist() -> None:
    assert riley.data.speckle_texture_path().is_file()
    assert riley.data.cal_target_texture_path().is_file()
    assert riley.data.sphere200_case_path().is_dir()
    assert riley.data.platehole_csv_case_path().is_dir()
    assert riley.data.platehole_exodus_path().is_file()
    assert riley.data.stereocal_case_path().is_dir()
    assert riley.data.rabbit_case_path("riley", "tri3").is_dir()


def _save_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",", fmt="%.8f")


def test_load_coord_csv_coord_major(tmp_path: Path) -> None:
    coords = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float64)
    _save_csv(tmp_path / "coords.csv", coords.T)

    coords_loaded = riley.load_coord_csv(
        tmp_path / "coords.csv",
        orientation=riley.ECoordCsvOrientation.COORD_MAJOR,
    )

    assert coords_loaded.flags.c_contiguous
    np.testing.assert_allclose(coords_loaded, coords)


def test_load_connect_csv_one_based_node_major(tmp_path: Path) -> None:
    connect = np.array(((1, 2, 3), (3, 4, 1)), dtype=np.float64)
    _save_csv(tmp_path / "connect.csv", connect.T)

    connect_loaded = riley.load_connect_csv(
        tmp_path / "connect.csv",
        orientation=riley.EConnectCsvOrientation.NODE_MAJOR,
        indexing=riley.EConnectIndexing.ONE_BASED,
    )

    assert connect_loaded.flags.c_contiguous
    np.testing.assert_array_equal(
        connect_loaded,
        np.array(((0, 1, 2), (2, 3, 0)), dtype=np.uintp),
    )


def test_load_disp_csvs_node_major(tmp_path: Path) -> None:
    disp_x = np.array(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)), dtype=np.float64)
    disp_y = disp_x + 10.0
    disp_z = disp_x + 20.0
    _save_csv(tmp_path / "field_disp_x.csv", disp_x)
    _save_csv(tmp_path / "field_disp_y.csv", disp_y)
    _save_csv(tmp_path / "field_disp_z.csv", disp_z)

    disp = riley.load_disp_csvs(
        tmp_path / "field_disp_x.csv",
        tmp_path / "field_disp_y.csv",
        tmp_path / "field_disp_z.csv",
    )

    assert disp is not None
    assert disp.shape == (2, 3, 3)
    np.testing.assert_allclose(disp[:, :, 0], disp_x.T)
    np.testing.assert_allclose(disp[:, :, 1], disp_y.T)
    np.testing.assert_allclose(disp[:, :, 2], disp_z.T)


def test_load_sim_csvs_round_trip(tmp_path: Path) -> None:
    coords = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    connect = np.array(((0.0, 1.0, 2.0),), dtype=np.float64)
    uvs = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    disp_x = np.array(((0.0,), (0.1,), (0.2,)), dtype=np.float64)
    disp_y = np.array(((0.0,), (0.0,), (0.0,)), dtype=np.float64)
    disp_z = np.array(((0.0,), (0.0,), (0.0,)), dtype=np.float64)

    _save_csv(tmp_path / "coords.csv", coords)
    _save_csv(tmp_path / "connect.csv", connect)
    _save_csv(tmp_path / "uvs.csv", uvs)
    _save_csv(tmp_path / "field_disp_x.csv", disp_x)
    _save_csv(tmp_path / "field_disp_y.csv", disp_y)
    _save_csv(tmp_path / "field_disp_z.csv", disp_z)

    sim_data = riley.load_sim_csvs(tmp_path)
    coords_loaded, connect_loaded, uvs_loaded, disp_loaded = sim_data

    assert sim_data.coords is coords_loaded

    np.testing.assert_allclose(coords_loaded, coords)
    np.testing.assert_array_equal(connect_loaded, connect.astype(np.uintp))
    np.testing.assert_allclose(uvs_loaded, uvs)
    assert disp_loaded is not None
    assert disp_loaded.shape == (1, 3, 3)


def test_load_connect_csv_rejects_fractional_indices(tmp_path: Path) -> None:
    _save_csv(tmp_path / "connect.csv", np.array(((0.0, 1.5, 2.0),)))

    with pytest.raises(ValueError, match="integer"):
        riley.load_connect_csv(tmp_path / "connect.csv")


def test_load_connect_csv_auto_rejects_ambiguous_table(tmp_path: Path) -> None:
    _save_csv(tmp_path / "connect.csv", np.array(((1.0, 2.0, 3.0),)))

    with pytest.raises(ValueError, match="ambiguous"):
        riley.load_connect_csv(tmp_path / "connect.csv")


@pytest.mark.parametrize("indices", [((-1.0, 0.0, 1.0),), ((0.0, 1.0, 3.0),)])
def test_load_connect_csv_rejects_invalid_range(
    tmp_path: Path,
    indices: tuple[tuple[float, ...], ...],
) -> None:
    _save_csv(tmp_path / "connect.csv", np.asarray(indices))

    with pytest.raises(ValueError, match="negative|out-of-range"):
        riley.load_connect_csv(
            tmp_path / "connect.csv",
            indexing=riley.EConnectIndexing.ZERO_BASED,
            node_count=3,
        )


def test_load_disp_csvs_rejects_supplied_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="x-component"):
        riley.load_disp_csvs(tmp_path / "missing.csv", None, None)


def test_load_coord_csv_rejects_non_finite_values(tmp_path: Path) -> None:
    _save_csv(tmp_path / "coords.csv", np.array(((0.0, np.nan, 1.0),)))

    with pytest.raises(ValueError, match="non-finite"):
        riley.load_coord_csv(tmp_path / "coords.csv")


def test_load_sim_csvs_rejects_mismatched_uv_nodes(tmp_path: Path) -> None:
    _save_csv(tmp_path / "coords.csv", np.zeros((3, 3)))
    _save_csv(tmp_path / "connect.csv", np.array(((0.0, 1.0, 2.0),)))
    _save_csv(tmp_path / "uvs.csv", np.zeros((2, 2)))

    with pytest.raises(ValueError, match="UV and coordinate"):
        riley.load_sim_csvs(tmp_path)


def test_load_sim_csvs_rejects_mismatched_displacement_nodes(
    tmp_path: Path,
) -> None:
    _save_csv(tmp_path / "coords.csv", np.zeros((3, 3)))
    _save_csv(tmp_path / "connect.csv", np.array(((0.0, 1.0, 2.0),)))
    _save_csv(tmp_path / "field_disp_x.csv", np.zeros((2, 1)))

    with pytest.raises(ValueError, match="Displacement and coordinate"):
        riley.load_sim_csvs(tmp_path)

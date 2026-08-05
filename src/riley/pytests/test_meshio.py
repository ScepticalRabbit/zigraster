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
        orientation=riley.CoordCsvOrientation.coord_major,
    )

    assert coords_loaded.flags.c_contiguous
    np.testing.assert_allclose(coords_loaded, coords)


def test_load_connect_csv_one_based_node_major(tmp_path: Path) -> None:
    connect = np.array(((1, 2, 3), (3, 4, 1)), dtype=np.float64)
    _save_csv(tmp_path / "connect.csv", connect.T)

    connect_loaded = riley.load_connect_csv(
        tmp_path / "connect.csv",
        orientation=riley.ConnectCsvOrientation.node_major,
        indexing=riley.ConnectIndexing.one_based,
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

    coords_loaded, connect_loaded, uvs_loaded, disp_loaded = riley.load_sim_csvs(
        tmp_path,
    )

    np.testing.assert_allclose(coords_loaded, coords)
    np.testing.assert_array_equal(connect_loaded, connect.astype(np.uintp))
    np.testing.assert_allclose(uvs_loaded, uvs)
    assert disp_loaded is not None
    assert disp_loaded.shape == (1, 3, 3)

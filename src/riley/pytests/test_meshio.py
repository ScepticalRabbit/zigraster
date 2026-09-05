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


def test_load_exodus_platehole() -> None:
    path = riley.data.platehole_exodus_path()
    sim = riley.load_exodus(
        path,
        connect_keys=("connect1",),
        nodal_keys=("stress_yy", "vonmises_stress"),
    )

    assert sim.coords.shape == (4032, 3)
    assert sim.coords.dtype == np.float64
    assert sim.coords.flags.c_contiguous

    assert "connect1" in sim.connect
    assert sim.connect["connect1"].shape == (672, 20)
    assert sim.connect["connect1"].dtype == np.int64
    assert sim.elem_types["connect1"] is riley.EElemType.HEX20

    assert sim.disp is not None
    assert len(sim.disp) == 3
    for component in sim.disp:
        assert component.shape == (4032, 64)
        assert component.dtype == np.float64

    assert sim.time is not None
    assert sim.time.shape == (64,)

    assert "stress_yy" in sim.nodal_vars
    assert "vonmises_stress" in sim.nodal_vars
    assert sim.nodal_vars["stress_yy"].shape == (4032, 64)
    assert sim.nodal_vars["vonmises_stress"].shape == (4032, 64)


def test_load_exodus_platehole_nodal_keys_all() -> None:
    path = riley.data.platehole_exodus_path()
    sim = riley.load_exodus(path, nodal_keys="all")
    expected_keys = {
        "disp_x", "disp_y", "disp_z", "stress_yy", "vonmises_stress",
    }
    assert expected_keys.issubset(set(sim.nodal_vars.keys()))


@pytest.mark.parametrize(
    ("case_name", "expected_elem_type", "expected_node_count"),
    (
        ("tet4", riley.EElemType.TET4, 4),
        ("tet10", riley.EElemType.TET10, 10),
        ("hex8", riley.EElemType.HEX8, 8),
        ("hex20", riley.EElemType.HEX20, 20),
        ("hex27", riley.EElemType.HEX27, 27),
    ),
)
def test_load_exodus_cube_fixtures(
    case_name: str,
    expected_elem_type: riley.EElemType,
    expected_node_count: int,
) -> None:
    exo_path = riley.data.cube_exodus_path(case_name)
    csv_path = riley.data.cube_case_path(case_name)

    sim = riley.load_exodus(
        exo_path,
        nodal_keys=("temperature", "strain_xx", "strain_yy", "strain_zz"),
    )

    csv_coords = riley.load_csv(csv_path / "coords.csv")
    csv_connect = riley.load_csv(csv_path / "connectivity.csv", dtype=np.int64)

    converted = riley.convert_mesh(
        sim.coords,
        sim.connect["connect1"],
        riley.ConnectConvention(
            elem_type=expected_elem_type,
            elem_axis=riley.EConnectAxis.ROW,
            index_base=1,
            node_order=riley.ENodeOrder.EXODUS,
        ),
    )

    riley.verify_mesh(converted)
    np.testing.assert_allclose(converted.coords, csv_coords)
    if case_name != "hex27":
        np.testing.assert_array_equal(converted.connect, csv_connect)

    assert sim.elem_types["connect1"] is expected_elem_type
    assert sim.connect["connect1"].shape[1] == expected_node_count

    assert sim.disp is not None
    assert len(sim.disp) == 3
    num_nodes = sim.coords.shape[0]
    for comp in sim.disp:
        assert comp.shape == (num_nodes, 21)

    assert sim.time is not None
    assert sim.time.shape == (21,)

    assert sim.nodal_vars["temperature"].shape == (num_nodes, 21)
    assert sim.nodal_vars["strain_xx"].shape == (num_nodes, 21)
    assert sim.nodal_vars["strain_yy"].shape == (num_nodes, 21)
    assert sim.nodal_vars["strain_zz"].shape == (num_nodes, 21)


@pytest.mark.parametrize(
    ("type_str", "node_count", "expected"),
    (
        ("HEX", 8, riley.EElemType.HEX8),
        ("HEX", 20, riley.EElemType.HEX20),
        ("HEX", 27, riley.EElemType.HEX27),
        ("HEX8", 8, riley.EElemType.HEX8),
        ("HEX20", 20, riley.EElemType.HEX20),
        ("HEX27", 27, riley.EElemType.HEX27),
        ("TETRA", 4, riley.EElemType.TET4),
        ("TETRA", 10, riley.EElemType.TET10),
        ("TET4", 4, riley.EElemType.TET4),
        ("TET10", 10, riley.EElemType.TET10),
        ("TETRA4", 4, riley.EElemType.TET4),
        ("TETRA10", 10, riley.EElemType.TET10),
        ("QUAD", 4, riley.EElemType.QUAD4),
        ("QUAD", 8, riley.EElemType.QUAD8),
        ("QUAD", 9, riley.EElemType.QUAD9),
        ("QUAD4", 4, riley.EElemType.QUAD4),
        ("QUAD8", 8, riley.EElemType.QUAD8),
        ("QUAD9", 9, riley.EElemType.QUAD9),
        ("TRI", 3, riley.EElemType.TRI3),
        ("TRI", 6, riley.EElemType.TRI6),
        ("TRI", 7, riley.EElemType.TRI7),
        ("TRIANGLE", 3, riley.EElemType.TRI3),
        ("TRIANGLE", 6, riley.EElemType.TRI6),
        ("TRIANGLE", 7, riley.EElemType.TRI7),
        ("TRI3", 3, riley.EElemType.TRI3),
        ("TRI6", 6, riley.EElemType.TRI6),
        ("TRI7", 7, riley.EElemType.TRI7),
        (None, 20, riley.EElemType.HEX20),
        (None, 27, riley.EElemType.HEX27),
        (None, 10, riley.EElemType.TET10),
        (None, 6, riley.EElemType.TRI6),
        (None, 7, riley.EElemType.TRI7),
        (None, 9, riley.EElemType.QUAD9),
        (None, 3, riley.EElemType.TRI3),
    ),
)
def test_parse_exodus_elem_type_valid(
    type_str: str | None,
    node_count: int,
    expected: riley.EElemType,
) -> None:
    result = riley.parse_exodus_elem_type(type_str, node_count)
    assert result is expected


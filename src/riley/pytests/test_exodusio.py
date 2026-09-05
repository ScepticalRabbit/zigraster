from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

import riley
from riley.python import exodusio


@pytest.mark.parametrize(
    ("type_name", "node_count", "expected"),
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
    ),
)
def test_parse_exodus_elem_type(
    type_name: str,
    node_count: int,
    expected: riley.EElemType,
) -> None:
    assert riley.parse_exodus_elem_type(type_name, node_count) is expected


@pytest.mark.parametrize(
    ("type_name", "node_count", "message"),
    (
        ("HEX8", 0, "Invalid node count"),
        ("HEX8", -5, "Invalid node count"),
        ("UNKNOWN_ELEM", 8, "Cannot determine EElemType"),
        ("HEX20", 8, "Cannot determine EElemType"),
        (None, 8, "element type attribute is missing"),
        (None, 20, "element type attribute is missing"),
    ),
)
def test_parse_exodus_elem_type_rejects_invalid_input(
    type_name: str | None,
    node_count: int,
    message: str,
) -> None:
    with pytest.raises(exodusio.ExodusError, match=message):
        exodusio.parse_exodus_elem_type(type_name, node_count)


def test_load_exodus_platehole() -> None:
    sim = riley.load_exodus(
        riley.data.platehole_exodus_path(),
        disp_keys=("disp_x", "disp_y", "disp_z"),
        nodal_keys=("stress_yy", "vonmises_stress"),
    )
    block = sim.blocks["connect1"]
    assert sim.coords.shape == (4032, 3)
    assert sim.coords.dtype == np.float64
    assert sim.coords.flags.c_contiguous
    assert block.connect.shape == (672, 20)
    assert block.connect.dtype == np.int64
    assert block.elem_type is riley.EElemType.HEX20
    assert sim.disp is not None
    assert len(sim.disp) == 3
    assert all(component.shape == (4032, 64) for component in sim.disp)
    assert sim.time is not None
    assert sim.time.shape == (64,)
    assert sim.nodal_vars["stress_yy"].shape == (4032, 64)
    assert sim.nodal_vars["vonmises_stress"].shape == (4032, 64)


def test_load_exodus_nodal_keys_all() -> None:
    sim = riley.load_exodus(
        riley.data.platehole_exodus_path(), nodal_keys="all"
    )
    expected = {
        "disp_x", "disp_y", "disp_z", "stress_yy", "vonmises_stress",
    }
    assert expected.issubset(sim.nodal_vars)


def test_load_exodus_defaults_to_no_displacement() -> None:
    sim = riley.load_exodus(riley.data.platehole_exodus_path())
    assert sim.disp is None


@pytest.mark.parametrize("disp_count", (1, 2, 3))
def test_load_exodus_accepts_up_to_three_displacement_components(
    disp_count: int,
) -> None:
    disp_names = ("disp_x", "disp_y", "disp_z")[:disp_count]
    sim = riley.load_exodus(
        riley.data.platehole_exodus_path(), disp_keys=disp_names
    )
    assert sim.disp is not None
    assert len(sim.disp) == disp_count


@pytest.mark.parametrize(
    ("case_name", "elem_type", "node_count"),
    (
        ("tet4", riley.EElemType.TET4, 4),
        ("tet10", riley.EElemType.TET10, 10),
        ("hex8", riley.EElemType.HEX8, 8),
        ("hex20", riley.EElemType.HEX20, 20),
        ("hex27", riley.EElemType.HEX27, 27),
    ),
)
def test_load_exodus_cube_and_convert_mesh(
    case_name: str,
    elem_type: riley.EElemType,
    node_count: int,
) -> None:
    sim = riley.load_exodus(
        riley.data.cube_exodus_path(case_name),
        disp_keys=("disp_x", "disp_y", "disp_z"),
        nodal_keys=("temperature", "strain_xx", "strain_yy", "strain_zz"),
    )
    block = sim.blocks["connect1"]
    converted = riley.convert_mesh(
        sim.coords,
        block.connect,
        riley.ConnectConvention(
            elem_type, riley.EConnectAxis.ROW, 1, riley.ENodeOrder.EXODUS
        ),
    )
    riley.verify_mesh(converted)
    assert block.elem_type is elem_type
    assert block.connect.shape[1] == node_count
    assert sim.disp is not None
    assert all(field.shape[1] == 21 for field in sim.disp)
    assert all(field.shape[1] == 21 for field in sim.nodal_vars.values())


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"connect_keys": ()}, "connect_keys must not be empty"),
        ({"connect_keys": ("connect1", "connect1")}, "duplicate keys"),
        ({"disp_keys": ("x", "y", "z", "w")}, "at most 3 keys"),
        ({"nodal_keys": ("",)}, "non-empty strings"),
    ),
)
def test_load_exodus_rejects_invalid_options(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(exodusio.ExodusError, match=message):
        exodusio.load_exodus(riley.data.platehole_exodus_path(), **options)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"connect_keys": "connect99"}, "missing connectivity tables"),
        ({"nodal_keys": "unknown_var"}, "missing nodal variables"),
        ({"disp_keys": ("u_x", "u_y")}, "missing nodal variables"),
    ),
)
def test_load_exodus_rejects_missing_variables(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(exodusio.ExodusError, match=message):
        exodusio.load_exodus(riley.data.platehole_exodus_path(), **options)


def _write_minimal_exodus(
    path: Path,
    *,
    connect_dtype: str = "i4",
    connect_values: tuple[int | float, ...] = (1, 2, 3),
    elem_type: str | None = "TRI3",
) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("num_nodes", 3)
        dataset.createDimension("num_dim", 2)
        dataset.createDimension("num_elem", 1)
        dataset.createDimension("num_nodes_per_elem", 3)
        coord = dataset.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        coord[:] = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        connect = dataset.createVariable(
            "connect1", connect_dtype, ("num_elem", "num_nodes_per_elem")
        )
        connect[:] = connect_values
        if elem_type is not None:
            connect.elem_type = elem_type


def test_load_exodus_rejects_float_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "float_connect.e"
    _write_minimal_exodus(path, connect_dtype="f8")
    with pytest.raises(exodusio.ExodusError, match="integer dtype"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_invalid_connectivity_index(tmp_path: Path) -> None:
    path = tmp_path / "invalid_connect.e"
    _write_minimal_exodus(path, connect_values=(0, 2, 3))
    with pytest.raises(exodusio.ExodusError, match="invalid node index"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_missing_element_type(tmp_path: Path) -> None:
    path = tmp_path / "missing_type.e"
    _write_minimal_exodus(path, elem_type=None)
    with pytest.raises(exodusio.ExodusError, match="attribute is missing"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_missing_file() -> None:
    with pytest.raises(exodusio.ExodusError, match="Exodus file not found"):
        exodusio.load_exodus("non_existent_file.e")

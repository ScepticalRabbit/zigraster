from __future__ import annotations

import numpy as np
import pytest

import riley


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
    assert all(field.shape[1] == 5 for field in sim.disp)
    assert all(field.shape[1] == 5 for field in sim.nodal_vars.values())

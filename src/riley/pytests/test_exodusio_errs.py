from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import riley
from riley.pytests.common import write_minimal_exodus
from riley.python import exodusio


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


def test_load_exodus_rejects_float_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "float_connect.e"
    write_minimal_exodus(path, connect_dtype="f8")
    with pytest.raises(exodusio.ExodusError, match="integer dtype"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_invalid_connectivity_index(tmp_path: Path) -> None:
    path = tmp_path / "invalid_connect.e"
    write_minimal_exodus(path, connect_values=(0, 2, 3))
    with pytest.raises(exodusio.ExodusError, match="invalid node index"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_missing_element_type(tmp_path: Path) -> None:
    path = tmp_path / "missing_type.e"
    write_minimal_exodus(path, elem_type=None)
    with pytest.raises(exodusio.ExodusError, match="attribute is missing"):
        exodusio.load_exodus(path)


def test_load_exodus_rejects_missing_file() -> None:
    with pytest.raises(exodusio.ExodusError, match="Exodus file not found"):
        exodusio.load_exodus("non_existent_file.e")

"""Exodus II netCDF loader and container for simulation data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import netCDF4
import numpy as np

from riley.python.meshconstants import (
    EXODUS_AMBIGUOUS_TYPE_MAP,
    EXODUS_ELEM_TYPE_STR_MAP,
    RILEY_ELEM_TOP_MAP,
)
from riley.python.meshconv import EElemType, MeshError


@dataclass(frozen=True, slots=True)
class ExodusSim:
    """Simulation dataset loaded from an Exodus II netCDF file."""

    coords: np.ndarray
    connect: dict[str, np.ndarray]
    elem_types: dict[str, EElemType]
    disp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    nodal_vars: dict[str, np.ndarray] = field(default_factory=dict)
    time: np.ndarray | None = None


def parse_exodus_elem_type(
    type_str: str | None,
    node_count: int,
) -> EElemType:
    """Parse an Exodus element type attribute and node count into EElemType.

    Parameters
    ----------
    type_str : str | None
        Element type string from the Exodus variable attribute.
    node_count : int
        Number of nodes per element (column width of connectivity table).

    Returns
    -------
    EElemType
        Matching Riley element type.
    """
    if node_count <= 0:
        raise ValueError(f"Invalid node count: {node_count}.")

    clean_str = (
        type_str.strip().upper()
        if type_str is not None and isinstance(type_str, str)
        else ""
    )

    if (clean_str, node_count) in EXODUS_AMBIGUOUS_TYPE_MAP:
        return EXODUS_AMBIGUOUS_TYPE_MAP[(clean_str, node_count)]

    if clean_str in EXODUS_ELEM_TYPE_STR_MAP:
        candidate = EXODUS_ELEM_TYPE_STR_MAP[clean_str]
        if RILEY_ELEM_TOP_MAP[candidate].node_count == node_count:
            return candidate

    if not clean_str:
        if node_count == 20:
            return EElemType.HEX20
        if node_count == 27:
            return EElemType.HEX27
        if node_count == 10:
            return EElemType.TET10
        if node_count == 6:
            return EElemType.TRI6
        if node_count == 7:
            return EElemType.TRI7
        if node_count == 9:
            return EElemType.QUAD9
        if node_count == 3:
            return EElemType.TRI3

    raise MeshError(
        f"Cannot determine EElemType for Exodus element '{clean_str}' "
        f"with {node_count} nodes per element."
    )


def _read_exodus_names(var: netCDF4.Variable) -> list[str]:
    raw = var[:]
    names: list[str] = []
    for row in raw:
        chars: list[str] = []
        for char in row:
            if isinstance(char, bytes):
                chars.append(char.decode("latin1", errors="replace"))
            elif isinstance(char, str):
                chars.append(char)
        names.append("".join(chars).strip())
    return names


def load_exodus(
    path: str | Path,
    connect_keys: Sequence[str] = ("connect1",),
    disp_keys: Sequence[str] | None = ("disp_x", "disp_y", "disp_z"),
    nodal_keys: Sequence[str] | str | None = None,
) -> ExodusSim:
    """Load mesh geometry and nodal simulation fields from an Exodus file.

    Parameters
    ----------
    path : str | Path
        Path to the Exodus II netCDF dataset.
    connect_keys : Sequence[str], optional
        Connectivity block variable names to load. Default is ("connect1",).
    disp_keys : Sequence[str] | None, optional
        Nodal variable names corresponding to (ux, uy, uz) displacements.
        Default is ("disp_x", "disp_y", "disp_z"). Set to None to skip.
    nodal_keys : Sequence[str] | str | None, optional
        Additional nodal variable names to load into `nodal_vars`.
        Can be a sequence of variable names, "all" to load all nodal fields,
        or None.

    Returns
    -------
    ExodusSim
        Frozen container holding coordinates, element blocks, and fields.
    """
    exodus_file = Path(path)
    if not exodus_file.is_file():
        raise FileNotFoundError(f"Exodus file not found: {exodus_file}")

    if isinstance(connect_keys, str):
        connect_keys = (connect_keys,)

    with netCDF4.Dataset(str(exodus_file), mode="r") as dataset:
        if "coordx" in dataset.variables and "coordy" in dataset.variables:
            cx = dataset.variables["coordx"][:]
            cy = dataset.variables["coordy"][:]
            if "coordz" in dataset.variables:
                cz = dataset.variables["coordz"][:]
                coords_raw = np.column_stack((cx, cy, cz))
            else:
                coords_raw = np.column_stack((cx, cy, np.zeros_like(cx)))
        elif "coord" in dataset.variables:
            raw_coord = dataset.variables["coord"][:]
            if raw_coord.shape[0] == 3:
                coords_raw = raw_coord.T
            elif raw_coord.shape[0] == 2:
                coords_raw = np.column_stack(
                    (raw_coord[0], raw_coord[1], np.zeros_like(raw_coord[0]))
                )
            else:
                raise ValueError(
                    f"Unexpected coordinate dimension: {raw_coord.shape[0]}."
                )
        else:
            raise ValueError(
                f"Exodus file '{exodus_file}' contains no coordinates."
            )

        coords = np.ascontiguousarray(coords_raw, dtype=np.float64)

        connect_dict: dict[str, np.ndarray] = {}
        elem_types_dict: dict[str, EElemType] = {}

        for key in connect_keys:
            if key not in dataset.variables:
                raise KeyError(
                    f"Connectivity table '{key}' not found in '{exodus_file}'."
                )
            var = dataset.variables[key]
            conn_raw = np.asarray(var[:], dtype=np.int64)
            if conn_raw.ndim != 2:
                raise ValueError(
                    f"Connectivity table '{key}' must be two-dimensional."
                )
            elem_type_attr = getattr(var, "elem_type", None)
            parsed_elem = parse_exodus_elem_type(
                elem_type_attr, conn_raw.shape[1]
            )
            connect_dict[key] = np.ascontiguousarray(conn_raw)
            elem_types_dict[key] = parsed_elem

        name_to_idx: dict[str, int] = {}
        if "name_nod_var" in dataset.variables:
            var_names = _read_exodus_names(dataset.variables["name_nod_var"])
            for idx, name in enumerate(var_names, start=1):
                name_to_idx[name] = idx

        disp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if disp_keys is not None:
            if len(disp_keys) not in (2, 3):
                raise ValueError("disp_keys must contain 2 or 3 keys.")
            disp_is_default = tuple(disp_keys) == (
                "disp_x", "disp_y", "disp_z"
            )
            all_found = all(k in name_to_idx for k in disp_keys)
            if not all_found:
                if not disp_is_default:
                    missing = [k for k in disp_keys if k not in name_to_idx]
                    raise KeyError(
                        f"Displacement variable(s) {missing} not found."
                    )
            else:
                var_x = dataset.variables[
                    f"vals_nod_var{name_to_idx[disp_keys[0]]}"
                ]
                var_y = dataset.variables[
                    f"vals_nod_var{name_to_idx[disp_keys[1]]}"
                ]
                raw_x = np.asarray(var_x[:], dtype=np.float64)
                raw_y = np.asarray(var_y[:], dtype=np.float64)
                ux = np.ascontiguousarray(
                    raw_x.T if raw_x.ndim == 2 else raw_x[:, None]
                )
                uy = np.ascontiguousarray(
                    raw_y.T if raw_y.ndim == 2 else raw_y[:, None]
                )
                if len(disp_keys) == 3:
                    var_z = dataset.variables[
                        f"vals_nod_var{name_to_idx[disp_keys[2]]}"
                    ]
                    raw_z = np.asarray(var_z[:], dtype=np.float64)
                    uz = np.ascontiguousarray(
                        raw_z.T if raw_z.ndim == 2 else raw_z[:, None]
                    )
                else:
                    uz = np.zeros_like(ux)
                disp = (ux, uy, uz)

        nodal_vars: dict[str, np.ndarray] = {}
        if nodal_keys == "all":
            for name, idx in name_to_idx.items():
                var_data = dataset.variables[f"vals_nod_var{idx}"]
                raw_var = np.asarray(var_data[:], dtype=np.float64)
                nodal_vars[name] = np.ascontiguousarray(
                    raw_var.T if raw_var.ndim == 2 else raw_var[:, None]
                )
        elif nodal_keys is not None:
            if isinstance(nodal_keys, str):
                nodal_keys = (nodal_keys,)
            for key in nodal_keys:
                if key not in name_to_idx:
                    raise KeyError(
                        f"Nodal variable '{key}' not found in '{exodus_file}'."
                    )
                idx = name_to_idx[key]
                var_data = dataset.variables[f"vals_nod_var{idx}"]
                raw_var = np.asarray(var_data[:], dtype=np.float64)
                nodal_vars[key] = np.ascontiguousarray(
                    raw_var.T if raw_var.ndim == 2 else raw_var[:, None]
                )

        time_arr: np.ndarray | None = None
        if "time_whole" in dataset.variables:
            time_arr = np.ascontiguousarray(
                dataset.variables["time_whole"][:], dtype=np.float64
            )

    return ExodusSim(
        coords=coords,
        connect=connect_dict,
        elem_types=elem_types_dict,
        disp=disp,
        nodal_vars=nodal_vars,
        time=time_arr,
    )


__all__ = [
    "ExodusSim",
    "load_exodus",
    "parse_exodus_elem_type",
]

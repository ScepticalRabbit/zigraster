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
from riley.python.meshconv import EElemType


class ExodusError(ValueError):
    pass


@dataclass(slots=True)
class ExodusBlock:
    connect: np.ndarray
    elem_type: EElemType


@dataclass(slots=True)
class ExodusSim:
    coords: np.ndarray
    blocks: dict[str, ExodusBlock]
    disp: tuple[np.ndarray, ...] | None = None
    nodal_vars: dict[str, np.ndarray] = field(default_factory=dict)
    time: np.ndarray | None = None


def parse_exodus_elem_type(
    type_str: str | None,
    node_count: int,
) -> EElemType:
    if node_count <= 0:
        raise ExodusError(f"Invalid node count: {node_count}.")
    if not isinstance(type_str, str) or not type_str.strip():
        raise ExodusError("Exodus element type attribute is missing or empty.")

    type_name = type_str.strip().upper()
    type_key = (type_name, node_count)
    if type_key in EXODUS_AMBIGUOUS_TYPE_MAP:
        return EXODUS_AMBIGUOUS_TYPE_MAP[type_key]

    if type_name in EXODUS_ELEM_TYPE_STR_MAP:
        elem_type = EXODUS_ELEM_TYPE_STR_MAP[type_name]
        if RILEY_ELEM_TOP_MAP[elem_type].node_count == node_count:
            return elem_type

    raise ExodusError(
        f"Cannot determine EElemType for Exodus element '{type_name}' "
        f"with {node_count} nodes per element."
    )


def _normalize_keys(
    keys: Sequence[str] | str,
    option_name: str,
) -> tuple[str, ...]:
    if isinstance(keys, str):
        keys_out = (keys,)
    else:
        try:
            keys_out = tuple(keys)
        except TypeError as error:
            raise ExodusError(
                f"{option_name} must be a string or sequence of strings."
            ) from error

    if not keys_out:
        raise ExodusError(f"{option_name} must not be empty.")
    invalid_keys = [
        key for key in keys_out
        if not isinstance(key, str) or not key.strip()
    ]
    if invalid_keys:
        raise ExodusError(f"{option_name} must contain non-empty strings.")
    if len(set(keys_out)) != len(keys_out):
        raise ExodusError(f"{option_name} must not contain duplicate keys.")
    return keys_out


def _normalize_load_options(
    connect_keys: Sequence[str] | str,
    disp_keys: Sequence[str] | str | None,
    nodal_keys: Sequence[str] | str | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...] | str | None]:
    connect_keys_out = _normalize_keys(connect_keys, "connect_keys")

    disp_keys_out: tuple[str, ...] = ()
    if disp_keys is not None:
        disp_keys_out = _normalize_keys(disp_keys, "disp_keys")
        if len(disp_keys_out) > 3:
            raise ExodusError("disp_keys must contain at most 3 keys.")

    nodal_keys_out: tuple[str, ...] | str | None = None
    if nodal_keys == "all":
        nodal_keys_out = "all"
    elif nodal_keys is not None:
        nodal_keys_out = _normalize_keys(nodal_keys, "nodal_keys")

    return connect_keys_out, disp_keys_out, nodal_keys_out


def _as_unmasked(values: np.ndarray, variable_name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ExodusError(
            f"Exodus variable '{variable_name}' contains masked values."
        )
    return np.asarray(values)


def _read_exodus_names(variable: netCDF4.Variable) -> list[str]:
    names_raw = np.ma.filled(variable[:], b"")
    if names_raw.ndim != 2:
        raise ExodusError("Nodal-variable names must be two-dimensional.")
    names = [str(name).strip() for name in netCDF4.chartostring(names_raw)]
    if any(not name for name in names):
        raise ExodusError("Nodal-variable names must not be empty.")
    if len(set(names)) != len(names):
        raise ExodusError("Nodal-variable names must be unique.")
    return names


def _build_nodal_index(dataset: netCDF4.Dataset) -> dict[str, int]:
    if "name_nod_var" not in dataset.variables:
        return {}
    names = _read_exodus_names(dataset.variables["name_nod_var"])
    return {name: index for index, name in enumerate(names, start=1)}


def _verify_dataset_schema(
    dataset: netCDF4.Dataset,
    exodus_file: Path,
    connect_keys: tuple[str, ...],
    disp_keys: tuple[str, ...],
    nodal_keys: tuple[str, ...] | str | None,
    nodal_index: dict[str, int],
) -> None:
    has_component_coords = all(
        name in dataset.variables for name in ("coordx", "coordy")
    )
    if not has_component_coords and "coord" not in dataset.variables:
        raise ExodusError(
            f"Exodus file '{exodus_file}' contains no coordinates."
        )

    missing_connect = [
        key for key in connect_keys if key not in dataset.variables
    ]
    requested_nodal = list(disp_keys)
    if nodal_keys not in (None, "all"):
        requested_nodal.extend(nodal_keys)
    missing_nodal = [
        name for name in requested_nodal if name not in nodal_index
    ]

    names_with_values = (
        tuple(nodal_index) if nodal_keys == "all" else requested_nodal
    )
    missing_values = []
    for name in names_with_values:
        if name not in nodal_index:
            continue
        value_name = f"vals_nod_var{nodal_index[name]}"
        if value_name not in dataset.variables:
            missing_values.append(value_name)

    schema_errors = []
    if missing_connect:
        schema_errors.append(f"missing connectivity tables {missing_connect}")
    if missing_nodal:
        schema_errors.append(f"missing nodal variables {missing_nodal}")
    if missing_values:
        schema_errors.append(f"missing nodal value arrays {missing_values}")
    if schema_errors:
        raise ExodusError(
            f"Exodus file '{exodus_file}' has an invalid schema: "
            + "; ".join(schema_errors)
            + "."
        )


def _read_coords(dataset: netCDF4.Dataset) -> np.ndarray:
    has_component_coords = all(
        name in dataset.variables for name in ("coordx", "coordy")
    )
    if has_component_coords:
        coord_x = _as_unmasked(dataset.variables["coordx"][:], "coordx")
        coord_y = _as_unmasked(dataset.variables["coordy"][:], "coordy")
        if "coordz" in dataset.variables:
            coord_z = _as_unmasked(dataset.variables["coordz"][:], "coordz")
        else:
            coord_z = np.zeros_like(coord_x)
        components = (coord_x, coord_y, coord_z)
        if any(component.ndim != 1 for component in components):
            raise ExodusError("Coordinate components must be one-dimensional.")
        if any(component.shape != coord_x.shape for component in components):
            raise ExodusError("Coordinate components must have equal lengths.")
        coords_raw = np.column_stack(components)
    else:
        coord_table = _as_unmasked(dataset.variables["coord"][:], "coord")
        valid_shape = (
            coord_table.ndim == 2 and coord_table.shape[0] in (2, 3)
        )
        if not valid_shape:
            raise ExodusError(
                "Exodus coordinate table must have shape (2|3, nodes)."
            )
        coords_raw = coord_table.T
        if coord_table.shape[0] == 2:
            coords_raw = np.column_stack(
                (coords_raw, np.zeros(coords_raw.shape[0]))
            )

    coords = np.ascontiguousarray(coords_raw, dtype=np.float64)
    if coords.shape[0] == 0:
        raise ExodusError("Exodus coordinates must contain at least one node.")
    if not np.all(np.isfinite(coords)):
        raise ExodusError("Exodus coordinates must contain finite values.")
    return coords


def _read_blocks(
    dataset: netCDF4.Dataset,
    connect_keys: tuple[str, ...],
    node_count: int,
) -> dict[str, ExodusBlock]:
    blocks: dict[str, ExodusBlock] = {}
    for key in connect_keys:
        connect_var = dataset.variables[key]
        if not np.issubdtype(connect_var.dtype, np.integer):
            raise ExodusError(
                f"Connectivity table '{key}' must have an integer dtype."
            )
        connect_raw = _as_unmasked(connect_var[:], key)
        if connect_raw.ndim != 2 or connect_raw.shape[0] == 0:
            raise ExodusError(
                f"Connectivity table '{key}' must be a non-empty 2D array."
            )
        invalid_index = np.any(connect_raw < 1) or np.any(
            connect_raw > node_count
        )
        if invalid_index:
            raise ExodusError(
                f"Connectivity table '{key}' contains an invalid node index."
            )
        elem_type_name = getattr(connect_var, "elem_type", None)
        elem_type = parse_exodus_elem_type(
            elem_type_name, connect_raw.shape[1]
        )
        connect = np.ascontiguousarray(connect_raw, dtype=np.int64)
        blocks[key] = ExodusBlock(connect, elem_type)
    return blocks


def _read_nodal_field(
    dataset: netCDF4.Dataset,
    nodal_index: dict[str, int],
    field_name: str,
    node_count: int,
) -> np.ndarray:
    variable_name = f"vals_nod_var{nodal_index[field_name]}"
    field_raw = _as_unmasked(dataset.variables[variable_name][:], variable_name)
    if field_raw.ndim == 1:
        field_raw = field_raw[None, :]
    valid_shape = (
        field_raw.ndim == 2 and field_raw.shape[1] == node_count
    )
    if not valid_shape:
        raise ExodusError(
            f"Nodal variable '{field_name}' must have shape (time, nodes)."
        )
    field = np.ascontiguousarray(field_raw.T, dtype=np.float64)
    if not np.all(np.isfinite(field)):
        raise ExodusError(
            f"Nodal variable '{field_name}' must contain finite values."
        )
    return field


def _read_time(dataset: netCDF4.Dataset) -> np.ndarray | None:
    if "time_whole" not in dataset.variables:
        return None
    time_raw = _as_unmasked(dataset.variables["time_whole"][:], "time_whole")
    if time_raw.ndim != 1:
        raise ExodusError("Exodus time must be one-dimensional.")
    time = np.ascontiguousarray(time_raw, dtype=np.float64)
    if not np.all(np.isfinite(time)):
        raise ExodusError("Exodus time must contain finite values.")
    return time


def _verify_time_dimensions(
    disp: tuple[np.ndarray, ...] | None,
    nodal_vars: dict[str, np.ndarray],
    time: np.ndarray | None,
) -> None:
    fields = list(nodal_vars.values())
    if disp is not None:
        fields.extend(disp)
    if not fields:
        return
    time_counts = {field.shape[1] for field in fields}
    if len(time_counts) != 1:
        raise ExodusError("Nodal variables have inconsistent time axes.")
    if time is not None and time.shape[0] != time_counts.pop():
        raise ExodusError(
            "Exodus time length does not match the nodal-variable time axis."
        )


def load_exodus(
    path: str | Path,
    *,
    connect_keys: Sequence[str] | str = ("connect1",),
    disp_keys: Sequence[str] | str | None = None,
    nodal_keys: Sequence[str] | str | None = None,
) -> ExodusSim:
    connect_keys_in, disp_keys_in, nodal_keys_in = _normalize_load_options(
        connect_keys, disp_keys, nodal_keys
    )
    exodus_file = Path(path)
    if not exodus_file.is_file():
        raise ExodusError(f"Exodus file not found: {exodus_file}")

    try:
        with netCDF4.Dataset(exodus_file, mode="r") as dataset:
            nodal_index = _build_nodal_index(dataset)
            _verify_dataset_schema(
                dataset, exodus_file, connect_keys_in, disp_keys_in,
                nodal_keys_in, nodal_index,
            )
            coords = _read_coords(dataset)
            blocks = _read_blocks(dataset, connect_keys_in, coords.shape[0])
            disp = None
            if disp_keys_in:
                disp = tuple(
                    _read_nodal_field(
                        dataset, nodal_index, name, coords.shape[0]
                    )
                    for name in disp_keys_in
                )

            if nodal_keys_in == "all":
                nodal_names = tuple(nodal_index)
            elif nodal_keys_in is None:
                nodal_names = ()
            else:
                nodal_names = nodal_keys_in
            nodal_vars = {
                name: _read_nodal_field(
                    dataset, nodal_index, name, coords.shape[0]
                )
                for name in nodal_names
            }
            time = _read_time(dataset)
    except ExodusError:
        raise
    except (OSError, RuntimeError) as error:
        raise ExodusError(
            f"Could not read Exodus file '{exodus_file}': {error}"
        ) from error

    _verify_time_dimensions(disp, nodal_vars, time)
    return ExodusSim(coords, blocks, disp, nodal_vars, time)


__all__ = [
    "ExodusBlock",
    "ExodusError",
    "ExodusSim",
    "load_exodus",
    "parse_exodus_elem_type",
]

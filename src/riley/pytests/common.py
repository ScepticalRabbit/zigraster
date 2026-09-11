from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

import riley


def save_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",", fmt="%.8f")


def coords_3d(elem_type: riley.EElemType) -> np.ndarray:
    coords = elem_type.get_para_coords()
    if coords.shape[1] == 2:
        coords = np.column_stack((coords, np.zeros(coords.shape[0])))
    return coords


def function_shader() -> riley.FunctionShader:
    return riley.FunctionShader(riley.FuncShaderBuiltin.constant)


def write_minimal_exodus(
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
        coord = dataset.createVariable(
            "coord", "f8", ("num_dim", "num_nodes")
        )
        coord[:] = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        connect = dataset.createVariable(
            "connect1", connect_dtype, ("num_elem", "num_nodes_per_elem")
        )
        connect[:] = connect_values
        if elem_type is not None:
            connect.elem_type = elem_type

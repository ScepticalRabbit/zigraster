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
from typing import Mapping

import numpy as np

from riley.python.enums import (
    ConnectCsvOrientation,
    ConnectIndexing,
    CoordCsvOrientation,
    FieldCsvOrientation,
)


def _load_csv_matrix(path: str | Path, skip_rows: int) -> np.ndarray:
    matrix = np.loadtxt(
        Path(path),
        delimiter=",",
        dtype=np.float64,
        ndmin=2,
        skiprows=skip_rows,
    )
    return np.asarray(matrix, dtype=np.float64)


def _ensure_contiguous_f64(array_in: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array_in, dtype=np.float64)


def _infer_one_based(connect: np.ndarray) -> bool:
    if connect.size == 0:
        return False
    if np.any(connect == 0):
        return False
    return bool(np.min(connect) >= 1)


def _normalise_point_table(
    matrix: np.ndarray,
    orientation: CoordCsvOrientation,
    output_dims: int,
) -> np.ndarray:
    points = matrix
    if orientation == CoordCsvOrientation.coord_major:
        points = points.T
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point table, got shape {points.shape}.")
    if points.shape[1] > output_dims:
        raise ValueError(
            f"Point table has {points.shape[1]} columns, expected at most "
            f"{output_dims}.",
        )

    points_out = np.zeros((points.shape[0], output_dims), dtype=np.float64)
    points_out[:, :points.shape[1]] = points
    return _ensure_contiguous_f64(points_out)


def load_coord_csv(
    path: str | Path,
    *,
    skip_rows: int = 0,
    orientation: CoordCsvOrientation = CoordCsvOrientation.node_major,
) -> np.ndarray:
    coords_raw = _load_csv_matrix(path, skip_rows)
    return _normalise_point_table(coords_raw, orientation, 3)


def load_connect_csv(
    path: str | Path,
    *,
    skip_rows: int = 0,
    orientation: ConnectCsvOrientation = ConnectCsvOrientation.elem_major,
    indexing: ConnectIndexing = ConnectIndexing.auto,
) -> np.ndarray:
    connect_raw = _load_csv_matrix(path, skip_rows)
    if orientation == ConnectCsvOrientation.node_major:
        connect_raw = connect_raw.T

    connect = np.rint(connect_raw).astype(np.int64, copy=False)
    if indexing == ConnectIndexing.one_based:
        connect = connect - 1
    elif indexing == ConnectIndexing.auto and _infer_one_based(connect):
        connect = connect - 1

    if np.any(connect < 0):
        raise ValueError("Connectivity contains negative node indices.")

    return np.ascontiguousarray(connect, dtype=np.uintp)


def load_field_csv(
    path: str | Path,
    *,
    skip_rows: int = 0,
    orientation: FieldCsvOrientation = FieldCsvOrientation.node_major,
) -> np.ndarray:
    field_raw = _load_csv_matrix(path, skip_rows)
    if orientation == FieldCsvOrientation.node_major:
        field_raw = field_raw.T
    return _ensure_contiguous_f64(field_raw)


def load_field_csvs(
    field_paths: Mapping[str, str | Path],
    *,
    skip_rows: int = 0,
    orientation: FieldCsvOrientation = FieldCsvOrientation.node_major,
) -> dict[str, np.ndarray]:
    fields_out: dict[str, np.ndarray] = {}
    for field_name, field_path in field_paths.items():
        fields_out[field_name] = load_field_csv(
            field_path,
            skip_rows=skip_rows,
            orientation=orientation,
        )
    return fields_out


def load_disp_csvs(
    path_x: str | Path | None,
    path_y: str | Path | None,
    path_z: str | Path | None,
    *,
    skip_rows: int = 0,
    orientation: FieldCsvOrientation = FieldCsvOrientation.node_major,
) -> np.ndarray | None:
    disp_paths = {
        axis_name: axis_path
        for axis_name, axis_path in (
            ("x", path_x),
            ("y", path_y),
            ("z", path_z),
        )
        if axis_path is not None and Path(axis_path).is_file()
    }
    if not disp_paths:
        return None

    disp_fields = load_field_csvs(
        disp_paths,
        skip_rows=skip_rows,
        orientation=orientation,
    )
    disp_shape = next(iter(disp_fields.values())).shape
    disp = np.zeros((disp_shape[0], disp_shape[1], 3), dtype=np.float64)

    axis_inds = {"x": 0, "y": 1, "z": 2}
    for axis_name, values in disp_fields.items():
        if values.shape != disp_shape:
            raise ValueError("All displacement CSVs must have the same shape.")
        disp[:, :, axis_inds[axis_name]] = values

    return _ensure_contiguous_f64(disp)


def load_sim_csvs(
    data_dir: str | Path,
    *,
    coords_name: str = "coords.csv",
    connect_name: str = "connect.csv",
    uvs_name: str = "uvs.csv",
    disp_x_name: str = "field_disp_x.csv",
    disp_y_name: str = "field_disp_y.csv",
    disp_z_name: str = "field_disp_z.csv",
    skip_rows: int = 0,
    coord_orientation: CoordCsvOrientation = CoordCsvOrientation.node_major,
    connect_orientation: ConnectCsvOrientation = ConnectCsvOrientation.elem_major,
    connect_indexing: ConnectIndexing = ConnectIndexing.auto,
    uv_orientation: CoordCsvOrientation = CoordCsvOrientation.node_major,
    field_orientation: FieldCsvOrientation = FieldCsvOrientation.node_major,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    data_path = Path(data_dir)

    coords = load_coord_csv(
        data_path / coords_name,
        skip_rows=skip_rows,
        orientation=coord_orientation,
    )
    connect = load_connect_csv(
        data_path / connect_name,
        skip_rows=skip_rows,
        orientation=connect_orientation,
        indexing=connect_indexing,
    )

    uvs: np.ndarray | None = None
    uvs_path = data_path / uvs_name
    if uvs_path.is_file():
        uvs_raw = _load_csv_matrix(uvs_path, skip_rows)
        uvs = _normalise_point_table(uvs_raw, uv_orientation, 2)[:, :2]

    disp = load_disp_csvs(
        data_path / disp_x_name,
        data_path / disp_y_name,
        data_path / disp_z_name,
        skip_rows=skip_rows,
        orientation=field_orientation,
    )

    return coords, connect, uvs, disp


__all__ = [
    "load_connect_csv",
    "load_coord_csv",
    "load_disp_csvs",
    "load_field_csv",
    "load_field_csvs",
    "load_sim_csvs",
]

# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Load Riley mesh and field arrays from CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from riley.python._verifio import _validate_finite_f64


class ECsvOrient(Enum):
    """Supported row semantics for simulation CSV tables."""
    NODE_MAJOR = "node_major"
    COORD_MAJOR = "coord_major"
    ELEM_MAJOR = "elem_major"
    FRAME_MAJOR = "frame_major"


class EConnectIndexing(Enum):
    """Connectivity index convention."""
    AUTO = "auto"
    ZERO_BASED = "zero_based"
    ONE_BASED = "one_based"


@dataclass(frozen=True, slots=True)
class SimCsvData:
    """Simulation arrays loaded from a directory of CSV files.

    The object remains iterable for compatibility with tuple unpacking.
    """
    coords: np.ndarray
    connect: np.ndarray
    uvs: np.ndarray | None
    disp: np.ndarray | None

    def __iter__(self) -> Iterator[np.ndarray | None]:
        """Iterate over arrays in the legacy return order."""
        yield self.coords
        yield self.connect
        yield self.uvs
        yield self.disp


def _load_csv_matrix(path: str | Path, skip_rows: int) -> np.ndarray:
    """Load a finite, two-dimensional floating-point CSV table."""
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative.")

    matrix = np.loadtxt(
        Path(path), delimiter=",", dtype=np.float64, ndmin=2,
        skiprows=skip_rows,
    )

    return _validate_finite_f64(matrix, f"CSV table '{path}'")


def _infer_one_based(connect: np.ndarray, node_count: int | None) -> bool:
    """Infer one-based indexing when the evidence is conclusive."""
    if connect.size == 0 or np.any(connect == 0):
        return False

    if node_count is not None:
        return bool(np.max(connect) == node_count)

    raise ValueError(
        "AUTO connectivity indexing is ambiguous without a node count; "
        "select ZERO_BASED or ONE_BASED explicitly.",
    )


def _normalise_point_table(
    matrix: np.ndarray,
    orient: ECsvOrient,
    output_dims: int,
) -> np.ndarray:
    """Orient and zero-pad a point table to the requested dimensions."""

    if orient is ECsvOrient.COORD_MAJOR:
        points = matrix.T
    elif orient is ECsvOrient.NODE_MAJOR:
        points = matrix
    else:
        raise ValueError(f"Unsupported point CSV orientation: {orient}.")

    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("Point table must not be empty.")

    if points.shape[1] > output_dims:
        raise ValueError(
            f"Point table has {points.shape[1]} columns, expected at most "
            f"{output_dims}.",
        )

    points_out = np.zeros((points.shape[0], output_dims), dtype=np.float64)
    points_out[:, : points.shape[1]] = points

    return points_out


def load_coord_csv(
    path: str | Path,
    skip_rows: int = 0,
    orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
) -> np.ndarray:
    """Load coordinates as a contiguous ``(nodes, 3)`` array."""
    return _normalise_point_table(
        _load_csv_matrix(path, skip_rows), orient, 3,
    )


def load_connect_csv(
    path: str | Path,
    skip_rows: int = 0,
    orient: ECsvOrient = ECsvOrient.ELEM_MAJOR,
    indexing: EConnectIndexing = EConnectIndexing.AUTO,
    node_count: int | None = None,
) -> np.ndarray:
    """Load connectivity as a contiguous platform-index array.

    ``AUTO`` requires ``node_count`` unless the table contains zero, because a
    positive-only table is otherwise ambiguous.
    """

    connect_raw = _load_csv_matrix(path, skip_rows)
    if orient is ECsvOrient.NODE_MAJOR:
        connect_raw = connect_raw.T
    elif orient is not ECsvOrient.ELEM_MAJOR:
        raise ValueError(f"Unsupported connectivity CSV orientation: {orient}.")

    rounded = np.rint(connect_raw)
    if not np.all(connect_raw == rounded):
        raise ValueError("Connectivity must contain integer node indices.")

    connect = connect_raw.astype(np.int64, copy=False)

    if indexing is EConnectIndexing.ONE_BASED:
        connect = connect - 1
    elif indexing is EConnectIndexing.AUTO:
        if _infer_one_based(connect, node_count):
            connect = connect - 1
    elif indexing is not EConnectIndexing.ZERO_BASED:
        raise ValueError(f"Unsupported connectivity indexing: {indexing}.")

    if np.any(connect < 0):
        raise ValueError("Connectivity contains negative node indices.")

    node_count_invalid = node_count is not None and node_count <= 0

    if node_count is not None:
        node_count_invalid |= np.any(connect >= node_count)

    if node_count_invalid:
        raise ValueError("Connectivity contains an out-of-range node index.")

    return np.ascontiguousarray(connect, dtype=np.uintp)


def load_field_csv(
    path: str | Path,
    skip_rows: int = 0,
    orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
) -> np.ndarray:
    """Load a scalar field as a ``(frames, nodes)`` array."""

    field_raw = _load_csv_matrix(path, skip_rows)
    if orient is ECsvOrient.NODE_MAJOR:
        field_raw = field_raw.T
    elif orient is not ECsvOrient.FRAME_MAJOR:
        raise ValueError(f"Unsupported field CSV orientation: {orient}.")

    return np.ascontiguousarray(field_raw, dtype=np.float64)


def load_field_csvs(
    field_paths: Mapping[str, str | Path],
    skip_rows: int = 0,
    orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
) -> dict[str, np.ndarray]:
    """Load several named scalar fields."""

    fields_out: dict[str, np.ndarray] = {}
    for name, path in field_paths.items():
        fields_out[name] = load_field_csv(path, skip_rows, orient)

    return fields_out


def load_disp_csvs(
    path_x: str | Path | None,
    path_y: str | Path | None,
    path_z: str | Path | None,
    skip_rows: int = 0,
    orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
) -> np.ndarray | None:
    """Load displacement components as ``(frames, nodes, 3)``."""

    paths_in = {"x": path_x, "y": path_y, "z": path_z}
    for axis_name, path in paths_in.items():
        path_exists = True

        if path is not None:
            path_exists = Path(path).is_file()

        if not path_exists:
            raise FileNotFoundError(
                f"Displacement {axis_name}-component CSV not found: {path}",
            )

    disp_paths: dict[str, str | Path] = {}
    for name, path in paths_in.items():
        if path is not None:
            disp_paths[name] = path

    if not disp_paths:
        return None

    fields = load_field_csvs(disp_paths, skip_rows, orient)
    shape = next(iter(fields.values())).shape
    disp = np.zeros((*shape, 3), dtype=np.float64)
    axis_indices = {"x": 0, "y": 1, "z": 2}

    for name, values in fields.items():
        if values.shape != shape:
            raise ValueError("All displacement CSVs must have the same shape.")
        disp[:, :, axis_indices[name]] = values

    return disp


def load_sim_csvs(
    data_dir: str | Path,
    coords_name: str = "coords.csv",
    connect_name: str = "connect.csv",
    uvs_name: str = "uvs.csv",
    disp_x_name: str = "field_disp_x.csv",
    disp_y_name: str = "field_disp_y.csv",
    disp_z_name: str = "field_disp_z.csv",
    skip_rows: int = 0,
    coord_orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
    connect_orient: ECsvOrient = (
        ECsvOrient.ELEM_MAJOR
    ),
    connect_indexing: EConnectIndexing = EConnectIndexing.AUTO,
    uv_orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
    field_orient: ECsvOrient = ECsvOrient.NODE_MAJOR,
) -> SimCsvData:
    """Load and cross-validate a simulation CSV directory."""
    data_path = Path(data_dir)

    coords = load_coord_csv(
        data_path / coords_name, skip_rows, coord_orient,
    )

    connect = load_connect_csv(
        data_path / connect_name, skip_rows, connect_orient,
        connect_indexing, node_count=coords.shape[0],
    )

    uvs = None
    uvs_path = data_path / uvs_name
    if uvs_path.is_file():
        uvs = _normalise_point_table(
            _load_csv_matrix(uvs_path, skip_rows), uv_orient, 2,
        )
        if uvs.shape[0] != coords.shape[0]:
            raise ValueError("UV and coordinate node counts must match.")

    disp_paths_out: list[Path] = []
    for name in (disp_x_name, disp_y_name, disp_z_name):
        disp_paths_out.append(data_path / name)

    disp_paths = tuple(disp_paths_out)
    disp_paths_optional: list[Path | None] = []
    for path in disp_paths:
        disp_paths_optional.append(path if path.is_file() else None)

    disp = load_disp_csvs(
        *disp_paths_optional,
        skip_rows=skip_rows, orient=field_orient,
    )

    if disp is not None and disp.shape[1] != coords.shape[0]:
        raise ValueError("Displacement and coordinate node counts must match.")

    return SimCsvData(coords, connect, uvs, disp)


__all__ = [
    "EConnectIndexing", "ECsvOrient", "SimCsvData", "load_connect_csv",
    "load_coord_csv", "load_disp_csvs", "load_field_csv",
    "load_field_csvs", "load_sim_csvs",
]

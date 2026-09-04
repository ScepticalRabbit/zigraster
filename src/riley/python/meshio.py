# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Parse numeric CSV files into NumPy arrays without mesh operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_csv(
    path: str | Path,
    dtype: npt.DTypeLike = np.float64,
    skip_rows: int = 0,
) -> np.ndarray:
    """Load a numeric CSV file as a contiguous two-dimensional array.

    Parameters
    ----------
    path : str | Path
        CSV file to read.
    dtype : numpy.typing.DTypeLike, optional
        NumPy output data type. The default is ``np.float64``.
    skip_rows : int, optional
        Number of leading rows to skip. The default is zero.

    Returns
    -------
    np.ndarray
        Parsed two-dimensional array. No transpose, index shift, padding,
        mesh verification or field-axis conversion is performed.
    """
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative.")

    array = np.loadtxt(
        Path(path),
        delimiter=",",
        dtype=dtype,
        ndmin=2,
        skiprows=skip_rows,
    )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"CSV table '{path}' contains non-finite values.")
    return np.ascontiguousarray(array)


def load_coord_csv(path: str | Path, skip_rows: int = 0) -> np.ndarray:
    """Load a coordinate CSV exactly as a ``float64`` NumPy array."""
    return load_csv(path, np.float64, skip_rows)


def load_connect_csv(path: str | Path, skip_rows: int = 0) -> np.ndarray:
    """Load a connectivity CSV exactly as an ``int64`` NumPy array."""
    connect_raw = load_csv(path, np.float64, skip_rows)
    connect_rounded = np.rint(connect_raw)
    if not np.array_equal(connect_raw, connect_rounded):
        raise ValueError(
            "Connectivity CSV must contain integer node indices."
        )
    return np.ascontiguousarray(connect_rounded, dtype=np.int64)


def load_field_csv(path: str | Path, skip_rows: int = 0) -> np.ndarray:
    """Load a field CSV exactly as a ``float64`` NumPy array."""
    return load_csv(path, np.float64, skip_rows)


__all__ = [
    "load_connect_csv",
    "load_coord_csv",
    "load_csv",
    "load_field_csv",
]

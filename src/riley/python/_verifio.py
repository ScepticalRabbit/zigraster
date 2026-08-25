# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Shared array verification for Riley's Python IO and geometry tools."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_finite_f64(
    values: object,
    name: str,
    shape: tuple[int, ...] | None = None,
    *,
    contiguous: bool = False,
) -> np.ndarray:
    """Convert values to a finite float64 array of an optional shape."""
    if contiguous:
        values_out = np.ascontiguousarray(values, dtype=np.float64)
    else:
        values_out = np.asarray(values, dtype=np.float64)

    if shape is not None and values_out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    finite = np.isfinite(values_out)
    if not np.all(finite):
        raise ValueError(f"{name} must not contain non-finite values.")

    return values_out


def _validate_coords(
    coords: np.ndarray,
    name: str = "coords",
    *,
    contiguous_f64: bool = False,
) -> np.ndarray:
    """Return a finite, non-empty ``(nodes, 3)`` coordinate array."""
    if contiguous_f64:
        coords_out = np.ascontiguousarray(coords, dtype=np.float64)
    else:
        coords_out = np.asarray(coords)

    coords_are_2d = coords_out.ndim == 2
    coords_have_nodes = coords_are_2d and coords_out.shape[0] > 0
    coords_have_xyz = coords_are_2d and coords_out.shape[1] == 3
    if not coords_have_nodes or not coords_have_xyz:
        raise ValueError(
            f"{name} must have shape (nodes, 3) and not be empty.",
        )

    finite = np.isfinite(coords_out)
    if not np.all(finite):
        raise ValueError(f"{name} must not contain non-finite values.")

    return coords_out


def _validate_vec3(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return a finite float64 vector with three components."""
    return _validate_finite_f64(values, name, (3,))

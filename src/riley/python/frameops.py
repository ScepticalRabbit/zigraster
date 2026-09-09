# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from __future__ import annotations

import numpy as np


def frames_first_last_idxs(frames_num: int) -> np.ndarray:
    """Return first and last frame indices without duplicating one frame.

    Parameters
    ----------
    frames_num : int
        Total number of available frames in the sequence (must be >= 1).

    Returns
    -------
    numpy.ndarray
        1D array of shape `(1,)` if `frames_num == 1`, or shape `(2,)` if
        `frames_num > 1`, with dtype `np.intp` containing the first (0) and
        last (`frames_num - 1`) frame indices.

    Raises
    ------
    ValueError
        If `frames_num < 1`.
    """
    if frames_num < 1:
        raise ValueError("At least one frame is required.")
    if frames_num == 1:
        return np.array((0,), dtype=np.intp)
    return np.array((0, frames_num - 1), dtype=np.intp)


def frames_evenly_spaced_idxs(
    frames_num: int,
    frames_max: int,
) -> np.ndarray:
    """Return at most `frames_max` evenly spaced frame indices.

    Parameters
    ----------
    frames_num : int
        Total number of available frames in the sequence (must be >= 1).
    frames_max : int
        Maximum number of frame indices to select (must be >= 1).

    Returns
    -------
    numpy.ndarray
        1D array of shape `(min(frames_num, frames_max),)` and dtype `np.intp`
        containing evenly spaced integer frame indices spanning from index 0
        to `frames_num - 1`.

    Raises
    ------
    ValueError
        If `frames_num < 1` or `frames_max < 1`.
    """
    if frames_num < 1:
        raise ValueError("At least one frame is required.")
    if frames_max < 1:
        raise ValueError("The frame limit must be positive.")
    selected_num = min(frames_num, frames_max)
    if selected_num == 1:
        return np.array((0,), dtype=np.intp)
    return np.array([
        frame * (frames_num - 1) // (selected_num - 1)
        for frame in range(selected_num)
    ], dtype=np.intp)


def frames_select(
    values: np.ndarray,
    frame_idxs: np.ndarray,
) -> np.ndarray:
    """Return an independent contiguous copy of selected frames along axis 0.

    Parameters
    ----------
    values : numpy.ndarray
        Multi-dimensional array of shape `(T, ...)` containing sequence or
        nodal/field data where the first axis `(T)` corresponds to frame
        or time steps.
    frame_idxs : numpy.ndarray
        1D array of shape `(K,)` and integer dtype containing frame indices
        to extract along axis 0.

    Returns
    -------
    numpy.ndarray
        C-contiguous array of shape `(K, ...)` matching `values.dtype`
        containing the extracted frame subset.

    Raises
    ------
    ValueError
        If `values` contains no frames or `frame_idxs` is not a non-empty
        1D array.
    IndexError
        If any index in `frame_idxs` is negative or exceeds the frame count
        of `values`.
    """
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Frame data must contain at least one frame.")
    if frame_idxs.ndim != 1 or frame_idxs.size < 1:
        raise ValueError("Frame indices must be a non-empty 1D array.")
    if np.any(frame_idxs < 0) or np.any(frame_idxs >= values.shape[0]):
        raise IndexError("A selected frame index is out of bounds.")
    return np.ascontiguousarray(values[frame_idxs])

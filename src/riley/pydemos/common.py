# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def first_last_frame_indices(frames_num: int) -> np.ndarray:
    """Return the first and last frame indices without duplicating one frame."""
    if frames_num < 1:
        raise ValueError("At least one frame is required.")
    if frames_num == 1:
        return np.array((0,), dtype=np.intp)
    return np.array((0, frames_num - 1), dtype=np.intp)


def evenly_spaced_frame_indices(
    frames_num: int,
    frames_max: int,
) -> np.ndarray:
    """Return at most ``frames_max`` indices spanning the full sequence."""
    if frames_num < 1:
        raise ValueError("At least one frame is required.")
    if frames_max < 1:
        raise ValueError("The frame limit must be positive.")

    selected_num = min(frames_num, frames_max)
    if selected_num == 1:
        return np.array((0,), dtype=np.intp)
    return np.array(
        [
            frame * (frames_num - 1) // (selected_num - 1)
            for frame in range(selected_num)
        ],
        dtype=np.intp,
    )


def select_frames(
    values: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Return a contiguous copy of selected frames from an array."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Frame data must contain at least one frame.")
    if frame_indices.ndim != 1 or frame_indices.size < 1:
        raise ValueError("Frame indices must be a non-empty 1D array.")
    if np.any(frame_indices < 0) or np.any(frame_indices >= values.shape[0]):
        raise IndexError("A selected frame index is out of bounds.")
    return np.ascontiguousarray(values[frame_indices])


def make_demo_out_dir(case_name: str, *, clean: bool = True) -> Path:
    out_dir = Path.cwd() / "out-riley-py" / case_name
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

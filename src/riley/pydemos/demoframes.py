"""Small frame-selection utilities shared by Python demonstrations."""

import numpy as np


def first_last_frame_indices(frames_num: int) -> np.ndarray:
    """Return first and last frame indices without duplicating one frame."""
    if frames_num < 1:
        raise ValueError("At least one frame is required.")
    if frames_num == 1:
        return np.array((0,), dtype=np.intp)
    return np.array((0, frames_num - 1), dtype=np.intp)


def evenly_spaced_frame_indices(frames_num: int, frames_max: int) -> np.ndarray:
    """Return at most ``frames_max`` indices spanning the sequence."""
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


def select_frames(values: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    """Return a contiguous copy of selected frames."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Frame data must contain at least one frame.")
    if frame_indices.ndim != 1 or frame_indices.size < 1:
        raise ValueError("Frame indices must be a non-empty 1D array.")
    if np.any(frame_indices < 0) or np.any(frame_indices >= values.shape[0]):
        raise IndexError("A selected frame index is out of bounds.")
    return np.ascontiguousarray(values[frame_indices])

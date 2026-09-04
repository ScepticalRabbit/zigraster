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

from riley.python.meshconv import (
    ConnectConvention,
    EConnectAxis,
    EElementType,
    convert_mesh,
)
from riley.python.meshio import (
    load_connect_csv,
    load_coord_csv,
    load_field_csv,
)


def load_demo_arrays(
    data_dir: Path,
    elem_type: EElementType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load Riley's node-major demonstration files for the renderer."""
    coords = load_coord_csv(data_dir / "coords.csv")
    connect_raw = load_connect_csv(data_dir / "connect.csv")
    convention = ConnectConvention(elem_type, EConnectAxis.ROW, 0)
    mesh = convert_mesh(coords, connect_raw, convention)

    uvs = None
    uvs_path = data_dir / "uvs.csv"
    if uvs_path.is_file():
        uvs = load_coord_csv(uvs_path)

    component_arrays = []
    for axis_name in ("x", "y", "z"):
        field_path = data_dir / f"field_disp_{axis_name}.csv"
        if field_path.is_file():
            component_arrays.append(load_field_csv(field_path).T)
    disp = None
    if component_arrays:
        reference_shape = component_arrays[0].shape
        for component in component_arrays:
            if component.shape != reference_shape:
                raise ValueError(
                    "Demo displacement components must have equal shapes."
                )
        while len(component_arrays) < 3:
            component_arrays.append(np.zeros(reference_shape))
        disp = np.ascontiguousarray(np.stack(component_arrays, axis=2))

    return mesh.coords, mesh.connect, uvs, disp


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

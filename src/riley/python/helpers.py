# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from PIL import Image

if TYPE_CHECKING:
    from riley.cython.riley import RasterConfig, SaveStrategy


def load_texture(texture_path: str | Path) -> np.ndarray:
    """Load an image as a contiguous eight-bit greyscale texture."""
    with Image.open(Path(texture_path)) as image_in:
        image_grey = image_in.convert("L")
        image_u8 = np.asarray(image_grey, dtype=np.uint8)
    return np.ascontiguousarray(image_u8, dtype=np.uint8)


def create_raster_config(
    num_frames: int,
    total_threads: int = 1,
    save_strategy: SaveStrategy | int = 2,
) -> RasterConfig:
    """Create an offline raster configuration balanced over frames."""
    from riley.cython.riley import (
        GeometrySchedulingMode,
        HullMode,
        ImageFormat,
        ImageSaveMode,
        NewtonSeedMode,
        NewtonSeedReuse,
        RasterConfig,
        RenderMode,
        ReportMode,
        SaveStrategy,
        ScaleStrategy,
    )

    if not isinstance(num_frames, Integral) or isinstance(num_frames, bool):
        raise TypeError("num_frames must be an integer.")
    if (
        not isinstance(total_threads, Integral)
        or isinstance(total_threads, bool)
    ):
        raise TypeError("total_threads must be an integer.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if total_threads <= 0:
        raise ValueError("total_threads must be positive.")
    if not isinstance(save_strategy, (int, SaveStrategy)):
        raise TypeError("save_strategy must be a SaveStrategy value.")

    frames_available = int(num_frames)
    total_threads = int(total_threads)
    if total_threads < frames_available:
        render_group_count = total_threads
    else:
        render_group_count = 1
        for group_count in range(1, frames_available + 1):
            if total_threads % group_count == 0:
                render_group_count = group_count
    workers_per_group = total_threads // render_group_count

    return RasterConfig(
        render_mode=RenderMode.offline,
        total_threads=total_threads,
        geom_scheduling_mode=GeometrySchedulingMode.spread,
        max_raster_workers_per_job=workers_per_group,
        save_strategy=SaveStrategy(save_strategy),
        image_save_mode=ImageSaveMode.grey,
        hull_mode=HullMode.on_no_fallback,
        newton_seed_mode=NewtonSeedMode.centroid,
        newton_seed_reuse=NewtonSeedReuse.off,
        report=ReportMode.bench,
        save_format=ImageFormat.bmp,
        save_bits=8,
        save_scaling=ScaleStrategy.auto,
    )

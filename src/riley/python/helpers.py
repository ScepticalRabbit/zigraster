# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
import numpy as np
from pathlib import Path

from PIL import Image


def load_texture(texture_path: str | Path) -> np.ndarray:
    with Image.open(Path(texture_path)) as image_in:
        image_grey = image_in.convert("L")
        image_u8 = np.asarray(image_grey, dtype=np.uint8)
    return np.ascontiguousarray(image_u8, dtype=np.uint8)


def create_raster_config(
    num_frames: int,
    total_threads: int = 1,
    save_strategy: int = 2,  # both
) -> "RasterConfig":
    from riley.cyth.riley import RasterConfig

    total_threads = max(1, int(total_threads))
    frames_available = max(1, int(num_frames))
    if total_threads < frames_available:
        render_group_count = total_threads
    else:
        render_group_count = 1
        for group_count in range(1, frames_available + 1):
            if total_threads % group_count == 0:
                render_group_count = group_count
    workers_per_group = total_threads // render_group_count

    return RasterConfig(
        render_mode=1,  # offline
        total_threads=total_threads,
        geom_scheduling_mode=0,  # spread
        max_raster_workers_per_job=workers_per_group,
        save_strategy=save_strategy,
        image_save_mode=0,  # grey
        hull_mode=1,  # on_no_fallback
        newton_seed_mode=0,  # centroid
        newton_seed_reuse=0,  # off
        report=1,  # bench
        save_format=3,  # bmp
        save_bits=8,
        save_scaling=1,  # auto
    )

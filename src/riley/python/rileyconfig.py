# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Construct common Riley raster configurations."""

from numbers import Integral

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


def create_raster_config(
    num_frames: int,
    total_threads: int = 1,
    save_strategy: SaveStrategy = SaveStrategy.both,
) -> RasterConfig:
    """Create an offline raster configuration balanced over frames.

    Parameters
    ----------
    num_frames : int
        Number of frames that will be rendered.
    total_threads : int, optional
        Total worker threads available. The default is one.
    save_strategy : SaveStrategy, optional
        Destination for rendered images. The default is
        ``SaveStrategy.both``.

    Returns
    -------
    RasterConfig
        Offline configuration with workers balanced across frame groups.
    """
    if not isinstance(num_frames, Integral) or isinstance(num_frames, bool):
        raise TypeError("num_frames must be an integer.")
    if not isinstance(total_threads, Integral) or isinstance(
        total_threads,
        bool,
    ):
        raise TypeError("total_threads must be an integer.")
    if not isinstance(save_strategy, SaveStrategy):
        raise TypeError("save_strategy must be a SaveStrategy member.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if total_threads <= 0:
        raise ValueError("total_threads must be positive.")

    frames_available = int(num_frames)
    threads_available = int(total_threads)
    if threads_available < frames_available:
        render_group_count = threads_available
    else:
        render_group_count = 1
        for group_count in range(1, frames_available + 1):
            if threads_available % group_count == 0:
                render_group_count = group_count

    workers_per_group = threads_available // render_group_count
    return RasterConfig(
        render_mode=RenderMode.offline,
        total_threads=threads_available,
        geom_scheduling_mode=GeometrySchedulingMode.spread,
        max_raster_workers_per_job=workers_per_group,
        save_strategy=save_strategy,
        image_save_mode=ImageSaveMode.grey,
        hull_mode=HullMode.on_no_fallback,
        newton_seed_mode=NewtonSeedMode.centroid,
        newton_seed_reuse=NewtonSeedReuse.off,
        report=ReportMode.bench,
        save_format=ImageFormat.bmp,
        save_bits=8,
        save_scaling=ScaleStrategy.auto,
    )


__all__ = ["create_raster_config"]

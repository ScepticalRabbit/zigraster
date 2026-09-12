# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

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
    ValidateInput,
)


def create_raster_config(
    num_frames: int,
    total_threads: int = 1,
    save_strategy: SaveStrategy = SaveStrategy.both,
    validate_input: ValidateInput = ValidateInput.fast,
) -> RasterConfig:
    """Create an offline RasterConfig balanced across frames and workers.

    Parameters
    ----------
    num_frames : int
        Number of frames to render. Must be a positive integer.
    total_threads : int, default=1
        Total number of worker threads available. Must be positive.
    save_strategy : SaveStrategy, default=SaveStrategy.both
        Strategy for retaining and writing rendered frame buffers.

    Returns
    -------
    RasterConfig
        Configured rasteriser settings.

    Raises
    ------
    TypeError
        If `num_frames` or `total_threads` is not an integer, or
        `save_strategy` is not a `SaveStrategy` member.
    ValueError
        If `num_frames` or `total_threads` is not positive.
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

    if not isinstance(validate_input, ValidateInput):
        raise TypeError("validate_input must be a ValidateInput member.")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if total_threads <= 0:
        raise ValueError("total_threads must be positive.")


    # We get best parallelisation from Riley when 1 thread works on 1 frame
    # so parallelisation over camera and frames is best. If we have only 1
    # frame we put all workers into the raster loop.
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
        validate_input=validate_input,
        report=ReportMode.bench,
        save_format=ImageFormat.bmp,
        save_bits=8,
        save_scaling=ScaleStrategy.auto,
    )


__all__ = ["create_raster_config"]

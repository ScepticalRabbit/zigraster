// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const RenderMode = rastcfg.RenderMode;
const HullMode = rastcfg.HullMode;

pub const REL_TOL: F = if (F == f32) 1.0e-3 else 1e-6;
pub const ABS_TOL: F = if (F == f32) 1.0e-3 else 1e-6;
pub const RENDER_MODE: RenderMode = .in_order;
pub const HULL_MODE: HullMode = .on_no_fallback;
// Includes the caller thread. TOTAL_THREADS = 2 means caller + 1 helper.
pub const TOTAL_THREADS: u16 = 1;
pub const FRAME_BATCH_SIZE_PER_GROUP: u16 = 1;
pub const MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP: u16 = 1;
pub const MAX_GEOM_WORKERS_PER_JOB: u16 = 1;
pub const MAX_RASTER_WORKERS_PER_JOB: u16 = 1;
pub const GEOM_SCHEDULING_MODE: rastcfg.GeometrySchedulingMode = .auto;
pub const TEST_CASE_VERBOSE: bool = false;

pub const RasterConfigMode = enum {
    gold,
    preview,
    testing,
    bench,
};

pub fn getRasterConfig(mode: RasterConfigMode) rastcfg.RasterConfig {
    var config = rastcfg.RasterConfig{
        .render_mode = RENDER_MODE,
        .frame_batch_size_per_group = FRAME_BATCH_SIZE_PER_GROUP,
        .max_geom_jobs_in_flight_per_group = MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP,
        .max_geom_workers_per_job = MAX_GEOM_WORKERS_PER_JOB,
        .geom_scheduling_mode = GEOM_SCHEDULING_MODE,
        .max_raster_workers_per_job = MAX_RASTER_WORKERS_PER_JOB,
        .hull_mode = HULL_MODE,
    };

    switch (mode) {
        .gold, .preview => {
            config.total_threads = 1;
            config.max_geom_workers_per_job = 1;
            config.max_raster_workers_per_job = 1;
            config.max_geom_jobs_in_flight_per_group = 1;
            config.frame_batch_size_per_group = 1;
            config.report = .off;
        },
        .testing => {
            config.total_threads = TOTAL_THREADS;
            config.max_geom_workers_per_job = MAX_GEOM_WORKERS_PER_JOB;
            config.max_raster_workers_per_job = MAX_RASTER_WORKERS_PER_JOB;
            config.max_geom_jobs_in_flight_per_group =
                MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP;
            config.frame_batch_size_per_group = FRAME_BATCH_SIZE_PER_GROUP;
            config.report = .off;
        },
        .bench => {
            config.total_threads = 1;
            config.max_geom_workers_per_job = 1;
            config.max_raster_workers_per_job = 1;
            config.max_geom_jobs_in_flight_per_group = 1;
            config.frame_batch_size_per_group = 1;
            config.report = .bench;
        },
    }

    return config;
}

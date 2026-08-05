// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const iio = @import("imageio.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

// Parallelism convention:
// Let each render group g have W_g work-capable threads, including the caller.
// The configured global active-thread budget is:
//   T_total_max = sum_g W_g
// The configured raster active-thread budget is:
//   T_raster_max = sum_g min(W_g, max_raster_workers_per_job)
// The configured geometry active-thread budget depends on scheduling mode:
//   spread: T_geom_max = sum_g min(
//       W_g,
//       max_geom_jobs_in_flight_per_group * max_geom_workers_per_job,
//   )
//   pack:   T_geom_max = sum_g min(W_g, max_geom_workers_per_job)
// For `.auto`, the mode resolves at runtime from the scene size in
// `scalingpolicy.resolveGeometrySchedulingMode(...)`.
// Compatibility note:
// - `total_threads` below is only the single-render-group wrapper budget
// - render-group topology itself lives outside RasterConfig
// - so for the wrapper path, W_0 = total_threads and T_total_max = total_threads

pub const RasterConfig = struct {
    // Outer scheduling mode for frame-camera jobs.
    render_mode: RenderMode = .in_order,
    // Single-render-group compatibility budget. User-facing thread counts
    // always include the caller thread.
    total_threads: u16 = 1,
    // Maximum number of frame-camera jobs assigned to one render group batch.
    frame_batch_size_per_group: u16 = 1,
    // Maximum number of geometry jobs a render group may have active at once.
    max_geom_jobs_in_flight_per_group: u16 = 1,
    // Maximum number of workers a single geometry job may use internally.
    max_geom_workers_per_job: u16 = 1,
    // Policy for distributing render-group workers across geometry jobs.
    geom_scheduling_mode: GeometrySchedulingMode = .auto,
    // Maximum number of workers the single active raster job in a render group
    // may use.
    max_raster_workers_per_job: u16 = 1,
    save_strategy: SaveStrategy = .memory,
    disk_save_overlap: bool = false,
    image_save_mode: ImageSaveMode = .multifield,
    image_save_opts: []const iio.ImageSaveOpts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .none },
    },
    tile_size_override: ?u16 = null,
    tile_size_min: u16 = 1,
    tile_size_max: u16 = 256,
    // Test/development override for exercising the extended raster domain
    // without changing the camera PSF or resolve operation.
    raster_halo_px_override: ?u16 = null,
    background_value: F = 0.0,
    hull_mode: HullMode = .on_no_fallback,
    newton_seed_mode: NewtonSeedMode = .centroid,
    newton_seed_reuse: NewtonSeedReuse = .off,
    report: ReportMode = .bench,
    full_stats_opts: FullStatsOpts = .{},
    save_frame_buff_count: usize = buildconfig.SaveFrameBuffCount,
};

pub const RenderMode = enum {
    // Preserve timestep order. Geometry/raster work may run in parallel across
    // cameras, but later timesteps do not advance until the current timestep
    // has fully completed.
    in_order,
    // Permit batches of frame-camera jobs to be scheduled without timestep
    // ordering constraints. This is the throughput-oriented mode used by the
    // grouped outer scheduler.
    offline,
};

pub const GeometrySchedulingMode = enum {
    // Prefer many geometry jobs in flight, spreading the render-group workers
    // across jobs before increasing per-job worker count.
    spread,
    // Prefer fewer geometry jobs in flight, packing workers into one job
    // before starting additional geometry jobs.
    pack,
    // Resolve at runtime from the scene size in scalingpolicy.zig:
    // smaller scenes def to spread, larger scenes def to pack.
    auto,
};

pub const SaveStrategy = enum {
    disk,
    memory,
    both,
    none,
};

pub const ImageSaveMode = enum {
    grey,
    rgb,
    multifield,
};

pub const ReportMode = enum {
    off,
    bench,
    full_stats,
};

pub const HullMode = enum {
    off,
    on_no_fallback,
    on_convex_fallback,
};

pub const NewtonSeedMode = enum {
    centroid,
    hull,
};

pub const NewtonSeedReuse = enum {
    off,
    last_conv,
};

pub const FullStatsOpts = struct {
    formats: []const iio.ImageSaveOpts = &[_]iio.ImageSaveOpts{
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
        .{ .format = .csv, .bits = null, .scaling = .none },
    },
    save_solver_csv: bool = false,
    save_iter_map: bool = true,
    save_xi_map: bool = true,
    save_eta_map: bool = true,
    save_conv_map: bool = true,
    save_jac_det_map: bool = true,
    save_tile_timing_map: bool = true,
    save_tile_density_map: bool = true,
    save_tile_occupancy_map: bool = true,
    save_depth_map: bool = true,
    save_earlyout_map: bool = true,
    save_pixel_occupancy_map: bool = true,
    save_normals_map: bool = false,
};

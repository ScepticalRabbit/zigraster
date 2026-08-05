// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const benchargs = @import("dev_support/benchargs.zig");
const benchdicuq = @import("dev_support/benchdicuq.zig");
const common = @import("dev_support/benchcommon.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const riley = @import("riley/zig/riley.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const F = buildconfig.F;

const DEFAULT_OUT_DIR = "out/bench_stats_dicuq";
const DEFAULT_IMAGE_OUT_DIR = "out/bench_images_dicuq";
const DEFAULT_DATA_DIR = "data/FE/platehole3d_6mr_63f/";
const DEFAULT_PIXELS_NUM = [2]u32{ 2464, 2056 };
const DEFAULT_SUB_SAMPLE: u32 = 2;
const DEFAULT_FOCAL_LENG: F = @floatCast(50.0e-3);
const DEFAULT_PIXELS_SIZE = [2]F{
    @floatCast(3.45e-6),
    @floatCast(3.45e-6),
};
const DEFAULT_FOV_SCALE: F = @floatCast(0.65);
const DEFAULT_STEREO_ANG: F = 20.0;
const DEFAULT_TEX_PATH = "texture/speckle.bmp";
const DEFAULT_RENDER_MODE = rastcfg.RenderMode.offline;
const DEFAULT_SAVE_STRATEGY = rastcfg.SaveStrategy.disk;
const DEFAULT_RUNS: usize = 1;
const DEFAULT_RENDER_GROUP_COUNT: u16 = 1;
const DEFAULT_TOTAL_THREADS: u16 = 1;
const DEFAULT_FRAME_BATCH_SIZE_PER_GROUP: u16 = 1;
const DEFAULT_MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP: u16 = 1;
const DEFAULT_MAX_GEOM_WORKERS_PER_JOB: u16 = 1;
const DEFAULT_GEOM_SCHEDULING_MODE = rastcfg.GeometrySchedulingMode.spread;
const DEFAULT_MAX_RASTER_WORKERS_PER_JOB: u16 = 1;

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    const base_raster_config = benchdicuq.getBaseRasterConfig();

    var default_bench_args = benchargs.defaultBenchArgs(
        DEFAULT_OUT_DIR,
        base_raster_config,
    );
    default_bench_args.image_out_dir = DEFAULT_IMAGE_OUT_DIR;
    default_bench_args.render_mode = DEFAULT_RENDER_MODE;
    default_bench_args.save_strategy = DEFAULT_SAVE_STRATEGY;
    default_bench_args.runs = DEFAULT_RUNS;
    default_bench_args.render_group_count = DEFAULT_RENDER_GROUP_COUNT;
    default_bench_args.total_threads = DEFAULT_TOTAL_THREADS;
    default_bench_args.frame_batch_size_per_group =
        DEFAULT_FRAME_BATCH_SIZE_PER_GROUP;
    default_bench_args.max_geom_jobs_in_flight_per_group =
        DEFAULT_MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP;
    default_bench_args.max_geom_workers_per_job =
        DEFAULT_MAX_GEOM_WORKERS_PER_JOB;
    default_bench_args.geom_scheduling_mode =
        DEFAULT_GEOM_SCHEDULING_MODE;
    default_bench_args.max_raster_workers_per_job =
        DEFAULT_MAX_RASTER_WORKERS_PER_JOB;

    default_bench_args.pixels_num = DEFAULT_PIXELS_NUM;
    default_bench_args.sub_sample = DEFAULT_SUB_SAMPLE;

    // Change .save_strategy here to write to disk!
    const bench_args = try benchargs.parseArgsWithDefaults(
        init.minimal.args.vector,
        default_bench_args,
    );
    if (bench_args.total_threads == 0) {
        return error.InvalidTotalThreads;
    }
    if (bench_args.render_group_count == 0) {
        return error.InvalidRenderGroupCount;
    }
    if (bench_args.total_threads < bench_args.render_group_count) {
        return error.InvalidRenderGroupConfiguration;
    }
    if (@rem(bench_args.total_threads, bench_args.render_group_count) != 0) {
        return error.UnevenRenderGroupPartition;
    }

    const workers_per_group =
        @divExact(bench_args.total_threads, bench_args.render_group_count);
    if (workers_per_group == 0) {
        return error.InvalidRenderGroupConfiguration;
    }

    const managed_ios = try outer_alloc.alloc(
        std.Io.Threaded,
        bench_args.render_group_count,
    );
    const managed_save_ios = try outer_alloc.alloc(
        std.Io.Threaded,
        bench_args.render_group_count,
    );
    defer {
        for (managed_save_ios) |*managed_io| {
            managed_io.deinit();
        }
        outer_alloc.free(managed_save_ios);
        for (managed_ios) |*managed_io| {
            managed_io.deinit();
        }
        outer_alloc.free(managed_ios);
    }
    const render_groups = try outer_alloc.alloc(
        riley.RenderGroupSpec,
        bench_args.render_group_count,
    );
    defer outer_alloc.free(render_groups);
    const render_group_workers = try outer_alloc.alloc(
        u16,
        bench_args.render_group_count,
    );
    defer outer_alloc.free(render_group_workers);

    for (0..bench_args.render_group_count) |ii| {
        managed_ios[ii] = riley.getThreadedIo(
            outer_alloc,
            init.minimal,
            workers_per_group,
        );
        managed_save_ios[ii] = riley.getThreadedIo(
            outer_alloc,
            init.minimal,
            1,
        );
        render_groups[ii] = .{
            .io = managed_ios[ii].io(),
            .save_frame_io = managed_save_ios[ii].io(),
            .workers = workers_per_group,
        };
        render_group_workers[ii] = workers_per_group;
    }
    const io = render_groups[0].io;

    const samp_cfg = try benchdicuq.makeSampleConfig(bench_args);

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();
    const dicuq_defaults = benchdicuq.DicuqDefaults{
        .data_dir = DEFAULT_DATA_DIR,
        .pixels_num = bench_args.pixels_num,
        .sub_sample = bench_args.sub_sample,
        .focal_leng = DEFAULT_FOCAL_LENG,
        .pixels_size = DEFAULT_PIXELS_SIZE,
        .fov_scale = DEFAULT_FOV_SCALE,
        .stereo_ang = DEFAULT_STEREO_ANG,
        .tex_path = DEFAULT_TEX_PATH,
    };

    const prepared_benchmark = try benchdicuq.prepareBenchmark(
        aa,
        io,
        dicuq_defaults,
        samp_cfg,
    );

    const raster_config = benchargs.applyRasterConfig(
        base_raster_config,
        bench_args,
    );
    const actual_tile_size = common.calcActualTileSize(
        raster_config,
        bench_args.pixels_num,
        bench_args.sub_sample,
        0,
    );
    try common.writeBenchmarkConfig(
        outer_alloc,
        io,
        bench_args.out_dir,
        bench_args.image_out_dir,
        "bench_dicuq.zig",
        init.minimal.args.vector,
        bench_args.subpixel_center_map,
        raster_config,
        render_group_workers,
        bench_args.pixels_num,
        bench_args.sub_sample,
        bench_args.runs,
        prepared_benchmark.fov_scale,
        actual_tile_size,
    );

    const case_name = try benchdicuq.calcCaseName(outer_alloc, samp_cfg);
    defer outer_alloc.free(case_name);

    std.debug.print(
        "Starting DIC UQ Benchmark ({d}x{d}, {d} runs, {d} total threads, {d} render groups, {d} workers/group)...\n",
        .{
            bench_args.pixels_num[0],
            bench_args.pixels_num[1],
            bench_args.runs,
            bench_args.total_threads,
            bench_args.render_group_count,
            workers_per_group,
        },
    );
    std.debug.print("Case: {s}\n", .{case_name});

    var e2e_rows_by_run = try outer_alloc.alloc(
        []benchdicuq.DicuqE2ERow,
        bench_args.runs,
    );
    var e2e_rows_filled: usize = 0;
    defer {
        for (0..e2e_rows_filled) |rr| {
            outer_alloc.free(e2e_rows_by_run[rr]);
        }
        outer_alloc.free(e2e_rows_by_run);
    }

    for (0..bench_args.runs) |rr| {
        const out_dir_path = if (bench_args.save_strategy == .disk or
            bench_args.save_strategy == .both)
            if (bench_args.image_out_dir.len > 0)
                bench_args.image_out_dir
            else
                bench_args.out_dir
        else
            null;
        var run_result = try benchdicuq.runBenchmark(
            outer_alloc,
            io,
            render_groups,
            &prepared_benchmark.camera_inputs,
            prepared_benchmark.mesh_input,
            raster_config,
            out_dir_path,
        );
        defer run_result.deinit(outer_alloc);

        for (run_result.frame_rows) |*frame_row| {
            frame_row.run_idx = rr;
        }
        for (run_result.e2e_rows) |*e2e_row| {
            e2e_row.run_idx = rr;
        }

        try benchdicuq.writeRunCSVs(
            outer_alloc,
            io,
            bench_args.out_dir,
            case_name,
            rr,
            prepared_benchmark.camera_inputs.len,
            run_result,
        );
        e2e_rows_by_run[rr] = try outer_alloc.dupe(
            benchdicuq.DicuqE2ERow,
            run_result.e2e_rows,
        );
        e2e_rows_filled += 1;
    }

    try benchdicuq.writeE2EOverRunsCSVs(
        outer_alloc,
        io,
        bench_args.out_dir,
        case_name,
        prepared_benchmark.camera_inputs.len,
        e2e_rows_by_run,
    );
}

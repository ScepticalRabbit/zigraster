// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const camera = @import("../riley/zig/camera.zig");
const common = @import("../dev_support/tests.zig");
const gengold_zoo = @import("../gengold/gen_gold_featurezoo.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

pub const ZooThreadingCase = struct {
    name: []const u8,
    render_mode: rastcfg.RenderMode = .in_order,
    buffer_mode: rastcfg.BufferMode = .tile_local,
    max_geom_workers_per_job: u16 = 1,
    max_raster_workers_per_job: u16 = 1,
    frame_batch_size_per_group: u16 = 1,
    workers_per_group: []const u16,
};

pub const zoo_threading_cases = [_]ZooThreadingCase{
    .{
        .name = "1grp_1geom_1rast",
        .workers_per_group = &.{1},
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 1,
        .buffer_mode = .tile_local,
        .render_mode = .in_order,
    },
    .{
        .name = "1grp_4geom_4rast",
        .workers_per_group = &.{4},
        .max_geom_workers_per_job = 4,
        .max_raster_workers_per_job = 4,
        .buffer_mode = .tile_local,
        .render_mode = .in_order,
    },
    .{
        .name = "1grp_1geom_4rast_tilelocal",
        .workers_per_group = &.{4},
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 4,
        .buffer_mode = .tile_local,
        .render_mode = .in_order,
    },
    .{
        .name = "1grp_1geom_4rast_globalsubpx",
        .workers_per_group = &.{4},
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 4,
        .buffer_mode = .global_subpx_full,
        .render_mode = .in_order,
    },
    .{
        .name = "1grp_1geom_4rast_stripe",
        .workers_per_group = &.{4},
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 4,
        .buffer_mode = .global_subpx_stripe,
        .render_mode = .in_order,
    },
    .{
        .name = "2grp_1geom_2rast",
        .workers_per_group = &.{ 2, 2 },
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 2,
        .frame_batch_size_per_group = 2,
        .buffer_mode = .tile_local,
        .render_mode = .in_order,
    },
    .{
        .name = "4grp_1geom_1rast_inorder",
        .workers_per_group = &.{ 1, 1, 1, 1 },
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 1,
        .frame_batch_size_per_group = 2,
        .buffer_mode = .tile_local,
        .render_mode = .in_order,
    },
    .{
        .name = "4grp_1geom_1rast_offline",
        .workers_per_group = &.{ 1, 1, 1, 1 },
        .max_geom_workers_per_job = 1,
        .max_raster_workers_per_job = 1,
        .frame_batch_size_per_group = 2,
        .buffer_mode = .tile_local,
        .render_mode = .offline,
    },
};

pub fn runZooMonoCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    case: ZooThreadingCase,
    texture_grey: texops.Tex(u8, 1),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try gengold_zoo.buildZooScene(
        u8,
        1,
        8,
        aa,
        io,
        texture_grey,
    );
    const cameras = gengold_zoo.buildAllZooCameras(meshes);

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/featurezoo_mono",
        .{gold_dir_root},
    );

    var total_workers_count: u16 = 0;
    for (case.workers_per_group) |workers_count| {
        total_workers_count += workers_count;
    }

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;
    run_config.render_mode = case.render_mode;
    run_config.buffer_mode = case.buffer_mode;
    run_config.total_threads = total_workers_count;
    run_config.max_geom_workers_per_job = case.max_geom_workers_per_job;
    run_config.max_raster_workers_per_job = case.max_raster_workers_per_job;
    run_config.frame_batch_size_per_group = case.frame_batch_size_per_group;

    var render_groups_buf: [8]riley.RenderGroupSpec = undefined;
    for (case.workers_per_group, 0..) |workers_count, ii| {
        render_groups_buf[ii] = .{ .io = io, .workers = workers_count };
    }
    const render_groups = render_groups_buf[0..case.workers_per_group.len];

    const start_time = Timestamp.now(io, .awake);
    const result = try riley.raster(
        aa,
        render_groups,
        &cameras,
        meshes,
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;

    const cameras_num = render_result.dims[0];
    const frames_num = render_result.dims[1];

    for (0..cameras_num) |cc| {
        for (0..frames_num) |ff| {
            const gold_path = try common.findGoldPath(
                aa,
                io,
                gold_dir,
                cc,
                ff,
                0,
                false,
            );

            common.compareNDArrayToGold(
                aa,
                io,
                &render_result,
                cc,
                ff,
                0,
                1,
                gold_path,
                tcfg.REL_TOL,
                tcfg.ABS_TOL,
            ) catch |err| {
                if (tcfg.TEST_CASE_VERBOSE) {
                    std.debug.print(
                        "FAIL featurezoo_mono {s} cam {d} frame {d} ({d:.2} ms)\n",
                        .{ case.name, cc, ff, duration_ms },
                    );
                }
                return err;
            };
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS featurezoo_mono {s} ({d:.2} ms, {d} cams x {d} frames)\n",
            .{ case.name, duration_ms, cameras_num, frames_num },
        );
    }
}

pub fn runZooRgbCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    case: ZooThreadingCase,
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try gengold_zoo.buildZooScene(
        u8,
        3,
        8,
        aa,
        io,
        texture_rgb,
    );
    const all_cameras = gengold_zoo.buildAllZooCameras(meshes);
    const rgb_cameras = [_]CameraInput{ all_cameras[0], all_cameras[1] };

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/featurezoo_rgb",
        .{gold_dir_root},
    );

    var total_workers_count: u16 = 0;
    for (case.workers_per_group) |workers_count| {
        total_workers_count += workers_count;
    }

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;
    run_config.render_mode = case.render_mode;
    run_config.buffer_mode = case.buffer_mode;
    run_config.total_threads = total_workers_count;
    run_config.max_geom_workers_per_job = case.max_geom_workers_per_job;
    run_config.max_raster_workers_per_job = case.max_raster_workers_per_job;
    run_config.frame_batch_size_per_group = case.frame_batch_size_per_group;

    var render_groups_buf: [8]riley.RenderGroupSpec = undefined;
    for (case.workers_per_group, 0..) |workers_count, ii| {
        render_groups_buf[ii] = .{ .io = io, .workers = workers_count };
    }
    const render_groups = render_groups_buf[0..case.workers_per_group.len];

    const start_time = Timestamp.now(io, .awake);
    const result = try riley.raster(
        aa,
        render_groups,
        &rgb_cameras,
        meshes,
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;

    const cameras_num = render_result.dims[0];
    const frames_num = render_result.dims[1];

    for (0..cameras_num) |cc| {
        for (0..frames_num) |ff| {
            for (0..3) |ch| {
                const gold_path = try common.findGoldPath(
                    aa,
                    io,
                    gold_dir,
                    cc,
                    ff,
                    ch,
                    false,
                );

                common.compareNDArrayToGold(
                    aa,
                    io,
                    &render_result,
                    cc,
                    ff,
                    ch,
                    1,
                    gold_path,
                    tcfg.REL_TOL,
                    tcfg.ABS_TOL,
                ) catch |err| {
                    if (tcfg.TEST_CASE_VERBOSE) {
                        std.debug.print(
                            "FAIL featurezoo_rgb {s} cam {d} frame {d} ch {d} ({d:.2} ms)\n",
                            .{ case.name, cc, ff, ch, duration_ms },
                        );
                    }
                    return err;
                };
            }
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS featurezoo_rgb {s} ({d:.2} ms, {d} cams x {d} frames)\n",
            .{ case.name, duration_ms, cameras_num, frames_num },
        );
    }
}

pub fn runZooMonoTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_grey: texops.Tex(u8, 1),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    for (zoo_threading_cases) |case| {
        try runZooMonoCaseTest(
            allocator,
            io,
            case,
            texture_grey,
            gold_dir_root,
            config,
        );
    }
}

pub fn runZooRgbTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_rgb: texops.Tex(u8, 3),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    for (zoo_threading_cases) |case| {
        try runZooRgbCaseTest(
            allocator,
            io,
            case,
            texture_rgb,
            gold_dir_root,
            config,
        );
    }
}

test "Basic Suite: featurezoo cases" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.basic);

    const texture_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speck128_mono_u8.bmp",
        .bmp,
    );
    defer texture_grey.deinit(allocator);

    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speck128_rgb_u8.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    try runZooMonoTest(allocator, io, texture_grey, gold_dir_root, config);
    try runZooRgbTest(allocator, io, texture_rgb, gold_dir_root, config);
}

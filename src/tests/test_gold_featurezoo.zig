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

pub fn runZooMonoTest(
    allocator: std.mem.Allocator,
    io: std.Io,
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

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
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
                        "FAIL featurezoo_mono cam {d} frame {d} ({d:.2} ms)\n",
                        .{ cc, ff, duration_ms },
                    );
                }
                return err;
            };
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS featurezoo_mono ({d:.2} ms, {d} cams x {d} frames)\n",
            .{ duration_ms, cameras_num, frames_num },
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

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
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
                            "FAIL featurezoo_rgb cam {d} frame {d} ch {d} ({d:.2} ms)\n",
                            .{ cc, ff, ch, duration_ms },
                        );
                    }
                    return err;
                };
            }
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS featurezoo_rgb ({d:.2} ms, {d} cams x {d} frames)\n",
            .{ duration_ms, cameras_num, frames_num },
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

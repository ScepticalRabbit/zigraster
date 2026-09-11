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
const common_full = @import("../gengold/gen_gold_full_common.zig");
const gengold_tiling = @import("../gengold/gen_gold_full_tiling.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

pub const FULL_TILING_REL_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;
pub const FULL_TILING_ABS_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;

const tile_sizes = [_]?u16{ null, 8, 16, 32 };
const buffer_modes = [_]rastcfg.BufferMode{
    .tile_local,
    .global_subpx_full,
    .global_subpx_stripe,
};
const worker_counts = [_]u16{ 1, 4 };

fn formatTilingCaseTag(
    allocator: std.mem.Allocator,
    pixel_num: [2]u32,
    ssaa: u8,
    tile_size: ?u16,
    buffer_mode: rastcfg.BufferMode,
    workers: u16,
) ![]const u8 {
    const tile_str = if (tile_size) |ts|
        try std.fmt.allocPrint(allocator, "tile{d}", .{ts})
    else
        try std.fmt.allocPrint(allocator, "tileauto", .{});
    defer allocator.free(tile_str);

    const buf_str = switch (buffer_mode) {
        .tile_local => "tilelocal",
        .global_subpx_full => "globalsubpx",
        .global_subpx_stripe => "globalstripe",
    };

    return std.fmt.allocPrint(
        allocator,
        "res_{d}x{d}_ssaa{d}_{s}_{s}_w{d}",
        .{ pixel_num[0], pixel_num[1], ssaa, tile_str, buf_str, workers },
    );
}

fn runTilingCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    pixel_num: [2]u32,
    ssaa: u8,
    tile_size: ?u16,
    buffer_mode: rastcfg.BufferMode,
    workers: u16,
    meshes: []const MeshInput,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const cam_inp = common_full.createScene2Camera(pixel_num, ssaa);

    const ref_gold_dir_name = try gengold_tiling.formatTilingGoldDirName(
        aa,
        pixel_num,
        ssaa,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, ref_gold_dir_name },
    );

    const case_tag = try formatTilingCaseTag(
        aa,
        pixel_num,
        ssaa,
        tile_size,
        buffer_mode,
        workers,
    );

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.tile_size_override = tile_size;
    run_config.buffer_mode = buffer_mode;
    run_config.total_threads = workers;
    run_config.max_raster_workers_per_job = workers;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = workers },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
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
    ) / 1.0e6;

    const gold_path = try common.findGoldPath(
        aa,
        io,
        gold_dir,
        0,
        0,
        0,
        false,
    );

    common.compareNDArrayToGold(
        aa,
        io,
        &render_result,
        0,
        0,
        0,
        1,
        gold_path,
        FULL_TILING_REL_TOL,
        FULL_TILING_ABS_TOL,
    ) catch |err| {
        const fail_dir_name = try std.fmt.allocPrint(
            aa,
            "full_tiling/{s}",
            .{case_tag},
        );
        try common.saveComparisonArtifactsFromResult(
            aa,
            io,
            common.default_fails_root,
            fail_dir_name,
            &render_result,
            0,
            0,
            0,
            gold_path,
            1,
        );
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print(
                "FAIL {s} ({d:.2} ms)\n",
                .{ case_tag, duration_ms },
            );
        }
        return err;
    };

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({d:.2} ms)\n",
            .{ case_tag, duration_ms },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.testing);
    config.background_value = 127.5;
    const gold_dir_root = policy.goldRoot(.full_tiling);

    var textures = try common_full.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep2 = try common_full.prepareScene2(allocator, io, .tri3);
    defer prep2.deinit(allocator);

    const meshes = common_full.buildScene2Meshes(&prep2, &textures);

    for (gengold_tiling.resolutions) |res| {
        for (gengold_tiling.ssaa_values) |ssaa| {
            for (tile_sizes) |tile_size| {
                for (buffer_modes) |buffer_mode| {
                    for (worker_counts) |workers| {
                        try runTilingCaseTest(
                            allocator,
                            io,
                            res,
                            ssaa,
                            tile_size,
                            buffer_mode,
                            workers,
                            &meshes,
                            gold_dir_root,
                            config,
                        );
                    }
                }
            }
        }
    }

    try runAdditionalTilingAndParityTests(allocator, io, &textures, config);
}

fn runAdditionalTilingAndParityTests(
    allocator: std.mem.Allocator,
    io: std.Io,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    // 1. tri3 vs tri3opt parity across irregular sensor sizes
    const irregular_resolutions = [_][2]u32{
        .{ 153, 97 },
        .{ 117, 83 },
    };

    for (irregular_resolutions) |res| {
        var prep_tri3 = try common_full.prepareScene2(allocator, io, .tri3);
        defer prep_tri3.deinit(allocator);

        var prep_tri3opt = try common_full.prepareScene2(allocator, io, .tri3opt);
        defer prep_tri3opt.deinit(allocator);

        const meshes_tri3 = common_full.buildScene2Meshes(&prep_tri3, textures);
        const meshes_tri3opt = common_full.buildScene2Meshes(&prep_tri3opt, textures);

        const cam_inp = common_full.createScene2Camera(res, 2);

        var run_config = config;
        run_config.save_strategy = .memory;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        var arena_tri3 = std.heap.ArenaAllocator.init(allocator);
        defer arena_tri3.deinit();
        const aa_tri3 = arena_tri3.allocator();

        const result_tri3 = try riley.raster(
            aa_tri3,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_tri3,
            run_config,
            null,
        );
        const img_tri3 = result_tri3 orelse return error.NoResult;

        var arena_opt = std.heap.ArenaAllocator.init(allocator);
        defer arena_opt.deinit();
        const aa_opt = arena_opt.allocator();

        const result_opt = try riley.raster(
            aa_opt,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_tri3opt,
            run_config,
            null,
        );
        const img_opt = result_opt orelse return error.NoResult;

        try std.testing.expectEqualSlices(usize, img_tri3.dims, img_opt.dims);
        for (img_tri3.slice, img_opt.slice) |v_tri3, v_opt| {
            try std.testing.expect(@abs(v_tri3 - v_opt) <= 1.0e-3);
        }
    }

    // 2. SSAA=8 in-memory equivalence between tile_local and global_subpx_full
    {
        var prep_scene2 = try common_full.prepareScene2(allocator, io, .tri3);
        defer prep_scene2.deinit(allocator);
        const meshes = common_full.buildScene2Meshes(&prep_scene2, textures);
        const cam_inp = common_full.createScene2Camera(.{ 128, 80 }, 8);

        var config_tile = config;
        config_tile.save_strategy = .memory;
        config_tile.buffer_mode = .tile_local;

        var config_global = config_tile;
        config_global.buffer_mode = .global_subpx_full;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        var arena_tile = std.heap.ArenaAllocator.init(allocator);
        defer arena_tile.deinit();
        const aa_tile = arena_tile.allocator();

        const res_tile = try riley.raster(
            aa_tile,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes,
            config_tile,
            null,
        );
        const img_tile = res_tile orelse return error.NoResult;

        var arena_global = std.heap.ArenaAllocator.init(allocator);
        defer arena_global.deinit();
        const aa_global = arena_global.allocator();

        const res_global = try riley.raster(
            aa_global,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes,
            config_global,
            null,
        );
        const img_global = res_global orelse return error.NoResult;

        try std.testing.expectEqualSlices(usize, img_tile.dims, img_global.dims);
        for (img_tile.slice, img_global.slice) |v_t, v_g| {
            try std.testing.expect(@abs(v_t - v_g) <= FULL_TILING_ABS_TOL);
        }
    }

    // 3. Tile / Stripe override sweeps with partial boundary regions
    {
        var prep_scene2 = try common_full.prepareScene2(allocator, io, .tri3);
        defer prep_scene2.deinit(allocator);
        const meshes = common_full.buildScene2Meshes(&prep_scene2, textures);
        const cam_inp = common_full.createScene2Camera(.{ 137, 89 }, 2);

        const tile_overrides = [_]u16{ 64, 128 };
        const stripe_overrides = [_]u16{ 256, 512 };

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        for (tile_overrides) |to_val| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            var run_config = config;
            run_config.save_strategy = .memory;
            run_config.buffer_mode = .global_subpx_full;
            run_config.global_subpx_tile_size_override = to_val;

            const res = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                run_config,
                null,
            );
            const img = res orelse return error.NoResult;
            try std.testing.expect(img.slice.len > 0);
        }

        for (stripe_overrides) |so_val| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            var run_config = config;
            run_config.save_strategy = .memory;
            run_config.buffer_mode = .global_subpx_stripe;
            run_config.global_subpx_stripe_size_override = so_val;

            const res = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                run_config,
                null,
            );
            const img = res orelse return error.NoResult;
            try std.testing.expect(img.slice.len > 0);
        }
    }
}


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
const gengold_io = @import("../gengold/gen_gold_full_image_output.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const imageops = @import("../riley/zig/imageops.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const report = @import("../riley/zig/report.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const NDArray = ndarray.NDArray(F);
const Timestamp = std.Io.Clock.Timestamp;

pub const FULL_IMAGE_OUTPUT_REL_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;
pub const FULL_IMAGE_OUTPUT_ABS_TOL: F = if (F == f32) 1.0e-3 else 1.0e-5;

const save_strategies = [_]rastcfg.SaveStrategy{
    .memory,
    .disk,
    .both,
    .none,
};

const image_save_modes = [_]rastcfg.ImageSaveMode{
    .grey,
    .rgb,
    .multifield,
};

const test_formats = [_]iio.ImageFormat{
    .fimg,
    .csv,
    .bmp,
    .tiff,
};

const report_modes = [_]rastcfg.ReportMode{
    .off,
    .bench,
    .full_stats,
};

const scale_strategies = [_]enum {
    none,
    auto,
    fixed,
    frac,
}{
    .none,
    .auto,
    .fixed,
    .frac,
};

fn runBaseComparison(
    allocator: std.mem.Allocator,
    io: std.Io,
    bc: gengold_io.ImageOutputBaseCase,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        bc.is_rgb,
        bc.is_u16,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes);

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.image_save_mode = if (bc.is_rgb) .rgb else .grey;
    run_config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = 1 },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
        &meshes,
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

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, bc.tag },
    );

    const expected_channels: usize = if (bc.is_rgb) 3 else 1;
    try std.testing.expectEqual(@as(usize, 1), render_result.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), render_result.dims[1]);
    try std.testing.expectEqual(expected_channels, render_result.dims[2]);
    try std.testing.expectEqual(@as(usize, 100), render_result.dims[3]);
    try std.testing.expectEqual(@as(usize, 160), render_result.dims[4]);

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
        expected_channels,
        gold_path,
        FULL_IMAGE_OUTPUT_REL_TOL,
        FULL_IMAGE_OUTPUT_ABS_TOL,
    ) catch |err| {
        const fail_dir_name = try std.fmt.allocPrint(
            aa,
            "full_image_output/{s}",
            .{bc.tag},
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
            expected_channels,
        );
        if (tcfg.TEST_CASE_VERBOSE) {
            std.debug.print("FAIL base {s} ({d:.2} ms)\n", .{ bc.tag, duration_ms });
        }
        return err;
    };

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print("PASS base {s} ({d:.2} ms)\n", .{ bc.tag, duration_ms });
    }
}

fn runFactorialSaveStrategy(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    const temp_dir_base = "temp-tests/image_output_strategy";
    var dir = try orch.openDirEnsured(io, temp_dir_base);
    dir.close(io);

    const meshes = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        false,
        false,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes);

    for (save_strategies) |strat| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        const out_subdir = try std.fmt.allocPrint(
            aa,
            "{s}/{s}",
            .{ temp_dir_base, @tagName(strat) },
        );
        var sub_dir = try orch.openDirEnsured(io, out_subdir);
        sub_dir.close(io);

        var run_config = config;
        run_config.save_strategy = strat;
        run_config.image_save_mode = .grey;
        run_config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .none },
        };

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes,
            run_config,
            out_subdir,
        );

        switch (strat) {
            .memory => {
                try std.testing.expect(result != null);
                if (result) |arr| aa.free(arr.slice);
            },
            .disk => {
                try std.testing.expect(result == null);
            },
            .both => {
                try std.testing.expect(result != null);
                if (result) |arr| aa.free(arr.slice);
            },
            .none => {
                try std.testing.expect(result == null);
            },
        }
    }
}

fn runFactorialFormatsAndModes(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    const temp_dir_base = "temp-tests/image_output_formats";
    var dir = try orch.openDirEnsured(io, temp_dir_base);
    dir.close(io);

    const meshes_mono = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        false,
        false,
    );
    const meshes_rgb = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        true,
        false,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes_mono);

    for (image_save_modes) |mode| {
        for (test_formats) |fmt| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            const out_subdir = try std.fmt.allocPrint(
                aa,
                "{s}/{s}_{s}",
                .{ temp_dir_base, @tagName(mode), @tagName(fmt) },
            );
            var sub_dir = try orch.openDirEnsured(io, out_subdir);
            sub_dir.close(io);

            var run_config = config;
            run_config.save_strategy = .both;
            run_config.image_save_mode = mode;
            const bits: ?u8 = switch (fmt) {
                .bmp, .tiff => 8,
                .fimg, .csv => null,
                .ppm => 8,
            };
            run_config.image_save_opts = &[_]iio.ImageSaveOpts{
                .{ .format = fmt, .bits = bits, .scaling = .none },
            };

            const meshes = if (mode == .rgb) &meshes_rgb else &meshes_mono;
            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = 1 },
            };

            const result = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                meshes,
                run_config,
                out_subdir,
            );

            try std.testing.expect(result != null);
            if (result) |arr| aa.free(arr.slice);
        }
    }
}

fn runFactorialScalingAndReports(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    const meshes = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        false,
        false,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes);

    for (scale_strategies) |scale_strat| {
        for (report_modes) |rep_mode| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const aa = arena.allocator();

            var run_config = config;
            run_config.save_strategy = .memory;
            run_config.image_save_mode = .grey;
            run_config.report = rep_mode;
            const scaling: imageops.ScaleStrategy = switch (scale_strat) {
                .none => .none,
                .auto => .auto,
                .fixed => .{ .fixed = .{ 0.0, 255.0 } },
                .frac => .{ .frac = .{ 0.05, 0.95 } },
            };
            run_config.image_save_opts = &[_]iio.ImageSaveOpts{
                .{ .format = .bmp, .bits = 8, .scaling = scaling },
            };

            const render_groups = [_]riley.RenderGroupSpec{
                .{ .io = io, .workers = 1 },
            };

            const result = try riley.raster(
                aa,
                &render_groups,
                &[_]CameraInput{cam_inp},
                &meshes,
                run_config,
                null,
            );

            try std.testing.expect(result != null);
            if (result) |arr| aa.free(arr.slice);
        }
    }
}

fn runEntryPointEquivalence(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        false,
        false,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes);
    const cam_inps = [_]CameraInput{cam_inp};

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.image_save_mode = .grey;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = 1 },
    };

    // 1. raster (allocates output array)
    const result_raster = try riley.raster(
        aa,
        &render_groups,
        &cam_inps,
        &meshes,
        run_config,
        null,
    );
    const img_raster = result_raster orelse return error.NoResult;

    // 2. rasterReport (allocates output array, captures bench)
    var bench_capt = [_]report.FrameBenchCapture{
        std.mem.zeroes(report.FrameBenchCapture),
    };
    const result_report = try riley.rasterReport(
        aa,
        &render_groups,
        &cam_inps,
        &meshes,
        run_config,
        null,
        bench_capt[0..],
    );
    const img_report = result_report orelse return error.NoResult;

    // 3. rasterInto (renders into pre-allocated output array)
    const dims = try riley.calcAllFramesImageDims(
        &cam_inps,
        &meshes,
        run_config,
    );
    var img_into = try NDArray.initFlat(aa, dims[0..]);
    try riley.rasterInto(
        aa,
        &render_groups,
        &cam_inps,
        &meshes,
        run_config,
        null,
        &img_into,
    );

    // 4. rasterReportInto (renders into pre-allocated array, captures bench)
    var bench_capt_into = [_]report.FrameBenchCapture{
        std.mem.zeroes(report.FrameBenchCapture),
    };
    var img_report_into = try NDArray.initFlat(aa, dims[0..]);
    try riley.rasterReportInto(
        aa,
        &render_groups,
        &cam_inps,
        &meshes,
        run_config,
        null,
        &img_report_into,
        bench_capt_into[0..],
    );

    // Assert exact identical outputs across all four entry points
    try std.testing.expectEqualSlices(usize, img_raster.dims, img_report.dims);
    try std.testing.expectEqualSlices(usize, img_raster.dims, img_into.dims);
    try std.testing.expectEqualSlices(
        usize,
        img_raster.dims,
        img_report_into.dims,
    );

    try std.testing.expectEqualSlices(F, img_raster.slice, img_report.slice);
    try std.testing.expectEqualSlices(F, img_raster.slice, img_into.slice);
    try std.testing.expectEqualSlices(
        F,
        img_raster.slice,
        img_report_into.slice,
    );
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.testing);
    config.background_value = 127.5;
    const gold_dir_root = policy.goldRoot(.full_image_output);

    var textures = try common_full.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    var prep = try common_full.prepareScene2(allocator, io, .tri3);
    defer prep.deinit(allocator);

    for (gengold_io.base_cases) |bc| {
        try runBaseComparison(
            allocator,
            io,
            bc,
            &prep,
            &textures,
            gold_dir_root,
            config,
        );
    }

    try runFactorialSaveStrategy(allocator, io, &prep, &textures, config);
    try runFactorialFormatsAndModes(allocator, io, &prep, &textures, config);
    try runFactorialScalingAndReports(allocator, io, &prep, &textures, config);
    try runEntryPointEquivalence(allocator, io, &prep, &textures, config);
    try runAdditionalImageOutputTests(allocator, io, &prep, &textures, config);
}

fn runAdditionalImageOutputTests(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    const temp_dir_base = "temp-tests/image_output_additional";
    var dir = try orch.openDirEnsured(io, temp_dir_base);
    dir.close(io);

    const meshes_mono = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        false,
        false,
    );
    const meshes_rgb = common_full.buildScene2ImageOutputMeshes(
        prep,
        textures,
        true,
        false,
    );
    const cam_inp = common_full.createScene2ImageOutputCamera(&meshes_mono);

    // 1. Disk vs Memory output reload round-trip
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        const out_subdir = try std.fmt.allocPrint(
            aa,
            "{s}/reload_roundtrip",
            .{temp_dir_base},
        );
        var sub_dir = try orch.openDirEnsured(io, out_subdir);
        sub_dir.close(io);

        var run_config = config;
        run_config.save_strategy = .both;
        run_config.image_save_mode = .grey;
        run_config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .fimg, .bits = null, .scaling = .none },
        };

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_mono,
            run_config,
            out_subdir,
        );
        const mem_img = result orelse return error.NoResult;

        const fimg_path = try std.fmt.allocPrint(
            aa,
            "{s}/cam0_frame0_field0.fimg",
            .{out_subdir},
        );
        var disk_img = try iio.loadFIMG(aa, io, fimg_path);
        defer {
            aa.free(disk_img.slice);
            disk_img.deinit(aa);
        }

        const rows_n = mem_img.dims[3];
        const cols_n = mem_img.dims[4];
        for (0..rows_n) |rr| {
            for (0..cols_n) |cc| {
                const mem_val = mem_img.get(&[_]usize{ 0, 0, 0, rr, cc });
                const disk_val = disk_img.get(&[_]usize{ 0, rr, cc });
                try std.testing.expectEqual(mem_val, disk_val);
            }
        }
    }

    // 2. save_frame_buff_count sweep
    const buff_counts = [_]usize{ 1, 2, 4 };
    for (buff_counts) |bc_val| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        var run_config = config;
        run_config.save_strategy = .memory;
        run_config.save_frame_buff_count = bc_val;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_mono,
            run_config,
            null,
        );
        const img = result orelse return error.NoResult;
        try std.testing.expect(img.slice.len > 0);
    }

    // 3. Field conversions: 1-field expanded RGB, 3-field reduced grey
    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        var run_config = config;
        run_config.save_strategy = .memory;
        run_config.image_save_mode = .rgb;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        // 1-field mono mesh -> rgb output
        const res_expanded = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_mono,
            run_config,
            null,
        );
        const img_expanded = res_expanded orelse return error.NoResult;
        try std.testing.expectEqual(@as(usize, 3), img_expanded.dims[2]);

        // 3-field rgb mesh -> grey output
        run_config.image_save_mode = .grey;
        const res_reduced = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_inp},
            &meshes_rgb,
            run_config,
            null,
        );
        const img_reduced = res_reduced orelse return error.NoResult;
        try std.testing.expectEqual(@as(usize, 1), img_reduced.dims[2]);
    }
}


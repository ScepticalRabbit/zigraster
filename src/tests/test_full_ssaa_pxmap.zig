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
const common_full = @import("../dev_support/fullfixtures.zig");
const common_test = @import("../dev_support/tests.zig");
const fullcase_ssaa_pxmap = @import("fullcase_ssaa_pxmap.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

pub fn runFullSsaaPxmapCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene1Prepared,
    ssaa: u32,
    dist_case: fullcase_ssaa_pxmap.DistCase,
    psf_case: fullcase_ssaa_pxmap.PsfCase,
    pxmap_case: fullcase_ssaa_pxmap.PxMapCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const case_name = try fullcase_ssaa_pxmap.formatSsaaPxmapCaseName(
        aa,
        ssaa,
        dist_case.tag,
        psf_case.tag,
        pxmap_case.tag,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_name },
    );

    const mesh = fullcase_ssaa_pxmap.buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    var camera_input = prep.camera_input;
    camera_input.sub_sample = ssaa;
    camera_input.distortion = dist_case.distortion;
    camera_input.psf = psf_case.psf;
    camera_input.subpixel_center_map = pxmap_case.map_mode;

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = common_full.grey_background_scene1;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
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

    const gold_path = try common_test.findGoldPath(
        aa,
        io,
        gold_dir,
        0,
        0,
        0,
        false,
    );

    common_test.compareNDArrayToGold(
        aa,
        io,
        &render_result,
        0,
        0,
        0,
        1,
        gold_path,
        tcfg.FULL_GOLD_REL_TOL,
        tcfg.FULL_GOLD_ABS_TOL,
    ) catch |err| {
        const fail_dir_name = try std.fmt.allocPrint(
            aa,
            "full_ssaa_pxmap/{s}",
            .{case_name},
        );
        try common_test.saveComparisonArtifactsFromResult(
            aa,
            io,
            common_test.default_fails_root,
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
                .{ case_name, duration_ms },
            );
        }
        return err;
    };

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({d:.2} ms)\n",
            .{ case_name, duration_ms },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.full_ssaa_pxmap);

    var prep = try common_full.prepareScene1(allocator, io);
    defer prep.deinit(allocator);

    for (fullcase_ssaa_pxmap.ssaa_levels) |ssaa| {
        for (fullcase_ssaa_pxmap.dist_cases) |dist_case| {
            for (fullcase_ssaa_pxmap.psf_cases) |psf_case| {
                for (fullcase_ssaa_pxmap.pxmap_cases) |pxmap_case| {
                    try runFullSsaaPxmapCaseTest(
                        allocator,
                        io,
                        &prep,
                        ssaa,
                        dist_case,
                        psf_case,
                        pxmap_case,
                        gold_dir_root,
                        config,
                    );
                }
            }
        }
    }

    try runPxmapEquivalenceTests(allocator, io, &prep, config);
}

fn runPxmapEquivalenceTests(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common_full.Scene1Prepared,
    config: rastcfg.RasterConfig,
) !void {
    const mesh = fullcase_ssaa_pxmap.buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    // 1. No-distortion control: full_in_mem, per_tile, and affine_jac must match exactly
    const pxmap_modes = [_]camera.SubPixelCenterMap{
        .full_in_mem,
        .per_tile,
        .affine_jac,
    };

    var base_renders: [3]?ndarray.NDArray(F) = [_]?ndarray.NDArray(F){ null, null, null };
    var arenas: [3]std.heap.ArenaAllocator = undefined;
    defer {
        for (0..3) |ii| {
            if (base_renders[ii]) |_| arenas[ii].deinit();
        }
    }

    for (pxmap_modes, 0..) |mode, ii| {
        arenas[ii] = std.heap.ArenaAllocator.init(allocator);
        const aa = arenas[ii].allocator();

        var cam_input = prep.camera_input;
        cam_input.sub_sample = 2;
        cam_input.distortion = .none;
        cam_input.psf = .{ .pixel_box = .{} };
        cam_input.subpixel_center_map = mode;

        var run_config = config;
        run_config.save_strategy = .memory;
        run_config.background_value = common_full.grey_background_scene1;

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = 1 },
        };

        const result = try riley.raster(
            aa,
            &render_groups,
            &[_]CameraInput{cam_input},
            &meshes,
            run_config,
            null,
        );

        base_renders[ii] = result;
    }

    const ref_img = base_renders[0] orelse return error.NoResult;
    for (1..3) |ii| {
        const test_img = base_renders[ii] orelse return error.NoResult;
        try std.testing.expectEqualSlices(usize, ref_img.dims, test_img.dims);
        for (ref_img.slice, test_img.slice) |ref_val, test_val| {
            try std.testing.expect(@abs(ref_val - test_val) <= tcfg.FULL_GOLD_ABS_TOL);
        }
    }
}
